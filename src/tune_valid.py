from __future__ import annotations

import argparse
import itertools
import math
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import (  # noqa: E402
    TextClassificationDataset,
    apply_configured_text_features,
    get_label_names,
    group_overlap,
    read_excel_dataset,
    split_train_valid,
    summarize_dataframe,
)
from src.losses import SupConLoss  # noqa: E402
from src.model import RobertaSupConClassifier  # noqa: E402
from src.train import move_batch_to_device, run_eval  # noqa: E402
from src.utils import compute_metrics, ensure_dir, load_config, save_json, set_seed  # noqa: E402


def build_trials(search_space: dict) -> list[dict]:
    keys = list(search_space)
    values = [search_space[key] for key in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def merge_trial_config(base_cfg: dict, trial: dict, output_dir: Path, trial_index: int) -> dict:
    cfg = deepcopy(base_cfg)
    cfg.update(trial)
    cfg["output_dir"] = str(output_dir / f"trial_{trial_index:03d}")
    return cfg


def train_valid_only(cfg: dict, train_all_df: pd.DataFrame, label_names: list[str], tokenizer) -> dict:
    set_seed(int(cfg["seed"]))
    output_dir = ensure_dir(cfg["output_dir"])
    logs_dir = ensure_dir(output_dir / "logs")

    split = split_train_valid(
        train_all_df,
        validation_split=float(cfg["validation_split"]),
        seed=int(cfg["seed"]),
        strategy=str(cfg["split_strategy"]),
    )
    save_json(
        {
            "train_all": summarize_dataframe(train_all_df, label_names),
            "train_split": summarize_dataframe(split.train_df, label_names),
            "valid_split": summarize_dataframe(split.valid_df, label_names),
            "split_method": split.split_method,
            "group_overlap": group_overlap(split.train_df, split.valid_df),
        },
        logs_dir / "data_summary.json",
    )

    train_dataset = TextClassificationDataset(split.train_df, tokenizer, int(cfg["max_length"]), has_labels=True)
    valid_dataset = TextClassificationDataset(split.valid_df, tokenizer, int(cfg["max_length"]), has_labels=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(cfg["eval_batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RobertaSupConClassifier(
        model_name=cfg["model_name"],
        num_labels=len(label_names),
        dropout=float(cfg["dropout"]),
        contrastive_dim=int(cfg["contrastive_dim"]),
        pooling=str(cfg["pooling"]),
        use_location_priority_branch=bool(cfg.get("use_location_priority_branch", False)),
        location_priority_embedding_dim=int(cfg.get("location_priority_embedding_dim", 8)),
    ).to(device)

    ce_loss_fn = nn.CrossEntropyLoss()
    supcon_loss_fn = SupConLoss(temperature=float(cfg["temperature"]))
    optimizer = AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))

    gradient_accumulation_steps = int(cfg["gradient_accumulation_steps"])
    update_steps_per_epoch = math.ceil(len(train_loader) / gradient_accumulation_steps)
    total_training_steps = update_steps_per_epoch * int(cfg["epochs"])
    warmup_steps = int(total_training_steps * float(cfg["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    best_metric_name = str(cfg["main_metric"])
    best_metric = -1.0
    best_macro_f1 = -1.0
    best_epoch = -1
    best_metrics = None
    best_eval = None
    patience = int(cfg["early_stopping_patience"])
    bad_epochs = 0
    history = []
    lambda_contrast = float(cfg["lambda_contrast"])

    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        train_cls_loss = 0.0
        train_contrast_loss = 0.0
        train_examples = 0
        train_valid_anchors = 0

        progress = tqdm(train_loader, desc=f"trial epoch {epoch}", leave=False)
        for step, batch in enumerate(progress, start=1):
            batch = move_batch_to_device(batch, device)
            labels = batch["labels"]
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
                location_priority_ids=batch.get("location_priority_ids"),
            )
            cls_loss = ce_loss_fn(outputs["logits"], labels)
            contrast_loss, valid_anchors = supcon_loss_fn(outputs["contrastive_embedding"], labels)
            loss = cls_loss + lambda_contrast * contrast_loss
            (loss / gradient_accumulation_steps).backward()

            if step % gradient_accumulation_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["max_grad_norm"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            batch_size = labels.size(0)
            train_loss += float(loss.item()) * batch_size
            train_cls_loss += float(cls_loss.item()) * batch_size
            train_contrast_loss += float(contrast_loss.item()) * batch_size
            train_examples += batch_size
            train_valid_anchors += valid_anchors

        valid_eval = run_eval(model, valid_loader, ce_loss_fn, supcon_loss_fn, device, lambda_contrast)
        valid_metrics = compute_metrics(valid_eval.pop("y_true"), valid_eval.pop("y_pred"), label_names)
        current_metric = float(valid_metrics[best_metric_name])
        current_macro_f1 = float(valid_metrics["macro_f1"])
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss / max(train_examples, 1),
            "train_classification_loss": train_cls_loss / max(train_examples, 1),
            "train_contrastive_loss": train_contrast_loss / max(train_examples, 1),
            "train_valid_contrastive_anchors": int(train_valid_anchors),
            "valid_eval": valid_eval,
            "valid_metrics": valid_metrics,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(epoch_record)

        if (current_metric, current_macro_f1) > (best_metric, best_macro_f1):
            best_metric = current_metric
            best_macro_f1 = current_macro_f1
            best_epoch = epoch
            best_metrics = valid_metrics
            best_eval = valid_eval
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    result = {
        "config": cfg,
        "best_epoch": best_epoch,
        "best_valid_metric_name": best_metric_name,
        "best_valid_metric": best_metric,
        "best_valid_macro_f1": best_macro_f1,
        "best_valid_eval": best_eval,
        "best_valid_metrics": best_metrics,
        "history": history,
    }
    save_json(result, logs_dir / "valid_tuning_result.json")
    return result


def get_search_space(search_mode: str) -> dict:
    if search_mode == "default":
        return {
            "learning_rate": [1.0e-5, 1.5e-5, 2.0e-5, 3.0e-5],
            "lambda_contrast": [0.05, 0.1, 0.2],
            "temperature": [0.07, 0.1],
            "dropout": [0.1],
            "gradient_accumulation_steps": [2],
        }
    if search_mode == "max_length_only":
        return {
            "max_length": [96, 128, 192, 256, 320],
        }
    if search_mode == "joint_compact":
        return {
            "learning_rate": [2.0e-5, 3.0e-5],
            "lambda_contrast": [0.1, 0.2],
            "temperature": [0.07, 0.1],
            "dropout": [0.1, 0.2],
            "gradient_accumulation_steps": [1, 2],
            "max_length": [192, 256],
        }
    raise ValueError(f"Unsupported search mode: {search_mode}")


def tune(config_path: str, output_dir: str, search_mode: str):
    base_cfg = load_config(config_path)
    base_cfg["pooling"] = "cls"
    base_cfg.pop("use_location_priority", None)
    base_cfg.pop("use_location_priority_branch", None)
    base_cfg.pop("metadata_columns", None)

    output_dir_path = ensure_dir(output_dir)
    logs_dir = ensure_dir(output_dir_path / "logs")
    configs_dir = ensure_dir(output_dir_path / "configs")
    tuned_name = output_dir_path.name
    if tuned_name.endswith("_tuning"):
        tuned_name = f"{tuned_name[:-len('_tuning')]}_tuned"
    else:
        tuned_name = f"{Path(config_path).stem}_tuned"

    search_space = get_search_space(search_mode)
    trials = build_trials(search_space)

    train_all_df = apply_configured_text_features(read_excel_dataset(base_cfg["train_path"]), base_cfg)
    label_names = get_label_names(train_all_df)
    tokenizer = AutoTokenizer.from_pretrained(base_cfg["model_name"], use_fast=False)

    records = []
    best_record = None
    for trial_index, trial in enumerate(trials, start=1):
        cfg = merge_trial_config(base_cfg, trial, output_dir_path / "trials", trial_index)
        print(
            f"trial={trial_index}/{len(trials)} "
            f"lr={cfg['learning_rate']} lambda={cfg['lambda_contrast']} "
            f"tau={cfg['temperature']} dropout={cfg['dropout']} "
            f"accum={cfg['gradient_accumulation_steps']} max_length={cfg['max_length']}"
        )
        result = train_valid_only(cfg, train_all_df, label_names, tokenizer)
        record = {
            "trial_index": trial_index,
            "trial": trial,
            "best_epoch": result["best_epoch"],
            "best_valid_metric_name": result["best_valid_metric_name"],
            "best_valid_metric": result["best_valid_metric"],
            "best_valid_macro_f1": result["best_valid_macro_f1"],
            "output_dir": cfg["output_dir"],
        }
        records.append(record)
        save_json({"records": records}, logs_dir / "tuning_records.json")
        print(
            f"trial={trial_index} best_epoch={record['best_epoch']} "
            f"valid_acc={record['best_valid_metric']:.4f} "
            f"valid_macro_f1={record['best_valid_macro_f1']:.4f}"
        )
        if best_record is None or (
            record["best_valid_metric"],
            record["best_valid_macro_f1"],
            -record["trial_index"],
        ) > (
            best_record["best_valid_metric"],
            best_record["best_valid_macro_f1"],
            -best_record["trial_index"],
        ):
            best_record = record

    if best_record is None:
        raise RuntimeError("No tuning trial completed.")

    best_cfg = deepcopy(base_cfg)
    best_cfg.update(best_record["trial"])
    best_cfg["output_dir"] = f"outputs/{tuned_name}"
    with open(configs_dir / f"{tuned_name}.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(best_cfg, f, allow_unicode=True, sort_keys=False)
    with open(ROOT / "configs" / f"{tuned_name}.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(best_cfg, f, allow_unicode=True, sort_keys=False)

    save_json(
        {
            "search_space": search_space,
            "search_mode": search_mode,
            "best_record": best_record,
            "records": records,
            "note": "This tuning run used only the training file and its validation split. Test data was not loaded.",
        },
        logs_dir / "tuning_summary.json",
    )
    print("best trial:", best_record)
    print(f"saved best config to configs/{tuned_name}.yaml")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/roberta_supcon_cls.yaml")
    parser.add_argument("--output-dir", default="outputs/roberta_supcon_cls_tuning")
    parser.add_argument(
        "--search-mode",
        choices=["default", "max_length_only", "joint_compact"],
        default="default",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tune(args.config, args.output_dir, args.search_mode)
