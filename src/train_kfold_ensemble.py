from __future__ import annotations

import argparse
import math
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import TextClassificationDataset, apply_configured_text_features, get_label_names, read_excel_dataset  # noqa: E402
from src.losses import SupConLoss  # noqa: E402
from src.model import RobertaSupConClassifier  # noqa: E402
from src.train import move_batch_to_device, run_eval  # noqa: E402
from src.utils import compute_metrics, ensure_dir, load_config, save_json, set_seed  # noqa: E402


def parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_splits(df: pd.DataFrame, n_splits: int, seed: int, strategy: str):
    y = df["label_id"].astype(int).to_numpy()
    if "group" in strategy and "leakage_group_id" in df.columns:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(df, y, groups=df["leakage_group_id"]))

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(df, y))


def predict_probabilities(model, dataloader, device) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
                location_priority_ids=batch.get("location_priority_ids"),
            )
            probs = torch.softmax(outputs["logits"], dim=1)
            chunks.append(probs.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def train_one_fold(
    cfg: dict,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_names: list[str],
    tokenizer,
    fold_output_dir: Path,
    seed: int,
    fold_index: int,
):
    set_seed(seed)
    checkpoint_dir = ensure_dir(fold_output_dir / "checkpoints")
    logs_dir = ensure_dir(fold_output_dir / "logs")

    train_dataset = TextClassificationDataset(train_df, tokenizer, int(cfg["max_length"]), has_labels=True)
    valid_dataset = TextClassificationDataset(valid_df, tokenizer, int(cfg["max_length"]), has_labels=True)
    test_dataset = TextClassificationDataset(test_df, tokenizer, int(cfg["max_length"]), has_labels=True)

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
    test_loader = DataLoader(
        test_dataset,
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
    best_epoch = -1
    best_state_dict = None
    bad_epochs = 0
    patience = int(cfg["early_stopping_patience"])
    lambda_contrast = float(cfg["lambda_contrast"])
    history = []

    print(
        f"member seed={seed} fold={fold_index} device={device} "
        f"train={len(train_df)} valid={len(valid_df)}"
    )

    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        train_cls_loss = 0.0
        train_contrast_loss = 0.0
        train_examples = 0
        train_valid_anchors = 0

        progress = tqdm(train_loader, desc=f"s{seed} f{fold_index} e{epoch}", leave=False)
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
            progress.set_postfix(loss=f"{train_loss / max(train_examples, 1):.4f}")

        valid_eval = run_eval(model, valid_loader, ce_loss_fn, supcon_loss_fn, device, lambda_contrast)
        valid_metrics = compute_metrics(valid_eval.pop("y_true"), valid_eval.pop("y_pred"), label_names)
        current_metric = float(valid_metrics[best_metric_name])

        record = {
            "epoch": epoch,
            "train_loss": train_loss / max(train_examples, 1),
            "train_classification_loss": train_cls_loss / max(train_examples, 1),
            "train_contrastive_loss": train_contrast_loss / max(train_examples, 1),
            "train_valid_contrastive_anchors": int(train_valid_anchors),
            "valid_eval": valid_eval,
            "valid_metrics": valid_metrics,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(record)
        print(
            f"seed={seed} fold={fold_index} epoch={epoch} "
            f"valid_{best_metric_name}={current_metric:.4f} "
            f"valid_macro_f1={valid_metrics['macro_f1']:.4f}"
        )

        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            bad_epochs = 0
            best_state_dict = deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state_dict is None:
        raise RuntimeError("No checkpoint was selected.")

    model.load_state_dict(best_state_dict)
    checkpoint = {
        "model_state_dict": best_state_dict,
        "config": cfg,
        "label_names": label_names,
        "seed": seed,
        "fold_index": fold_index,
        "best_epoch": best_epoch,
        "best_metric_name": best_metric_name,
        "best_metric": best_metric,
    }
    torch.save(checkpoint, checkpoint_dir / "best_model.pt")
    save_json({"history": history, "best_epoch": best_epoch, "best_metric": best_metric}, logs_dir / "training_history.json")

    valid_probs = predict_probabilities(model, valid_loader, device)
    test_probs = predict_probabilities(model, test_loader, device)
    return {
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "valid_probs": valid_probs,
        "test_probs": test_probs,
    }


def run_ensemble(config_path: str, folds: int, seeds: list[int], output_dir: str | None, use_plain_split: bool):
    cfg = load_config(config_path)
    if output_dir:
        cfg["output_dir"] = output_dir
    output_dir_path = ensure_dir(cfg["output_dir"])
    logs_dir = ensure_dir(output_dir_path / "logs")
    prediction_dir = ensure_dir(output_dir_path / "predictions")
    members_dir = ensure_dir(output_dir_path / "members")

    raw_train_df = read_excel_dataset(cfg["train_path"])
    raw_test_df = read_excel_dataset(cfg["test_path"])
    label_names = get_label_names(raw_train_df)
    metadata_columns = cfg.get("metadata_columns", [])
    train_df = apply_configured_text_features(raw_train_df, cfg)
    test_df = apply_configured_text_features(raw_test_df, cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=False)
    num_labels = len(label_names)
    test_prob_sum = np.zeros((len(test_df), num_labels), dtype=np.float64)
    oof_prob_sum = np.zeros((len(train_df), num_labels), dtype=np.float64)
    oof_counts = np.zeros(len(train_df), dtype=np.int64)
    fold_records = []
    member_count = 0

    split_strategy = "stratified_kfold" if use_plain_split else str(cfg["split_strategy"])
    for seed in seeds:
        splits = build_splits(train_df, folds, seed, split_strategy)
        for fold_index, (train_idx, valid_idx) in enumerate(splits, start=1):
            member_count += 1
            fold_output_dir = ensure_dir(members_dir / f"seed_{seed}_fold_{fold_index}")
            result = train_one_fold(
                cfg=cfg,
                train_df=train_df.iloc[train_idx].reset_index(drop=True),
                valid_df=train_df.iloc[valid_idx].reset_index(drop=True),
                test_df=test_df,
                label_names=label_names,
                tokenizer=tokenizer,
                fold_output_dir=fold_output_dir,
                seed=seed + fold_index,
                fold_index=fold_index,
            )
            test_prob_sum += result["test_probs"]
            oof_prob_sum[valid_idx] += result["valid_probs"]
            oof_counts[valid_idx] += 1
            fold_records.append(
                {
                    "seed": seed,
                    "fold_index": fold_index,
                    "best_epoch": result["best_epoch"],
                    "best_valid_metric": result["best_metric"],
                    "train_size": int(len(train_idx)),
                    "valid_size": int(len(valid_idx)),
                }
            )

    test_probs = test_prob_sum / max(member_count, 1)
    test_pred = test_probs.argmax(axis=1).tolist()
    test_metrics = compute_metrics(raw_test_df["label_id"].astype(int).tolist(), test_pred, label_names)

    covered = oof_counts > 0
    oof_probs = np.zeros_like(oof_prob_sum)
    oof_probs[covered] = oof_prob_sum[covered] / oof_counts[covered, None]
    oof_pred = oof_probs.argmax(axis=1)
    oof_metrics = compute_metrics(
        raw_train_df.loc[covered, "label_id"].astype(int).tolist(),
        oof_pred[covered].tolist(),
        label_names,
    )

    summary = {
        "config_path": config_path,
        "folds": folds,
        "seeds": seeds,
        "member_count": member_count,
        "split_strategy": split_strategy,
        "metadata_columns": metadata_columns,
        "fold_records": fold_records,
        "oof_metrics": oof_metrics,
        "test_metrics": test_metrics,
    }
    save_json(summary, logs_dir / "ensemble_metrics.json")

    pred_df = raw_test_df.copy()
    pred_df["pred_label_id"] = test_pred
    pred_df["pred_label"] = [label_names[i] for i in test_pred]
    for label_id, label_name in enumerate(label_names):
        pred_df[f"prob_{label_name}"] = test_probs[:, label_id]
    pred_df.to_excel(prediction_dir / "kfold_ensemble_test_predictions.xlsx", index=False)
    pred_df.to_csv(prediction_dir / "kfold_ensemble_test_predictions.csv", index=False, encoding="utf-8-sig")

    wrong_df = pred_df[pred_df["label_id"] != pred_df["pred_label_id"]].copy()
    wrong_df.to_excel(prediction_dir / "kfold_ensemble_wrong_predictions.xlsx", index=False)
    wrong_df.to_csv(prediction_dir / "kfold_ensemble_wrong_predictions.csv", index=False, encoding="utf-8-sig")

    print(f"members={member_count}")
    print(f"oof_accuracy={oof_metrics['accuracy']:.4f}, oof_macro_f1={oof_metrics['macro_f1']:.4f}")
    print(f"test_accuracy={test_metrics['accuracy']:.4f}, test_macro_f1={test_metrics['macro_f1']:.4f}")
    print(f"saved outputs to {output_dir_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/roberta_supcon_cls.yaml")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--plain-split", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_ensemble(
        config_path=args.config,
        folds=args.folds,
        seeds=parse_seeds(args.seeds),
        output_dir=args.output_dir,
        use_plain_split=args.plain_split,
    )
