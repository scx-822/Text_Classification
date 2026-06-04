from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
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
    read_excel_dataset,
    summarize_dataframe,
)
from src.losses import SupConLoss  # noqa: E402
from src.model import RobertaSupConClassifier  # noqa: E402
from src.train import move_batch_to_device, run_eval  # noqa: E402
from src.utils import compute_metrics, ensure_dir, load_config, save_json, set_seed  # noqa: E402


def train_select_on_train(config_path: str, epochs: int | None, output_dir: str | None):
    cfg = load_config(config_path)
    if epochs is not None:
        cfg["epochs"] = epochs
    if output_dir is not None:
        cfg["output_dir"] = output_dir

    set_seed(int(cfg["seed"]))
    output_dir_path = ensure_dir(cfg["output_dir"])
    checkpoint_dir = ensure_dir(output_dir_path / "checkpoints")
    logs_dir = ensure_dir(output_dir_path / "logs")
    prediction_dir = ensure_dir(output_dir_path / "predictions")

    train_df = read_excel_dataset(cfg["train_path"])
    test_df = read_excel_dataset(cfg["test_path"])
    train_df = apply_configured_text_features(train_df, cfg)
    test_df = apply_configured_text_features(test_df, cfg)
    label_names = get_label_names(train_df)

    save_json(
        {
            "train_all": summarize_dataframe(train_df, label_names),
            "train_as_validation": summarize_dataframe(train_df, label_names),
            "test": summarize_dataframe(test_df, label_names),
            "mode": "train_full_and_select_epoch_on_train",
            "note": "All training rows are also used as the validation criterion. Test data is evaluated after every epoch for diagnostics.",
        },
        logs_dir / "data_summary.json",
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=False)
    train_dataset = TextClassificationDataset(train_df, tokenizer, int(cfg["max_length"]), has_labels=True)
    test_dataset = TextClassificationDataset(test_df, tokenizer, int(cfg["max_length"]), has_labels=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    train_eval_loader = DataLoader(
        train_dataset,
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

    main_metric_name = str(cfg["main_metric"])
    lambda_contrast = float(cfg["lambda_contrast"])
    best_metric = -1.0
    best_macro_f1 = -1.0
    best_epoch = -1
    history = []

    print(
        f"device={device}, mode=train_select_on_train, train={len(train_df)}, "
        f"test={len(test_df)}, epochs={cfg['epochs']}"
    )
    print(f"labels={label_names}")

    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        train_cls_loss = 0.0
        train_contrast_loss = 0.0
        train_examples = 0
        train_valid_anchors = 0

        progress = tqdm(train_loader, desc=f"train-select epoch {epoch}", leave=False)
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

        train_eval = run_eval(model, train_eval_loader, ce_loss_fn, supcon_loss_fn, device, lambda_contrast)
        train_y_true = train_eval.pop("y_true")
        train_y_pred = train_eval.pop("y_pred")
        train_metrics = compute_metrics(train_y_true, train_y_pred, label_names)

        epoch_test_eval = run_eval(model, test_loader, ce_loss_fn, supcon_loss_fn, device, lambda_contrast)
        epoch_test_y_true = epoch_test_eval.pop("y_true")
        epoch_test_y_pred = epoch_test_eval.pop("y_pred")
        epoch_test_metrics = compute_metrics(epoch_test_y_true, epoch_test_y_pred, label_names)
        epoch_checkpoint_path = checkpoint_dir / f"epoch_{epoch:02d}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "label_names": label_names,
                "epoch": epoch,
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "test_accuracy": epoch_test_metrics["accuracy"],
                "test_macro_f1": epoch_test_metrics["macro_f1"],
            },
            epoch_checkpoint_path,
        )
        epoch_pred_df = test_df.copy()
        epoch_pred_df["pred_label_id"] = epoch_test_y_pred
        epoch_pred_df["pred_label"] = [label_names[i] for i in epoch_test_y_pred]
        epoch_xlsx_path = prediction_dir / f"epoch_{epoch:02d}_test_predictions.xlsx"
        epoch_csv_path = prediction_dir / f"epoch_{epoch:02d}_test_predictions.csv"
        epoch_pred_df.to_excel(epoch_xlsx_path, index=False)
        epoch_pred_df.to_csv(
            epoch_csv_path,
            index=False,
            encoding="utf-8-sig",
        )

        current_metric = float(train_metrics[main_metric_name])
        current_macro_f1 = float(train_metrics["macro_f1"])
        is_best = (current_metric, current_macro_f1) > (best_metric, best_macro_f1)
        if is_best:
            best_metric = current_metric
            best_macro_f1 = current_macro_f1
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "label_names": label_names,
                    "best_epoch": best_epoch,
                    "best_metric_name": main_metric_name,
                    "best_metric": best_metric,
                    "best_macro_f1": best_macro_f1,
                },
                checkpoint_dir / "best_train_model.pt",
            )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss / max(train_examples, 1),
            "train_classification_loss": train_cls_loss / max(train_examples, 1),
            "train_contrastive_loss": train_contrast_loss / max(train_examples, 1),
            "train_valid_contrastive_anchors": int(train_valid_anchors),
            "train_eval": train_eval,
            "train_metrics": train_metrics,
            "epoch_test_eval": epoch_test_eval,
            "epoch_test_metrics": epoch_test_metrics,
            "learning_rate": scheduler.get_last_lr()[0],
            "checkpoint_path": str(epoch_checkpoint_path),
            "prediction_xlsx_path": str(epoch_xlsx_path),
            "prediction_csv_path": str(epoch_csv_path),
            "is_best_train_epoch": is_best,
        }
        history.append(epoch_record)
        save_json(
            {
                "history": history,
                "best_epoch": best_epoch,
                "best_train_metric_name": main_metric_name,
                "best_train_metric": best_metric,
                "best_train_macro_f1": best_macro_f1,
            },
            logs_dir / "training_history.json",
        )

        print(
            f"epoch={epoch} train_loss={epoch_record['train_loss']:.4f} "
            f"train_{main_metric_name}={current_metric:.4f} "
            f"train_macro_f1={current_macro_f1:.4f} "
            f"test_accuracy={epoch_test_metrics['accuracy']:.4f} "
            f"test_macro_f1={epoch_test_metrics['macro_f1']:.4f}"
        )

    best_checkpoint = torch.load(checkpoint_dir / "best_train_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    test_eval = run_eval(model, test_loader, ce_loss_fn, supcon_loss_fn, device, lambda_contrast)
    test_y_true = test_eval.pop("y_true")
    test_y_pred = test_eval.pop("y_pred")
    test_metrics = compute_metrics(test_y_true, test_y_pred, label_names)
    test_report = {
        "best_epoch": best_epoch,
        "best_train_metric_name": main_metric_name,
        "best_train_metric": best_metric,
        "best_train_macro_f1": best_macro_f1,
        "test_eval": test_eval,
        "test_metrics": test_metrics,
        "note": "Each epoch was evaluated on test for diagnostics. The final reported checkpoint is selected by train-set metrics.",
    }
    save_json(test_report, logs_dir / "test_metrics.json")

    pred_df = test_df.copy()
    pred_df["pred_label_id"] = test_y_pred
    pred_df["pred_label"] = [label_names[i] for i in test_y_pred]
    pred_df.to_excel(prediction_dir / "train_selected_test_predictions.xlsx", index=False)
    pred_df.to_csv(prediction_dir / "train_selected_test_predictions.csv", index=False, encoding="utf-8-sig")

    print(
        f"best_epoch={best_epoch}, best_train_{main_metric_name}={best_metric:.4f}, "
        f"best_train_macro_f1={best_macro_f1:.4f}"
    )
    print(f"test_accuracy={test_metrics['accuracy']:.4f}, test_macro_f1={test_metrics['macro_f1']:.4f}")
    print(f"saved outputs to {output_dir_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/roberta_supcon_cls_joint_tuned.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_select_on_train(args.config, args.epochs, args.output_dir)
