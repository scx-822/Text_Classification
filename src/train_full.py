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


def train_full(config_path: str, epochs: int | None, lambda_contrast: float | None, output_dir: str | None):
    cfg = load_config(config_path)
    if epochs is not None:
        cfg["epochs"] = epochs
    if lambda_contrast is not None:
        cfg["lambda_contrast"] = lambda_contrast
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
            "test": summarize_dataframe(test_df, label_names),
            "mode": "full_train_no_validation",
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

    lambda_contrast_value = float(cfg["lambda_contrast"])
    history = []
    print(
        f"device={device}, mode=full_train, train={len(train_df)}, test={len(test_df)}, "
        f"epochs={cfg['epochs']}, lambda={lambda_contrast_value}"
    )

    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        train_cls_loss = 0.0
        train_contrast_loss = 0.0
        train_examples = 0
        train_valid_anchors = 0

        progress = tqdm(train_loader, desc=f"full epoch {epoch}", leave=False)
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
            loss = cls_loss + lambda_contrast_value * contrast_loss
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

        test_eval = run_eval(model, test_loader, ce_loss_fn, supcon_loss_fn, device, lambda_contrast_value)
        y_true = test_eval.pop("y_true")
        y_pred = test_eval.pop("y_pred")
        test_metrics = compute_metrics(y_true, y_pred, label_names)
        epoch_checkpoint_path = checkpoint_dir / f"epoch_{epoch:02d}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "label_names": label_names,
                "epoch": epoch,
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
            },
            epoch_checkpoint_path,
        )

        epoch_pred_df = test_df.copy()
        epoch_pred_df["pred_label_id"] = y_pred
        epoch_pred_df["pred_label"] = [label_names[i] for i in y_pred]
        epoch_xlsx_path = prediction_dir / f"epoch_{epoch:02d}_test_predictions.xlsx"
        epoch_csv_path = prediction_dir / f"epoch_{epoch:02d}_test_predictions.csv"
        epoch_pred_df.to_excel(epoch_xlsx_path, index=False)
        epoch_pred_df.to_csv(epoch_csv_path, index=False, encoding="utf-8-sig")

        record = {
            "epoch": epoch,
            "train_loss": train_loss / max(train_examples, 1),
            "train_classification_loss": train_cls_loss / max(train_examples, 1),
            "train_contrastive_loss": train_contrast_loss / max(train_examples, 1),
            "train_valid_contrastive_anchors": int(train_valid_anchors),
            "test_eval": test_eval,
            "test_metrics": test_metrics,
            "learning_rate": scheduler.get_last_lr()[0],
            "checkpoint_path": str(epoch_checkpoint_path),
            "prediction_xlsx_path": str(epoch_xlsx_path),
            "prediction_csv_path": str(epoch_csv_path),
        }
        history.append(record)
        save_json({"history": history}, logs_dir / "full_training_history.json")
        print(
            f"epoch={epoch} train_loss={record['train_loss']:.4f} "
            f"test_acc={test_metrics['accuracy']:.4f} test_macro_f1={test_metrics['macro_f1']:.4f}"
        )

    final_record = history[-1]
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "label_names": label_names,
        "epoch": int(cfg["epochs"]),
        "test_accuracy": final_record["test_metrics"]["accuracy"],
        "test_macro_f1": final_record["test_metrics"]["macro_f1"],
    }
    torch.save(checkpoint, checkpoint_dir / "final_model.pt")
    save_json({"history": history}, logs_dir / "full_training_history.json")
    save_json(final_record, logs_dir / "test_metrics.json")

    test_pred = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
                location_priority_ids=batch.get("location_priority_ids"),
            )
            test_pred.extend(outputs["logits"].argmax(dim=1).detach().cpu().tolist())

    pred_df = test_df.copy()
    pred_df["pred_label_id"] = test_pred
    pred_df["pred_label"] = [label_names[i] for i in test_pred]
    pred_df.to_excel(prediction_dir / "full_train_test_predictions.xlsx", index=False)
    pred_df.to_csv(prediction_dir / "full_train_test_predictions.csv", index=False, encoding="utf-8-sig")
    print(
        f"final test_accuracy={final_record['test_metrics']['accuracy']:.4f}, "
        f"test_macro_f1={final_record['test_metrics']['macro_f1']:.4f}"
    )
    print(f"saved outputs to {output_dir_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/roberta_supcon_cls.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lambda-contrast", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_full(args.config, args.epochs, args.lambda_contrast, args.output_dir)
