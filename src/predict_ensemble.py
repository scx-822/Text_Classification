from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import TextClassificationDataset, apply_configured_text_features, read_excel_dataset  # noqa: E402
from src.model import RobertaSupConClassifier  # noqa: E402
from src.train import move_batch_to_device  # noqa: E402
from src.utils import compute_metrics, ensure_dir, save_json  # noqa: E402


def parse_checkpoint_paths(value: str) -> list[Path]:
    return [Path(item.strip()) for item in value.split(",") if item.strip()]


def predict_probs_for_checkpoint(checkpoint_path: Path, raw_df):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    label_names = checkpoint["label_names"]

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=False)
    df = apply_configured_text_features(raw_df, cfg)
    dataset = TextClassificationDataset(df, tokenizer, int(cfg["max_length"]), has_labels="label_id" in df.columns)
    loader = DataLoader(dataset, batch_size=int(cfg["eval_batch_size"]), shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RobertaSupConClassifier(
        model_name=cfg["model_name"],
        num_labels=len(label_names),
        dropout=float(cfg["dropout"]),
        contrastive_dim=int(cfg["contrastive_dim"]),
        pooling=str(cfg["pooling"]),
        use_location_priority_branch=bool(cfg.get("use_location_priority_branch", False)),
        location_priority_embedding_dim=int(cfg.get("location_priority_embedding_dim", 8)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    chunks = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
                location_priority_ids=batch.get("location_priority_ids"),
            )
            chunks.append(torch.softmax(outputs["logits"], dim=1).cpu().numpy())
    return np.concatenate(chunks, axis=0), label_names


def predict_ensemble(checkpoints: list[Path], data_path: str, output_dir: str):
    raw_df = read_excel_dataset(data_path)
    prob_sum = None
    label_names = None

    for checkpoint_path in checkpoints:
        probs, current_label_names = predict_probs_for_checkpoint(checkpoint_path, raw_df)
        if label_names is None:
            label_names = current_label_names
        elif label_names != current_label_names:
            raise ValueError(f"Label names mismatch in {checkpoint_path}")
        prob_sum = probs if prob_sum is None else prob_sum + probs

    if prob_sum is None or label_names is None:
        raise ValueError("No checkpoints were provided.")

    probs = prob_sum / len(checkpoints)
    pred = probs.argmax(axis=1).tolist()
    output_dir_path = ensure_dir(output_dir)
    metrics = None
    if "label_id" in raw_df.columns:
        metrics = compute_metrics(raw_df["label_id"].astype(int).tolist(), pred, label_names)
        save_json(metrics, output_dir_path / "ensemble_metrics.json")
        print(f"accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}")

    result = raw_df.copy()
    result["pred_label_id"] = pred
    result["pred_label"] = [label_names[i] for i in pred]
    for label_id, label_name in enumerate(label_names):
        result[f"prob_{label_name}"] = probs[:, label_id]
    result.to_excel(output_dir_path / "checkpoint_ensemble_predictions.xlsx", index=False)
    result.to_csv(output_dir_path / "checkpoint_ensemble_predictions.csv", index=False, encoding="utf-8-sig")

    if metrics is not None:
        wrong = result[result["label_id"] != result["pred_label_id"]].copy()
        wrong.to_excel(output_dir_path / "checkpoint_ensemble_wrong_predictions.xlsx", index=False)
        wrong.to_csv(output_dir_path / "checkpoint_ensemble_wrong_predictions.csv", index=False, encoding="utf-8-sig")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--data", default="Dataset/roberta_test.xlsx")
    parser.add_argument("--output-dir", default="outputs/checkpoint_ensemble")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict_ensemble(parse_checkpoint_paths(args.checkpoints), args.data, args.output_dir)
