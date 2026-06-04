from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import TextClassificationDataset, apply_configured_text_features, read_excel_dataset  # noqa: E402
from src.model import RobertaSupConClassifier  # noqa: E402
from src.utils import ensure_dir  # noqa: E402


def predict(checkpoint_path: str, data_path: str, output_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    label_names = checkpoint["label_names"]

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=False)
    df = read_excel_dataset(data_path)
    df = apply_configured_text_features(df, cfg)
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

    predictions = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            location_priority_ids = batch.get("location_priority_ids")
            if location_priority_ids is not None:
                location_priority_ids = location_priority_ids.to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                location_priority_ids=location_priority_ids,
            )
            predictions.extend(outputs["logits"].argmax(dim=1).cpu().tolist())

    result = df.copy()
    result["pred_label_id"] = predictions
    result["pred_label"] = [label_names[i] for i in predictions]
    output_path = Path(output_path) # type: ignore
    ensure_dir(output_path.parent) # type: ignore
    if output_path.suffix.lower() == ".csv": # type: ignore
        result.to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        result.to_excel(output_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/roberta_supcon_cls/checkpoints/best_model.pt")
    parser.add_argument("--data", default="Dataset/roberta_test.xlsx")
    parser.add_argument("--output", default="outputs/predictions/predictions.xlsx")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(args.checkpoint, args.data, args.output)
