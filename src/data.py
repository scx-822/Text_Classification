from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import Dataset


REQUIRED_COLUMNS = ["text", "label", "label_id"]
LOCATION_PRIORITY_TO_ID = {"low": 0, "medium": 1, "high": 2}


@dataclass
class SplitResult:
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    split_method: str


class TextClassificationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int, has_labels: bool = True):
        self.texts = df["text"].astype(str).tolist()
        self.has_labels = has_labels
        self.labels = df["label_id"].astype(int).tolist() if has_labels and "label_id" in df.columns else None
        self.orig_ids = df["orig_id"].tolist() if "orig_id" in df.columns else list(range(len(df)))
        if "location_priority" in df.columns:
            self.location_priority_ids = (
                df["location_priority"]
                .map(LOCATION_PRIORITY_TO_ID)
                .fillna(LOCATION_PRIORITY_TO_ID["medium"])
                .astype(int)
                .tolist()
            )
        else:
            self.location_priority_ids = None
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoded = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["orig_id"] = torch.tensor(int(self.orig_ids[idx]), dtype=torch.long)
        if self.location_priority_ids is not None:
            item["location_priority_ids"] = torch.tensor(self.location_priority_ids[idx], dtype=torch.long)
        if self.has_labels:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def read_excel_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_excel(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    df = df.copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""].reset_index(drop=True)
    df["label_id"] = df["label_id"].astype(int)
    return df


def add_metadata_to_text(df: pd.DataFrame, metadata_columns: list[str] | None = None) -> pd.DataFrame:
    if not metadata_columns:
        return df

    available_columns = [col for col in metadata_columns if col in df.columns]
    if not available_columns:
        return df

    df = df.copy()

    def build_text(row: pd.Series) -> str:
        metadata_parts = []
        for col in available_columns:
            value = row[col]
            if pd.notna(value) and str(value).strip():
                metadata_parts.append(f"[{col}] {str(value).strip()}")
        if not metadata_parts:
            return str(row["text"]).strip()
        return " ".join(metadata_parts + [f"[text] {str(row['text']).strip()}"])

    df["text"] = df.apply(build_text, axis=1)
    return df


LOW_PRIORITY_LOCATION_KEYWORDS = [
    "管片厂",
    "内业",
    "资料",
    "设计文件",
    "料库",
    "拌和站",
    "试验室",
    "加工场",
    "存放场",
    "材料",
    "交底",
]

HIGH_PRIORITY_LOCATION_KEYWORDS = [
    "营业线",
    "基坑",
    "隧道",
    "盾构",
    "桥",
    "路基",
    "支架",
    "竖井",
    "梁",
    "墩",
]


def infer_location_priority(location: object) -> str:
    if pd.isna(location):
        return "medium"

    text = str(location)
    if any(keyword in text for keyword in LOW_PRIORITY_LOCATION_KEYWORDS):
        return "low"
    if any(keyword in text for keyword in HIGH_PRIORITY_LOCATION_KEYWORDS):
        return "high"
    return "medium"


def add_location_priority_feature(df: pd.DataFrame, enabled: bool = False) -> pd.DataFrame:
    if not enabled or "location" not in df.columns:
        return df

    df = df.copy()
    df["location_priority"] = df["location"].apply(infer_location_priority)
    return df


def add_location_priority_to_text(
    df: pd.DataFrame,
    enabled: bool = False,
    include_location: bool = False,
) -> pd.DataFrame:
    if not enabled or "location" not in df.columns:
        return df

    df = add_location_priority_feature(df, enabled=True)

    def build_text(row: pd.Series) -> str:
        priority = row["location_priority"]
        prefix_parts = [f"[location_priority] {priority}"]
        if include_location and pd.notna(row["location"]) and str(row["location"]).strip():
            prefix_parts.append(f"[location] {str(row['location']).strip()}")
        return " ".join(prefix_parts + [f"[text] {str(row['text']).strip()}"])

    df["text"] = df.apply(build_text, axis=1)
    return df


def apply_configured_text_features(df: pd.DataFrame, cfg: Mapping) -> pd.DataFrame:
    df = add_location_priority_feature(df, enabled=bool(cfg.get("use_location_priority_branch", False)))
    df = add_location_priority_to_text(
        df,
        enabled=bool(cfg.get("use_location_priority", False)),
        include_location=bool(cfg.get("location_priority_include_location", False)),
    )
    return add_metadata_to_text(df, cfg.get("metadata_columns", []))


def get_label_names(train_df: pd.DataFrame) -> list[str]:
    mapping = (
        train_df[["label_id", "label"]]
        .drop_duplicates()
        .sort_values("label_id")
        .set_index("label_id")["label"]
        .to_dict()
    )
    return [str(mapping[i]) for i in sorted(mapping)]


def split_train_valid(
    df: pd.DataFrame,
    validation_split: float,
    seed: int,
    strategy: str = "group_aware_split_if_feasible",
) -> SplitResult:
    df = df.reset_index(drop=True)
    y = df["label_id"].astype(int)

    if "group" in strategy and "leakage_group_id" in df.columns:
        n_splits = max(2, round(1 / validation_split))
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        try:
            train_idx, valid_idx = next(splitter.split(df, y, groups=df["leakage_group_id"]))
            train_df = df.iloc[train_idx].reset_index(drop=True)
            valid_df = df.iloc[valid_idx].reset_index(drop=True)
            if valid_df["label_id"].nunique() == df["label_id"].nunique():
                return SplitResult(train_df=train_df, valid_df=valid_df, split_method="stratified_group_kfold")
        except ValueError:
            pass

    train_df, valid_df = train_test_split(
        df,
        test_size=validation_split,
        random_state=seed,
        stratify=y,
    )
    return SplitResult(
        train_df=train_df.reset_index(drop=True),
        valid_df=valid_df.reset_index(drop=True),
        split_method="stratified_train_test_split",
    )


def summarize_dataframe(df: pd.DataFrame, label_names: list[str] | None = None) -> dict:
    summary = {
        "num_rows": int(len(df)),
        "columns": list(df.columns),
        "text_length": {
            "min": int(df["text"].astype(str).str.len().min()),
            "p50": float(df["text"].astype(str).str.len().quantile(0.5)),
            "p90": float(df["text"].astype(str).str.len().quantile(0.9)),
            "p95": float(df["text"].astype(str).str.len().quantile(0.95)),
            "max": int(df["text"].astype(str).str.len().max()),
        },
        "label_counts": {
            str(k): int(v) for k, v in df["label_id"].value_counts().sort_index().to_dict().items()
        },
    }
    if label_names:
        summary["label_names"] = label_names
    if "leakage_group_id" in df.columns:
        summary["num_groups"] = int(df["leakage_group_id"].nunique())
    if "location_priority" in df.columns:
        summary["location_priority_counts"] = {
            str(k): int(v)
            for k, v in df["location_priority"].value_counts().sort_index().to_dict().items()
        }
    return summary


def group_overlap(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> list[str]:
    if "leakage_group_id" not in train_df.columns or "leakage_group_id" not in valid_df.columns:
        return []
    train_groups = set(train_df["leakage_group_id"].astype(str))
    valid_groups = set(valid_df["leakage_group_id"].astype(str))
    return sorted(train_groups & valid_groups)
