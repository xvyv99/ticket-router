"""Data loading utilities."""

from typing import List

import pandas as pd

from ticket_router_base.config import OUTPUT_DIR
from ticket_router_base.datasets.base import BaseDataset
from ticket_router_base.types import Record


def load_dataset(dataset: BaseDataset) -> List[Record]:
    """Load a dataset via its descriptor and return standardized Records."""
    return dataset.load()


def load_test_set() -> List[Record]:
    """Load the pre-built test set from JSONL.

    Supports both new format (labels dict) and legacy format.
    """
    test_path = OUTPUT_DIR / "test_set.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(f"Test set not found at {test_path}")
    return _load_jsonl_records(test_path)


def load_train_set() -> List[Record]:
    """Load the pre-built train set from JSONL.

    Supports both new format (labels dict) and legacy format.
    """
    train_path = OUTPUT_DIR / "train_set.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Train set not found at {train_path}")
    return _load_jsonl_records(train_path)


def _load_jsonl_records(path) -> List[Record]:
    """Load JSONL and convert rows to Record, with legacy-format fallback."""
    df = pd.read_json(path, orient="records", lines=True, encoding="utf-8")
    records: List[Record] = []
    for _, row in df.iterrows():
        # new format: labels dict
        if "labels" in row and isinstance(row["labels"], dict):
            records.append(
                Record(
                    request_id=row.get("request_id", ""),
                    title=row.get("title") or row.get("subject"),
                    body=row.get("body", ""),
                    language=row.get("language"),
                    labels=row["labels"],
                    discrete_features=row.get("discrete_features", {}),
                    generation_target=row.get("generation_target") or row.get("answer"),
                )
            )
        else:
            # legacy format
            labels = {}
            if "queue" in row:
                labels["queue"] = str(row["queue"]) if pd.notna(row["queue"]) else ""
            if "priority" in row:
                labels["priority"] = (
                    str(row["priority"]) if pd.notna(row["priority"]) else ""
                )
            discrete = {}
            if "tag_1" in row:
                discrete["tag_1"] = row.get("tag_1")
            if "tag_2" in row:
                discrete["tag_2"] = row.get("tag_2")
            records.append(
                Record(
                    request_id=row.get("request_id", ""),
                    title=row.get("subject"),
                    body=row.get("body", ""),
                    language=row.get("language"),
                    labels=labels,
                    discrete_features=discrete,
                    generation_target=row.get("answer"),
                )
            )
    return records
