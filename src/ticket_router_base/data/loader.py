"""Data loading utilities."""

from typing import List
from pathlib import Path

import pandas as pd

from ticket_router_base.config import TRAIN_SET_PATH, TEST_SET_PATH
from ticket_router_base.data.base import BaseDataset
from ticket_router_base.types import Record


def load_dataset(dataset: BaseDataset) -> List[Record]:
    """Load a dataset via its descriptor and return standardized Records."""
    return dataset.load(None)


def load_test_set() -> List[Record]:
    """Load the pre-built test set from JSONL.

    Supports both new format (labels dict) and legacy format.
    """
    if not TEST_SET_PATH.exists():
        raise FileNotFoundError(f"Test set not found at {TEST_SET_PATH}")
    return _load_jsonl_records(TEST_SET_PATH)


def load_train_set() -> List[Record]:
    """Load the pre-built train set from JSONL.

    Supports both new format (labels dict) and legacy format.
    """
    if not TRAIN_SET_PATH.exists():
        raise FileNotFoundError(f"Train set not found at {TRAIN_SET_PATH}")
    return _load_jsonl_records(TRAIN_SET_PATH)


def _load_jsonl_records(path: Path) -> List[Record]:
    """Load JSONL and convert rows to Record, with legacy-format fallback."""
    df = pd.read_json(path, orient="records", lines=True, encoding="utf-8")
    records: List[Record] = []
    for _, row in df.iterrows():
        # new format: labels dict

        request_id: str | None = row.get("request_id")
        assert request_id is not None

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
    return records
