"""Load prediction JSONL files into typed PredSave dataclasses."""

import json
from pathlib import Path
from typing import List

from ticket_router_base.types import (
    ErrorFlag,
    GroundRecord,
    Language,
    PredSave,
    Prediction,
    Priority,
    Queue,
)


def load_pred_saves(path: Path) -> List[PredSave]:
    """Load a prediction JSONL file into a list of PredSave instances.

    Each line in the file must be a JSON object matching the PredSave schema.
    """
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    results: List[PredSave] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            results.append(_parse_pred_save(raw))
    return results


def _parse_prediction(raw: dict) -> Prediction:
    """Deserialize a Prediction dict, with safe fallback for null values."""
    queue_raw = raw.get("queue")
    priority_raw = raw.get("priority")

    # fallback for null/ missing queue or priority
    queue = Queue.CUSTOMER_SERVICE
    if queue_raw is not None:
        queue = Queue(str(queue_raw))

    priority = Priority.LOW
    if priority_raw is not None:
        priority = Priority(str(priority_raw))

    error_val = raw.get("error", 0)
    error = ErrorFlag(error_val) if isinstance(error_val, int) else ErrorFlag.SUCCESS

    return Prediction(
        request_id=raw.get("request_id", ""),
        queue=queue,
        priority=priority,
        tag_1=raw.get("tag_1"),
        tag_2=raw.get("tag_2"),
        answer=raw.get("answer"),
        queue_confidence=raw.get("queue_confidence"),
        priority_confidence=raw.get("priority_confidence"),
        raw_output=raw.get("raw_output"),
        error=error,
    )


def _parse_ground_record(raw: dict) -> GroundRecord:
    """Deserialize a GroundRecord dict."""
    return GroundRecord(
        queue=Queue(str(raw["queue"])),
        priority=Priority(str(raw["priority"])),
        tag_1=raw.get("tag_1"),
        tag_2=raw.get("tag_2"),
        answer=raw.get("answer"),
    )


def _parse_pred_save(raw: dict) -> PredSave:
    """Deserialize a top-level PredSave dict."""
    return PredSave(
        request_id=raw["request_id"],
        language=Language(raw["language"]),
        predicted=_parse_prediction(raw["predicted"]),
        ground_truth=_parse_ground_record(raw["ground_truth"]),
    )
