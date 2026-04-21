"""Load prediction JSONL files into typed PredSave dataclasses."""

import json
from pathlib import Path
from typing import List

from ticket_router_base.types import (
    ErrorFlag,
    GroundRecord,
    PredSave,
    Prediction,
)


def load_pred_saves(path: Path) -> List[PredSave]:
    """Load a prediction JSONL file into a list of PredSave instances.

    Supports both the new format (labels dict) and the legacy format
    (queue/priority/answer/tag_1/tag_2 fields).
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
    """Deserialize a Prediction dict with legacy-format fallback."""
    error_val = raw.get("error", 0)
    error = ErrorFlag(error_val) if isinstance(error_val, int) else ErrorFlag.SUCCESS

    # new format: labels dict
    labels = raw.get("labels")
    if labels is not None:
        return Prediction(
            request_id=raw.get("request_id", ""),
            labels=labels,
            discrete_features=raw.get("discrete_features", {}),
            generation_target=raw.get("generation_target"),
            confidences=raw.get("confidences", {}),
            raw_output=raw.get("raw_output"),
            error=error,
        )

    # legacy format: queue/priority/answer/tag_1/tag_2 fields
    labels = {}
    if "queue" in raw:
        labels["queue"] = str(raw["queue"]) if raw["queue"] is not None else ""
    if "priority" in raw:
        labels["priority"] = str(raw["priority"]) if raw["priority"] is not None else ""

    discrete = {}
    if "tag_1" in raw:
        discrete["tag_1"] = raw.get("tag_1")
    if "tag_2" in raw:
        discrete["tag_2"] = raw.get("tag_2")

    confidences = {}
    if "queue_confidence" in raw:
        confidences["queue"] = raw.get("queue_confidence")
    if "priority_confidence" in raw:
        confidences["priority"] = raw.get("priority_confidence")

    return Prediction(
        request_id=raw.get("request_id", ""),
        labels=labels,
        discrete_features=discrete,
        generation_target=raw.get("answer"),
        confidences=confidences,
        raw_output=raw.get("raw_output"),
        error=error,
    )


def _parse_ground_record(raw: dict) -> GroundRecord:
    """Deserialize a GroundRecord dict with legacy-format fallback."""
    # new format
    labels = raw.get("labels")
    if labels is not None:
        return GroundRecord(
            labels=labels,
            discrete_features=raw.get("discrete_features", {}),
            generation_target=raw.get("generation_target"),
        )

    # legacy format
    labels = {}
    if "queue" in raw:
        labels["queue"] = str(raw["queue"]) if raw["queue"] is not None else ""
    if "priority" in raw:
        labels["priority"] = str(raw["priority"]) if raw["priority"] is not None else ""

    discrete = {}
    if "tag_1" in raw:
        discrete["tag_1"] = raw.get("tag_1")
    if "tag_2" in raw:
        discrete["tag_2"] = raw.get("tag_2")

    return GroundRecord(
        labels=labels,
        discrete_features=discrete,
        generation_target=raw.get("answer"),
    )


def _parse_pred_save(raw: dict) -> PredSave:
    """Deserialize a top-level PredSave dict with legacy-format fallback."""
    language = raw.get("language")
    # legacy format may have Language enum as dict with value key
    if isinstance(language, dict):
        language = language.get("value")

    return PredSave(
        request_id=raw["request_id"],
        language=language,
        predicted=_parse_prediction(raw["predicted"]),
        ground_truth=_parse_ground_record(raw["ground_truth"]),
    )
