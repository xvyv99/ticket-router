"""Load prediction JSONL files into typed PredSave dataclasses."""

from pathlib import Path
from typing import List

from ticket_router_base.types import (
    PredSave,
)
from ticket_router_base.utils import JSONLLogger


def load_pred_saves(path: Path) -> List[PredSave]:
    """Load a prediction JSONL file into a list of PredSave instances."""
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    results: List[PredSave] = []
    with JSONLLogger(path, mode="r") as logger:
        for pred_save_raw in logger.read():
            pred_save_parsed = PredSave.model_validate_json(
                pred_save_raw
            )  # validate against the new format

            results.append(pred_save_parsed)
    return results
