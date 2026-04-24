"""Tests for eval.loader — JSONL loading and PredSave deserialization."""

import json
from pathlib import Path

import pytest

from ticket_router_base.eval.loader import load_pred_saves, _parse_pred_save
from ticket_router_base.types import (
    ErrorFlag,
)


class TestLoadPredSaves:
    """Tests for load_pred_saves from JSONL file."""

    def test_load_valid_jsonl(self, tmp_path: Path) -> None:
        """Load a JSONL file with two valid PredSave records."""
        path = tmp_path / "preds.jsonl"
        records = [
            {
                "request_id": "Test-0000",
                "language": "en",
                "predicted": {
                    "labels": {"queue": "Technical Support", "priority": "high"},
                    "discrete_features": {},
                    "generation_target": None,
                    "request_id": "Test-0000",
                    "confidences": {"queue": 0.5, "priority": 0.6},
                    "raw_output": None,
                    "error": 0,
                },
                "ground_truth": {
                    "labels": {"queue": "Technical Support", "priority": "high"},
                    "discrete_features": {
                        "tag_1": "Network Issue",
                        "tag_2": "Hardware Failure",
                    },
                    "generation_target": "Hello",
                    "request_id": "Test-0000",
                    "subject": "Subject",
                    "body": "Body",
                    "language": "en",
                },
            },
            {
                "request_id": "Test-0001",
                "language": "de",
                "predicted": {
                    "labels": {"queue": "Billing and Payments", "priority": "low"},
                    "discrete_features": {},
                    "generation_target": None,
                    "request_id": "Test-0001",
                    "confidences": {},
                    "raw_output": '{"queue": "Billing and Payments"}',
                    "error": 1,
                },
                "ground_truth": {
                    "labels": {"queue": "Billing and Payments", "priority": "low"},
                    "discrete_features": {"tag_1": "Billing Issue", "tag_2": None},
                    "generation_target": "Danke",
                    "request_id": "Test-0001",
                    "subject": "Betreff",
                    "body": "Inhalt",
                    "language": "de",
                },
            },
        ]
        with path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        result = load_pred_saves(path)
        assert len(result) == 2

        # first record
        assert result[0].language == "en"
        assert result[0].predicted.labels == {
            "queue": "Technical Support",
            "priority": "high",
        }
        assert result[0].predicted.error == ErrorFlag.SUCCESS
        assert result[0].ground_truth.labels == {
            "queue": "Technical Support",
            "priority": "high",
        }

        # second record
        assert result[1].language == "de"
        assert result[1].predicted.labels == {
            "queue": "Billing and Payments",
            "priority": "low",
        }
        assert result[1].predicted.error == ErrorFlag.JSON_ERR
        assert result[1].ground_truth.discrete_features == {
            "tag_1": "Billing Issue",
            "tag_2": None,
        }

    def test_load_empty_file(self, tmp_path: Path) -> None:
        """Empty JSONL file returns empty list."""
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        result = load_pred_saves(path)
        assert result == []

    def test_load_file_not_found(self, tmp_path: Path) -> None:
        """Non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_pred_saves(tmp_path / "nonexistent.jsonl")


class TestParsePredSave:
    """Tests for _parse_pred_save edge cases."""

    def test_error_flag_mapping(self) -> None:
        """Error integer maps correctly to ErrorFlag."""
        raw = {
            "language": "en",
            "predicted": {
                "labels": {"queue": "IT Support"},
                "discrete_features": {},
                "generation_target": None,
                "confidences": {},
                "raw_output": None,
                "error": 3,  # JSON_ERR | CLASSIFICATION_REGEX_ERR
            },
            "ground_truth": {
                "labels": {"queue": "IT Support"},
                "discrete_features": {},
                "generation_target": "OK",
                "subject": "S",
                "body": "B",
                "language": "en",
            },
        }
        ps = _parse_pred_save(raw)
        assert ps.predicted.error == (
            ErrorFlag.JSON_ERR | ErrorFlag.CLASSIFICATION_REGEX_ERR
        )
