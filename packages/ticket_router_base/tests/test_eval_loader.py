"""Tests for eval.loader — JSONL loading and PredSave deserialization."""

import json
from pathlib import Path

import pytest

from ticket_router_base.eval.loader import load_pred_saves, _parse_pred_save
from ticket_router_base.types import (
    ErrorFlag,
    Language,
    Priority,
    Queue,
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
                    "queue": "Technical Support",
                    "priority": "high",
                    "tag_1": None,
                    "tag_2": None,
                    "answer": None,
                    "request_id": "Test-0000",
                    "queue_confidence": 0.5,
                    "priority_confidence": 0.6,
                    "raw_output": None,
                    "error": 0,
                },
                "ground_truth": {
                    "queue": "Technical Support",
                    "priority": "high",
                    "tag_1": "Network Issue",
                    "tag_2": "Hardware Failure",
                    "answer": "Hello",
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
                    "queue": "Billing and Payments",
                    "priority": "low",
                    "tag_1": None,
                    "tag_2": None,
                    "answer": None,
                    "request_id": "Test-0001",
                    "queue_confidence": None,
                    "priority_confidence": None,
                    "raw_output": '{"queue": "Billing and Payments"}',
                    "error": 1,
                },
                "ground_truth": {
                    "queue": "Billing and Payments",
                    "priority": "low",
                    "tag_1": "Billing Issue",
                    "tag_2": None,
                    "answer": "Danke",
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
        assert result[0].request_id == "Test-0000"
        assert result[0].language == Language.ENGLISH
        assert result[0].predicted.queue == Queue.TECHNICAL_SUPPORT
        assert result[0].predicted.priority == Priority.HIGH
        assert result[0].predicted.error == ErrorFlag.SUCCESS
        assert result[0].ground_truth.queue == Queue.TECHNICAL_SUPPORT

        # second record
        assert result[1].request_id == "Test-0001"
        assert result[1].language == Language.GERMAN
        assert result[1].predicted.queue == Queue.BILLING_AND_PAYMENTS
        assert result[1].predicted.priority == Priority.LOW
        assert result[1].predicted.error == ErrorFlag.JSON_ERR
        assert result[1].ground_truth.tag_1 == "Billing Issue"
        assert result[1].ground_truth.tag_2 is None

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

    def test_null_queue_fallback(self) -> None:
        """Null predicted queue falls back to CUSTOMER_SERVICE."""
        raw = {
            "request_id": "T-001",
            "language": "en",
            "predicted": {
                "queue": None,
                "priority": "medium",
                "tag_1": None,
                "tag_2": None,
                "answer": None,
                "request_id": "T-001",
                "queue_confidence": None,
                "priority_confidence": None,
                "raw_output": None,
                "error": 0,
            },
            "ground_truth": {
                "queue": "IT Support",
                "priority": "medium",
                "tag_1": None,
                "tag_2": None,
                "answer": "OK",
                "request_id": "T-001",
                "subject": "S",
                "body": "B",
                "language": "en",
            },
        }
        ps = _parse_pred_save(raw)
        assert ps.predicted.queue == Queue.CUSTOMER_SERVICE
        assert ps.predicted.priority == Priority.MEDIUM

    def test_null_priority_fallback(self) -> None:
        """Null predicted priority falls back to LOW."""
        raw = {
            "request_id": "T-002",
            "language": "en",
            "predicted": {
                "queue": "General Inquiry",
                "priority": None,
                "tag_1": None,
                "tag_2": None,
                "answer": None,
                "request_id": "T-002",
                "queue_confidence": None,
                "priority_confidence": None,
                "raw_output": None,
                "error": 0,
            },
            "ground_truth": {
                "queue": "General Inquiry",
                "priority": "low",
                "tag_1": None,
                "tag_2": None,
                "answer": "OK",
                "request_id": "T-002",
                "subject": "S",
                "body": "B",
                "language": "en",
            },
        }
        ps = _parse_pred_save(raw)
        assert ps.predicted.queue == Queue.GENERAL_INQUIRY
        assert ps.predicted.priority == Priority.LOW

    def test_error_flag_mapping(self) -> None:
        """Error integer maps correctly to ErrorFlag."""
        raw = {
            "request_id": "T-003",
            "language": "en",
            "predicted": {
                "queue": "IT Support",
                "priority": "high",
                "tag_1": None,
                "tag_2": None,
                "answer": None,
                "request_id": "T-003",
                "queue_confidence": None,
                "priority_confidence": None,
                "raw_output": None,
                "error": 3,  # JSON_ERR | QUEUE_REGEX_ERR
            },
            "ground_truth": {
                "queue": "IT Support",
                "priority": "high",
                "tag_1": None,
                "tag_2": None,
                "answer": "OK",
                "request_id": "T-003",
                "subject": "S",
                "body": "B",
                "language": "en",
            },
        }
        ps = _parse_pred_save(raw)
        assert ps.predicted.error == (ErrorFlag.JSON_ERR | ErrorFlag.QUEUE_REGEX_ERR)
