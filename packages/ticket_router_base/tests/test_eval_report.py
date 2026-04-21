"""Tests for eval.report — EvaluationReport serialization and output."""

import json
from pathlib import Path

from ticket_router_base.eval.evaluator import TaskEvaluator
from ticket_router_base.eval.report import EvaluationReport
from ticket_router_base.types import (
    ErrorFlag,
    GroundRecord,
    Language,
    PredSave,
    Prediction,
    Priority,
    Queue,
    Task,
)


def _make_pred_save(
    request_id: str,
    language: str,
    pred_queue: Queue,
    pred_priority: Priority,
    gt_queue: Queue,
    gt_priority: Priority,
) -> PredSave:
    return PredSave(
        request_id=request_id,
        language=Language(language),
        predicted=Prediction(
            request_id=request_id,
            queue=pred_queue,
            priority=pred_priority,
            tag_1=None,
            tag_2=None,
            answer=None,
            queue_confidence=None,
            priority_confidence=None,
            raw_output=None,
            error=ErrorFlag.SUCCESS,
        ),
        ground_truth=GroundRecord(
            queue=gt_queue,
            priority=gt_priority,
            tag_1=None,
            tag_2=None,
            answer=None,
        ),
    )


class TestEvaluationReport:
    """Tests for EvaluationReport dataclass."""

    def _build_report(self) -> EvaluationReport:
        """Helper to build a minimal EvaluationReport."""
        pred_saves = [
            _make_pred_save(
                "T-0",
                "en",
                Queue.TECHNICAL_SUPPORT,
                Priority.HIGH,
                Queue.TECHNICAL_SUPPORT,
                Priority.HIGH,
            ),
            _make_pred_save(
                "T-1",
                "de",
                Queue.PRODUCT_SUPPORT,
                Priority.MEDIUM,
                Queue.PRODUCT_SUPPORT,
                Priority.MEDIUM,
            ),
        ]
        evaluator = TaskEvaluator()
        queue_result = evaluator.evaluate(pred_saves, Task.QUEUE)
        priority_result = evaluator.evaluate(pred_saves, Task.PRIORITY)

        return EvaluationReport(
            model_name="test_model",
            pred_file_path="outputs/test_predictions.jsonl",
            queue_result=queue_result,
            priority_result=priority_result,
            error_summary={"SUCCESS": 2},
            total_samples=2,
        )

    def test_to_dict(self) -> None:
        """to_dict() should return a fully nested dict."""
        report = self._build_report()
        d = report.to_dict()

        assert d["model_name"] == "test_model"
        assert d["total_samples"] == 2
        assert "queue_result" in d
        assert "priority_result" in d
        assert d["error_summary"] == {"SUCCESS": 2}

    def test_to_json_roundtrip(self) -> None:
        """to_json() output must be valid JSON."""
        report = self._build_report()
        s = report.to_json()
        loaded = json.loads(s)

        assert loaded["model_name"] == "test_model"
        assert loaded["queue_result"]["task"] == "queue"
        assert loaded["priority_result"]["task"] == "priority"

    def test_to_json_writes_file(self, tmp_path: Path) -> None:
        """to_json(path) should write to disk."""
        report = self._build_report()
        out_path = tmp_path / "report.json"
        report.to_json(out_path)

        assert out_path.exists()
        loaded = json.loads(out_path.read_text(encoding="utf-8"))
        assert loaded["model_name"] == "test_model"
