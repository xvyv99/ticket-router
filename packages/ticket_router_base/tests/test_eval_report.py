"""Tests for eval.report — EvaluationReport serialization and output."""

import json
from pathlib import Path

from ticket_router_base.data.base import BaseDataset, ClassificationTask
from ticket_router_base.eval.evaluator import TaskEvaluator
from ticket_router_base.eval.report import EvaluationReport
from ticket_router_base.types import (
    ErrorFlag,
    GroundRecord,
    PredSave,
    Prediction,
)


def _make_pred_save(
    request_id: str,
    language: str,
    pred_labels: dict[str, str],
    gt_labels: dict[str, str],
) -> PredSave:
    return PredSave(
        request_id=request_id,
        language=language,
        predicted=Prediction(
            request_id=request_id,
            labels=pred_labels,
            discrete_features={},
            generation_target=None,
            confidences={},
            raw_output=None,
            error=ErrorFlag.SUCCESS,
        ),
        ground_truth=GroundRecord(
            labels=gt_labels,
            discrete_features={},
            generation_target=None,
        ),
    )


class _FakeDataset(BaseDataset):
    name = "fake"
    csv_path = Path("/dev/null")
    body_column = "body"
    classification_tasks = [
        ClassificationTask("queue", "queue", ["A", "B"]),
    ]


class TestEvaluationReport:
    """Tests for EvaluationReport dataclass."""

    def _build_report(self) -> EvaluationReport:
        """Helper to build a minimal EvaluationReport."""
        pred_saves = [
            _make_pred_save("T-0", "en", {"queue": "A"}, {"queue": "A"}),
            _make_pred_save("T-1", "de", {"queue": "B"}, {"queue": "B"}),
        ]
        dataset = _FakeDataset()
        evaluator = TaskEvaluator()
        task_results = evaluator.evaluate(pred_saves, dataset)

        return EvaluationReport(
            model_name="test_model",
            pred_file_path="outputs/test_predictions.jsonl",
            dataset_name="fake",
            task_results=task_results,
            error_summary={"SUCCESS": 2},
            total_samples=2,
        )

    def test_to_dict(self) -> None:
        """to_dict() should return a fully nested dict."""
        report = self._build_report()
        d = report.to_dict()

        assert d["model_name"] == "test_model"
        assert d["total_samples"] == 2
        assert "task_results" in d
        assert d["error_summary"] == {"SUCCESS": 2}

    def test_to_json_roundtrip(self) -> None:
        """to_json() output must be valid JSON."""
        report = self._build_report()
        s = report.to_json()
        loaded = json.loads(s)

        assert loaded["model_name"] == "test_model"
        assert loaded["dataset_name"] == "fake"
        assert len(loaded["task_results"]) == 1

    def test_to_json_writes_file(self, tmp_path: Path) -> None:
        """to_json(path) should write to disk."""
        report = self._build_report()
        out_path = tmp_path / "report.json"
        report.to_json(out_path)

        assert out_path.exists()
        loaded = json.loads(out_path.read_text(encoding="utf-8"))
        assert loaded["model_name"] == "test_model"
