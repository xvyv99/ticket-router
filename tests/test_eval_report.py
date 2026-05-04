"""Tests for eval.report — EvaluationReport serialization and output."""

import json
from pathlib import Path

from ticket_router_base.data import BaseDataset, ClassificationTask, TaskDescriptor
from ticket_router.eval.evaluator import TaskEvaluator
from ticket_router.eval.report import EvaluationReport
from ticket_router_base.types import (
    ErrorFlag,
    GroundRecord,
    PredSave,
    Prediction,
)


def _make_pred_save(
    language: str,
    pred_labels: dict[str, str],
    gt_labels: dict[str, str],
) -> PredSave:
    return PredSave(
        language=language,
        predicted=Prediction(
            request_id="test",
            labels=pred_labels,
            discrete_features={},
            generation_target=None,
            sensitive_attributes={},
            confidences={},
            raw_output=None,
            error=ErrorFlag.SUCCESS,
        ),
        ground_truth=GroundRecord(
            labels=gt_labels,
            discrete_features={},
            generation_target=None,
            sensitive_attributes={},
        ),
    )


class _FakeDataset(BaseDataset):
    name = "fake"
    csv_path = Path("/dev/null")
    body_column = "body"
    task_descriptor = TaskDescriptor(
        classification_tasks=[
            ClassificationTask(name="queue", target_column="queue", labels=["A", "B"]),
        ],
    )
    sensitive_columns = ["language"]
    stratified_columns = ["language"]
    discrete_feature_columns = []

    def load_df(self, dataset_path=None, sample_num=None):
        raise NotImplementedError

    def load(self, dataset_path=None, sample_num=None):
        raise NotImplementedError


class TestEvaluationReport:
    """Tests for EvaluationReport dataclass."""

    def _build_report(self) -> EvaluationReport:
        """Helper to build a minimal EvaluationReport."""
        pred_saves = [
            _make_pred_save("en", {"queue": "A"}, {"queue": "A"}),
            _make_pred_save("de", {"queue": "B"}, {"queue": "B"}),
        ]
        dataset = _FakeDataset()
        evaluator = TaskEvaluator()
        task_results = evaluator.evaluate(pred_saves, dataset)

        return EvaluationReport(
            model_name="test_model",
            dataset=dataset,
            file_path=Path("outputs/test_predictions.jsonl"),
            task_results=task_results,
            error_summary={"SUCCESS": 2},
            total_samples=2,
        )

    def test_to_dict(self) -> None:
        """asdict() should return a fully nested dict."""
        from dataclasses import asdict
        report = self._build_report()
        d = asdict(report)

        assert d["model_name"] == "test_model"
        assert d["total_samples"] == 2
        assert "task_results" in d
        assert d["error_summary"] == {"SUCCESS": 2}

    def test_to_json_roundtrip(self) -> None:
        """report JSON serialization must be valid JSON."""
        from dataclasses import asdict

        def _serialize(obj):
            if isinstance(obj, BaseDataset):
                return {"name": obj.name}
            return str(obj)

        report = self._build_report()
        s = json.dumps(asdict(report), default=_serialize)
        loaded = json.loads(s)

        assert loaded["model_name"] == "test_model"
        assert loaded["dataset"]["name"] == "fake"
        assert len(loaded["task_results"]) == 1

    def test_to_json_writes_file(self, tmp_path: Path) -> None:
        """JSON report should be writable to disk."""
        from dataclasses import asdict

        def _serialize(obj):
            if isinstance(obj, BaseDataset):
                return {"name": obj.name}
            return str(obj)

        report = self._build_report()
        out_path = tmp_path / "report.json"
        out_path.write_text(json.dumps(asdict(report), default=_serialize), encoding="utf-8")

        assert out_path.exists()
        loaded = json.loads(out_path.read_text(encoding="utf-8"))
        assert loaded["model_name"] == "test_model"
