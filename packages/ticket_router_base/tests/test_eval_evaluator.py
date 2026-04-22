"""Tests for eval.evaluator — TaskEvaluator with multi-dimension aggregation."""

import pytest

from ticket_router_base.data import BaseDataset, ClassificationTask
from ticket_router_base.eval.evaluator import TaskEvaluator
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
    """Helper to construct a minimal PredSave for testing."""
    return PredSave(
        language=language,
        predicted=Prediction(
            request_id="test",
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
    """Minimal fake dataset for evaluator tests."""

    name = "fake"
    csv_path = __import__("pathlib").Path("/dev/null")
    body_column = "body"
    classification_tasks = [
        ClassificationTask("queue", "queue", ["A", "B", "C"]),
        ClassificationTask("priority", "priority", ["high", "low"]),
    ]


class TestTaskEvaluatorQueue:
    """Tests for TaskEvaluator."""

    def test_overall_accuracy(self) -> None:
        """3 out of 4 predictions correct -> accuracy=0.75."""
        pred_saves = [
            _make_pred_save("en", {"queue": "A"}, {"queue": "A"}),
            _make_pred_save("en", {"queue": "B"}, {"queue": "B"}),
            _make_pred_save("de", {"queue": "C"}, {"queue": "C"}),
            _make_pred_save("de", {"queue": "A"}, {"queue": "B"}),
        ]
        dataset = _FakeDataset()
        evaluator = TaskEvaluator()
        results = evaluator.evaluate(pred_saves, dataset)

        queue_result = [r for r in results if r.task_name == "queue"][0]
        assert queue_result.overall.accuracy == pytest.approx(0.75)
        assert queue_result.overall.support == 4

    def test_by_language(self) -> None:
        """By-language breakdown should only contain languages present in data."""
        pred_saves = [
            _make_pred_save("en", {"queue": "A"}, {"queue": "A"}),
            _make_pred_save("en", {"queue": "B"}, {"queue": "B"}),
            _make_pred_save("de", {"queue": "C"}, {"queue": "A"}),
        ]
        dataset = _FakeDataset()
        evaluator = TaskEvaluator()
        results = evaluator.evaluate(pred_saves, dataset)

        queue_result = [r for r in results if r.task_name == "queue"][0]
        assert set(queue_result.by_language.keys()) == {"en", "de"}
        assert queue_result.by_language["en"].accuracy == 1.0
        assert queue_result.by_language["de"].accuracy == 0.0

    def test_by_strata(self) -> None:
        """By-strata breakdown for queue task."""
        pred_saves = [
            _make_pred_save("en", {"queue": "A"}, {"queue": "A"}),
            _make_pred_save("en", {"queue": "A"}, {"queue": "A"}),
            _make_pred_save("en", {"queue": "B"}, {"queue": "B"}),
            _make_pred_save("en", {"queue": "C"}, {"queue": "B"}),
        ]
        dataset = _FakeDataset()
        evaluator = TaskEvaluator()
        results = evaluator.evaluate(pred_saves, dataset)

        queue_result = [r for r in results if r.task_name == "queue"][0]
        assert queue_result.by_strata is not None
        assert queue_result.by_strata["A"].accuracy == 1.0
        assert queue_result.by_strata["B"].accuracy == pytest.approx(0.5)

    def test_multiple_tasks(self) -> None:
        """Evaluator should return results for all classification tasks."""
        pred_saves = [
            _make_pred_save(
                "en",
                {"queue": "A", "priority": "high"},
                {"queue": "A", "priority": "high"},
            ),
            _make_pred_save(
                "en",
                {"queue": "B", "priority": "low"},
                {"queue": "B", "priority": "high"},
            ),
        ]
        dataset = _FakeDataset()
        evaluator = TaskEvaluator()
        results = evaluator.evaluate(pred_saves, dataset)

        assert len(results) == 2
        task_names = {r.task_name for r in results}
        assert task_names == {"queue", "priority"}

    def test_single_sample(self) -> None:
        """Evaluation on a single sample should work."""
        pred_saves = [
            _make_pred_save("en", {"queue": "A"}, {"queue": "A"}),
        ]
        dataset = _FakeDataset()
        evaluator = TaskEvaluator()
        results = evaluator.evaluate(pred_saves, dataset)

        queue_result = [r for r in results if r.task_name == "queue"][0]
        assert queue_result.overall.accuracy == 1.0
        assert queue_result.overall.support == 1
