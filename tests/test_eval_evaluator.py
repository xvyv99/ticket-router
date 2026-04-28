"""Tests for eval.evaluator — TaskEvaluator with multi-dimension aggregation."""

import pytest

from ticket_router_base.data import BaseDataset, ClassificationTask, TaskDescriptor
from ticket_router_eval.evaluator import TaskEvaluator
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
    """Helper to construct a minimal PredSave for testing.

    Ensures all tasks defined in _FakeDataset have labels to avoid
    AssertionError in _extract_labels.
    """
    # Fill in defaults for all tasks so evaluator can extract labels
    all_pred = {"queue": "A", "priority": "high"}
    all_gt = {"queue": "A", "priority": "high"}
    all_pred.update(pred_labels)
    all_gt.update(gt_labels)
    return PredSave(
        language=language,
        predicted=Prediction(
            request_id="test",
            labels=all_pred,
            discrete_features={},
            generation_target=None,
            sensitive_attributes={},
            confidences={},
            raw_output=None,
            error=ErrorFlag.SUCCESS,
        ),
        ground_truth=GroundRecord(
            labels=all_gt,
            discrete_features={},
            generation_target=None,
            sensitive_attributes={"language": language},
        ),
    )


class _FakeDataset(BaseDataset):
    """Minimal fake dataset for evaluator tests."""

    name = "fake"
    csv_path = __import__("pathlib").Path("/dev/null")
    body_column = "body"
    task_descriptor = TaskDescriptor(
        classification_tasks=[
            ClassificationTask(name="queue", target_column="queue", labels=["A", "B", "C"]),
            ClassificationTask(name="priority", target_column="priority", labels=["high", "low"]),
        ],
    )
    sensitive_columns = ["language"]
    stratified_columns = ["language"]
    discrete_feature_columns = []

    def load_df(self, dataset_path=None, sample_num=None):
        raise NotImplementedError

    def load(self, dataset_path=None, sample_num=None):
        raise NotImplementedError


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
        assert queue_result.performance.accuracy == pytest.approx(0.75)
        assert queue_result.performance.support == 4

    def test_fairness_audit(self) -> None:
        """Fairness metrics should be computed for each sensitive attribute."""
        pred_saves = [
            _make_pred_save("en", {"queue": "A"}, {"queue": "A"}),
            _make_pred_save("en", {"queue": "B"}, {"queue": "B"}),
            _make_pred_save("de", {"queue": "C"}, {"queue": "A"}),
        ]
        dataset = _FakeDataset()
        evaluator = TaskEvaluator()
        results = evaluator.evaluate(pred_saves, dataset)

        queue_result = [r for r in results if r.task_name == "queue"][0]
        assert "language" in queue_result.fairness
        lang_fairness = queue_result.fairness["language"]
        assert lang_fairness.accuracy_gap >= 0.0

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
        assert queue_result.performance.accuracy == 1.0
        assert queue_result.performance.support == 1
