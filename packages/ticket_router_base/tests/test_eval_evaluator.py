"""Tests for eval.evaluator — TaskEvaluator with multi-dimension aggregation."""

import pytest

from ticket_router_base.eval.evaluator import TaskEvaluator
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
    """Helper to construct a minimal PredSave for testing."""
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


class TestTaskEvaluatorQueue:
    """Tests for TaskEvaluator on QUEUE task."""

    def test_overall_accuracy(self) -> None:
        """3 out of 4 queue predictions correct -> accuracy=0.75."""
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
                "en",
                Queue.PRODUCT_SUPPORT,
                Priority.MEDIUM,
                Queue.PRODUCT_SUPPORT,
                Priority.MEDIUM,
            ),
            _make_pred_save(
                "T-2",
                "de",
                Queue.CUSTOMER_SERVICE,
                Priority.LOW,
                Queue.CUSTOMER_SERVICE,
                Priority.LOW,
            ),
            _make_pred_save(
                "T-3",
                "de",
                Queue.IT_SUPPORT,
                Priority.HIGH,
                Queue.TECHNICAL_SUPPORT,
                Priority.HIGH,
            ),
        ]
        evaluator = TaskEvaluator()
        result = evaluator.evaluate(pred_saves, Task.QUEUE)

        assert result.task == Task.QUEUE
        assert result.overall.accuracy == pytest.approx(0.75)
        assert result.overall.support == 4

    def test_by_language(self) -> None:
        """By-language breakdown should only contain languages present in data."""
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
                "en",
                Queue.PRODUCT_SUPPORT,
                Priority.MEDIUM,
                Queue.PRODUCT_SUPPORT,
                Priority.MEDIUM,
            ),
            _make_pred_save(
                "T-2",
                "de",
                Queue.CUSTOMER_SERVICE,
                Priority.LOW,
                Queue.TECHNICAL_SUPPORT,
                Priority.LOW,
            ),
        ]
        evaluator = TaskEvaluator()
        result = evaluator.evaluate(pred_saves, Task.QUEUE)

        assert set(result.by_language.keys()) == {"en", "de"}
        # en: 2/2 correct
        assert result.by_language["en"].accuracy == 1.0
        # de: 0/1 correct
        assert result.by_language["de"].accuracy == 0.0

    def test_by_queue_for_queue_task(self) -> None:
        """By-queue breakdown for queue task: each queue shows its own recall."""
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
                "en",
                Queue.TECHNICAL_SUPPORT,
                Priority.HIGH,
                Queue.TECHNICAL_SUPPORT,
                Priority.HIGH,
            ),
            _make_pred_save(
                "T-2",
                "en",
                Queue.PRODUCT_SUPPORT,
                Priority.MEDIUM,
                Queue.PRODUCT_SUPPORT,
                Priority.MEDIUM,
            ),
            _make_pred_save(
                "T-3",
                "en",
                Queue.CUSTOMER_SERVICE,
                Priority.LOW,
                Queue.PRODUCT_SUPPORT,
                Priority.LOW,
            ),
        ]
        evaluator = TaskEvaluator()
        result = evaluator.evaluate(pred_saves, Task.QUEUE)

        assert result.by_queue is not None
        # Technical Support: 2 samples, both correct
        assert result.by_queue["Technical Support"].accuracy == 1.0
        # Product Support: 2 samples, 1 correct (T-2), 1 misclassified as CUSTOMER_SERVICE (T-3)
        assert result.by_queue["Product Support"].accuracy == pytest.approx(0.5)


class TestTaskEvaluatorPriority:
    """Tests for TaskEvaluator on PRIORITY task."""

    def test_overall_priority(self) -> None:
        """2 out of 3 priority predictions correct -> accuracy=2/3."""
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
                "en",
                Queue.PRODUCT_SUPPORT,
                Priority.MEDIUM,
                Queue.PRODUCT_SUPPORT,
                Priority.MEDIUM,
            ),
            _make_pred_save(
                "T-2",
                "de",
                Queue.CUSTOMER_SERVICE,
                Priority.LOW,
                Queue.CUSTOMER_SERVICE,
                Priority.HIGH,
            ),
        ]
        evaluator = TaskEvaluator()
        result = evaluator.evaluate(pred_saves, Task.PRIORITY)

        assert result.task == Task.PRIORITY
        assert result.overall.accuracy == pytest.approx(2 / 3)

    def test_by_queue_for_priority_task(self) -> None:
        """By-queue breakdown for priority task shows priority accuracy per queue."""
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
                "en",
                Queue.TECHNICAL_SUPPORT,
                Priority.LOW,
                Queue.TECHNICAL_SUPPORT,
                Priority.HIGH,
            ),
            _make_pred_save(
                "T-2",
                "en",
                Queue.PRODUCT_SUPPORT,
                Priority.MEDIUM,
                Queue.PRODUCT_SUPPORT,
                Priority.MEDIUM,
            ),
        ]
        evaluator = TaskEvaluator()
        result = evaluator.evaluate(pred_saves, Task.PRIORITY)

        assert result.by_queue is not None
        # Technical Support: 2 samples, 1 correct -> accuracy=0.5
        assert result.by_queue["Technical Support"].accuracy == pytest.approx(0.5)
        # Product Support: 1 sample, correct -> accuracy=1.0
        assert result.by_queue["Product Support"].accuracy == 1.0


class TestTaskEvaluatorEdgeCases:
    """Edge cases for TaskEvaluator."""

    def test_single_sample(self) -> None:
        """Evaluation on a single sample should work."""
        pred_saves = [
            _make_pred_save(
                "T-0",
                "en",
                Queue.TECHNICAL_SUPPORT,
                Priority.HIGH,
                Queue.TECHNICAL_SUPPORT,
                Priority.HIGH,
            ),
        ]
        evaluator = TaskEvaluator()
        result = evaluator.evaluate(pred_saves, Task.QUEUE)

        assert result.overall.accuracy == 1.0
        assert result.overall.support == 1
        assert result.by_language["en"].accuracy == 1.0
