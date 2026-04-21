"""Task-level evaluation with multi-dimension aggregation."""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from ticket_router_base.types import PredSave, Task

from .metrics_core import ClassificationMetrics, compute_classification_metrics


@dataclass(frozen=True)
class TaskEvaluationResult:
    """Evaluation result for a single task across multiple dimensions."""

    task: Task
    overall: ClassificationMetrics
    by_language: Dict[str, ClassificationMetrics]
    by_queue: Dict[str, ClassificationMetrics] | None

    def to_dict(self) -> dict:
        return {
            "task": self.task.value,
            "overall": self.overall.to_dict(),
            "by_language": {k: v.to_dict() for k, v in self.by_language.items()},
            "by_queue": (
                {k: v.to_dict() for k, v in self.by_queue.items()}
                if self.by_queue is not None
                else None
            ),
        }


class TaskEvaluator:
    """Evaluate predictions for a single task with global and breakdown metrics."""

    def evaluate(self, pred_saves: List[PredSave], task: Task) -> TaskEvaluationResult:
        """Evaluate the given task across all samples and breakdown dimensions."""
        # global metrics
        y_true_all, y_pred_all = self._extract_labels(pred_saves, task)
        labels = sorted(set(y_true_all) | set(y_pred_all))
        overall = compute_classification_metrics(y_true_all, y_pred_all, labels=labels)

        # by language
        by_language: Dict[str, ClassificationMetrics] = {}
        lang_groups: Dict[str, List[PredSave]] = {}
        for ps in pred_saves:
            lang = ps.language.value
            lang_groups.setdefault(lang, []).append(ps)
        for lang, group in sorted(lang_groups.items()):
            yt, yp = self._extract_labels(group, task)
            by_language[lang] = compute_classification_metrics(yt, yp, labels=labels)

        # by ground-truth queue
        by_queue: Dict[str, ClassificationMetrics] = {}
        queue_groups: Dict[str, List[PredSave]] = {}
        for ps in pred_saves:
            gt_queue = ps.ground_truth.queue.value
            queue_groups.setdefault(gt_queue, []).append(ps)
        for q, group in sorted(queue_groups.items()):
            yt, yp = self._extract_labels(group, task)
            by_queue[q] = compute_classification_metrics(yt, yp, labels=labels)

        return TaskEvaluationResult(
            task=task,
            overall=overall,
            by_language=by_language,
            by_queue=by_queue,
        )

    def _extract_labels(
        self, pred_saves: List[PredSave], task: Task
    ) -> Tuple[List[str], List[str]]:
        """Extract y_true and y_pred string lists for the given task."""
        y_true: List[str] = []
        y_pred: List[str] = []
        for ps in pred_saves:
            if task == Task.QUEUE:
                y_true.append(ps.ground_truth.queue.value)
                y_pred.append(ps.predicted.queue.value)
            elif task == Task.PRIORITY:
                y_true.append(ps.ground_truth.priority.value)
                y_pred.append(ps.predicted.priority.value)
            else:
                raise ValueError(f"Unsupported task for evaluation: {task}")
        return y_true, y_pred
