"""Task-level evaluation with multi-dimension aggregation."""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from ticket_router_base.datasets.base import BaseDataset
from ticket_router_base.types import PredSave

from .metrics_core import ClassificationMetrics, compute_classification_metrics


@dataclass(frozen=True)
class TaskEvaluationResult:
    """Evaluation result for a single task across multiple dimensions."""

    task_name: str
    overall: ClassificationMetrics
    by_language: Dict[str, ClassificationMetrics]
    by_strata: Dict[str, ClassificationMetrics] | None

    def to_dict(self) -> dict:
        return {
            "task_name": self.task_name,
            "overall": self.overall.to_dict(),
            "by_language": {k: v.to_dict() for k, v in self.by_language.items()},
            "by_strata": (
                {k: v.to_dict() for k, v in self.by_strata.items()}
                if self.by_strata is not None
                else None
            ),
        }


class TaskEvaluator:
    """Evaluate predictions for all classification tasks in a dataset."""

    def evaluate(
        self, pred_saves: List[PredSave], dataset: BaseDataset
    ) -> List[TaskEvaluationResult]:
        """Evaluate all classification tasks defined by the dataset descriptor."""
        results: List[TaskEvaluationResult] = []
        for task in dataset.classification_tasks:
            result = self._evaluate_task(pred_saves, task.name)
            results.append(result)
        return results

    def _evaluate_task(
        self, pred_saves: List[PredSave], task_name: str
    ) -> TaskEvaluationResult:
        """Evaluate a single classification task with global and breakdown metrics."""
        y_true_all, y_pred_all = self._extract_labels(pred_saves, task_name)
        labels = sorted(set(y_true_all) | set(y_pred_all))
        overall = compute_classification_metrics(y_true_all, y_pred_all, labels=labels)

        # by language
        by_language: Dict[str, ClassificationMetrics] = {}
        lang_groups: Dict[str, List[PredSave]] = {}
        for ps in pred_saves:
            lang = ps.language or "unknown"
            lang_groups.setdefault(lang, []).append(ps)
        for lang, group in sorted(lang_groups.items()):
            yt, yp = self._extract_labels(group, task_name)
            by_language[lang] = compute_classification_metrics(yt, yp, labels=labels)

        # by ground-truth label (formerly by_queue)
        by_strata: Dict[str, ClassificationMetrics] = {}
        strata_groups: Dict[str, List[PredSave]] = {}
        for ps in pred_saves:
            gt_label = ps.ground_truth.labels.get(task_name, "unknown")
            strata_groups.setdefault(gt_label, []).append(ps)
        for label_val, group in sorted(strata_groups.items()):
            yt, yp = self._extract_labels(group, task_name)
            by_strata[label_val] = compute_classification_metrics(yt, yp, labels=labels)

        return TaskEvaluationResult(
            task_name=task_name,
            overall=overall,
            by_language=by_language,
            by_strata=by_strata,
        )

    def _extract_labels(
        self, pred_saves: List[PredSave], task_name: str
    ) -> Tuple[List[str], List[str]]:
        """Extract y_true and y_pred for a specific task from PredSave list."""
        y_true: List[str] = []
        y_pred: List[str] = []
        for ps in pred_saves:
            y_true.append(ps.ground_truth.labels.get(task_name, ""))
            y_pred.append(ps.predicted.labels.get(task_name, ""))
        return y_true, y_pred
