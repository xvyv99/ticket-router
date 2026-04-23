"""Task-level evaluation with multi-dimension aggregation."""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from ticket_router_base.data import BaseDataset
from ticket_router_base.types import PredSave

from .metrics import (
    ClassificationMetrics,
    OrdinalMetrics,
    compute_classification_metrics,
    compute_ordinal_metrics,
)
from .fairness_metrics import FairnessMetrics, compute_fairness_metrics


@dataclass(frozen=True)
class TaskEvaluationResult:
    """Evaluation result for a single task across multiple dimensions."""

    task_name: str
    performance: ClassificationMetrics | OrdinalMetrics
    fairness: Dict[str, FairnessMetrics]  # fairness audit by sensitive attribute


class TaskEvaluator:
    """Evaluate predictions for all classification tasks in a dataset."""

    def evaluate(
        self, pred_saves: List[PredSave], dataset: BaseDataset
    ) -> List[TaskEvaluationResult]:
        """Evaluate all classification and ordinal tasks defined by the dataset descriptor."""
        results: List[TaskEvaluationResult] = []
        for task in dataset.classification_tasks:
            result = self._evaluate_task(
                pred_saves, task.name, dataset, is_ordinal=False
            )
            results.append(result)
        for task in dataset.ordinal_tasks:
            result = self._evaluate_task(
                pred_saves, task.name, dataset, is_ordinal=True
            )
            results.append(result)

        return results

    def _evaluate_task(
        self,
        pred_saves: List[PredSave],
        task_name: str,
        dataset: BaseDataset,
        is_ordinal: bool = False,
    ) -> TaskEvaluationResult:
        """Evaluate a single classification or ordinal task with global and breakdown metrics."""
        y_true_all, y_pred_all = self._extract_labels(pred_saves, task_name)
        labels = sorted(set(y_true_all) | set(y_pred_all))

        perf: ClassificationMetrics | OrdinalMetrics | None = None
        if is_ordinal:
            perf = compute_ordinal_metrics(y_true_all, y_pred_all, labels=labels)
        else:
            perf = compute_classification_metrics(y_true_all, y_pred_all, labels=labels)

        # fairness audit
        fairness_dict: Dict[str, FairnessMetrics] = {}

        for sensitive_attr in dataset.sensitive_columns:
            sensitive_lst = []

            for ps in pred_saves:
                sensitive_val = ps.ground_truth.sensitive_attributes.get(sensitive_attr)
                assert sensitive_val is not None
                sensitive_lst.append(sensitive_val)

            fairness_res = compute_fairness_metrics(
                y_true_all,
                y_pred_all,
                sensitive_lst,
                labels=labels,
            )
            fairness_dict[sensitive_attr] = fairness_res

        return TaskEvaluationResult(
            task_name=task_name,
            performance=perf,
            fairness=fairness_dict,
        )

    def _extract_labels(
        self, pred_saves: List[PredSave], task_col_name: str
    ) -> Tuple[List[str | None], List[str | None]]:
        """Extract y_true and y_pred for a specific task from PredSave list."""
        y_true: List[str | None] = []
        y_pred: List[str | None] = []
        # TODO: None value handling - currently treated as a separate category, but may want to exclude or impute
        for ps in pred_saves:
            y_true.append(ps.ground_truth.labels.get(task_col_name, None))
            y_pred.append(ps.predicted.labels.get(task_col_name, None))
        return y_true, y_pred
