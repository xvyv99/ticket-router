"""Task-level evaluation with multi-dimension aggregation."""

from typing import Dict, List, Tuple, TypeGuard
from logging import getLogger

from pydantic import BaseModel

from ticket_router_base.data import BaseDataset
from ticket_router_base.types import PredSave

from .metrics import (
    ClassificationMetrics,
    OrdinalMetrics,
    compute_classification_metrics,
    compute_ordinal_metrics,
)
from .fairness_metrics import FairnessMetrics, compute_fairness_metrics

logger = getLogger(__name__)


def is_ordinal_metrics(
    x: ClassificationMetrics | OrdinalMetrics,
) -> TypeGuard[OrdinalMetrics]:
    return isinstance(x, OrdinalMetrics)


class TaskEvaluationResult(BaseModel):
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

    @staticmethod
    def _evaluate_perf(
        y_true: List[str], y_pred: List[str], is_ordinal: bool
    ) -> ClassificationMetrics | OrdinalMetrics:
        """Evaluate performance metrics for a single task."""
        labels = sorted(set(y_true) | set(y_pred))
        if is_ordinal:
            return compute_ordinal_metrics(y_true, y_pred, labels=labels)
        else:
            return compute_classification_metrics(y_true, y_pred, labels=labels)

    def _evaluate_fairness(
        self,
        y_true: List[str],
        y_pred: List[str],
        sensitive_lst: List[str],
        labels: List[str],
    ) -> FairnessMetrics:
        # TODO
        ...

    def _evaluate_task(
        self,
        pred_saves: List[PredSave],
        task_name: str,
        dataset: BaseDataset,
        is_ordinal: bool = False,
    ) -> TaskEvaluationResult:
        """Evaluate a single classification or ordinal task with global and breakdown metrics."""

        print(f"Evaluating task '{task_name}' with {len(pred_saves)} samples...")

        y_true_all, y_pred_all = self._extract_labels(pred_saves, task_name)
        labels = sorted(set(y_true_all) | set(y_pred_all))

        perf = self._evaluate_perf(y_true_all, y_pred_all, is_ordinal=is_ordinal)

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
    ) -> Tuple[List[str], List[str]]:
        """Extract y_true and y_pred for a specific task from PredSave list."""
        y_true: List[str] = []
        y_pred: List[str] = []
        # TODO: None value handling - currently treated as a separate category, but may want to exclude or impute
        for ps in pred_saves:
            ground = ps.ground_truth.labels.get(task_col_name)
            assert ground is not None, (
                f"Missing ground truth for task '{task_col_name}' in PredSave with id {ps.predicted.request_id}"
            )
            pred = ps.predicted.labels.get(task_col_name)
            assert pred is not None, (
                f"Missing prediction for task '{task_col_name}' in PredSave with id {ps.predicted.request_id}"
            )

            y_true.append(ground)
            y_pred.append(pred)
        return y_true, y_pred
