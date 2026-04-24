"""Fairness metrics using fairlearn (primary) and aif360 (supplementary)."""

from typing import Dict, List, cast
import math

from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from fairlearn.metrics import MetricFrame

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric


class FairlearnMetrics(BaseModel):
    """Fairlearn MetricFrame results for performance parity."""

    accuracy_by_group: Dict[str, float]
    macro_f1_by_group: Dict[str, float]
    accuracy_gap: float
    accuracy_ratio: float
    macro_f1_gap: float
    macro_f1_ratio: float


class AIF360Metrics(BaseModel):
    """AIF360 one-vs-rest aggregated metrics for binary classification tasks."""

    avg_disparate_impact: float | None
    avg_statistical_parity_difference: float | None
    avg_equal_opportunity_difference: float | None
    avg_average_odds_difference: float | None


class FairnessMetrics(FairlearnMetrics, AIF360Metrics):
    """Fairness audit results for a single task with sensitive attribute."""

    @staticmethod
    def merge_from(
        fairlearn: FairlearnMetrics, aif360: AIF360Metrics
    ) -> "FairnessMetrics":
        """Merge separate Fairlearn and AIF360 results into a single dataclass."""
        return FairnessMetrics(**fairlearn.model_dump(), **aif360.model_dump())


def _fairlearn_metrics(
    y_true: List[str],
    y_pred: List[str],
    sensitive: List[str],
    labels: List[str],
) -> FairlearnMetrics:
    """Compute per-group accuracy and macro_f1 via fairlearn MetricFrame."""

    def _macro_f1(y_true, y_pred, labels):
        # Wrapper to handle zero-division gracefully in fairlearn MetricFrame
        return f1_score(y_true, y_pred, average="macro", labels=labels)

    mf = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "macro_f1": lambda yt, yp: _macro_f1(yt, yp, labels),
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive,
    )

    accuracy_by_group = mf.by_group["accuracy"].to_dict()
    macro_f1_by_group = mf.by_group["macro_f1"].to_dict()

    diff = mf.difference(method="between_groups").to_dict()
    ratio = mf.ratio(method="between_groups").to_dict()

    return FairlearnMetrics(
        accuracy_by_group=cast(Dict[str, float], accuracy_by_group),
        macro_f1_by_group=cast(Dict[str, float], macro_f1_by_group),
        accuracy_gap=float(diff["accuracy"]),
        accuracy_ratio=float(ratio["accuracy"]),
        macro_f1_gap=float(diff["macro_f1"]),
        macro_f1_ratio=float(ratio["macro_f1"]),
    )


def _aif360_ovr_metrics(
    y_true: List[str],
    y_pred: List[str],
    sensitive: List[str],
    labels: List[str],
) -> AIF360Metrics:
    """Compute aggregated one-vs-rest fairness metrics via aif360.

    For each class c, we binarize y_true and y_pred to (is_c, is_not_c),
    then compute pairwise (unprivileged, privileged) metrics across all sensitive attributes.
    Results are averaged across classes and sensitive attribute pairs.
    """

    n_classes = len(labels)

    # Encode sensitive attributes and labels to integers
    sensitive_values = sorted(set(sensitive))
    sensitive2idx = {v: i for i, v in enumerate(sensitive_values)}
    label2idx = {label: i for i, label in enumerate(labels)}

    sensitive_idx = np.array([sensitive2idx[s] for s in sensitive])

    for y in y_true + y_pred:
        assert y in label2idx, f"Found label '{y}' not in provided labels list"

    y_true_idx = np.array([label2idx[y] for y in y_true])
    y_pred_idx = np.array([label2idx[y] for y in y_pred])

    di_vals: List[float] = []
    spd_vals: List[float] = []
    eod_vals: List[float] = []
    aod_vals: List[float] = []

    for c in range(n_classes):
        y_true_bin = (y_true_idx == c).astype(int)
        y_pred_bin = (y_pred_idx == c).astype(int)

        df = pd.DataFrame(
            {"label": y_true_bin, "sensitive": sensitive_idx, "pred": y_pred_bin}
        )

        dataset = BinaryLabelDataset(
            df=df,
            label_names=["label"],
            protected_attribute_names=["sensitive"],
            favorable_label=1,
            unfavorable_label=0,
        )

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_pred_bin.reshape(-1, 1)

        for u_val in range(len(sensitive_values)):
            for p_val in range(len(sensitive_values)):
                if u_val == p_val:
                    continue

                metric = ClassificationMetric(
                    dataset,
                    dataset_pred,
                    unprivileged_groups=[{"sensitive": u_val}],
                    privileged_groups=[{"sensitive": p_val}],
                )
                di = metric.disparate_impact()
                spd = metric.statistical_parity_difference()
                eod = metric.equal_opportunity_difference()
                aod = metric.average_odds_difference()

                if not math.isnan(di) and not math.isinf(di):
                    di_vals.append(float(di))
                if not math.isnan(spd):
                    spd_vals.append(float(spd))
                if not math.isnan(eod):
                    eod_vals.append(float(eod))
                if not math.isnan(aod):
                    aod_vals.append(float(aod))

    def _avg(vals: List[float]) -> float | None:
        return sum(vals) / len(vals) if len(vals) > 0 else None

    return AIF360Metrics(
        avg_disparate_impact=_avg(di_vals),
        avg_statistical_parity_difference=_avg(spd_vals),
        avg_equal_opportunity_difference=_avg(eod_vals),
        avg_average_odds_difference=_avg(aod_vals),
    )


def compute_fairness_metrics(
    y_true: List[str],
    y_pred: List[str],
    sensitive: List[str],
    labels: List[str],
) -> FairnessMetrics:
    """Compute fairness metrics for a single task.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        sensitive: Sensitive attribute values .
        labels: Ordered list of valid labels for this task.
        is_ordinal: If True, skip aif360 binary metrics.

    Returns:
        FairnessMetrics dataclass with both fairlearn and aif360 results.
    """

    fl = _fairlearn_metrics(y_true, y_pred, sensitive, labels)
    aif360 = _aif360_ovr_metrics(y_true, y_pred, sensitive, labels)

    return FairnessMetrics.merge_from(fl, aif360)
