"""Typed classification metrics with sklearn backend.

This module re-exports no symbols from eval.metrics.py; it is a clean replacement
with strict dataclass typing.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass(frozen=True)
class PerClassMetrics:
    """Per-class classification metrics."""

    precision: float
    recall: float
    f1: float
    support: int  # number of ground-truth samples for this class


@dataclass(frozen=True)
class ClassificationMetrics:
    """Complete classification metrics for a single task."""

    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: Dict[str, PerClassMetrics]
    confusion_matrix: List[List[int]]
    support: int  # total number of samples
    labels: List[str]  # label order, matching confusion_matrix rows/cols


@dataclass(frozen=True)
class OrdinalMetrics(ClassificationMetrics):
    """Ordinal classification metrics that respect label ordering."""

    mae: float  # Mean Absolute Error in label-index space
    qwk: float  # Quadratic Weighted Kappa


def compute_classification_metrics(
    y_true: List[str], y_pred: List[str], labels: List[str] | None = None
) -> ClassificationMetrics:
    """Compute classification metrics and return a typed dataclass.

    The underlying computation uses sklearn, but the return type is a strict
    dataclass with per-class breakdowns.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true ({len(y_true)}) and y_pred ({len(y_pred)}) must have same length"
        )

    if len(y_true) == 0:
        raise ValueError("y_true and y_pred must not be empty")

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_precision = float(
        precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    )
    macro_recall = float(
        recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    )
    macro_f1 = float(
        f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    # per-class metrics — compute all at once to avoid repeated sklearn calls
    per_class: Dict[str, PerClassMetrics] = {}
    _precisions = precision_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    _recalls = recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    _f1s = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    assert isinstance(_precisions, np.ndarray), (
        "Expected precision_score with average=None to return np.ndarray"
    )
    assert isinstance(_recalls, np.ndarray), (
        "Expected recall_score with average=None to return np.ndarray"
    )
    assert isinstance(_f1s, np.ndarray), (
        "Expected f1_score with average=None to return np.ndarray"
    )

    label2idx = {label: i for i, label in enumerate(labels)}
    for label in labels:
        idx = label2idx[label]
        p = float(_precisions[idx])
        r = float(_recalls[idx])
        f = float(_f1s[idx])
        support = sum(1 for yt in y_true if yt == label)
        per_class[label] = PerClassMetrics(precision=p, recall=r, f1=f, support=support)

    return ClassificationMetrics(
        accuracy=accuracy,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        per_class=per_class,
        confusion_matrix=cm,
        support=len(y_true),
        labels=labels,
    )


def compute_ordinal_metrics(
    y_true: List[str], y_pred: List[str], labels: List[str]
) -> OrdinalMetrics:
    """Compute ordinal classification metrics respecting label order.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        labels: Ordered list of labels from lowest to highest.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true ({len(y_true)}) and y_pred ({len(y_pred)}) must have same length"
        )
    if len(y_true) == 0:
        raise ValueError("y_true and y_pred must not be empty")

    # classification metrics (reuse same logic)
    cls = compute_classification_metrics(y_true, y_pred, labels=labels)

    # convert labels to ordered integer indices
    label2idx = {label: i for i, label in enumerate(labels)}

    # QWK requires at least 2 distinct ratings in both vectors
    assert len(label2idx) >= 2, (
        "At least 2 distinct labels are required for ordinal metrics"
    )

    yt_idx = [label2idx.get(y, -1) for y in y_true]
    yp_idx = [label2idx.get(y, -1) for y in y_pred]

    assert all(p >= 0 for p in yp_idx), "Invalid predicted label detected"
    assert all(t >= 0 for t in yt_idx), "Invalid true label detected"

    mae = float(np.mean([abs(t - p) for t, p in zip(yt_idx, yp_idx)]))
    qwk = float(cohen_kappa_score(yt_idx, yp_idx, weights="quadratic"))

    return OrdinalMetrics(
        accuracy=cls.accuracy,
        macro_f1=cls.macro_f1,
        macro_precision=cls.macro_precision,
        macro_recall=cls.macro_recall,
        mae=mae,
        qwk=qwk,
        per_class=cls.per_class,
        confusion_matrix=cls.confusion_matrix,
        support=cls.support,
        labels=labels,
    )
