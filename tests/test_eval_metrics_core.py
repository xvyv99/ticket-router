"""Tests for eval.metrics_core — typed classification metrics with sklearn backend."""

import pytest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from ticket_router_eval.metrics import (
    PerClassMetrics,
    compute_classification_metrics,
)


class TestComputeClassificationMetrics:
    """Tests for compute_classification_metrics correctness."""

    def test_perfect_prediction(self) -> None:
        """All predictions correct -> all metrics = 1.0."""
        y_true = ["a", "b", "c", "a", "b", "c"]
        y_pred = ["a", "b", "c", "a", "b", "c"]
        result = compute_classification_metrics(y_true, y_pred)

        assert result.accuracy == 1.0
        assert result.macro_precision == 1.0
        assert result.macro_recall == 1.0
        assert result.macro_f1 == 1.0
        assert result.support == 6
        for label in ["a", "b", "c"]:
            pcm = result.per_class[label]
            assert pcm.precision == 1.0
            assert pcm.recall == 1.0
            assert pcm.f1 == 1.0

    def test_all_wrong_single_class(self) -> None:
        """All predictions misclassified to a single class."""
        y_true = ["a", "a", "b", "b", "c", "c"]
        y_pred = ["a", "a", "a", "a", "a", "a"]
        result = compute_classification_metrics(y_true, y_pred)

        assert result.accuracy == 2 / 6
        # class "a": precision=2/6, recall=2/2=1.0
        # class "b": precision=0/0=0, recall=0/2=0
        # class "c": precision=0/0=0, recall=0/2=0
        assert result.per_class["a"].precision == pytest.approx(2 / 6)
        assert result.per_class["a"].recall == 1.0
        assert result.per_class["b"].precision == 0.0
        assert result.per_class["b"].recall == 0.0
        assert result.per_class["c"].precision == 0.0
        assert result.per_class["c"].recall == 0.0

    def test_binary_balanced(self) -> None:
        """Manually construct TP=8, FP=2, FN=3, TN=7 and verify metrics."""
        y_true = ["pos"] * 11 + ["neg"] * 9
        y_pred = ["pos"] * 8 + ["neg"] * 3 + ["pos"] * 2 + ["neg"] * 7
        # pos: TP=8, FP=2, FN=3 -> precision=8/10, recall=8/11
        # neg: TN=7, FP=3, FN=2 -> precision=7/9, recall=7/9
        result = compute_classification_metrics(y_true, y_pred, labels=["pos", "neg"])

        assert result.per_class["pos"].precision == pytest.approx(8 / 10)
        assert result.per_class["pos"].recall == pytest.approx(8 / 11)
        expected_f1_pos = 2 * (8 / 10) * (8 / 11) / ((8 / 10) + (8 / 11))
        assert result.per_class["pos"].f1 == pytest.approx(expected_f1_pos)

    def test_multi_class_imbalanced(self) -> None:
        """Macro F1 should be lower than accuracy when minority classes fail."""
        y_true = ["a"] * 90 + ["b"] * 9 + ["c"] * 1
        y_pred = ["a"] * 85 + ["b"] * 5 + ["b"] * 4 + ["c"] * 5 + ["a"] * 1
        result = compute_classification_metrics(y_true, y_pred)

        # macro penalizes minority class performance
        assert result.macro_f1 < result.accuracy

    def test_class_with_zero_support(self) -> None:
        """A label in labels but not in y_true should have zero support and zero metrics."""
        y_true = ["a", "a", "b", "b"]
        y_pred = ["a", "a", "b", "b"]
        result = compute_classification_metrics(y_true, y_pred, labels=["a", "b", "c"])

        assert result.per_class["c"].support == 0
        assert result.per_class["c"].precision == 0.0
        assert result.per_class["c"].recall == 0.0
        assert result.per_class["c"].f1 == 0.0
        # macro should include the zero class, pulling average down
        assert result.macro_f1 < 1.0

    def test_empty_input_raises(self) -> None:
        """Empty y_true/y_pred should raise ValueError."""
        with pytest.raises(ValueError):
            compute_classification_metrics([], [])

    def test_length_mismatch_raises(self) -> None:
        """Mismatched lengths should raise ValueError."""
        with pytest.raises(ValueError):
            compute_classification_metrics(["a", "b"], ["a"])

    def test_confusion_matrix_labels_order(self) -> None:
        """Confusion matrix rows/cols must strictly follow labels order."""
        y_true = ["b", "a", "b", "c", "a", "c"]
        y_pred = ["a", "a", "b", "c", "c", "c"]
        labels = ["a", "b", "c"]
        result = compute_classification_metrics(y_true, y_pred, labels=labels)

        # manual cm for labels=[a, b, c]
        # a: true=2, pred [a,a,c] -> a:1, b:0, c:1
        # b: true=2, pred [a,b] -> a:1, b:1, c:0
        # c: true=2, pred [c,c] -> a:0, b:0, c:2
        expected_cm = [[1, 0, 1], [1, 1, 0], [0, 0, 2]]
        assert result.confusion_matrix == expected_cm
        assert result.labels == labels

    def test_against_sklearn(self) -> None:
        """Cross-validate against sklearn on random data."""
        import random

        random.seed(42)
        labels = ["x", "y", "z"]
        y_true = [random.choice(labels) for _ in range(200)]
        y_pred = [random.choice(labels) for _ in range(200)]

        result = compute_classification_metrics(y_true, y_pred, labels=labels)

        assert result.accuracy == pytest.approx(
            float(accuracy_score(y_true, y_pred)), abs=1e-10
        )
        assert result.macro_precision == pytest.approx(
            float(
                precision_score(
                    y_true, y_pred, average="macro", labels=labels, zero_division=0
                )
            ),
            abs=1e-10,
        )
        assert result.macro_recall == pytest.approx(
            float(
                recall_score(
                    y_true, y_pred, average="macro", labels=labels, zero_division=0
                )
            ),
            abs=1e-10,
        )
        assert result.macro_f1 == pytest.approx(
            float(
                f1_score(
                    y_true, y_pred, average="macro", labels=labels, zero_division=0
                )
            ),
            abs=1e-10,
        )
        assert (
            result.confusion_matrix
            == confusion_matrix(y_true, y_pred, labels=labels).tolist()
        )


class TestPerClassMetrics:
    def test_to_dict(self) -> None:
        pcm = PerClassMetrics(precision=0.5, recall=0.6, f1=0.55, support=10)
        d = pcm.model_dump()
        assert d == {"precision": 0.5, "recall": 0.6, "f1": 0.55, "support": 10}


class TestClassificationMetrics:
    def test_to_dict_roundtrip(self) -> None:
        y_true = ["a", "b", "a"]
        y_pred = ["a", "b", "b"]
        result = compute_classification_metrics(y_true, y_pred)
        d = result.model_dump()

        assert d["accuracy"] == result.accuracy
        assert d["macro_f1"] == result.macro_f1
        assert d["labels"] == result.labels
        assert "per_class" in d
        assert "confusion_matrix" in d
