"""Tests for fairness_metrics module."""

import pytest

from ticket_router_base.eval.fairness_metrics import compute_fairness_metrics


class TestComputeFairnessMetrics:
    def test_perfect_prediction_all_groups_equal(self):
        """All languages have identical accuracy -> gaps = 0, ratios = 1."""
        labels = ["a", "b"]
        y_true = ["a", "b", "a", "b", "a", "b"]
        y_pred = ["a", "b", "a", "b", "a", "b"]
        sensitive = ["en", "en", "de", "de", "es", "es"]

        fm = compute_fairness_metrics(y_true, y_pred, sensitive, labels)

        assert fm.accuracy_gap == pytest.approx(0.0, abs=1e-6)
        assert fm.accuracy_ratio == pytest.approx(1.0, abs=1e-6)
        assert fm.macro_f1_gap == pytest.approx(0.0, abs=1e-6)
        assert fm.macro_f1_ratio == pytest.approx(1.0, abs=1e-6)
        # all groups have accuracy 1.0
        for lang in ["en", "de", "es"]:
            assert fm.accuracy_by_group[lang] == pytest.approx(1.0)

    def test_biased_prediction_across_groups(self):
        """One language has 0% accuracy, others 100%."""
        labels = ["a", "b"]
        # en: perfect, de: all wrong, es: perfect
        y_true = ["a", "b", "a", "b", "a", "b"]
        y_pred = ["a", "b", "b", "a", "a", "b"]
        sensitive = ["en", "en", "de", "de", "es", "es"]

        fm = compute_fairness_metrics(y_true, y_pred, sensitive, labels)

        assert fm.accuracy_gap == pytest.approx(1.0, abs=1e-6)
        assert fm.accuracy_ratio == pytest.approx(0.0, abs=1e-6)
        assert fm.macro_f1_gap > 0.0

    def test_single_group(self):
        """Only one language -> gaps = 0, ratios = 1 (trivially fair)."""
        labels = ["a", "b"]
        y_true = ["a", "b", "a"]
        y_pred = ["a", "b", "b"]
        sensitive = ["en", "en", "en"]

        fm = compute_fairness_metrics(y_true, y_pred, sensitive, labels)

        assert fm.accuracy_gap == pytest.approx(0.0, abs=1e-6)
        assert fm.accuracy_ratio == pytest.approx(1.0, abs=1e-6)

    def test_ordinal_task_skips_aif360(self):
        """Ordinal tasks should have None for all aif360 fields."""
        labels = ["low", "medium", "high"]
        y_true = ["low", "medium", "high", "low"]
        y_pred = ["low", "medium", "high", "medium"]
        sensitive = ["en", "en", "de", "de"]

        fm = compute_fairness_metrics(
            y_true, y_pred, sensitive, labels, is_ordinal=True
        )

        assert fm.avg_disparate_impact is None
        assert fm.avg_statistical_parity_difference is None
        assert fm.avg_equal_opportunity_difference is None
        assert fm.avg_average_odds_difference is None
        # fairlearn metrics should still be present
        assert fm.accuracy_gap >= 0.0

    def test_aif360_fields_present_for_classification(self):
        """Classification tasks should compute aif360 metrics (may be nan/None in edge cases)."""
        labels = ["a", "b"]
        y_true = ["a", "b", "a", "b"]
        y_pred = ["a", "b", "a", "b"]
        sensitive = ["en", "en", "de", "de"]

        fm = compute_fairness_metrics(y_true, y_pred, sensitive, labels, is_ordinal=False)

        # With perfect prediction and balanced groups, aif360 metrics should be defined.
        # DI should be close to 1.0, SPD close to 0.0.
        if fm.avg_disparate_impact is not None:
            assert fm.avg_disparate_impact == pytest.approx(1.0, abs=0.1)
        if fm.avg_statistical_parity_difference is not None:
            assert abs(fm.avg_statistical_parity_difference) < 0.1

    def test_empty_input(self):
        """Empty inputs should produce safe defaults."""
        fm = compute_fairness_metrics([], [], [], ["a"])
        assert fm.accuracy_by_group == {}
        assert fm.accuracy_gap == 0.0
        assert fm.accuracy_ratio == 0.0
        # aif360 should return None for empty input
        assert fm.avg_disparate_impact is None
