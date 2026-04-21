"""Integration tests for eval — end-to-end with real prediction files."""

import json
from pathlib import Path

import pytest

from ticket_router_base.data.dataset import MultilingualCustomerSupportDataset
from ticket_router_base.eval import evaluate_file
from ticket_router_base.eval.report import EvaluationReport


class TestEvaluateFile:
    """Integration tests using real model outputs."""

    @pytest.fixture
    def dataset(self):
        return MultilingualCustomerSupportDataset()

    @pytest.fixture
    def lr_pred_path(self) -> Path:
        """Path to LR predictions JSONL."""
        path = Path("../../outputs/supervised/lr_predictions.jsonl").resolve()
        if not path.exists():
            pytest.skip(f"LR predictions not found at {path}")
        return path

    def test_evaluate_lr_end_to_end(self, lr_pred_path: Path, dataset) -> None:
        """Full pipeline: load -> evaluate -> report -> JSON."""
        # NOTE: old pred JSONL uses legacy format (queue/priority fields).
        # This test will be skipped or need migration until JSONL is regenerated.
        report = evaluate_file(lr_pred_path, dataset)

        assert isinstance(report, EvaluationReport)
        assert report.model_name == "lr_predictions"
        assert report.total_samples > 0

        # queue result
        queue_result = [r for r in report.task_results if r.task_name == "queue"][0]
        assert queue_result.overall.accuracy >= 0.0
        assert queue_result.overall.accuracy <= 1.0
        assert queue_result.overall.support == report.total_samples

        # priority result
        priority_result = [r for r in report.task_results if r.task_name == "priority"][
            0
        ]
        assert priority_result.overall.accuracy >= 0.0
        assert priority_result.overall.accuracy <= 1.0

        # by_language breakdown
        assert len(queue_result.by_language) > 0
        for lang, metrics in queue_result.by_language.items():
            assert 0.0 <= metrics.accuracy <= 1.0

        # by_strata breakdown
        assert queue_result.by_strata is not None
        assert len(queue_result.by_strata) > 0

        # error summary
        assert len(report.error_summary) >= 1
        assert sum(report.error_summary.values()) == report.total_samples

    def test_evaluate_lr_json_roundtrip(
        self, lr_pred_path: Path, dataset, tmp_path: Path
    ) -> None:
        """Report JSON must be fully serializable and parseable."""
        report = evaluate_file(lr_pred_path, dataset)
        out_path = tmp_path / "lr_report.json"
        report.to_json(out_path)

        assert out_path.exists()
        loaded = json.loads(out_path.read_text(encoding="utf-8"))
        assert loaded["model_name"] == "lr_predictions"
        assert loaded["total_samples"] == report.total_samples
        assert "task_results" in loaded
        assert "error_summary" in loaded
