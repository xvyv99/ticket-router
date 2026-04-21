"""Integration tests for eval — end-to-end with real prediction files."""

import json
from pathlib import Path

import pytest

from ticket_router_base.eval import evaluate_file
from ticket_router_base.eval.report import EvaluationReport


class TestEvaluateFile:
    """Integration tests using real model outputs."""

    @pytest.fixture
    def lr_pred_path(self) -> Path:
        """Path to LR predictions JSONL (relative to project root)."""
        # tests run from packages/ticket_router_base, go up two levels to project root
        path = Path("../../outputs/supervised/lr_predictions.jsonl").resolve()
        if not path.exists():
            pytest.skip(f"LR predictions not found at {path}")
        return path

    def test_evaluate_lr_end_to_end(self, lr_pred_path: Path) -> None:
        """Full pipeline: load -> evaluate queue + priority -> report -> JSON."""
        report = evaluate_file(lr_pred_path)

        assert isinstance(report, EvaluationReport)
        assert report.model_name == "lr_predictions"
        assert report.total_samples > 0

        # queue result
        assert report.queue_result.task.value == "queue"
        assert report.queue_result.overall.accuracy >= 0.0
        assert report.queue_result.overall.accuracy <= 1.0
        assert report.queue_result.overall.support == report.total_samples

        # priority result
        assert report.priority_result.task.value == "priority"
        assert report.priority_result.overall.accuracy >= 0.0
        assert report.priority_result.overall.accuracy <= 1.0

        # by_language breakdown
        assert len(report.queue_result.by_language) > 0
        for lang, metrics in report.queue_result.by_language.items():
            assert 0.0 <= metrics.accuracy <= 1.0

        # by_queue breakdown
        assert report.queue_result.by_queue is not None
        assert len(report.queue_result.by_queue) > 0

        # error summary
        assert len(report.error_summary) >= 1
        assert sum(report.error_summary.values()) == report.total_samples

    def test_evaluate_lr_json_roundtrip(
        self, lr_pred_path: Path, tmp_path: Path
    ) -> None:
        """Report JSON must be fully serializable and parseable."""
        report = evaluate_file(lr_pred_path)
        out_path = tmp_path / "lr_report.json"
        report.to_json(out_path)

        assert out_path.exists()
        loaded = json.loads(out_path.read_text(encoding="utf-8"))

        assert loaded["model_name"] == "lr_predictions"
        assert loaded["total_samples"] == report.total_samples
        assert "queue_result" in loaded
        assert "priority_result" in loaded
        assert "error_summary" in loaded
