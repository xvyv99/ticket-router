"""Evaluation report serialization and console output."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


from .evaluator import TaskEvaluationResult


@dataclass(frozen=True)
class EvaluationReport:
    """Complete evaluation report for a model across all dataset tasks."""

    model_name: str
    pred_file_path: str
    dataset_name: str
    task_results: List[TaskEvaluationResult]
    error_summary: Dict[str, int]  # error flag name -> count
    total_samples: int

    def to_dict(self) -> dict:
        """Recursively convert to a plain dict for JSON serialization."""
        return {
            "model_name": self.model_name,
            "pred_file_path": self.pred_file_path,
            "dataset_name": self.dataset_name,
            "total_samples": self.total_samples,
            "error_summary": self.error_summary,
            "task_results": [tr.to_dict() for tr in self.task_results],
        }

    def to_json(self, path: Path | None = None) -> str:
        """Serialize to a JSON string; write to file if path is provided."""
        s = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(s, encoding="utf-8")
        return s

    def print_summary(self) -> None:
        """Print a concise summary table to the console."""
        print(f"\n{'=' * 70}")
        print(f"Evaluation Report: {self.model_name}")
        print(f"{'=' * 70}")
        print(f"Dataset: {self.dataset_name}")
        print(f"File: {self.pred_file_path}")
        print(f"Total samples: {self.total_samples}")
        print(f"Errors: {self.error_summary}")
        print()

        for tr in self.task_results:
            print(f"Task: {tr.task_name}")
            print(f"  Accuracy:    {tr.overall.accuracy:.4f}")
            print(f"  Macro F1:    {tr.overall.macro_f1:.4f}")
            if tr.ordinal is not None:
                print(f"  MAE:         {tr.ordinal.mae:.4f}")
                print(f"  QWK:         {tr.ordinal.qwk:.4f}")
            print()

        # language fairness (using the first task as representative)
        if self.task_results:
            first = self.task_results[0]
            print("By Language (Macro F1):")
            for lang, metrics in sorted(first.by_language.items()):
                print(f"  {lang:>10}: {metrics.macro_f1:.4f}")

        print(f"{'=' * 70}\n")
