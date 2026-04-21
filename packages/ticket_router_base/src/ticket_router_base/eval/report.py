"""Evaluation report serialization and console output."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from .evaluator import TaskEvaluationResult


@dataclass(frozen=True)
class EvaluationReport:
    """Complete evaluation report for a model across queue and priority tasks."""

    model_name: str
    pred_file_path: str
    queue_result: TaskEvaluationResult
    priority_result: TaskEvaluationResult
    error_summary: Dict[str, int]  # error flag name -> count
    total_samples: int

    def to_dict(self) -> dict:
        """Recursively convert to a plain dict for JSON serialization."""
        return {
            "model_name": self.model_name,
            "pred_file_path": self.pred_file_path,
            "total_samples": self.total_samples,
            "error_summary": self.error_summary,
            "queue_result": self.queue_result.to_dict(),
            "priority_result": self.priority_result.to_dict(),
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
        print(f"\n{'=' * 60}")
        print(f"Evaluation Report: {self.model_name}")
        print(f"{'=' * 60}")
        print(f"File: {self.pred_file_path}")
        print(f"Total samples: {self.total_samples}")
        print(f"Errors: {self.error_summary}")
        print()

        # queue summary
        q = self.queue_result.overall
        print("Queue Prediction:")
        print(f"  Accuracy:      {q.accuracy:.4f}")
        print(f"  Macro F1:      {q.macro_f1:.4f}")
        print(f"  Weighted F1:   {q.weighted_f1:.4f}")
        print()

        # priority summary
        p = self.priority_result.overall
        print("Priority Prediction:")
        print(f"  Accuracy:      {p.accuracy:.4f}")
        print(f"  Macro F1:      {p.macro_f1:.4f}")
        print(f"  Weighted F1:   {p.weighted_f1:.4f}")
        print()

        # language fairness
        print("By Language (Macro F1):")
        for lang, metrics in sorted(self.queue_result.by_language.items()):
            print(f"  {lang:>3}: Queue={metrics.macro_f1:.4f}", end="")
            if lang in self.priority_result.by_language:
                pm = self.priority_result.by_language[lang]
                print(f"  Priority={pm.macro_f1:.4f}")
            else:
                print()

        print(f"{'=' * 60}\n")
