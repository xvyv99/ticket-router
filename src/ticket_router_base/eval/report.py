"""Evaluation report serialization and console output."""

from typing import Dict, List

from pydantic import BaseModel

from .evaluator import TaskEvaluationResult, is_ordinal_metrics


class EvaluationReport(BaseModel):
    """Complete evaluation report for a model across all dataset tasks."""

    model_name: str
    pred_file_path: str
    dataset_name: str
    task_results: List[TaskEvaluationResult]
    error_summary: Dict[str, int]  # error flag name -> count
    total_samples: int

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

        for task in self.task_results:
            print(f"Task: {task.task_name}")
            print(f"  Accuracy:    {task.performance.accuracy:.4f}")
            print(f"  Macro F1:    {task.performance.macro_f1:.4f}")
            if is_ordinal_metrics(task.performance):
                print(f"  MAE:         {task.performance.mae:.4f}")
                print(f"  QWK:         {task.performance.qwk:.4f}")

                fm = task.fairness

                for key, fm in task.fairness.items():
                    print(f"  Fairness ({key}):")
                    print(f"    Acc Gap:   {fm.accuracy_gap:.4f}")
                    print(f"    Acc Ratio: {fm.accuracy_ratio:.4f}")
                    print(f"    F1 Gap:    {fm.macro_f1_gap:.4f}")
                    print(f"    F1 Ratio:  {fm.macro_f1_ratio:.4f}")
                    if fm.avg_disparate_impact is not None:
                        print(f"    Avg DI:    {fm.avg_disparate_impact:.4f}")
                    if fm.avg_statistical_parity_difference is not None:
                        print(
                            f"    Avg SPD:   {fm.avg_statistical_parity_difference:.4f}"
                        )

        print(f"{'=' * 70}\n")
