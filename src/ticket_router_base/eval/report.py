"""Evaluation report serialization and console output."""

from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path

from rich.console import Console
from rich.table import Table


from ticket_router_base.data import BaseDataset

from .evaluator import TaskEvaluationResult

console = Console()


@dataclass
class EvaluationReport:
    """Complete evaluation report for a model across all dataset tasks."""

    model_name: str
    dataset: BaseDataset

    file_path: Path
    task_results: List[TaskEvaluationResult]

    error_summary: Dict[str, int]  # error flag name -> count
    total_samples: int


def print_overall_report(reports: List[EvaluationReport]) -> None:
    if not reports:
        console.print("[red]No reports provided[/red]")
        return

    dataset_names = set(r.dataset.name for r in reports)
    if len(dataset_names) != 1:
        raise ValueError(f"Reports must share same dataset: {dataset_names}")

    all_tasks = sorted({t.task_name for r in reports for t in r.task_results})

    fairness_keys = sorted(
        {key for r in reports for t in r.task_results for key in t.fairness.keys()}
    )

    def get_task(
        report: EvaluationReport, task_name: str
    ) -> TaskEvaluationResult | None:
        return next(
            (t for t in report.task_results if t.task_name == task_name),
            None,
        )

    for task_name in all_tasks:
        table = Table(
            title=f"Task: {task_name}",
            show_lines=False,
        )

        table.add_column("Metric")

        for r in reports:
            table.add_column(r.model_name, justify="right")

        base_metrics = [
            ("Accuracy", lambda p: p.accuracy),
            ("Macro F1", lambda p: p.macro_f1),
            ("MAE", lambda p: getattr(p, "mae", None)),
            ("QWK", lambda p: getattr(p, "qwk", None)),
        ]

        for metric_name, getter in base_metrics:
            row = [metric_name]

            for r in reports:
                task = get_task(r, task_name)
                if task is None:
                    row.append("-")
                    continue

                val = getter(task.performance)
                row.append(f"{val:.4f}" if val is not None else "-")

            table.add_row(*row)

        fairness_metrics = [
            ("Acc Gap", lambda fm: fm.accuracy_gap),
            ("Acc Ratio", lambda fm: fm.accuracy_ratio),
            ("F1 Gap", lambda fm: fm.macro_f1_gap),
            ("F1 Ratio", lambda fm: fm.macro_f1_ratio),
            ("DI", lambda fm: fm.avg_disparate_impact),
            ("SPD", lambda fm: fm.avg_statistical_parity_difference),
        ]

        for key in fairness_keys:
            for metric_name, getter in fairness_metrics:
                row = [f"{key} {metric_name}"]

                for r in reports:
                    task = get_task(r, task_name)
                    if task is None:
                        row.append("-")
                        continue

                    fm = task.fairness.get(key)
                    if fm is None:
                        row.append("-")
                        continue

                    val = getter(fm)
                    row.append(f"{val:.4f}" if val is not None else "-")

                table.add_row(*row)

        console.print(table)
