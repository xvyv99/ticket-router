"""Evaluation report serialization and console output."""

from dataclasses import dataclass, field
from typing import Dict, List, Any
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

    file_path: Path | None
    task_results: List[TaskEvaluationResult]

    error_summary: Dict[str, int]  # error flag name -> count
    total_samples: int

    # Multi-run aggregation fields
    task_stds: List[TaskEvaluationResult] = field(default_factory=list)
    run_results: List[List[TaskEvaluationResult]] = field(default_factory=list)
    n_runs: int = 1

    # Config discovery fields
    cfg_id: str | None = None
    cfg_info: Dict[str, Any] | None = None


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

    def get_std_task(
        report: EvaluationReport, task_name: str
    ) -> TaskEvaluationResult | None:
        if not report.task_stds:
            return None
        return next(
            (t for t in report.task_stds if t.task_name == task_name),
            None,
        )

    def format_metric(val: float | None, std: float | None) -> str:
        if val is None:
            return "-"
        if std is not None and std > 0:
            return f"{val:.4f} ± {std:.4f}"
        return f"{val:.4f}"

    for task_name in all_tasks:
        table = Table(
            title=f"Task: {task_name}",
            show_lines=False,
        )

        table.add_column("Metric")

        for r in reports:
            if r.cfg_id is not None and r.cfg_info:
                # Show first 2 config entries as summary
                cfg_summary = ", ".join(
                    f"{k}={v}" for k, v in list(r.cfg_info.items())[:2]
                )
                # Escape '[' for rich markup; wrap in parens instead of brackets
                col_name = f"{r.model_name}\n{r.cfg_id} (n={r.n_runs})\n({cfg_summary})"
            elif r.n_runs > 1:
                col_name = f"{r.model_name} (n={r.n_runs})"
            else:
                col_name = r.model_name
            table.add_column(col_name, justify="right")

        base_metrics = [
            ("Accuracy", "accuracy"),
            ("Macro F1", "macro_f1"),
            ("MAE", "mae"),
            ("QWK", "qwk"),
        ]

        for metric_name, metric_key in base_metrics:
            row = [metric_name]

            for r in reports:
                task = get_task(r, task_name)
                if task is None:
                    row.append("-")
                    continue

                val = getattr(task.performance, metric_key, None)
                std_task = get_std_task(r, task_name)
                std = getattr(std_task.performance, metric_key, None) if std_task is not None else None
                row.append(format_metric(val, std))

            table.add_row(*row)

        fairness_metrics = [
            ("Acc Gap", "accuracy_gap"),
            ("Acc Ratio", "accuracy_ratio"),
            ("F1 Gap", "macro_f1_gap"),
            ("F1 Ratio", "macro_f1_ratio"),
            ("DI", "avg_disparate_impact"),
            ("SPD", "avg_statistical_parity_difference"),
        ]

        for key in fairness_keys:
            for metric_name, field_name in fairness_metrics:
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

                    val = getattr(fm, field_name, None)
                    std_task = get_std_task(r, task_name)
                    std_fm = std_task.fairness.get(key) if std_task is not None else None
                    std = getattr(std_fm, field_name, None) if std_fm is not None else None
                    row.append(format_metric(val, std))

                table.add_row(*row)

        console.print(table)
