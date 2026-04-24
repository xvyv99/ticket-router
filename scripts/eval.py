"""Evaluate existing model predictions and output a rich comparison table.

Usage:
    uv run python scripts/eval_models.py --dataset multilingual-customer-support
    uv run python scripts/eval_models.py --dataset french-gov-oss --pred-files lr:xgb
"""

from pathlib import Path
from argparse import ArgumentParser
from logging import basicConfig

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from ticket_router_base.eval import evaluate_file
from ticket_router_base.data import get_dataset, DATASET_REGISTRY
from ticket_router_base.config import LOGGING_FORMAT

console = Console()


def illustrate_metric(dataset_name: str, pred_files: str, pred_dir: Path) -> None:
    dataset = get_dataset(dataset_name)
    stems = [s.strip() for s in pred_files.split(":") if s.strip()]

    results = []
    for stem in stems:
        path = pred_dir / f"{stem}_predictions.jsonl"
        if not path.exists():
            console.print(f"[yellow][SKIP][/yellow] {stem}: {path} not found")
            continue

        report = evaluate_file(path, dataset, model_name=stem)
        results.append((stem, report))

    if not results:
        console.print("[red]No valid prediction files found.[/red]")
        return

    task_names = [tr.task_name for tr in results[0][1].task_results]

    # ── Overall Metrics ──────────────────────────────────────────────
    overall_table = Table(
        title="Model Evaluation Comparison — Overall Metrics",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    overall_table.add_column("Model", style="bold cyan", width=22)
    for tn in task_names:
        overall_table.add_column(f"{tn} Acc", justify="right", min_width=8)
        overall_table.add_column(f"{tn} MF1", justify="right", min_width=8)
        has_ordinal = any(
            tr.ordinal is not None
            for _, report in results
            for tr in report.task_results
            if tr.task_name == tn
        )
        if has_ordinal:
            overall_table.add_column(f"{tn} MAE", justify="right", min_width=8)
            overall_table.add_column(f"{tn} QWK", justify="right", min_width=8)
        has_fairness = any(
            tr.fairness is not None
            for _, report in results
            for tr in report.task_results
            if tr.task_name == tn
        )
        if has_fairness:
            overall_table.add_column(f"{tn} AccGap", justify="right", min_width=8)
            overall_table.add_column(f"{tn} AccRatio", justify="right", min_width=8)
            overall_table.add_column(f"{tn} F1Gap", justify="right", min_width=8)
            overall_table.add_column(f"{tn} F1Ratio", justify="right", min_width=8)

    for name, report in results:
        row = [name]
        for tr in report.task_results:
            row.append(f"{tr.overall.accuracy:.4f}")
            row.append(f"{tr.overall.macro_f1:.4f}")
            if tr.ordinal is not None:
                row.append(f"{tr.ordinal.mae:.4f}")
                row.append(f"{tr.ordinal.qwk:.4f}")
            if tr.fairness is not None:
                row.append(f"{tr.fairness.accuracy_gap:.4f}")
                row.append(f"{tr.fairness.accuracy_ratio:.4f}")
                row.append(f"{tr.fairness.macro_f1_gap:.4f}")
                row.append(f"{tr.fairness.macro_f1_ratio:.4f}")
        overall_table.add_row(*row)

    console.print(overall_table)
    # console.print(Panel(overall_table, border_style="blue"))

    # ── By-Language Fairness (first task) ────────────────────────────
    if results[0][1].task_results and results[0][1].task_results[0].by_language:
        first_task = results[0][1].task_results[0]
        languages = sorted(first_task.by_language.keys())

        lang_table = Table(
            title=f"{first_task.task_name} Macro-F1 by Language (Fairness Check)",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )
        lang_table.add_column("Model", style="bold cyan", width=22)
        for lang in languages:
            lang_table.add_column(str(lang), justify="right", min_width=10)

        for name, report in results:
            tr = report.task_results[0]
            row = [name]
            for lang in languages:
                m = tr.by_language.get(lang)
                val = m.macro_f1 if m else 0.0
                row.append(f"{val:.4f}")
            lang_table.add_row(*row)

        console.print(lang_table)

    # ── Error Summary ────────────────────────────────────────────────
    has_errors = any(
        report.error_summary != {"SUCCESS": report.total_samples}
        for _, report in results
    )
    if has_errors:
        err_table = Table(
            title="Error Summary",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold red",
        )
        err_table.add_column("Model", style="bold cyan", width=22)
        err_table.add_column("Errors", style="red")

        for name, report in results:
            if report.error_summary != {"SUCCESS": report.total_samples}:
                err_table.add_row(name, str(report.error_summary))

        console.print()
        console.print(Panel(err_table, border_style="red"))

    # ── Dataset info footer ──────────────────────────────────────────
    console.print()
    console.print(
        f"[dim]Dataset: {dataset.name}  |  Samples: {results[0][1].total_samples}  |  Models: {len(results)}[/dim]"
    )
    console.print()


def main() -> None:
    parser = ArgumentParser(description="Evaluate model predictions")
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_REGISTRY.keys()),
        default="multilingual-customer-support",
        help="Dataset to evaluate against",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=Path("../../outputs/supervised"),
        help="Directory containing prediction JSONL files",
    )
    parser.add_argument(
        "--pred-files",
        type=str,
        default="lr:xgb:mbert",
        help="Colon-separated list of prediction file stems (e.g. lr:xgb)",
    )
    args = parser.parse_args()

    illustrate_metric(args.dataset, args.pred_files, args.pred_dir)


if __name__ == "__main__":
    basicConfig(level="INFO", format=LOGGING_FORMAT)
    main()
