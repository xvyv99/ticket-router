"""Console and file output for interpretability evaluation results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from rich.console import Console
from rich.table import Table

from .interpretability import (
    TaskAttributionReport,
)

console = Console()


def print_interpretability_report(reports: Dict[str, TaskAttributionReport]) -> None:
    """Print a rich table summary of attribution results per task."""
    for task_name, report in reports.items():
        console.print(f"\n[bold cyan]Task: {task_name}[/bold cyan]")

        # Per-class top tokens
        table = Table(title=f"Top Tokens by Predicted Class ({task_name})")
        table.add_column("Class", style="bold")
        table.add_column("Samples")
        table.add_column("Top Positive Tokens", style="green")
        table.add_column("Top Negative Tokens", style="red")

        for class_label in sorted(report.class_summaries.keys()):
            summary = report.class_summaries[class_label]
            top = summary.top_tokens(k=10)
            pos = [(t, s) for t, s in top if s > 0]
            neg = [(t, s) for t, s in top if s < 0]
            pos_str = ", ".join(f"{t} ({s:+.3f})" for t, s in pos[:5])
            neg_str = ", ".join(f"{t} ({s:+.3f})" for t, s in neg[:5])
            table.add_row(class_label, str(summary.sample_count), pos_str, neg_str)

        console.print(table)

        # Sample-level breakdown for first few samples
        console.print(
            f"[dim]Sample attributions: {len(report.sample_attributions)} records[/dim]"
        )


def save_interpretability_results(
    reports: Dict[str, TaskAttributionReport],
    output_dir: Path,
) -> None:
    """Save attribution results to JSONL (per-sample) and JSON (class summaries).

    Args:
        reports: Dict mapping task_name -> TaskAttributionReport.
        output_dir: Directory to write output files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for task_name, report in reports.items():
        # Per-sample attributions as JSONL
        sample_path = output_dir / f"{task_name}_attributions.jsonl"
        with sample_path.open("w", encoding="utf-8") as f:
            for sa in report.sample_attributions:
                obj = {
                    "request_id": sa.request_id,
                    "text": sa.text,
                    "predicted_label": sa.predicted_label,
                    "true_label": sa.true_label,
                    "confidence": sa.confidence,
                    "top_positive": [(t.token, t.score) for t in sa.top_positive],
                    "top_negative": [(t.token, t.score) for t in sa.top_negative],
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        # Class summaries as JSON
        summary_path = output_dir / f"{task_name}_class_summary.json"
        summaries = {}
        for label, summary in report.class_summaries.items():
            summaries[label] = {
                "sample_count": summary.sample_count,
                "top_tokens": summary.top_tokens(k=20),
            }
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)

    console.print(f"[green]Saved interpretability results to {output_dir}[/green]")
