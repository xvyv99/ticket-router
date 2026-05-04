"""Console and file output for robustness evaluation results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

from rich.console import Console
from rich.table import Table

from .robustness import AdversarialExample, RobustnessMetrics

console = Console()


def print_robustness_report(metrics_list: List[RobustnessMetrics]) -> None:
    """Print a rich table summary of robustness results per task."""
    for metrics in metrics_list:
        attack_info = f"Attack: {metrics.attack_type}"
        if metrics.recipe:
            attack_info += f" | Recipe: {metrics.recipe}"
        if metrics.avg_perturbation_rate is not None:
            attack_info += f" | Avg Perturb Rate: {metrics.avg_perturbation_rate:.4f}"
        if metrics.avg_query_count is not None:
            attack_info += f" | Avg Queries: {metrics.avg_query_count:.1f}"

        console.print(
            f"\n[bold cyan]Task: {metrics.task_name}[/bold cyan] | {attack_info}"
        )

        # Main metrics table
        table = Table(title=f"Robustness Metrics ({metrics.task_name})")
        table.add_column("Metric", style="bold")
        table.add_column("Overall", justify="right")

        # Collect all languages and queues for columns
        lang_keys = sorted(metrics.per_language.keys())
        queue_keys = sorted(metrics.per_queue.keys())

        for lang in lang_keys:
            table.add_column(lang, justify="right")

        rows = [
            ("Clean Accuracy", metrics.clean_accuracy),
            ("Perturbed Accuracy", metrics.perturbed_accuracy),
            ("Accuracy Drop", metrics.accuracy_drop),
            ("Attack Success Rate", metrics.attack_success_rate),
        ]

        for label, overall_val in rows:
            row = [label, f"{overall_val:.4f}"]
            for lang in lang_keys:
                lm = metrics.per_language[lang]
                val = getattr(lm, _row_attr(label), 0.0)
                row.append(f"{val:.4f}")
            table.add_row(*row)

        console.print(table)

        # Per-queue breakdown if available
        if queue_keys:
            queue_table = Table(title=f"Per-Queue Breakdown ({metrics.task_name})")
            queue_table.add_column("Queue", style="bold")
            queue_table.add_column("Clean Acc", justify="right")
            queue_table.add_column("Perturbed Acc", justify="right")
            queue_table.add_column("Acc Drop", justify="right")
            queue_table.add_column("Attack Success", justify="right")

            for q in queue_keys:
                qm = metrics.per_queue[q]
                queue_table.add_row(
                    q,
                    f"{qm.clean_accuracy:.4f}",
                    f"{qm.perturbed_accuracy:.4f}",
                    f"{qm.accuracy_drop:.4f}",
                    f"{qm.attack_success_rate:.4f}",
                )
            console.print(queue_table)

        # Adversarial examples summary
        n_adv = len(metrics.adversarial_examples)
        n_success = sum(1 for ex in metrics.adversarial_examples if ex.success)
        console.print(
            f"[dim]Samples: {metrics.n_samples} | Correct: {metrics.n_correct} | "
            f"Attacked: {metrics.n_attacked} | "
            f"Adversarial Examples: {n_adv} ({n_success} successful)[/dim]"
        )


def _row_attr(label: str) -> str:
    """Map display label to RobustnessMetrics attribute name."""
    mapping = {
        "Clean Accuracy": "clean_accuracy",
        "Perturbed Accuracy": "perturbed_accuracy",
        "Accuracy Drop": "accuracy_drop",
        "Attack Success Rate": "attack_success_rate",
    }
    return mapping[label]


def save_robustness_to_csv(
    metrics_list: List[RobustnessMetrics], output_path: Path
) -> None:
    """Save robustness metrics to a tidy CSV file.

    Each row represents one (task, attack_type, granularity) combination.
    """
    rows: List[Dict[str, str | float | int | None]] = []

    for metrics in metrics_list:
        # Overall row
        rows.append(
            {
                "task_name": metrics.task_name,
                "attack_type": metrics.attack_type,
                "recipe": metrics.recipe,
                "granularity": "overall",
                "group": "all",
                "clean_accuracy": metrics.clean_accuracy,
                "perturbed_accuracy": metrics.perturbed_accuracy,
                "accuracy_drop": metrics.accuracy_drop,
                "attack_success_rate": metrics.attack_success_rate,
                "avg_perturbation_rate": metrics.avg_perturbation_rate,
                "avg_query_count": metrics.avg_query_count,
                "n_samples": metrics.n_samples,
                "n_correct": metrics.n_correct,
                "n_attacked": metrics.n_attacked,
            }
        )

        # Per-language rows
        for lang, lm in sorted(metrics.per_language.items()):
            rows.append(
                {
                    "task_name": metrics.task_name,
                    "attack_type": metrics.attack_type,
                    "recipe": metrics.recipe,
                    "granularity": "language",
                    "group": lang,
                    "clean_accuracy": lm.clean_accuracy,
                    "perturbed_accuracy": lm.perturbed_accuracy,
                    "accuracy_drop": lm.accuracy_drop,
                    "attack_success_rate": lm.attack_success_rate,
                    "avg_perturbation_rate": metrics.avg_perturbation_rate,
                    "avg_query_count": metrics.avg_query_count,
                    "n_samples": lm.n_samples,
                    "n_correct": lm.n_correct,
                    "n_attacked": lm.n_attacked,
                }
            )

        # Per-queue rows
        for q, qm in sorted(metrics.per_queue.items()):
            rows.append(
                {
                    "task_name": metrics.task_name,
                    "attack_type": metrics.attack_type,
                    "recipe": metrics.recipe,
                    "granularity": "queue",
                    "group": q,
                    "clean_accuracy": qm.clean_accuracy,
                    "perturbed_accuracy": qm.perturbed_accuracy,
                    "accuracy_drop": qm.accuracy_drop,
                    "attack_success_rate": qm.attack_success_rate,
                    "avg_perturbation_rate": metrics.avg_perturbation_rate,
                    "avg_query_count": metrics.avg_query_count,
                    "n_samples": qm.n_samples,
                    "n_correct": qm.n_correct,
                    "n_attacked": qm.n_attacked,
                }
            )

    if not rows:
        return

    fieldnames = [
        "task_name",
        "attack_type",
        "recipe",
        "granularity",
        "group",
        "clean_accuracy",
        "perturbed_accuracy",
        "accuracy_drop",
        "attack_success_rate",
        "avg_perturbation_rate",
        "avg_query_count",
        "n_samples",
        "n_correct",
        "n_attacked",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    console.print(f"[green]CSV saved to {output_path} ({len(rows)} rows)[/green]")


def save_robustness_to_excel(
    metrics_list: List[RobustnessMetrics], output_path: Path
) -> None:
    """Save robustness metrics to an Excel workbook with multiple sheets.

    Sheets:
        - "overview": one row per task with overall metrics
        - "per_language": one row per (task, language) combination
        - "per_queue": one row per (task, queue) combination
    """
    try:
        import pandas as pd
    except ImportError:
        console.print("[yellow]pandas not installed, skipping Excel export[/yellow]")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Overview sheet
    overview_rows: List[Dict] = []
    for m in metrics_list:
        overview_rows.append(
            {
                "task_name": m.task_name,
                "attack_type": m.attack_type,
                "recipe": m.recipe,
                "clean_accuracy": m.clean_accuracy,
                "perturbed_accuracy": m.perturbed_accuracy,
                "accuracy_drop": m.accuracy_drop,
                "attack_success_rate": m.attack_success_rate,
                "avg_perturbation_rate": m.avg_perturbation_rate,
                "avg_query_count": m.avg_query_count,
                "n_samples": m.n_samples,
                "n_correct": m.n_correct,
                "n_attacked": m.n_attacked,
            }
        )

    # Per-language sheet
    lang_rows: List[Dict] = []
    for m in metrics_list:
        for lang, lm in sorted(m.per_language.items()):
            lang_rows.append(
                {
                    "task_name": m.task_name,
                    "language": lang,
                    "clean_accuracy": lm.clean_accuracy,
                    "perturbed_accuracy": lm.perturbed_accuracy,
                    "accuracy_drop": lm.accuracy_drop,
                    "attack_success_rate": lm.attack_success_rate,
                    "n_samples": lm.n_samples,
                    "n_correct": lm.n_correct,
                    "n_attacked": lm.n_attacked,
                }
            )

    # Per-queue sheet
    queue_rows: List[Dict] = []
    for m in metrics_list:
        for q, qm in sorted(m.per_queue.items()):
            queue_rows.append(
                {
                    "task_name": m.task_name,
                    "queue": q,
                    "clean_accuracy": qm.clean_accuracy,
                    "perturbed_accuracy": qm.perturbed_accuracy,
                    "accuracy_drop": qm.accuracy_drop,
                    "attack_success_rate": qm.attack_success_rate,
                    "n_samples": qm.n_samples,
                    "n_correct": qm.n_correct,
                    "n_attacked": qm.n_attacked,
                }
            )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        if overview_rows:
            pd.DataFrame(overview_rows).to_excel(writer, sheet_name="overview", index=False)
        if lang_rows:
            pd.DataFrame(lang_rows).to_excel(
                writer, sheet_name="per_language", index=False
            )
        if queue_rows:
            pd.DataFrame(queue_rows).to_excel(
                writer, sheet_name="per_queue", index=False
            )

    console.print(f"[green]Excel saved to {output_path}[/green]")


def save_robustness_to_json(
    metrics_list: List[RobustnessMetrics], output_path: Path
) -> None:
    """Save robustness metrics to a JSON file."""
    data = []
    for metrics in metrics_list:
        m_dict = {
            "task_name": metrics.task_name,
            "attack_type": metrics.attack_type,
            "recipe": metrics.recipe,
            "clean_accuracy": metrics.clean_accuracy,
            "perturbed_accuracy": metrics.perturbed_accuracy,
            "accuracy_drop": metrics.accuracy_drop,
            "attack_success_rate": metrics.attack_success_rate,
            "avg_perturbation_rate": metrics.avg_perturbation_rate,
            "avg_query_count": metrics.avg_query_count,
            "n_samples": metrics.n_samples,
            "n_correct": metrics.n_correct,
            "n_attacked": metrics.n_attacked,
            "per_language": {
                k: {
                    "clean_accuracy": v.clean_accuracy,
                    "perturbed_accuracy": v.perturbed_accuracy,
                    "accuracy_drop": v.accuracy_drop,
                    "attack_success_rate": v.attack_success_rate,
                    "n_samples": v.n_samples,
                    "n_correct": v.n_correct,
                    "n_attacked": v.n_attacked,
                }
                for k, v in metrics.per_language.items()
            },
            "per_queue": {
                k: {
                    "clean_accuracy": v.clean_accuracy,
                    "perturbed_accuracy": v.perturbed_accuracy,
                    "accuracy_drop": v.accuracy_drop,
                    "attack_success_rate": v.attack_success_rate,
                    "n_samples": v.n_samples,
                    "n_correct": v.n_correct,
                    "n_attacked": v.n_attacked,
                }
                for k, v in metrics.per_queue.items()
            },
        }
        data.append(m_dict)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    console.print(f"[green]JSON saved to {output_path}[/green]")


def save_adversarial_examples_to_jsonl(
    metrics_list: List[RobustnessMetrics], output_dir: Path
) -> None:
    """Save adversarial examples per task as JSONL files.

    Each task gets its own file: ``{task_name}_adversarial_examples.jsonl``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for metrics in metrics_list:
        if not metrics.adversarial_examples:
            continue
        path = output_dir / f"{metrics.task_name}_adversarial_examples.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for ex in metrics.adversarial_examples:
                obj = {
                    "request_id": ex.request_id,
                    "task_name": ex.task_name,
                    "original_text": ex.original_text,
                    "adversarial_text": ex.adversarial_text,
                    "true_label": ex.true_label,
                    "clean_pred": ex.clean_pred,
                    "perturbed_pred": ex.perturbed_pred,
                    "language": ex.language,
                    "queue": ex.queue,
                    "success": ex.success,
                    "perturbation_rate": ex.perturbation_rate,
                    "query_count": ex.query_count,
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        console.print(
            f"[green]Saved {len(metrics.adversarial_examples)} adversarial examples "
            f"to {path}[/green]"
        )
