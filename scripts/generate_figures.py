#!/usr/bin/env python3
"""Generate all publication-quality figures for the Ticket Router project.

Usage:
    uv run python scripts/generate_figures.py                 # all figures
    uv run python scripts/generate_figures.py --eda           # EDA only
    uv run python scripts/generate_figures.py --fairness      # fairness + robustness(lang) only
    uv run python scripts/generate_figures.py --evaluation    # evaluation + robustness(recipe) only
    uv run python scripts/generate_figures.py --eda --fairness
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

from matplotlib.figure import Figure

from ticket_router.plot.data import (
    filter_best_configs,
    load_eda_data,
    load_eval_tidy,
    load_robustness_lang_csv,
    load_robustness_recipe_xlsx,
)
from ticket_router.plot.style import save_figure, set_academic_style

# ---------------------------------------------------------------------------
# Output base — all figures go under results/figures/<category>/
# ---------------------------------------------------------------------------
FIGURES_BASE = Path("results/figures")


# ===================================================================
# EDA
# ===================================================================

def _generate_eda() -> int:
    from ticket_router.plot.eda import (
        plot_class_distribution,
        plot_inferred_attributes,
        plot_language_queue_heatmap,
        plot_priority_by_attributes,
        plot_priority_by_language,
        plot_queue_by_attributes,
        plot_queue_priority_heatmap,
        plot_tag_coverage,
        plot_text_lengths,
        plot_text_vs_label,
    )

    out_dir = FIGURES_BASE / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = Path("outputs/multilingual-customer-support_test_split.parquet")
    jsonl_path = Path("outputs/infer_multilingual-customer-support_Qwen3-4B.jsonl")

    set_academic_style(font_size=10)

    print("Loading EDA data...")
    df = load_eda_data(parquet_path=parquet_path, jsonl_path=jsonl_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    figures: list[tuple[str, Figure]] = [
        ("01_class_distribution.png",         plot_class_distribution(df)),
        ("02_queue_priority_heatmap.png",      plot_queue_priority_heatmap(df)),
        ("03_language_queue_heatmap.png",      plot_language_queue_heatmap(df)),
        ("04_priority_by_language.png",        plot_priority_by_language(df)),
        ("05_text_lengths.png",                plot_text_lengths(df)),
        ("06_inferred_attributes.png",         plot_inferred_attributes(df)),
        ("07_queue_by_attributes.png",         plot_queue_by_attributes(df)),
        ("08_priority_by_attributes.png",      plot_priority_by_attributes(df)),
        ("09_text_vs_label.png",               plot_text_vs_label(df)),
        ("10_tag_coverage.png",                plot_tag_coverage(df)),
    ]

    for filename, fig in figures:
        save_figure(fig, out_dir / filename)
        print(f"  Saved: {filename}")

    return len(figures)


# ===================================================================
# Fairness + language-level robustness
# ===================================================================

def _generate_fairness() -> int:
    from ticket_router.plot.fairness import (
        plot_cross_task_fairness,
        plot_gap_with_errorbars,
        plot_language_di_matrices,
        plot_llm_scale_fairness,
        plot_pareto_frontier,
        plot_spearman_correlation,
        plot_thinking_vs_nothinking,
        plot_user_type_eod,
    )
    from ticket_router.plot.robustness import (
        plot_robustness_accuracy_drop_by_lang,
        plot_robustness_attack_success_by_lang,
        plot_robustness_clean_vs_perturbed_by_lang,
    )

    out_dir = FIGURES_BASE / "fairness"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_csv = sorted(glob.glob("results/eval_multilingual-customer-support_*.csv"))[-1]
    robustness_dir = Path("outputs/robustness/xlm-roberta/multilingual-customer-support")

    set_academic_style(font_size=11)

    print(f"Loading results from: {results_csv}")
    tidy_df = load_eval_tidy(results_csv)
    best_df = filter_best_configs(tidy_df)

    figures: list[tuple[str, Figure]] = [
        ("01_pareto.png", plot_pareto_frontier(best_df)),
    ]
    figures.extend(plot_language_di_matrices(tidy_df))
    figures.extend([
        ("04_user_type_eod.png",           plot_user_type_eod(tidy_df)),
        ("05_thinking_vs_nothinking.png",  plot_thinking_vs_nothinking(tidy_df)),
        ("06_cross_task.png",              plot_cross_task_fairness(best_df)),
        ("07_gap_errorbars.png",           plot_gap_with_errorbars(best_df)),
        ("08_spearman.png",                plot_spearman_correlation(best_df)),
        ("09_llm_scale.png",               plot_llm_scale_fairness(tidy_df)),
    ])

    print(f"Loading robustness data from: {robustness_dir}")
    rob_df = load_robustness_lang_csv(robustness_dir)
    if not rob_df.empty:
        figures.extend([
            ("robustness_01_clean_vs_perturbed.png",
             plot_robustness_clean_vs_perturbed_by_lang(rob_df)),
            ("robustness_02_attack_success.png",
             plot_robustness_attack_success_by_lang(rob_df)),
            ("robustness_03_accuracy_drop.png",
             plot_robustness_accuracy_drop_by_lang(rob_df)),
        ])
    else:
        print("Warning: No robustness data found, skipping robustness figures")

    for filename, fig in figures:
        save_figure(fig, out_dir / filename)
        print(f"  Saved: {filename}")

    return len(figures)


# ===================================================================
# Evaluation + recipe-level robustness
# ===================================================================

def _generate_evaluation() -> int:
    from ticket_router.plot.evaluation import (
        plot_accuracy_priority,
        plot_accuracy_queue,
        plot_fairness_heatmap,
        plot_macro_f1_priority,
        plot_macro_f1_queue,
        plot_scaling_curve,
    )
    from ticket_router.plot.robustness import (
        plot_robustness_recipe_attack_success,
        plot_robustness_recipe_clean_vs_perturbed,
    )

    out_dir = FIGURES_BASE / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_xlsx = sorted(glob.glob("results/eval_multilingual-customer-support_*.xlsx"))[-1]

    set_academic_style(font_size=11)

    print(f"Loading results from: {results_xlsx}")
    tidy_df = load_eval_tidy(results_xlsx)
    robust_df = load_robustness_recipe_xlsx()

    figures: list[tuple[str, Figure]] = [
        ("01a_accuracy_priority.png",           plot_accuracy_priority(tidy_df)),
        ("01b_accuracy_queue.png",              plot_accuracy_queue(tidy_df)),
        ("02a_macro_f1_priority.png",           plot_macro_f1_priority(tidy_df)),
        ("02b_macro_f1_queue.png",              plot_macro_f1_queue(tidy_df)),
        ("03_fairness_disparate_impact.png",    plot_fairness_heatmap(tidy_df)),
        ("04_robustness_attack_success_rate.png",
         plot_robustness_recipe_attack_success(robust_df)),
        ("05_clean_vs_perturbed_accuracy.png",
         plot_robustness_recipe_clean_vs_perturbed(robust_df)),
        ("06a_scaling_accuracy.png",            plot_scaling_curve(tidy_df, "accuracy")),
        ("06b_scaling_macro_f1.png",            plot_scaling_curve(tidy_df, "macro_f1")),
    ]

    for filename, fig in figures:
        save_figure(fig, out_dir / filename)
        print(f"  Saved: {filename}")

    return len(figures)


# ===================================================================
# Main
# ===================================================================

MODES = {
    "eda": _generate_eda,
    "fairness": _generate_fairness,
    "evaluation": _generate_evaluation,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures for Ticket Router.",
    )
    parser.add_argument(
        "--eda", action="store_true", help="Generate EDA figures.",
    )
    parser.add_argument(
        "--fairness", action="store_true",
        help="Generate fairness + language-level robustness figures.",
    )
    parser.add_argument(
        "--evaluation", action="store_true",
        help="Generate evaluation + recipe-level robustness figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # If no flags given, run all
    selected = [m for m in MODES if getattr(args, m)] or list(MODES.keys())

    total = 0
    print(f"Output base: {FIGURES_BASE.resolve()}\n")
    for mode in selected:
        print(f"{'=' * 60}")
        print(f"  {mode.upper()}")
        print(f"{'=' * 60}")
        n = MODES[mode]()
        total += n
        print()

    print(f"Done — {total} figures saved across {len(selected)} categories.")


if __name__ == "__main__":
    main()
