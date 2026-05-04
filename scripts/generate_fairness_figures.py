#!/usr/bin/env python3
"""Fairness visualizations for ticket router evaluation results.

Usage:
    uv run python scripts/generate_fairness_figures.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib.pyplot as plt

from ticket_router.plot.data import (
    filter_best_configs,
    load_eval_tidy,
    load_robustness_lang_csv,
)
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
from ticket_router.plot.style import save_figure, set_academic_style

OUTPUT_DIR = Path("results/figures")
RESULTS_CSV = sorted(glob.glob("results/eval_multilingual-customer-support_*.csv"))[-1]
ROBUSTNESS_DIR = Path("outputs/robustness/xlm-roberta/multilingual-customer-support")


def main() -> None:
    set_academic_style(font_size=11)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {RESULTS_CSV}")
    tidy_df = load_eval_tidy(RESULTS_CSV)
    best_df = filter_best_configs(tidy_df)

    figures: list[tuple[str, plt.Figure]] = [
        ("fairness_01_pareto.png", plot_pareto_frontier(best_df)),
    ]
    figures.extend(plot_language_di_matrices(tidy_df))
    figures.extend([
        ("fairness_04_user_type_eod.png", plot_user_type_eod(tidy_df)),
        ("fairness_05_thinking_vs_nothinking.png", plot_thinking_vs_nothinking(tidy_df)),
        ("fairness_06_cross_task.png", plot_cross_task_fairness(best_df)),
        ("fairness_07_gap_errorbars.png", plot_gap_with_errorbars(best_df)),
        ("fairness_08_spearman.png", plot_spearman_correlation(best_df)),
        ("fairness_09_llm_scale.png", plot_llm_scale_fairness(tidy_df)),
    ])

    print(f"Loading robustness data from: {ROBUSTNESS_DIR}")
    rob_df = load_robustness_lang_csv(ROBUSTNESS_DIR)
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
        save_figure(fig, OUTPUT_DIR / filename)
        print(f"Saved: {filename}")

    print(f"\nDone — {len(figures)} figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
