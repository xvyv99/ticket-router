#!/usr/bin/env python3
"""Generate evaluation result visualizations.

Usage:
    uv run python scripts/generate_evaluation_figures.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib.pyplot as plt

from ticket_router.plot.data import load_eval_tidy, load_robustness_recipe_xlsx
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
from ticket_router.plot.style import save_figure, set_academic_style

OUTPUT_DIR = Path("outputs/figures")
RESULTS_XLSX = sorted(glob.glob("results/eval_multilingual-customer-support_*.xlsx"))[-1]


def main() -> None:
    set_academic_style(font_size=11)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {RESULTS_XLSX}")
    tidy_df = load_eval_tidy(RESULTS_XLSX)
    robust_df = load_robustness_recipe_xlsx()

    figures: list[tuple[str, plt.Figure]] = [
        ("01a_accuracy_priority.png", plot_accuracy_priority(tidy_df)),
        ("01b_accuracy_queue.png", plot_accuracy_queue(tidy_df)),
        ("02a_macro_f1_priority.png", plot_macro_f1_priority(tidy_df)),
        ("02b_macro_f1_queue.png", plot_macro_f1_queue(tidy_df)),
        ("03_fairness_disparate_impact.png", plot_fairness_heatmap(tidy_df)),
        ("04_robustness_attack_success_rate.png",
         plot_robustness_recipe_attack_success(robust_df)),
        ("05_clean_vs_perturbed_accuracy.png",
         plot_robustness_recipe_clean_vs_perturbed(robust_df)),
        ("06a_scaling_accuracy.png", plot_scaling_curve(tidy_df, "accuracy")),
        ("06b_scaling_macro_f1.png", plot_scaling_curve(tidy_df, "macro_f1")),
    ]

    for filename, fig in figures:
        save_figure(fig, OUTPUT_DIR / filename)
        print(f"Saved: {filename}")

    print("\nAll figures generated successfully.")


if __name__ == "__main__":
    main()
