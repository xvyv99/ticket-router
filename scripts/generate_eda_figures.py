#!/usr/bin/env python3
"""EDA visualizations for the multilingual customer support ticket dataset.

Usage:
    uv run python scripts/generate_eda_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from ticket_router.plot.data import load_eda_data
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
from ticket_router.plot.style import save_figure, set_academic_style

OUTPUT_DIR = Path("results/figures")
PARQUET_PATH = Path("outputs/multilingual-customer-support_test_split.parquet")
JSONL_PATH = Path("outputs/infer_multilingual-customer-support_Qwen3-4B.jsonl")


def main() -> None:
    set_academic_style(font_size=10)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_eda_data(parquet_path=PARQUET_PATH, jsonl_path=JSONL_PATH)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"\nLanguage distribution:\n{df['language'].value_counts()}")
    print(f"\nUser type distribution:\n{df['user_type'].value_counts()}")
    print(f"\nTech proficiency distribution:\n{df['tech_proficiency'].value_counts()}")

    figures: list[tuple[str, plt.Figure]] = [
        ("eda_01_class_distribution.png",    plot_class_distribution(df)),
        ("eda_02_queue_priority_heatmap.png", plot_queue_priority_heatmap(df)),
        ("eda_03_language_queue_heatmap.png", plot_language_queue_heatmap(df)),
        ("eda_04_priority_by_language.png",   plot_priority_by_language(df)),
        ("eda_05_text_lengths.png",           plot_text_lengths(df)),
        ("eda_06_inferred_attributes.png",    plot_inferred_attributes(df)),
        ("eda_07_queue_by_attributes.png",    plot_queue_by_attributes(df)),
        ("eda_08_priority_by_attributes.png", plot_priority_by_attributes(df)),
        ("eda_09_text_vs_label.png",          plot_text_vs_label(df)),
        ("eda_10_tag_coverage.png",           plot_tag_coverage(df)),
    ]

    for filename, fig in figures:
        save_figure(fig, OUTPUT_DIR / filename)
        print(f"Saved: {filename}")

    print(f"\nDone — {len(figures)} figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
