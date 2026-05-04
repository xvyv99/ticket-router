"""Evaluation/performance plotting functions.

Includes metric bar charts, fairness heatmap, and scaling curve.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from ticket_router.plot.config import (
    PARADIGM_COLOR,
    PARADIGM_ORDER,
    SCALING_GROUP_COLORS,
    SCALING_GROUP_MARKERS,
    SCALING_GROUPS,
)
from ticket_router.plot.data import get_scaling_key, sort_by_paradigm


# ---------------------------------------------------------------------------
# Fig 01a — Accuracy — Priority
# ---------------------------------------------------------------------------

def plot_accuracy_priority(df: pd.DataFrame) -> plt.Figure:
    """Priority task accuracy, sorted by paradigm, colored by paradigm."""
    perf = df[
        (df["metric_category"] == "performance")
        & (df["metric_name"] == "accuracy")
        & (df["task_name"] == "priority")
    ].copy()
    perf = sort_by_paradigm(perf)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [PARADIGM_COLOR.get(p, "#999") for p in perf["paradigm"]]
    bars = ax.barh(perf["display_name"], perf["value"], height=0.7, color=colors)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Accuracy")
    ax.set_title("Accuracy — Priority (3-class)", fontweight="bold")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}", ha="left", va="center", fontsize=8)
    legend_elements = [
        Patch(facecolor=PARADIGM_COLOR[p], label=p)
        for p in PARADIGM_ORDER if p in perf["paradigm"].values
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 01b — Accuracy — Queue
# ---------------------------------------------------------------------------

def plot_accuracy_queue(df: pd.DataFrame) -> plt.Figure:
    """Queue task accuracy, sorted by paradigm, colored by paradigm."""
    perf = df[
        (df["metric_category"] == "performance")
        & (df["metric_name"] == "accuracy")
        & (df["task_name"] == "queue")
    ].copy()
    perf = sort_by_paradigm(perf)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [PARADIGM_COLOR.get(p, "#999") for p in perf["paradigm"]]
    bars = ax.barh(perf["display_name"], perf["value"], height=0.7, color=colors)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Accuracy")
    ax.set_title("Accuracy — Queue (10-class)", fontweight="bold")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}", ha="left", va="center", fontsize=8)
    legend_elements = [
        Patch(facecolor=PARADIGM_COLOR[p], label=p)
        for p in PARADIGM_ORDER if p in perf["paradigm"].values
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 02a — Macro-F1 — Priority
# ---------------------------------------------------------------------------

def plot_macro_f1_priority(df: pd.DataFrame) -> plt.Figure:
    """Priority task Macro-F1, sorted by paradigm, colored by paradigm."""
    perf = df[
        (df["metric_category"] == "performance")
        & (df["metric_name"] == "macro_f1")
        & (df["task_name"] == "priority")
    ].copy()
    perf = sort_by_paradigm(perf)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [PARADIGM_COLOR.get(p, "#999") for p in perf["paradigm"]]
    bars = ax.barh(perf["display_name"], perf["value"], height=0.7, color=colors)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Macro-F1")
    ax.set_title("Macro-F1 — Priority (3-class)", fontweight="bold")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}", ha="left", va="center", fontsize=8)
    legend_elements = [
        Patch(facecolor=PARADIGM_COLOR[p], label=p)
        for p in PARADIGM_ORDER if p in perf["paradigm"].values
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 02b — Macro-F1 — Queue
# ---------------------------------------------------------------------------

def plot_macro_f1_queue(df: pd.DataFrame) -> plt.Figure:
    """Queue task Macro-F1, sorted by paradigm, colored by paradigm."""
    perf = df[
        (df["metric_category"] == "performance")
        & (df["metric_name"] == "macro_f1")
        & (df["task_name"] == "queue")
    ].copy()
    perf = sort_by_paradigm(perf)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [PARADIGM_COLOR.get(p, "#999") for p in perf["paradigm"]]
    bars = ax.barh(perf["display_name"], perf["value"], height=0.7, color=colors)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Macro-F1")
    ax.set_title("Macro-F1 — Queue (10-class)", fontweight="bold")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}", ha="left", va="center", fontsize=8)
    legend_elements = [
        Patch(facecolor=PARADIGM_COLOR[p], label=p)
        for p in PARADIGM_ORDER if p in perf["paradigm"].values
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 03 — Disparate Impact Heatmap
# ---------------------------------------------------------------------------

def plot_fairness_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Rows=models (by paradigm), cols=sensitive_attrs, Disparate Impact values."""
    fairness = df[
        (df["metric_category"] == "fairness_pairwise")
        & (df["metric_name"] == "disparate_impact")
    ].copy()

    fairness = sort_by_paradigm(fairness)

    pivot = (
        fairness.groupby(["display_name", "sensitive_attr"])["value"]
        .mean()
        .unstack()
    )

    attr_order = ["language", "tech_proficiency", "user_type"]
    pivot = pivot[[a for a in attr_order if a in pivot.columns]]

    fig, ax = plt.subplots(figsize=(8, 10))
    cmap = sns.diverging_palette(10, 133, s=90, l=50, as_cmap=True)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        center=1.0,
        vmin=0,
        vmax=2.5,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Disparate Impact"},
    )
    ax.set_xlabel("Sensitive Attribute")
    ax.set_ylabel("")
    ax.set_title("Fairness: Disparate Impact by Model", fontweight="bold")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 06 — Scaling Curve
# ---------------------------------------------------------------------------

def plot_scaling_curve(df: pd.DataFrame, metric: str = "accuracy") -> plt.Figure:
    """Parameter count vs performance scaling curve.

    Each line = one model group:
    - Rule-Based: x=0.001
    - LR (tfidf): x=0.1 / LR (ST): x=100
    - XGBoost (tfidf): x=1 / XGBoost (ST): x=100
    - Supervised Encoder: mBERT(x=560) + XLM-R(x=250) shared line
    - Goal-Based LLM: zero-shot / few-shot per line, x=600/1700/4000
    """
    metric_df = df[
        (df["metric_category"] == "performance")
        & (df["metric_name"] == metric)
    ].copy()

    key_to_group: dict[str, tuple[str, float]] = dict(SCALING_GROUPS)

    raw_points: dict[str, list[tuple[float, float]]] = {}
    for _, row in metric_df.iterrows():
        key = get_scaling_key(row["model_name"], row["cfg"])
        if key is None or key not in key_to_group:
            continue
        group_name, param_count = key_to_group[key]
        raw_points.setdefault(group_name, []).append((param_count, row["value"]))

    def _build_line_data(
        raw: dict[str, list[tuple[float, float]]],
    ) -> dict[str, list[tuple[float, float]]]:
        result: dict[str, list[tuple[float, float]]] = {}
        for group, xs_vals in raw.items():
            by_x: dict[float, list[float]] = {}
            for x, v in xs_vals:
                by_x.setdefault(x, []).append(v)
            result[group] = sorted((x, sum(vs) / len(vs)) for x, vs in by_x.items())
        return result

    line_data = _build_line_data(raw_points)

    group_order = [
        "Rule-Based",
        "LR (tfidf)",
        "XGBoost (tfidf)",
        "LR (ST)",
        "XGBoost (ST)",
        "Supervised (Encoder)",
        "Goal-Based (LLM, zero-shot)",
        "Goal-Based (LLM, few-shot)",
    ]

    legend_names = {
        "Rule-Based": "Rule-Based",
        "LR (tfidf)": "LR (tfidf)",
        "LR (ST)": "LR (ST)",
        "XGBoost (tfidf)": "XGBoost (tfidf)",
        "XGBoost (ST)": "XGBoost (ST)",
        "Supervised (Encoder)": "mBERT + XLM-R",
        "Goal-Based (LLM, zero-shot)": "Qwen3 (zero-shot)",
        "Goal-Based (LLM, few-shot)": "Qwen3 (few-shot)",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for group in group_order:
        if group not in line_data:
            continue
        xs, ys = zip(*line_data[group])

        ax.plot(
            xs, ys,
            color=SCALING_GROUP_COLORS.get(group, "#999"),
            marker=SCALING_GROUP_MARKERS.get(group, "o"),
            markersize=8,
            label=legend_names.get(group, group),
            linewidth=2,
            zorder=3,
        )
        for x, y in zip(xs, ys):
            if group == "Supervised (Encoder)":
                label = "XLM-R" if x == 250 else "mBERT"
            elif group == "Goal-Based (LLM, zero-shot)":
                label = "0.6B" if x == 600 else ("1.7B" if x == 1700 else "4B")
            elif group == "Goal-Based (LLM, few-shot)":
                label = "0.6B" if x == 600 else ("1.7B" if x == 1700 else "4B")
            elif group == "Rule-Based":
                label = "Rule"
            else:
                label = group
            ax.annotate(
                label, (x, y),
                textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=7,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Parameter Count (log scale)")
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"Scaling Curve — {metric.capitalize()}", fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
