#!/usr/bin/env python3
"""EDA visualizations for the multilingual customer support ticket dataset.

Merges test_split.parquet with Qwen3-4B inferred attributes (JSONL).

Usage:
    uv run python scripts/generate_eda_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PARQUET_PATH = Path("outputs/multilingual-customer-support_test_split.parquet")
JSONL_PATH = Path("outputs/infer_multilingual-customer-support_Qwen3-4B.jsonl")
OUTPUT_DIR = Path("results/figures")

# ---------------------------------------------------------------------------
# Style — refined academic / modern
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
sns.set_context("paper", font_scale=1.1)

# ---------------------------------------------------------------------------
# Sophisticated colour palettes
# ---------------------------------------------------------------------------
# Language: desaturated jewel tones via "Set2" + manual tweaks
LANG_PALETTE = sns.color_palette("Set2", n_colors=5)
LANG_COLORS = {
    "en": LANG_PALETTE[0],
    "de": LANG_PALETTE[1],
    "es": LANG_PALETTE[2],
    "fr": LANG_PALETTE[3],
    "pt": LANG_PALETTE[4],
}

# Priority: warm-cool diverging (high=warm, low=cool)
PRIORITY_COLORS = {"high": "#d95f02", "medium": "#f2a340", "low": "#5ba3cf"}

# User type: complementary teal / coral from the PuBuGn family
USER_TYPE_COLORS = {"enterprise": "#276d86", "individual": "#cf7c46"}

# Tech proficiency: three-step sequential blue
TECH_COLORS = {"high": "#08519c", "medium": "#6baed6", "low": "#bdd7e7"}

QUEUE_ORDER = [
    "Technical Support", "Product Support", "Customer Service",
    "IT Support", "Billing and Payments", "Returns and Exchanges",
    "Service Outages and Maintenance", "Sales and Pre-Sales",
    "Human Resources", "General Inquiry",
]

# ---------------------------------------------------------------------------
# Data loading & merging
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Load parquet + JSONL and merge into a unified EDA DataFrame."""
    df = pd.read_parquet(PARQUET_PATH)

    records: list[dict] = []
    with open(JSONL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    infer_df = pd.DataFrame(records)
    infer_df["_idx"] = infer_df["request_id"].str.extract(r"-(\d+)$").astype(int)

    df = df.reset_index(drop=True)
    df["_idx"] = df.index
    df = df.merge(
        infer_df[["_idx", "user_type", "industry", "tech_proficiency"]],
        on="_idx", how="left",
    )
    df = df.drop(columns=["_idx"])

    # Derived fields
    df["subject_len"] = df["subject"].str.len()
    df["body_len"] = df["body"].str.len()
    df["answer_len"] = df["answer"].str.len()
    df["has_tag1"] = df["tag_1"].notna()
    df["has_tag2"] = df["tag_2"].notna()

    if "tech_proficiency" in df.columns:
        df["tech_proficiency"] = df["tech_proficiency"].str.lower()
    if "user_type" in df.columns:
        df["user_type"] = df["user_type"].str.lower()

    return df


# ---------------------------------------------------------------------------
# Fig 1 — Class distributions: Language, Queue, Priority (all bars, no pie)
# ---------------------------------------------------------------------------

def plot_class_distribution(df: pd.DataFrame) -> plt.Figure:
    """Three-panel overview: Language, Queue, Priority — all as bar charts."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # --- Panel A: Language (Bold qualitative — deep muted jewel tones) ---
    ax = axes[0]
    lang_counts = df["language"].value_counts()
    lang_order = lang_counts.index.tolist()
    lang_hex = ["#4a6fa5", "#4fa17e", "#cd7b3e", "#b55b7a", "#7b6eaa"]
    colors_a = dict(zip(lang_order, lang_hex))
    bar_colors_a = [colors_a[l] for l in lang_order]
    bars = ax.bar(lang_order, lang_counts.values, color=bar_colors_a,
                  edgecolor="white", linewidth=0.6, width=0.55)
    for bar, val in zip(bars, lang_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                str(val), ha="center", fontsize=9, fontweight="bold",
                color="#333333")
    ax.set_ylabel("Count")
    ax.set_ylim(0, lang_counts.max() * 1.18)
    ax.tick_params(axis="x", pad=6)
    sns.despine(ax=ax)

    # --- Panel B: Queue (horizontal, "mako" colormap) ---
    ax = axes[1]
    queue_counts = df["queue"].value_counts()
    queue_sorted = [q for q in QUEUE_ORDER if q in queue_counts.index]
    counts_sorted = [queue_counts[q] for q in queue_sorted]
    cmap_b = sns.color_palette("crest", as_cmap=True)
    norm_b = plt.Normalize(min(counts_sorted), max(counts_sorted))
    colors_b = [cmap_b(norm_b(v)) for v in counts_sorted]
    bars = ax.barh(queue_sorted[::-1], counts_sorted[::-1], color=colors_b[::-1],
                   edgecolor="white", linewidth=0.6, height=0.65)
    for bar, val in zip(bars, counts_sorted[::-1]):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=8, fontweight="bold",
                color="#333333")
    ax.set_xlabel("Count")
    ax.set_xlim(0, max(counts_sorted) * 1.20)
    sns.despine(ax=ax)

    # --- Panel C: Priority (horizontal bar, warm-cool gradient) ---
    ax = axes[2]
    prio_order = ["high", "medium", "low"]
    prio_counts = df["priority"].value_counts()
    prio_vals = [prio_counts.get(p, 0) for p in prio_order]
    prio_labels = ["High", "Medium", "Low"]
    colors_c = ["#d95f02", "#f2a340", "#5ba3cf"]
    bars = ax.barh(prio_labels[::-1], prio_vals[::-1], color=colors_c[::-1],
                   edgecolor="white", linewidth=0.6, height=0.5)
    for bar, val in zip(bars, prio_vals[::-1]):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                f"{val} ({val / sum(prio_vals) * 100:.1f}%)",
                va="center", fontsize=9, fontweight="bold", color="#333333")
    ax.set_xlabel("Count")
    ax.set_xlim(0, max(prio_vals) * 1.35)
    sns.despine(ax=ax)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 2 — Queue × Priority heatmap
# ---------------------------------------------------------------------------

def plot_queue_priority_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Cross-tabulation heatmap: Queue × Priority."""
    ct = pd.crosstab(df["queue"], df["priority"])
    queue_in_data = [q for q in QUEUE_ORDER if q in ct.index]
    ct = ct.loc[queue_in_data, ["high", "medium", "low"]]

    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = sns.cubehelix_palette(start=0.5, rot=-0.5, dark=0.15, light=0.92,
                                 hue=0.15, as_cmap=True)
    sns.heatmap(
        ct, annot=True, fmt="d", cmap=cmap,
        ax=ax, linewidths=0.6, linecolor="#f0f0f0",
        cbar_kws={"label": "Count", "shrink": 0.8},
        annot_kws={"fontsize": 10, "fontweight": "bold"},
    )
    ax.set_title("Queue × Priority Cross-Tabulation", fontweight="bold",
                 fontsize=13, color="#222", pad=12)
    ax.set_xlabel("Priority", labelpad=8)
    ax.set_ylabel("Queue", labelpad=8)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 3 — Language × Queue heatmap (row-normalized)
# ---------------------------------------------------------------------------

def plot_language_queue_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Language × Queue cross-tab, row-normalized with count annotations."""
    ct = pd.crosstab(df["language"], df["queue"])
    queue_in_data = [q for q in QUEUE_ORDER if q in ct.columns]
    ct = ct[queue_in_data]
    ct_norm = ct.div(ct.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(13, 4.8))
    cmap = sns.color_palette("crest", as_cmap=True)
    sns.heatmap(
        ct_norm, annot=ct.values, fmt="d",
        cmap=cmap, ax=ax, linewidths=0.6, linecolor="#f0f0f0",
        cbar_kws={"label": "Proportion", "shrink": 0.7},
        annot_kws={"fontsize": 9, "fontweight": "bold"},
        vmin=0, vmax=ct_norm.values.max(),
    )
    ax.set_title("Language × Queue Distribution  (row-normalized, count annotated)",
                 fontweight="bold", fontsize=13, color="#222", pad=12)
    ax.set_xlabel("Queue", labelpad=8)
    ax.set_ylabel("Language", labelpad=8)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 4 — Priority by Language (stacked bars)
# ---------------------------------------------------------------------------

def plot_priority_by_language(df: pd.DataFrame) -> plt.Figure:
    """Stacked bar: Priority distribution within each language (absolute + %)."""
    ct = pd.crosstab(df["language"], df["priority"])
    prio_order = ["high", "medium", "low"]
    ct = ct.reindex(columns=prio_order, fill_value=0)
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    colors = [PRIORITY_COLORS[p] for p in prio_order]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    # Panel A: absolute counts
    ax = axes[0]
    ct.plot(kind="bar", stacked=True, color=colors, edgecolor="white",
            linewidth=0.6, width=0.6, ax=ax)
    for i, (lang, row) in enumerate(ct.iterrows()):
        cum = 0
        for prio in prio_order:
            val = row[prio]
            if val > 5:
                ax.text(i, cum + val / 2, str(val), ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white")
                cum += val
    ax.set_title("(a) Absolute Counts", fontweight="bold", color="#222")
    ax.set_ylabel("Count")
    ax.set_xlabel("Language")
    ax.legend(title="Priority", frameon=True, facecolor="white",
              edgecolor="#ddd", fancybox=True)
    ax.tick_params(axis="x", rotation=0)
    sns.despine(ax=ax)

    # Panel B: percentage stacked
    ax = axes[1]
    ct_pct.plot(kind="bar", stacked=True, color=colors, edgecolor="white",
                linewidth=0.6, width=0.6, ax=ax)
    for i, (lang, row) in enumerate(ct_pct.iterrows()):
        cum = 0
        for prio in prio_order:
            val = row[prio]
            if val > 3:
                ax.text(i, cum + val / 2, f"{val:.0f}%", ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white")
                cum += val
    ax.set_title("(b) Percentage (per language)", fontweight="bold", color="#222")
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("Language")
    ax.legend(title="Priority", frameon=True, facecolor="white",
              edgecolor="#ddd", fancybox=True)
    ax.tick_params(axis="x", rotation=0)
    sns.despine(ax=ax)

    fig.suptitle("Priority Distribution by Language", fontweight="bold",
                 fontsize=14, color="#111", y=1.01)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 5 — Text length distributions
# ---------------------------------------------------------------------------

def plot_text_lengths(df: pd.DataFrame) -> plt.Figure:
    """Text length by language: histograms, boxplot, and scatter."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Subject length histogram + KDE overlay
    ax = axes[0, 0]
    for lang, color in LANG_COLORS.items():
        sub = df[df["language"] == lang]["subject_len"]
        if len(sub) > 1:
            sns.kdeplot(data=sub, ax=ax, color=color, label=lang,
                        linewidth=2, fill=True, alpha=0.12)
    ax.set_xlabel("Subject Length (characters)")
    ax.set_ylabel("Density")
    ax.set_title("(a) Subject Length by Language", fontweight="bold", color="#222")
    ax.legend(fontsize=7.5, frameon=True, facecolor="white",
              edgecolor="#ddd", fancybox=True)
    sns.despine(ax=ax)

    # Panel B: Body length KDE
    ax = axes[0, 1]
    for lang, color in LANG_COLORS.items():
        sub = df[df["language"] == lang]["body_len"]
        if len(sub) > 1:
            sns.kdeplot(data=sub, ax=ax, color=color, label=lang,
                        linewidth=2, fill=True, alpha=0.12)
    ax.set_xlabel("Body Length (characters)")
    ax.set_ylabel("Density")
    ax.set_title("(b) Body Length by Language", fontweight="bold", color="#222")
    ax.legend(fontsize=7.5, frameon=True, facecolor="white",
              edgecolor="#ddd", fancybox=True)
    sns.despine(ax=ax)

    # Panel C: Body length boxplot by language
    ax = axes[1, 0]
    lang_order = sorted(df["language"].unique())
    box_data = [df[df["language"] == l]["body_len"].dropna().values
                for l in lang_order]
    bp = ax.boxplot(box_data, tick_labels=lang_order, patch_artist=True,
                    widths=0.45, medianprops={"color": "#333", "linewidth": 1.5},
                    whiskerprops={"linewidth": 1},
                    capprops={"linewidth": 1},
                    boxprops={"linewidth": 0.8})
    for patch, lang in zip(bp["boxes"], lang_order):
        patch.set_facecolor(LANG_COLORS.get(lang, "#999"))
        patch.set_alpha(0.55)
    ax.set_ylabel("Body Length (characters)")
    ax.set_title("(c) Body Length Boxplot by Language", fontweight="bold", color="#222")
    sns.despine(ax=ax)

    # Panel D: Subject vs Body scatter
    ax = axes[1, 1]
    for lang, color in LANG_COLORS.items():
        sub = df[df["language"] == lang]
        ax.scatter(sub["subject_len"], sub["body_len"], alpha=0.35, s=12,
                   color=color, label=lang, edgecolors="none", rasterized=True)
    ax.set_xlabel("Subject Length (characters)")
    ax.set_ylabel("Body Length (characters)")
    ax.set_title("(d) Subject vs Body Length", fontweight="bold", color="#222")
    ax.legend(fontsize=7.5, frameon=True, facecolor="white",
              edgecolor="#ddd", fancybox=True)
    sns.despine(ax=ax)

    fig.suptitle("Text Length Distributions", fontweight="bold",
                 fontsize=14, color="#111")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 6 — Inferred attributes: user_type, tech_proficiency, industry
# ---------------------------------------------------------------------------

def plot_inferred_attributes(df: pd.DataFrame) -> plt.Figure:
    """Qwen3-4B inferred fields — all bars (no pie)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Panel A: User type (horizontal bar, side-by-side with counts+%) ---
    ax = axes[0, 0]
    ut = df["user_type"].value_counts()
    ut = ut.reindex(index=["enterprise", "individual"], fill_value=0)
    ut_labels = [s.capitalize() for s in ut.index]
    ut_colors = [USER_TYPE_COLORS.get(s, "#999") for s in ut.index]
    bars = ax.barh(ut_labels[::-1], ut.values[::-1], color=ut_colors[::-1],
                   edgecolor="white", linewidth=0.6, height=0.45)
    for bar, val, total in zip(bars, ut.values[::-1],
                                [ut.sum()] * len(ut)):
        pct = val / ut.sum() * 100
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                f"{val} ({pct:.1f}%)", va="center",
                fontsize=10, fontweight="bold", color="#333")
    ax.set_xlabel("Count")
    ax.set_title("(a) User Type", fontweight="bold", color="#222")
    ax.set_xlim(0, ut.max() * 1.35)
    sns.despine(ax=ax)

    # --- Panel B: Tech Proficiency (horizontal bar) ---
    ax = axes[0, 1]
    tp_order = ["high", "medium", "low"]
    tp_labels = ["High", "Medium", "Low"]
    tp = df["tech_proficiency"].value_counts()
    tp_vals = [tp.get(p, 0) for p in tp_order]
    actual_order = [l for l, v in zip(tp_labels, tp_vals) if v > 0]
    actual_vals = [v for v in tp_vals if v > 0]
    actual_colors = [TECH_COLORS[p] for p in tp_order if tp.get(p, 0) > 0]
    bars = ax.barh(actual_order[::-1], actual_vals[::-1],
                   color=actual_colors[::-1],
                   edgecolor="white", linewidth=0.6, height=0.45)
    for bar, val in zip(bars, actual_vals[::-1]):
        pct = val / sum(actual_vals) * 100
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                f"{val} ({pct:.1f}%)", va="center",
                fontsize=10, fontweight="bold", color="#333")
    ax.set_xlabel("Count")
    ax.set_title("(b) Tech Proficiency", fontweight="bold", color="#222")
    ax.set_xlim(0, max(actual_vals) * 1.35)
    sns.despine(ax=ax)

    # --- Panel C: Top 15 Industries ---
    ax = axes[1, 0]
    ind = df["industry"].value_counts().head(15)
    # Use a perceptually uniform sequential colormap
    ind_cmap = sns.color_palette("vlag_r", n_colors=15)
    bars = ax.barh(ind.index.str[:30][::-1], ind.values[::-1],
                   color=ind_cmap, edgecolor="white", linewidth=0.6)
    for bar, val in zip(bars, ind.values[::-1]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=8, fontweight="bold",
                color="#333")
    ax.set_xlabel("Count")
    ax.set_title("(c) Top 15 Industries", fontweight="bold", color="#222")
    sns.despine(ax=ax)

    # --- Panel D: Tech Proficiency × User Type stacked bar ---
    ax = axes[1, 1]
    ct_ut_tp = pd.crosstab(df["tech_proficiency"], df["user_type"])
    ct_ut_tp = ct_ut_tp.reindex(index=tp_order, fill_value=0)
    ct_ut_tp.index = ["High", "Medium", "Low"]
    ut_cols_present = [c for c in ct_ut_tp.columns if c in USER_TYPE_COLORS]
    ct_ut_tp.plot(
        kind="bar", stacked=True, ax=ax,
        color=[USER_TYPE_COLORS.get(c, "#999") for c in ct_ut_tp.columns],
        edgecolor="white", linewidth=0.6, width=0.55,
    )
    for i, (idx_name, row) in enumerate(ct_ut_tp.iterrows()):
        cum = 0
        for col in ct_ut_tp.columns:
            val = row[col]
            if val > 15:
                ax.text(i, cum + val / 2, str(val), ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")
                cum += val
    ax.set_title("(d) Tech Proficiency × User Type", fontweight="bold", color="#222")
    ax.set_ylabel("Count")
    ax.set_xlabel("Tech Proficiency")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="User Type", frameon=True, facecolor="white",
              edgecolor="#ddd", fancybox=True)
    sns.despine(ax=ax)

    fig.suptitle("Inferred Attributes  (Qwen3-4B)", fontweight="bold",
                 fontsize=14, color="#111")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 7 — Queue by User Type & Tech Proficiency
# ---------------------------------------------------------------------------

def plot_queue_by_attributes(df: pd.DataFrame) -> plt.Figure:
    """Queue composition by user_type and tech_proficiency (100% stacked)."""
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))

    queue_cmap = sns.color_palette("tab10", n_colors=10)

    # Panel A: Queue by user_type
    ax = axes[0]
    ct_ut = pd.crosstab(df["queue"], df["user_type"])
    queue_in_data = [q for q in QUEUE_ORDER if q in ct_ut.index]
    ct_ut = ct_ut.loc[queue_in_data]
    ct_ut_pct = ct_ut.div(ct_ut.sum(axis=0), axis=1) * 100
    ct_ut_pct.T.plot(kind="barh", stacked=True, ax=ax, color=queue_cmap,
                     edgecolor="white", linewidth=0.4, width=0.7)
    ax.set_title("(a) Queue Composition by User Type (%)",
                 fontweight="bold", color="#222")
    ax.set_xlabel("Percentage (%)")
    ax.legend(title="Queue", bbox_to_anchor=(1.02, 1), fontsize=7.5,
              frameon=True, facecolor="white", edgecolor="#ddd", fancybox=True)
    ax.set_xlim(0, 100)
    sns.despine(ax=ax)

    # Panel B: Queue by tech_proficiency
    ax = axes[1]
    ct_tp = pd.crosstab(df["queue"], df["tech_proficiency"])
    queue_in_data = [q for q in QUEUE_ORDER if q in ct_tp.index]
    ct_tp = ct_tp.loc[queue_in_data]
    ct_tp_pct = ct_tp.div(ct_tp.sum(axis=0), axis=1) * 100
    tp_order_display = ["high", "low"]
    tp_labels = ["High", "Low"]
    ct_tp_pct = ct_tp_pct.reindex(columns=tp_order_display, fill_value=0)
    ct_tp_pct.columns = tp_labels
    ct_tp_pct.T.plot(kind="barh", stacked=True, ax=ax, color=queue_cmap,
                     edgecolor="white", linewidth=0.4, width=0.5)
    ax.set_title("(b) Queue Composition by Tech Proficiency (%)",
                 fontweight="bold", color="#222")
    ax.set_xlabel("Percentage (%)")
    ax.legend(title="Queue", bbox_to_anchor=(1.02, 1), fontsize=7.5,
              frameon=True, facecolor="white", edgecolor="#ddd", fancybox=True)
    ax.set_xlim(0, 100)
    sns.despine(ax=ax)

    fig.suptitle("Queue Distribution by Inferred Attributes", fontweight="bold",
                 fontsize=14, color="#111")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 8 — Priority by Inferred Attributes
# ---------------------------------------------------------------------------

def plot_priority_by_attributes(df: pd.DataFrame) -> plt.Figure:
    """Priority split by user_type and tech_proficiency (100% stacked bar)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    prio_order = ["high", "medium", "low"]
    prio_labels = ["High", "Medium", "Low"]

    # Panel A: User Type
    ax = axes[0]
    ct_ut = pd.crosstab(df["user_type"], df["priority"])
    ct_ut = ct_ut.reindex(index=["enterprise", "individual"],
                          columns=prio_order, fill_value=0)
    ct_ut.index = ["Enterprise", "Individual"]
    ct_ut.columns = prio_labels
    ct_ut_pct = ct_ut.div(ct_ut.sum(axis=1), axis=0) * 100
    ct_ut_pct.plot(kind="bar", ax=ax,
                   color=[PRIORITY_COLORS[p] for p in prio_order],
                   edgecolor="white", linewidth=0.6, width=0.55)
    for i, (idx_name, row) in enumerate(ct_ut_pct.iterrows()):
        cum = 0
        for prio in prio_order:
            val = row[prio_labels[prio_order.index(prio)]]
            if val > 5:
                ax.text(i, cum + val / 2, f"{val:.0f}%", ha="center",
                        va="center", fontsize=9, fontweight="bold", color="white")
                cum += val
    ax.set_title("(a) Priority by User Type (%)", fontweight="bold", color="#222")
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Priority", frameon=True, facecolor="white",
              edgecolor="#ddd", fancybox=True)
    ax.set_ylim(0, 105)
    sns.despine(ax=ax)

    # Panel B: Tech Proficiency
    ax = axes[1]
    ct_tp = pd.crosstab(df["tech_proficiency"], df["priority"])
    tp_in_data = [p for p in ["high", "medium", "low"]
                  if p in ct_tp.index]
    ct_tp = ct_tp.reindex(index=tp_in_data, columns=prio_order, fill_value=0)
    ct_tp.index = [l.capitalize() for l in tp_in_data]
    ct_tp.columns = prio_labels
    ct_tp_pct = ct_tp.div(ct_tp.sum(axis=1), axis=0) * 100
    actual_colors = [PRIORITY_COLORS[p] for p in prio_order]
    ct_tp_pct.plot(kind="bar", ax=ax, color=actual_colors,
                   edgecolor="white", linewidth=0.6, width=0.55)
    for i, (idx_name, row) in enumerate(ct_tp_pct.iterrows()):
        cum = 0
        for prio in prio_order:
            val = row[prio_labels[prio_order.index(prio)]]
            if val > 5:
                ax.text(i, cum + val / 2, f"{val:.0f}%", ha="center",
                        va="center", fontsize=9, fontweight="bold", color="white")
                cum += val
    ax.set_title("(b) Priority by Tech Proficiency (%)",
                 fontweight="bold", color="#222")
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Priority", frameon=True, facecolor="white",
              edgecolor="#ddd", fancybox=True)
    ax.set_ylim(0, 105)
    sns.despine(ax=ax)

    fig.suptitle("Priority Distribution by Inferred Attributes", fontweight="bold",
                 fontsize=14, color="#111")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 9 — Body Length vs Priority / Queue
# ---------------------------------------------------------------------------

def plot_text_vs_label(df: pd.DataFrame) -> plt.Figure:
    """Boxplots: body length split by Priority and Queue."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Body length by priority
    ax = axes[0]
    prio_order = ["high", "medium", "low"]
    bp_data = [df[df["priority"] == p]["body_len"].dropna().values
               for p in prio_order]
    bp = ax.boxplot(bp_data, tick_labels=["High", "Medium", "Low"],
                    patch_artist=True, widths=0.45,
                    medianprops={"color": "#222", "linewidth": 1.5},
                    whiskerprops={"linewidth": 1},
                    capprops={"linewidth": 1},
                    boxprops={"linewidth": 0.8})
    for patch, prio in zip(bp["boxes"], prio_order):
        patch.set_facecolor(PRIORITY_COLORS[prio])
        patch.set_alpha(0.55)
    ax.set_ylabel("Body Length (characters)")
    ax.set_title("(a) Body Length by Priority", fontweight="bold", color="#222")
    sns.despine(ax=ax)

    # Panel B: Body length by queue
    ax = axes[1]
    queue_in_data = [q for q in QUEUE_ORDER if q in df["queue"].values]
    bp_data_q = [df[df["queue"] == q]["body_len"].dropna().values
                 for q in queue_in_data]
    queue_cmap = sns.color_palette("crest", n_colors=len(queue_in_data))
    bp_q = ax.boxplot(bp_data_q, tick_labels=[q[:22] for q in queue_in_data],
                      patch_artist=True, widths=0.45, vert=False,
                      medianprops={"color": "#222", "linewidth": 1.2},
                      whiskerprops={"linewidth": 1},
                      capprops={"linewidth": 1},
                      boxprops={"linewidth": 0.8})
    for patch, color in zip(bp_q["boxes"], queue_cmap):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    ax.set_xlabel("Body Length (characters)")
    ax.set_title("(b) Body Length by Queue", fontweight="bold", color="#222")
    sns.despine(ax=ax)

    fig.suptitle("Text Length vs Labels", fontweight="bold",
                 fontsize=14, color="#111")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 10 — Tag coverage analysis
# ---------------------------------------------------------------------------

def plot_tag_coverage(df: pd.DataFrame) -> plt.Figure:
    """Tag column non-null rates and top tag_1 values."""
    tag_cols = [f"tag_{i}" for i in range(1, 10)]
    coverage = {col: df[col].notna().mean() * 100 for col in tag_cols}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    # Panel A: Tag coverage rate
    ax = axes[0]
    labels = [f"tag_{i}" for i in range(1, 10)]
    values = [coverage[c] for c in tag_cols]
    # Gradient from low to full coverage
    cov_cmap = sns.color_palette("flare", n_colors=len(values))
    bars = ax.bar(labels, values, color=cov_cmap, edgecolor="white",
                  linewidth=0.6, width=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold",
                color="#333")
    ax.set_ylabel("Non-null Rate (%)")
    ax.set_title("(a) Tag Column Coverage", fontweight="bold", color="#222")
    ax.set_ylim(0, 108)
    sns.despine(ax=ax)

    # Panel B: Top-15 tag_1 values
    ax = axes[1]
    tag1_counts = df["tag_1"].value_counts().head(15)
    tag1_cmap = sns.color_palette("crest_r", n_colors=len(tag1_counts))
    bars = ax.barh(tag1_counts.index.str[:35][::-1], tag1_counts.values[::-1],
                   color=tag1_cmap, edgecolor="white", linewidth=0.6)
    for bar, val in zip(bars, tag1_counts.values[::-1]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=8, fontweight="bold", color="#333")
    ax.set_xlabel("Count")
    ax.set_title("(b) Top 15 tag_1 Values", fontweight="bold", color="#222")
    sns.despine(ax=ax)

    fig.suptitle("Tag Coverage Analysis", fontweight="bold",
                 fontsize=14, color="#111")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_data()
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
        path = OUTPUT_DIR / filename
        fig.savefig(path, format="png")
        plt.close(fig)
        print(f"Saved: {path}")

    print(f"\nDone — {len(figures)} figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
