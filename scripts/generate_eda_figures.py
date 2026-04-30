#!/usr/bin/env python3
"""EDA 可视化脚本 — 数据集探索性分析图表.

基于 outputs/multilingual-customer-support_test_split.parquet 及
outputs/infer_multilingual-customer-support_Qwen3-4B.jsonl 推导字段.

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
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
PARQUET_PATH = Path("outputs/multilingual-customer-support_test_split.parquet")
JSONL_PATH = Path("outputs/infer_multilingual-customer-support_Qwen3-4B.jsonl")
OUTPUT_DIR = Path("results/figures")

# ---------------------------------------------------------------------------
# 学术风格
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
sns.set_palette("muted")

# 调色板
LANG_COLORS = {"en": "#2196f3", "de": "#4caf50", "es": "#ff9800", "fr": "#e91e63", "pt": "#9c27b0"}
PRIORITY_COLORS = {"high": "#e53935", "medium": "#ff9800", "low": "#4caf50"}
USER_TYPE_COLORS = {"enterprise": "#1565c0", "individual": "#66bb6a"}
TECH_COLORS = {"high": "#e53935", "medium": "#ff9800", "low": "#4caf50"}

QUEUE_ORDER = [
    "Technical Support", "Product Support", "Customer Service",
    "IT Support", "Billing and Payments", "Returns and Exchanges",
    "Service Outages and Maintenance", "Sales and Pre-Sales",
    "Human Resources", "General Inquiry",
]


# ---------------------------------------------------------------------------
# 数据加载 & 合并
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """加载 parquet 和 jsonl, 合并为统一的 EDA DataFrame."""
    df = pd.read_parquet(PARQUET_PATH)

    # 加载 JSONL 推导字段
    records: list[dict] = []
    with open(JSONL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    infer_df = pd.DataFrame(records)
    # request_id 格式: "multilingual-customer-support-XXXXXX", 索引即 parquet 行号
    infer_df["_idx"] = infer_df["request_id"].str.extract(r"-(\d+)$").astype(int)

    # 合并: 按索引对齐
    df = df.reset_index(drop=True)
    df["_idx"] = df.index
    df = df.merge(
        infer_df[["_idx", "user_type", "industry", "tech_proficiency"]],
        on="_idx", how="left",
    )
    df = df.drop(columns=["_idx"])

    # 衍生字段
    df["subject_len"] = df["subject"].str.len()
    df["body_len"] = df["body"].str.len()
    df["answer_len"] = df["answer"].str.len()
    df["has_tag1"] = df["tag_1"].notna()
    df["has_tag2"] = df["tag_2"].notna()

    # 统一 tech_proficiency 大小写
    if "tech_proficiency" in df.columns:
        df["tech_proficiency"] = df["tech_proficiency"].str.lower()

    # 统一 user_type
    if "user_type" in df.columns:
        df["user_type"] = df["user_type"].str.lower()

    return df


# ---------------------------------------------------------------------------
# Fig 1: Language + Queue + Priority 三合一分布
# ---------------------------------------------------------------------------

def plot_class_distribution(df: pd.DataFrame) -> plt.Figure:
    """三面板: 语言分布 / Queue 分布 / Priority 饼图."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Language
    ax = axes[0]
    lang_counts = df["language"].value_counts()
    lang_order = lang_counts.index.tolist()
    colors_a = [LANG_COLORS.get(l, "#999") for l in lang_order]
    bars = ax.bar(lang_order, lang_counts.values, color=colors_a, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, lang_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                str(val), ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_title("(a) Language Distribution", fontweight="bold")
    ax.set_ylim(0, lang_counts.max() * 1.15)

    # Panel B: Queue
    ax = axes[1]
    queue_counts = df["queue"].value_counts()
    queue_sorted = [q for q in QUEUE_ORDER if q in queue_counts.index]
    counts_sorted = [queue_counts[q] for q in queue_sorted]
    cmap_b = plt.cm.viridis
    norm_b = plt.Normalize(min(counts_sorted), max(counts_sorted))
    colors_b = [cmap_b(norm_b(v)) for v in counts_sorted]
    bars = ax.barh(queue_sorted[::-1], counts_sorted[::-1], color=colors_b[::-1],
                   edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, counts_sorted[::-1]):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=8, fontweight="bold")
    ax.set_xlabel("Count")
    ax.set_title("(b) Queue Distribution", fontweight="bold")
    ax.set_xlim(0, max(counts_sorted) * 1.18)

    # Panel C: Priority
    ax = axes[2]
    prio_counts = df["priority"].value_counts()
    prio_order = ["high", "medium", "low"]
    prio_vals = [prio_counts.get(p, 0) for p in prio_order]
    colors_c = [PRIORITY_COLORS[p] for p in prio_order]
    wedges, texts, autotexts = ax.pie(
        prio_vals, labels=["High", "Medium", "Low"],
        colors=colors_c, autopct="%1.1f%%",
        startangle=90, pctdistance=0.6, explode=(0.02, 0.02, 0.02),
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
    ax.set_title("(c) Priority Distribution", fontweight="bold")

    fig.suptitle("Dataset Overview: Language, Queue & Priority", fontweight="bold", fontsize=14)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 2: Queue vs Priority 热力图
# ---------------------------------------------------------------------------

def plot_queue_priority_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Queue × Priority 交叉表热力图."""
    ct = pd.crosstab(df["queue"], df["priority"])
    # 按 queue 顺序排列
    queue_in_data = [q for q in QUEUE_ORDER if q in ct.index]
    ct = ct.loc[queue_in_data, ["high", "medium", "low"]]

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        ct, annot=True, fmt="d", cmap="YlOrRd",
        ax=ax, linewidths=0.5, cbar_kws={"label": "Count"},
    )
    ax.set_title("Queue × Priority Cross-Tabulation", fontweight="bold", fontsize=13)
    ax.set_xlabel("Priority")
    ax.set_ylabel("Queue")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 3: Language vs Queue 热力图 (按比例归一化)
# ---------------------------------------------------------------------------

def plot_language_queue_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Language × Queue 交叉表, 按行归一化 (每个语言内各 queue 占比)."""
    ct = pd.crosstab(df["language"], df["queue"])
    queue_in_data = [q for q in QUEUE_ORDER if q in ct.columns]
    ct = ct[queue_in_data]
    ct_norm = ct.div(ct.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    sns.heatmap(
        ct_norm, annot=ct.values, fmt="d",
        cmap="YlGnBu", ax=ax, linewidths=0.5,
        cbar_kws={"label": "Proportion"},
        vmin=0, vmax=ct_norm.values.max(),
    )
    ax.set_title("Language × Queue Distribution (row-normalized, count annotated)",
                 fontweight="bold", fontsize=13)
    ax.set_xlabel("Queue")
    ax.set_ylabel("Language")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 4: Priority 按 Language 分组堆叠柱状图
# ---------------------------------------------------------------------------

def plot_priority_by_language(df: pd.DataFrame) -> plt.Figure:
    """每种语言内 Priority 分布 (堆叠 + 百分比标注)."""
    ct = pd.crosstab(df["language"], df["priority"])
    prio_order = ["high", "medium", "low"]
    ct = ct.reindex(columns=prio_order, fill_value=0)
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    colors = [PRIORITY_COLORS[p] for p in prio_order]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: 绝对数量堆叠
    ax = axes[0]
    ct.plot(kind="bar", stacked=True, color=colors, edgecolor="white", linewidth=0.5, ax=ax)
    for i, (lang, row) in enumerate(ct.iterrows()):
        cum = 0
        for j, prio in enumerate(prio_order):
            val = row[prio]
            if val > 0:
                ax.text(i, cum + val / 2, str(val), ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white")
                cum += val
    ax.set_title("(a) Absolute Counts", fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_xlabel("Language")
    ax.legend(title="Priority")
    ax.tick_params(axis="x", rotation=0)

    # Panel B: 百分比堆叠
    ax = axes[1]
    ct_pct.plot(kind="bar", stacked=True, color=colors, edgecolor="white", linewidth=0.5, ax=ax)
    for i, (lang, row) in enumerate(ct_pct.iterrows()):
        cum = 0
        for j, prio in enumerate(prio_order):
            val = row[prio]
            if val > 2:
                ax.text(i, cum + val / 2, f"{val:.0f}%", ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white")
                cum += val
    ax.set_title("(b) Percentage (normalized per language)", fontweight="bold")
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("Language")
    ax.legend(title="Priority")
    ax.tick_params(axis="x", rotation=0)

    fig.suptitle("Priority Distribution by Language", fontweight="bold", fontsize=14)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 5: Text Length Distributions
# ---------------------------------------------------------------------------

def plot_text_lengths(df: pd.DataFrame) -> plt.Figure:
    """Subject / Body / Answer 长度分布 (直方图 + 箱线图)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Subject length histogram + KDE
    ax = axes[0, 0]
    for lang, color in LANG_COLORS.items():
        sub = df[df["language"] == lang]["subject_len"]
        if len(sub) > 0:
            ax.hist(sub, bins=30, alpha=0.4, color=color, label=lang, density=True)
    ax.set_xlabel("Subject Length (characters)")
    ax.set_ylabel("Density")
    ax.set_title("(a) Subject Length by Language", fontweight="bold")
    ax.legend(fontsize=7)

    # Panel B: Body length histogram + KDE
    ax = axes[0, 1]
    for lang, color in LANG_COLORS.items():
        sub = df[df["language"] == lang]["body_len"]
        if len(sub) > 0:
            ax.hist(sub, bins=30, alpha=0.4, color=color, label=lang, density=True)
    ax.set_xlabel("Body Length (characters)")
    ax.set_ylabel("Density")
    ax.set_title("(b) Body Length by Language", fontweight="bold")
    ax.legend(fontsize=7)

    # Panel C: Body length boxplot by language
    ax = axes[1, 0]
    lang_order = sorted(df["language"].unique())
    box_data = [df[df["language"] == l]["body_len"].dropna().values for l in lang_order]
    bp = ax.boxplot(box_data, tick_labels=lang_order, patch_artist=True, widths=0.5)
    for patch, lang in zip(bp["boxes"], lang_order):
        patch.set_facecolor(LANG_COLORS.get(lang, "#999"))
        patch.set_alpha(0.6)
    ax.set_ylabel("Body Length (characters)")
    ax.set_title("(c) Body Length Boxplot by Language", fontweight="bold")

    # Panel D: Subject vs Body scatter
    ax = axes[1, 1]
    for lang, color in LANG_COLORS.items():
        sub = df[df["language"] == lang]
        ax.scatter(sub["subject_len"], sub["body_len"], alpha=0.4, s=10,
                   color=color, label=lang, edgecolors="none")
    ax.set_xlabel("Subject Length")
    ax.set_ylabel("Body Length")
    ax.set_title("(d) Subject vs Body Length", fontweight="bold")
    ax.legend(fontsize=7)

    fig.suptitle("Text Length Distributions", fontweight="bold", fontsize=14)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 6: User Type & Tech Proficiency (from Qwen3-4B)
# ---------------------------------------------------------------------------

def plot_inferred_attributes(df: pd.DataFrame) -> plt.Figure:
    """Qwen3-4B 推导字段: user_type, tech_proficiency, industry 分布."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: User type pie
    ax = axes[0, 0]
    ut = df["user_type"].value_counts()
    colors_ut = [USER_TYPE_COLORS.get(u, "#999") for u in ut.index]
    wedges, texts, autotexts = ax.pie(
        ut.values, labels=ut.index.str.capitalize(),
        colors=colors_ut, autopct="%1.1f%%", startangle=90,
        pctdistance=0.6, explode=(0.03, 0.03),
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
    ax.set_title("(a) User Type", fontweight="bold")

    # Panel B: Tech proficiency bar
    ax = axes[0, 1]
    tp = df["tech_proficiency"].value_counts()
    tp_order = ["high", "medium", "low"]
    tp_vals = [tp.get(p, 0) for p in tp_order]
    colors_tp = [TECH_COLORS[p] for p in tp_order]
    bars = ax.bar(["High", "Medium", "Low"], tp_vals, color=colors_tp,
                  edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, tp_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                str(val), ha="center", fontweight="bold", fontsize=10)
    ax.set_ylabel("Count")
    ax.set_title("(b) Tech Proficiency", fontweight="bold")
    ax.set_ylim(0, max(tp_vals) * 1.15)

    # Panel C: Industry horizontal bar
    ax = axes[1, 0]
    ind = df["industry"].value_counts().head(15)
    bars = ax.barh(ind.index.str[:30][::-1], ind.values[::-1],
                   color="#5c6bc0", edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, ind.values[::-1]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=8, fontweight="bold")
    ax.set_xlabel("Count")
    ax.set_title("(c) Top 15 Industries", fontweight="bold")

    # Panel D: User type × Tech proficiency 堆叠
    ax = axes[1, 1]
    ct_ut_tp = pd.crosstab(df["tech_proficiency"], df["user_type"])
    ct_ut_tp = ct_ut_tp.reindex(index=tp_order, fill_value=0)
    ct_ut_tp.plot(kind="bar", stacked=True, ax=ax,
                  color=[USER_TYPE_COLORS.get(c, "#999") for c in ct_ut_tp.columns],
                  edgecolor="white", linewidth=0.5)
    for i, (tp_val, row) in enumerate(ct_ut_tp.iterrows()):
        cum = 0
        for j, col in enumerate(ct_ut_tp.columns):
            val = row[col]
            if val > 10:
                ax.text(i, cum + val / 2, str(val), ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")
                cum += val
    ax.set_title("(d) Tech Proficiency × User Type", fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_xlabel("Tech Proficiency")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="User Type")

    fig.suptitle("Inferred Attributes from Qwen3-4B", fontweight="bold", fontsize=14)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 7: Queue by User Type & Tech Proficiency
# ---------------------------------------------------------------------------

def plot_queue_by_attributes(df: pd.DataFrame) -> plt.Figure:
    """Queue 分布在 user_type / tech_proficiency 维度上的分组对比."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel A: Queue by user_type (百分比堆叠)
    ax = axes[0]
    ct_ut = pd.crosstab(df["queue"], df["user_type"])
    queue_in_data = [q for q in QUEUE_ORDER if q in ct_ut.index]
    ct_ut = ct_ut.loc[queue_in_data]
    ct_ut_pct = ct_ut.div(ct_ut.sum(axis=0), axis=1) * 100
    ct_ut_pct.T.plot(kind="barh", stacked=True, ax=ax,
                     colormap="tab10", edgecolor="white", linewidth=0.3)
    ax.set_title("(a) Queue Composition by User Type (%)", fontweight="bold")
    ax.set_xlabel("Percentage (%)")
    ax.legend(title="Queue", bbox_to_anchor=(1.02, 1), fontsize=7)

    # Panel B: Queue by tech_proficiency (百分比堆叠)
    ax = axes[1]
    ct_tp = pd.crosstab(df["queue"], df["tech_proficiency"])
    queue_in_data = [q for q in QUEUE_ORDER if q in ct_tp.index]
    ct_tp = ct_tp.loc[queue_in_data]
    ct_tp_pct = ct_tp.div(ct_tp.sum(axis=0), axis=1) * 100
    tp_order_display = ["high", "medium", "low"]
    ct_tp_pct = ct_tp_pct.reindex(columns=tp_order_display, fill_value=0)
    ct_tp_pct.T.plot(kind="barh", stacked=True, ax=ax,
                     colormap="tab10", edgecolor="white", linewidth=0.3)
    ax.set_title("(b) Queue Composition by Tech Proficiency (%)", fontweight="bold")
    ax.set_xlabel("Percentage (%)")
    ax.legend(title="Queue", bbox_to_anchor=(1.02, 1), fontsize=7)

    fig.suptitle("Queue Distribution by Inferred Attributes", fontweight="bold", fontsize=14)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 8: Priority by Inferred Attributes
# ---------------------------------------------------------------------------

def plot_priority_by_attributes(df: pd.DataFrame) -> plt.Figure:
    """Priority 在 user_type / tech_proficiency 维度上的分组对比."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    prio_order = ["high", "medium", "low"]

    # Panel A: user_type
    ax = axes[0]
    ct_ut = pd.crosstab(df["user_type"], df["priority"])
    ct_ut = ct_ut.reindex(index=["enterprise", "individual"], columns=prio_order, fill_value=0)
    ct_ut_pct = ct_ut.div(ct_ut.sum(axis=1), axis=0) * 100
    ct_ut_pct.plot(kind="bar", ax=ax, color=[PRIORITY_COLORS[p] for p in prio_order],
                   edgecolor="white", linewidth=0.5)
    for i, (idx, row) in enumerate(ct_ut_pct.iterrows()):
        cum = 0
        for j, prio in enumerate(prio_order):
            val = row[prio]
            if val > 5:
                ax.text(i, cum + val / 2, f"{val:.0f}%", ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")
                cum += val
    ax.set_title("(a) Priority by User Type (%)", fontweight="bold")
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("User Type")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Priority")

    # Panel B: tech_proficiency
    ax = axes[1]
    ct_tp = pd.crosstab(df["tech_proficiency"], df["priority"])
    ct_tp = ct_tp.reindex(index=["high", "medium", "low"], columns=prio_order, fill_value=0)
    ct_tp_pct = ct_tp.div(ct_tp.sum(axis=1), axis=0) * 100
    ct_tp_pct.plot(kind="bar", ax=ax, color=[PRIORITY_COLORS[p] for p in prio_order],
                   edgecolor="white", linewidth=0.5)
    for i, (idx, row) in enumerate(ct_tp_pct.iterrows()):
        cum = 0
        for j, prio in enumerate(prio_order):
            val = row[prio]
            if val > 5:
                ax.text(i, cum + val / 2, f"{val:.0f}%", ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")
                cum += val
    ax.set_title("(b) Priority by Tech Proficiency (%)", fontweight="bold")
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("Tech Proficiency")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Priority")

    fig.suptitle("Priority Distribution by Inferred Attributes", fontweight="bold", fontsize=14)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 9: Body Length vs Priority / Queue
# ---------------------------------------------------------------------------

def plot_text_vs_label(df: pd.DataFrame) -> plt.Figure:
    """Body 长度按 Priority 和 Queue 分组的箱线图."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Body length by priority
    ax = axes[0]
    prio_order = ["high", "medium", "low"]
    bp_data = [df[df["priority"] == p]["body_len"].dropna().values for p in prio_order]
    bp = ax.boxplot(bp_data, tick_labels=["High", "Medium", "Low"],
                    patch_artist=True, widths=0.5)
    for patch, prio in zip(bp["boxes"], prio_order):
        patch.set_facecolor(PRIORITY_COLORS[prio])
        patch.set_alpha(0.6)
    ax.set_ylabel("Body Length (characters)")
    ax.set_title("(a) Body Length by Priority", fontweight="bold")

    # Panel B: Body length by queue
    ax = axes[1]
    queue_in_data = [q for q in QUEUE_ORDER if q in df["queue"].values]
    bp_data_q = [df[df["queue"] == q]["body_len"].dropna().values for q in queue_in_data]
    bp_q = ax.boxplot(bp_data_q, tick_labels=[q[:20] for q in queue_in_data],
                      patch_artist=True, widths=0.5, vert=False)
    for patch in bp_q["boxes"]:
        patch.set_facecolor("#5c6bc0")
        patch.set_alpha(0.5)
    ax.set_xlabel("Body Length (characters)")
    ax.set_title("(b) Body Length by Queue", fontweight="bold")

    fig.suptitle("Text Length vs Labels", fontweight="bold", fontsize=14)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 10: Tag 覆盖分析
# ---------------------------------------------------------------------------

def plot_tag_coverage(df: pd.DataFrame) -> plt.Figure:
    """Tag 字段的非空率和 tag 值分布."""
    tag_cols = [f"tag_{i}" for i in range(1, 10)]
    coverage = {col: df[col].notna().mean() * 100 for col in tag_cols}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Tag coverage rate
    ax = axes[0]
    labels = [f"tag_{i}" for i in range(1, 10)]
    values = [coverage[c] for c in tag_cols]
    bars = ax.bar(labels, values, color="#5c6bc0", edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", fontsize=8, fontweight="bold")
    ax.set_ylabel("Non-null Rate (%)")
    ax.set_title("(a) Tag Column Coverage", fontweight="bold")
    ax.set_ylim(0, 105)

    # Panel B: Top tag_1 values
    ax = axes[1]
    tag1_counts = df["tag_1"].value_counts().head(15)
    bars = ax.barh(tag1_counts.index.str[:30][::-1], tag1_counts.values[::-1],
                   color="#26a69a", edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, tag1_counts.values[::-1]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=8, fontweight="bold")
    ax.set_xlabel("Count")
    ax.set_title("(b) Top 15 tag_1 Values", fontweight="bold")

    fig.suptitle("Tag Coverage Analysis", fontweight="bold", fontsize=14)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 11: 综合信息面板 (Summary Stats)
# ---------------------------------------------------------------------------

def plot_summary_stats(df: pd.DataFrame) -> plt.Figure:
    """汇总统计表格."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")

    stats = [
        ("Total Samples", f"{len(df):,}"),
        ("Languages", f"{df['language'].nunique()} ({', '.join(sorted(df['language'].unique()))})"),
        ("Queue Classes", str(df["queue"].nunique())),
        ("Priority Classes", str(df["priority"].nunique())),
        ("Avg Subject Length", f"{df['subject_len'].mean():.0f} ± {df['subject_len'].std():.0f}"),
        ("Avg Body Length", f"{df['body_len'].mean():.0f} ± {df['body_len'].std():.0f}"),
        ("Avg Answer Length", f"{df['answer_len'].mean():.0f} ± {df['answer_len'].std():.0f}"),
        ("Unique Business Types", str(df["business_type"].nunique())),
        ("User Type — Enterprise", f"{df['user_type'].value_counts().get('enterprise', 0)}"),
        ("User Type — Individual", f"{df['user_type'].value_counts().get('individual', 0)}"),
        ("Tech Proficiency — High/Medium/Low",
         f"{df['tech_proficiency'].value_counts().get('high', 0)} / "
         f"{df['tech_proficiency'].value_counts().get('medium', 0)} / "
         f"{df['tech_proficiency'].value_counts().get('low', 0)}"),
        ("Most Common Queue", f"{df['queue'].value_counts().index[0]} ({df['queue'].value_counts().iloc[0]})"),
        ("Most Common Priority", f"{df['priority'].value_counts().index[0]} ({df['priority'].value_counts().iloc[0]})"),
    ]

    col_labels = ["Metric", "Value"]
    table = ax.table(
        cellText=stats, colLabels=col_labels,
        cellLoc="left", loc="center",
        colWidths=[0.45, 0.55],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    # Style header
    for key, cell in table._cells.items():
        if key[0] == 0:
            cell.set_fontsize(11)
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#e0e0e0")
        else:
            cell.set_facecolor("#fafafa" if key[0] % 2 == 0 else "#f0f0f0")

    ax.set_title("Dataset Summary Statistics", fontweight="bold", fontsize=14, pad=20)
    return fig


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # 数据摘要
    print(f"\nLanguage distribution:\n{df['language'].value_counts()}")
    print(f"\nUser type distribution:\n{df['user_type'].value_counts()}")
    print(f"\nTech proficiency distribution:\n{df['tech_proficiency'].value_counts()}")

    # 生成图表
    figures: list[tuple[str, plt.Figure]] = [
        ("eda_01_class_distribution.png", plot_class_distribution(df)),
        ("eda_02_queue_priority_heatmap.png", plot_queue_priority_heatmap(df)),
        ("eda_03_language_queue_heatmap.png", plot_language_queue_heatmap(df)),
        ("eda_04_priority_by_language.png", plot_priority_by_language(df)),
        ("eda_05_text_lengths.png", plot_text_lengths(df)),
        ("eda_06_inferred_attributes.png", plot_inferred_attributes(df)),
        ("eda_07_queue_by_attributes.png", plot_queue_by_attributes(df)),
        ("eda_08_priority_by_attributes.png", plot_priority_by_attributes(df)),
        ("eda_09_text_vs_label.png", plot_text_vs_label(df)),
        ("eda_10_tag_coverage.png", plot_tag_coverage(df)),
        ("eda_11_summary_stats.png", plot_summary_stats(df)),
    ]

    for filename, fig in figures:
        path = OUTPUT_DIR / filename
        fig.savefig(path, format="png")
        plt.close(fig)
        print(f"Saved: {path}")

    print(f"\nAll {len(figures)} EDA figures generated in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
