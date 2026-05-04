#!/usr/bin/env python3
"""Fairness 可视化脚本 — 8 张论文级公平性分析图表.

Usage:
    uv run python results/generate_fairness_figures.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("results/figures")
RESULTS_CSV = sorted(glob.glob("results/eval_multilingual-customer-support_*.csv"))[-1]

PARADIGM_ORDER = [
    "Rule-Based",
    "Supervised (Non-Encoder)",
    "Supervised (Encoder)",
    "Goal-Based (LLM)",
    "Goal-Based (API)",
]

PARADIGM_COLOR = {
    "Rule-Based": "#4caf50",
    "Supervised (Non-Encoder)": "#2196f3",
    "Supervised (Encoder)": "#1565c0",
    "Goal-Based (LLM)": "#ff9800",
    "Goal-Based (API)": "#e65100",
}

MODEL_RENAME: dict[str, tuple[str, str]] = {
    "rule-based": ("Rule-Based", "Rule-Based"),
    "lr": ("LR", "Supervised (Non-Encoder)"),
    "xgb": ("XGBoost", "Supervised (Non-Encoder)"),
    "mbert": ("mBERT", "Supervised (Encoder)"),
    "xlm-roberta": ("XLM-RoBERTa", "Supervised (Encoder)"),
    "qwen-qwen3-0.6b": ("Qwen3-0.6B", "Goal-Based (LLM)"),
    "qwen-qwen3-1.7b": ("Qwen3-1.7B", "Goal-Based (LLM)"),
    "qwen-qwen3-4b": ("Qwen3-4B", "Goal-Based (LLM)"),
    "qwen3.5-flash(no thinking)": ("Qwen3.5-Flash (no thinking)", "Goal-Based (API)"),
    "qwen3.5-flash(thinking)": ("Qwen3.5-Flash (thinking)", "Goal-Based (API)"),
    "qwen3.5-plus(no thinking)": ("Qwen3.5-Plus (no thinking)", "Goal-Based (API)"),
    "qwen3.5-plus(thinking)": ("Qwen3.5-Plus (thinking)", "Goal-Based (API)"),
}

# 学术风格
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
sns.set_context("paper", font_scale=1.1)
sns.set_palette("muted")


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def _model_info(model_name: str) -> tuple[str, str]:
    return MODEL_RENAME.get(model_name, (model_name, "Unknown"))


def _parse_cfg_suffix(cfg: str | float | None) -> str:
    if cfg is None or (isinstance(cfg, float) and pd.isna(cfg)):
        return ""
    try:
        c = json.loads(cfg)
    except (json.JSONDecodeError, TypeError):
        return ""
    if "encoder_type" in c:
        return "ST" if c["encoder_type"] == "sentence_transformer" else "tfidf"
    elif "few_shot" in c:
        return "few-shot" if c["few_shot"] else "zero-shot"
    elif "enable_candidate_search" in c:
        return "cand-search" if c["enable_candidate_search"] else "no-cand"
    return ""


def _load_tidy() -> pd.DataFrame:
    """加载最新 CSV, 附加 display_name / paradigm / cfg_suffix 列."""
    df = pd.read_csv(RESULTS_CSV)
    base_names: list[str] = []
    paradigms: list[str] = []
    cfg_suffixes: list[str] = []
    for _, row in df.iterrows():
        base_name, paradigm = _model_info(row["model_name"])
        suffix = _parse_cfg_suffix(row["cfg"])
        cfg_suffixes.append(suffix)
        display_name = f"{base_name} ({suffix})" if suffix else base_name
        base_names.append(display_name)
        paradigms.append(paradigm)
    df["display_name"] = base_names
    df["paradigm"] = paradigms
    df["cfg_suffix"] = cfg_suffixes
    return df


def _filter_best_configs(df: pd.DataFrame) -> pd.DataFrame:
    """每个 model_name 只保留 macro_f1 最高的配置 (按 priority 任务)."""
    perf = df[(df["metric_category"] == "performance") & (df["metric_name"] == "macro_f1")]
    best_dfs: list[pd.DataFrame] = []
    for model in df["model_name"].unique():
        model_perf = perf[(perf["model_name"] == model) & (perf["task_name"] == "priority")]
        if model_perf.empty:
            model_perf = perf[perf["model_name"] == model]
        if model_perf.empty:
            continue
        best_idx = model_perf["value"].idxmax()
        best_cfg = df.loc[best_idx, "cfg"]
        model_rows = df[df["model_name"] == model]
        best_dfs.append(model_rows[model_rows["cfg"] == best_cfg])
    if not best_dfs:
        return df.iloc[:0]
    return pd.concat(best_dfs, ignore_index=True)


def _sort_by_paradigm(df: pd.DataFrame) -> pd.DataFrame:
    paradigm_order_idx = {p: i for i, p in enumerate(PARADIGM_ORDER)}
    df = df.copy()
    df["_paradigm_order"] = df["paradigm"].map(paradigm_order_idx)
    df = df.sort_values(["_paradigm_order", "display_name"]).drop(columns=["_paradigm_order"])
    return df


def _map_display_to_paradigm(name: str) -> str:
    """从 display_name 反查 paradigm."""
    for model_name, (base_name, paradigm) in MODEL_RENAME.items():
        if name.startswith(base_name):
            return paradigm
    return "Unknown"


# ---------------------------------------------------------------------------
# Fig 1: 不公平雷达图
# ---------------------------------------------------------------------------

def plot_unfairness_radar(df: pd.DataFrame | None = None) -> plt.Figure:
    """三属性不公平雷达图, 按模型家族分面.

    6 个指标归一化为 "越靠外 = 越不公平".
    """
    if df is None:
        df = _filter_best_configs(_load_tidy())

    fairness_fields: list[tuple[str, str]] = [
        ("accuracy_gap", "Acc Gap"),
        ("macro_f1_gap", "F1 Gap"),
        ("1-accuracy_ratio", "1-Acc Ratio"),
        ("1-macro_f1_ratio", "1-F1 Ratio"),
        ("|avg_di-1|", "|DI-1|"),
        ("|avg_eod|", "|EOD|"),
    ]
    raw_fields = [f for f, _ in fairness_fields]

    sensitive_attrs = ["language", "tech_proficiency", "user_type"]

    fair = df[df["metric_category"] == "fairness"].copy()

    # 按 display_name + sensitive_attr 聚合, 取 task 均值
    agg_rows: list[dict] = []
    for (model, sens), group in fair.groupby(["display_name", "sensitive_attr"]):
        row: dict = {"display_name": model, "sensitive_attr": sens}
        for field in raw_fields:
            sub = group[group["metric_name"] == field]
            if not sub.empty:
                row[field] = sub["value"].mean()
        # 派生指标
        if "accuracy_ratio" in row:
            row["1-accuracy_ratio"] = 1.0 - row["accuracy_ratio"]
        if "macro_f1_ratio" in row:
            row["1-macro_f1_ratio"] = 1.0 - row["macro_f1_ratio"]
        if "avg_disparate_impact" in row:
            row["|avg_di-1|"] = abs(row["avg_disparate_impact"] - 1.0)
        if "avg_equal_opportunity_difference" in row:
            row["|avg_eod|"] = abs(row["avg_equal_opportunity_difference"])
        agg_rows.append(row)

    radar_df = pd.DataFrame(agg_rows)
    if radar_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No fairness data available", ha="center", va="center")
        ax.axis("off")
        return fig

    # 归一化每个指标到 [0, 1]
    for field in raw_fields:
        if field in radar_df.columns:
            vmax = radar_df[field].max()
            if vmax and vmax > 0:
                radar_df[field] = radar_df[field] / vmax

    # 收集实际存在的 paradigm
    paradigms_in_data: list[str] = []
    for p in PARADIGM_ORDER:
        if any(_map_display_to_paradigm(n) == p for n in radar_df["display_name"]):
            paradigms_in_data.append(p)

    if not paradigms_in_data:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No paradigm data available", ha="center", va="center")
        ax.axis("off")
        return fig

    n_paradigms = len(paradigms_in_data)
    n_attrs = len(sensitive_attrs)

    fig, axes = plt.subplots(
        n_paradigms, n_attrs,
        figsize=(4 * n_attrs, 4 * n_paradigms),
        subplot_kw={"projection": "polar"},
    )
    if n_paradigms == 1 and n_attrs == 1:
        axes = np.array([[axes]])
    elif n_paradigms == 1:
        axes = axes.reshape(1, -1)
    elif n_attrs == 1:
        axes = axes.reshape(-1, 1)

    angles = np.linspace(0, 2 * np.pi, len(raw_fields), endpoint=False).tolist()
    angles += angles[:1]

    labels = [label for _, label in fairness_fields]

    for i, paradigm in enumerate(paradigms_in_data):
        paradigm_names = [
            n for n in radar_df["display_name"]
            if _map_display_to_paradigm(n) == paradigm
        ]
        paradigm_data = radar_df[radar_df["display_name"].isin(paradigm_names)]

        for j, attr in enumerate(sensitive_attrs):
            ax = axes[i, j]
            attr_data = paradigm_data[paradigm_data["sensitive_attr"] == attr]

            for _, row in attr_data.iterrows():
                values = [row.get(f, 0.0) for f in raw_fields]
                values += values[:1]
                ax.fill(angles, values, alpha=0.1)
                ax.plot(angles, values, "o-", linewidth=1.5, label=row["display_name"])

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, fontsize=7)
            ax.set_ylim(0, 1.1)
            ax.set_title(f"{paradigm} — {attr}", fontsize=10, pad=20)
            if i == 0 and j == 0:
                ax.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.3, 1.0))

    fig.suptitle("Unfairness Radar: Sensitive Attribute × Model Family", fontweight="bold", fontsize=14)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 2: Performance-Fairness Pareto Frontier
# ---------------------------------------------------------------------------

def plot_pareto_frontier(df: pd.DataFrame | None = None) -> plt.Figure:
    """Pareto Frontier: x=macro_f1, y=1-accuracy_ratio, 按 task × sensitive_attr 分面."""
    if df is None:
        df = _filter_best_configs(_load_tidy())

    perf = df[(df["metric_category"] == "performance") & (df["metric_name"] == "macro_f1")].copy()
    fair = df[(df["metric_category"] == "fairness") & (df["metric_name"] == "accuracy_ratio")].copy()

    if perf.empty or fair.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Insufficient data for Pareto analysis", ha="center", va="center")
        ax.axis("off")
        return fig

    # perf 行的 sensitive_attr 为空, 会与 fair 冲突; drop 后通过 display_name 去 fair 中找
    if "sensitive_attr" in perf.columns:
        perf = perf.drop(columns=["sensitive_attr"])

    merged = perf.merge(
        fair[["task_name", "model_name", "display_name", "paradigm", "sensitive_attr", "value"]],
        on=["task_name", "model_name", "display_name", "paradigm"],
        suffixes=("_perf", "_fair"),
    )
    # y 轴直接用 accuracy_ratio (越高越公平)
    merged["accuracy_ratio"] = merged["value_fair"]

    tasks = sorted(merged["task_name"].unique())
    sens_attrs = sorted(merged["sensitive_attr"].unique())

    fig, axes = plt.subplots(
        len(tasks), len(sens_attrs),
        figsize=(5 * len(sens_attrs), 5 * len(tasks)),
    )
    if len(tasks) == 1 and len(sens_attrs) == 1:
        axes = np.array([[axes]])
    elif len(tasks) == 1:
        axes = axes.reshape(1, -1)
    elif len(sens_attrs) == 1:
        axes = axes.reshape(-1, 1)

    for i, task in enumerate(tasks):
        for j, attr in enumerate(sens_attrs):
            ax = axes[i, j]
            sub = merged[(merged["task_name"] == task) & (merged["sensitive_attr"] == attr)]

            if sub.empty:
                ax.set_title(f"{task} — {attr}\n(no data)")
                ax.axis("off")
                continue

            for paradigm in PARADIGM_ORDER:
                paradigm_data = sub[sub["paradigm"] == paradigm]
                if paradigm_data.empty:
                    continue
                color = PARADIGM_COLOR.get(paradigm, "#999")
                ax.scatter(
                    paradigm_data["value_perf"], paradigm_data["accuracy_ratio"],
                    color=color, s=80, zorder=3,
                    edgecolors="black", linewidth=0.5,
                )
                for _, pt in paradigm_data.iterrows():
                    ax.annotate(
                        pt["display_name"].split("(")[0].strip(),
                        (pt["value_perf"], pt["accuracy_ratio"]),
                        fontsize=6, ha="center", va="bottom",
                        textcoords="offset points", xytext=(0, 5),
                    )

            # Pareto frontier: maximize perf AND maximize accuracy_ratio
            points = sub[["value_perf", "accuracy_ratio"]].dropna().values
            if len(points) > 1:
                sorted_idx = points[:, 0].argsort()[::-1]
                sorted_pts = points[sorted_idx]
                frontier_x, frontier_y = [float(sorted_pts[0, 0])], [float(sorted_pts[0, 1])]
                for px, py in sorted_pts[1:]:
                    py_f = float(py)
                    if py_f > frontier_y[-1]:  # accuracy_ratio 越大越好
                        frontier_x.append(float(px))
                        frontier_y.append(py_f)
                ax.plot(frontier_x, frontier_y, "k--", linewidth=1, alpha=0.5)

            ax.set_xlabel("Macro-F1 ↑")
            ax.set_ylabel("Accuracy Ratio ↑")
            ax.set_title(f"{task} — {attr}", fontweight="bold")

    fig.suptitle("Performance-Fairness Pareto Frontier", fontweight="bold", fontsize=14)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 3: Pairwise DI Heatmap
# ---------------------------------------------------------------------------

def _filter_pairwise_best(df: pd.DataFrame) -> pd.DataFrame:
    """从全量 pairwise 数据中筛出最优配置对应的行."""
    pairwise = df[df["metric_category"] == "fairness_pairwise"].copy()
    if pairwise.empty:
        return pairwise
    best_df = _filter_best_configs(df)
    best_model_names = set(best_df["model_name"].unique())
    pairwise_best_rows: list[pd.DataFrame] = []
    for model_name in best_model_names:
        model_best = best_df[best_df["model_name"] == model_name]
        if model_best.empty:
            continue
        best_cfg = model_best["cfg"].iloc[0]
        match = pairwise[(pairwise["model_name"] == model_name) & (pairwise["cfg"] == best_cfg)]
        pairwise_best_rows.append(match)
    models_covered = set(best_model_names)
    for model_name in pairwise["model_name"].unique():
        if model_name not in models_covered:
            pairwise_best_rows.append(pairwise[pairwise["model_name"] == model_name])
    return pd.concat(pairwise_best_rows, ignore_index=True) if pairwise_best_rows else pairwise


def plot_language_di_matrices(df: pd.DataFrame | None = None) -> list[tuple[str, plt.Figure]]:
    """5×5 语言对 Disparate Impact 矩阵, 每个范式独立一张图.

    Returns:
        List of (paradigm_label, Figure) tuples for separate saving.
    """
    if df is None:
        df = _load_tidy()

    pairwise = _filter_pairwise_best(df)
    language_pairs = pairwise[
        (pairwise["metric_name"] == "disparate_impact") &
        (pairwise["sensitive_attr"] == "language")
    ].copy()

    if language_pairs.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No language pairwise DI data", ha="center", va="center")
        ax.axis("off")
        return [("no_data", fig)]

    extracted = language_pairs["pair"].str.extract(r"(\w+)_vs_(\w+)")
    language_pairs["lang_a"] = extracted[0]
    language_pairs["lang_b"] = extracted[1]
    all_langs = sorted(set(language_pairs["lang_a"].unique()) | set(language_pairs["lang_b"].unique()))
    n_langs = len(all_langs)
    lang_idx = {l: i for i, l in enumerate(all_langs)}

    paradigms_with_data = sorted(
        set(language_pairs["paradigm"].unique()),
        key=lambda p: PARADIGM_ORDER.index(p) if p in PARADIGM_ORDER else 99,
    )

    cmap = sns.diverging_palette(10, 133, s=90, l=50, as_cmap=True)
    figures: list[tuple[str, plt.Figure]] = []

    for paradigm in paradigms_with_data:
        model_data = language_pairs[language_pairs["paradigm"] == paradigm]
        best_display = model_data["display_name"].iloc[0]
        model_data = model_data[model_data["display_name"] == best_display]

        matrix = np.full((n_langs, n_langs), np.nan)
        for _, row in model_data.iterrows():
            a, b = row["lang_a"], row["lang_b"]
            if a in lang_idx and b in lang_idx:
                i, j = lang_idx[a], lang_idx[b]
                matrix[i, j] = row["value"]
                matrix[j, i] = row["value"]

        np.fill_diagonal(matrix, 1.0)

        fig, ax = plt.subplots(figsize=(5.5, 5))
        im = ax.imshow(matrix, cmap=cmap, vmin=0.5, vmax=1.5, aspect="equal")
        ax.set_xticks(range(n_langs))
        ax.set_yticks(range(n_langs))
        ax.set_xticklabels(all_langs, rotation=45, fontsize=10)
        ax.set_yticklabels(all_langs, fontsize=10)
        ax.set_title(f"{paradigm}\n{best_display}", fontsize=11, fontweight="bold")

        for ii in range(n_langs):
            for jj in range(n_langs):
                if not np.isnan(matrix[ii, jj]):
                    ax.text(jj, ii, f"{matrix[ii, jj]:.2f}", ha="center", va="center", fontsize=9)

        plt.colorbar(im, ax=ax, shrink=0.8, label="DI")
        plt.tight_layout()

        # 生成简短的字母后缀 (a, b, c, ...)
        suffix = chr(ord("a") + len(figures))
        figures.append((f"fairness_02{suffix}_language_di_{paradigm.lower().replace(' ', '_').replace('(','').replace(')','')}.png", fig))

    return figures


def plot_user_type_eod(df: pd.DataFrame | None = None) -> plt.Figure:
    """User Type 成对 Equal Opportunity Difference 热力图."""
    if df is None:
        df = _load_tidy()

    pairwise = _filter_pairwise_best(df)
    ut_data = pairwise[
        (pairwise["metric_name"] == "equal_opportunity_difference") &
        (pairwise["sensitive_attr"] == "user_type")
    ].copy()

    if ut_data.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No user_type pairwise EOD data", ha="center", va="center")
        ax.axis("off")
        return fig

    pivot = ut_data.pivot_table(
        index="display_name", columns="pair", values="value", aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
        ax=ax, cbar_kws={"label": "EOD"}, linewidths=0.5,
    )
    ax.set_title("User Type: Equal Opportunity Difference by Pair", fontweight="bold", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 4: Thinking vs No-Thinking
# ---------------------------------------------------------------------------

def plot_thinking_vs_nothinking(df: pd.DataFrame | None = None) -> plt.Figure:
    """Thinking vs No-Thinking 公平性对比 + Qwen3 尺寸 few-shot 消融."""
    if df is None:
        df = _load_tidy()

    api_models = [
        "qwen3.5-plus(no thinking)", "qwen3.5-plus(thinking)",
        "qwen3.5-flash(no thinking)", "qwen3.5-flash(thinking)",
    ]

    gap_data = df[
        (df["model_name"].isin(api_models)) &
        (df["metric_category"] == "fairness") &
        (df["metric_name"] == "accuracy_gap") &
        (df["sensitive_attr"].isin(["language", "user_type"]))
    ].copy()

    gap_data["thinking"] = ~gap_data["model_name"].str.contains("no thinking")
    gap_data["model_family"] = gap_data["model_name"].str.extract(r"(plus|flash)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: grouped bar — thinking vs no thinking
    ax = axes[0]
    if not gap_data.empty:
        bar_df = gap_data.groupby(
            ["model_family", "sensitive_attr", "thinking"]
        )["value"].mean().reset_index()
        bar_df["label"] = bar_df["model_family"].str.capitalize() + " — " + bar_df["sensitive_attr"]

        labels_unique = sorted(bar_df["label"].unique())
        x = np.arange(len(labels_unique))
        width = 0.35

        no_think_vals = [
            bar_df[(bar_df["label"] == l) & (~bar_df["thinking"])]["value"].sum()
            for l in labels_unique
        ]
        think_vals = [
            bar_df[(bar_df["label"] == l) & (bar_df["thinking"])]["value"].sum()
            for l in labels_unique
        ]

        ax.bar(x - width / 2, no_think_vals, width, label="No Thinking", color="#4c72b0")
        ax.bar(x + width / 2, think_vals, width, label="Thinking", color="#c44e52")
        ax.set_xticks(x)
        ax.set_xticklabels(labels_unique, rotation=30, fontsize=9)
        ax.set_ylabel("Accuracy Gap")
        ax.legend()
        ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_title("Thinking vs No-Thinking: Fairness Gap", fontweight="bold")

    # Panel B: Qwen3 sizes few-shot vs zero-shot
    ax2 = axes[1]
    qwen_local = ["qwen-qwen3-0.6b", "qwen-qwen3-1.7b", "qwen-qwen3-4b"]
    qwen_data = df[
        (df["model_name"].isin(qwen_local)) &
        (df["metric_category"] == "fairness") &
        (df["metric_name"] == "accuracy_gap") &
        (df["sensitive_attr"] == "language")
    ].copy()

    if not qwen_data.empty:
        qwen_data["few_shot"] = qwen_data["cfg"].apply(
            lambda c: "few-shot" if (isinstance(c, str) and "true" in c.lower()) else "zero-shot"
        )
        qwen_data["size"] = qwen_data["model_name"].str.extract(r"(?i)(0\.6b|1\.7b|4b)")

        size_order = ["0.6B", "1.7B", "4B"]
        qwen_bar = qwen_data.groupby(["size", "few_shot"])["value"].mean().unstack().reindex(size_order)

        x2 = np.arange(len(qwen_bar))
        width2 = 0.35
        ax2.bar(x2 - width2 / 2, qwen_bar.get("zero-shot", [0] * len(x2)), width2,
                label="Zero-Shot", color="#4c72b0")
        ax2.bar(x2 + width2 / 2, qwen_bar.get("few-shot", [0] * len(x2)), width2,
                label="Few-Shot", color="#ff9800")
        ax2.set_xticks(x2)
        ax2.set_xticklabels(size_order)
        ax2.set_xlabel("Model Size")
        ax2.set_ylabel("Language Accuracy Gap")
        ax2.legend()
    ax2.set_title("Qwen3: Few-Shot vs Zero-Shot (Language)", fontweight="bold")

    fig.suptitle("LLM Fairness Ablation: Thinking Mode & Few-Shot", fontweight="bold", fontsize=14)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 5: Cross-Task Fairness Scatter
# ---------------------------------------------------------------------------

def plot_cross_task_fairness(df: pd.DataFrame | None = None) -> plt.Figure:
    """跨任务公平性散点图: priority gap vs queue gap."""
    if df is None:
        df = _filter_best_configs(_load_tidy())

    gap = df[
        (df["metric_category"] == "fairness") &
        (df["metric_name"].isin(["macro_f1_gap", "accuracy_gap"]))
    ].copy()

    if gap.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No fairness gap data", ha="center", va="center")
        ax.axis("off")
        return fig

    sens_attrs = sorted(gap["sensitive_attr"].unique())
    metrics = ["macro_f1_gap", "accuracy_gap"]

    fig, axes = plt.subplots(
        len(metrics), len(sens_attrs),
        figsize=(5 * len(sens_attrs), 5 * len(metrics)),
    )
    if len(metrics) == 1 and len(sens_attrs) == 1:
        axes = np.array([[axes]])
    elif len(metrics) == 1:
        axes = axes.reshape(1, -1)
    elif len(sens_attrs) == 1:
        axes = axes.reshape(-1, 1)

    for i, metric in enumerate(metrics):
        metric_data = gap[gap["metric_name"] == metric]
        for j, attr in enumerate(sens_attrs):
            ax = axes[i, j]
            attr_data = metric_data[metric_data["sensitive_attr"] == attr]

            pivot = attr_data.pivot_table(
                index=["display_name", "paradigm"],
                columns="task_name", values="value", aggfunc="mean",
            )
            if "priority" not in pivot.columns or "queue" not in pivot.columns:
                ax.set_title(f"{metric} — {attr}\n(insufficient data)")
                ax.axis("off")
                continue

            for paradigm in PARADIGM_ORDER:
                paradigm_data = pivot[pivot.index.get_level_values("paradigm") == paradigm]
                if paradigm_data.empty:
                    continue
                color = PARADIGM_COLOR.get(paradigm, "#999")
                ax.scatter(
                    paradigm_data["priority"], paradigm_data["queue"],
                    color=color, label=paradigm, s=80, zorder=3,
                    edgecolors="black", linewidth=0.5,
                )
                for (name, _), row in paradigm_data.iterrows():
                    short = name.split("(")[0].strip()
                    ax.annotate(
                        short, (row["priority"], row["queue"]),
                        fontsize=6, textcoords="offset points", xytext=(0, 5), ha="center",
                    )

            # y=x reference line
            all_vals = list(pivot["priority"]) + list(pivot["queue"])
            lo, hi = min(all_vals), max(all_vals)
            margin = (hi - lo) * 0.1 if hi > lo else 0.1
            ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                    "k--", alpha=0.3, linewidth=0.8)
            ax.set_xlabel(f"Priority {metric}")
            ax.set_ylabel(f"Queue {metric}")
            ax.set_title(f"{metric} — {attr}", fontweight="bold")
            if i == 0 and j == 0:
                ax.legend(fontsize=7)

    fig.suptitle("Cross-Task Fairness: Priority vs Queue", fontweight="bold", fontsize=14)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 6: Accuracy Gap with Error Bars
# ---------------------------------------------------------------------------

def plot_gap_with_errorbars(df: pd.DataFrame | None = None) -> plt.Figure:
    """带误差棒的 accuracy_gap, 按 sensitive_attr 分面."""
    if df is None:
        df = _filter_best_configs(_load_tidy())

    gap = df[
        (df["metric_category"] == "fairness") &
        (df["metric_name"] == "accuracy_gap")
    ].copy()

    if gap.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No accuracy_gap data", ha="center", va="center")
        ax.axis("off")
        return fig

    gap = _sort_by_paradigm(gap)
    sens_attrs = sorted(gap["sensitive_attr"].unique())

    fig, axes = plt.subplots(1, len(sens_attrs), figsize=(6 * len(sens_attrs), 8))
    if len(sens_attrs) == 1:
        axes = np.array([axes])

    for j, attr in enumerate(sens_attrs):
        ax = axes[j]
        attr_data = gap[gap["sensitive_attr"] == attr].copy()

        agg = attr_data.groupby(["display_name", "paradigm"]).agg(
            mean_val=("value", "mean"),
            mean_std=("std", lambda x: x.mean() if x.notna().any() else 0.0),
        ).reset_index()
        agg = agg.sort_values("mean_val")

        if agg.empty:
            ax.set_title(f"Accuracy Gap — {attr}\n(no data)")
            ax.axis("off")
            continue

        colors = [PARADIGM_COLOR.get(p, "#999") for p in agg["paradigm"]]
        ax.barh(agg["display_name"], agg["mean_val"], xerr=agg["mean_std"],
                color=colors, height=0.7, capsize=2)
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.set_xlabel("Accuracy Gap")
        ax.set_title(f"Accuracy Gap — {attr}", fontweight="bold")

        legend_elements = [
            Patch(facecolor=PARADIGM_COLOR[p], label=p)
            for p in PARADIGM_ORDER if p in agg["paradigm"].values
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

    fig.suptitle("Accuracy Gap with Error Bars (±1σ)", fontweight="bold", fontsize=14)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 7: Spearman Correlation Heatmap
# ---------------------------------------------------------------------------

def plot_spearman_correlation(df: pd.DataFrame | None = None) -> plt.Figure:
    """Performance-Fairness Spearman 秩相关矩阵."""
    if df is None:
        df = _filter_best_configs(_load_tidy())

    perf_metrics = ["accuracy", "macro_f1"]
    fair_metrics = [
        "accuracy_gap", "macro_f1_gap", "accuracy_ratio", "macro_f1_ratio",
        "avg_disparate_impact", "avg_equal_opportunity_difference",
    ]

    sens_attrs = sorted(df["sensitive_attr"].dropna().unique())
    if not sens_attrs:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No sensitive attribute data", ha="center", va="center")
        ax.axis("off")
        return fig

    fig, axes = plt.subplots(1, len(sens_attrs), figsize=(6 * len(sens_attrs), 5))
    if len(sens_attrs) == 1:
        axes = np.array([axes])

    for j, attr in enumerate(sens_attrs):
        ax = axes[j]

        corr_matrix = np.zeros((len(perf_metrics), len(fair_metrics)))
        p_matrix = np.zeros((len(perf_metrics), len(fair_metrics)))

        for pi, p_metric in enumerate(perf_metrics):
            perf_data = df[
                (df["metric_category"] == "performance") & (df["metric_name"] == p_metric)
            ]
            for fi, f_metric in enumerate(fair_metrics):
                fair_data = df[
                    (df["metric_category"] == "fairness") &
                    (df["metric_name"] == f_metric) &
                    (df["sensitive_attr"] == attr)
                ]
                merged = perf_data.merge(
                    fair_data[["task_name", "model_name", "value"]],
                    on=["task_name", "model_name"], suffixes=("_perf", "_fair"),
                )
                if len(merged) >= 3:
                    r, p = spearmanr(merged["value_perf"], merged["value_fair"])
                    corr_matrix[pi, fi] = r
                    p_matrix[pi, fi] = p
                else:
                    corr_matrix[pi, fi] = np.nan
                    p_matrix[pi, fi] = np.nan

        annot = [
            [f"{corr_matrix[ii, jj]:.2f}" + ("*" if p_matrix[ii, jj] < 0.05 else "")
             for jj in range(len(fair_metrics))]
            for ii in range(len(perf_metrics))
        ]

        sns.heatmap(
            corr_matrix, annot=annot, fmt="", cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, xticklabels=fair_metrics, yticklabels=perf_metrics,
            ax=ax, linewidths=0.5, cbar_kws={"label": "Spearman ρ"},
        )
        ax.set_title(f"Performance-Fairness Correlation — {attr}", fontweight="bold")
        ax.set_xlabel("Fairness Metric")
        ax.set_ylabel("Performance Metric")

    fig.suptitle("Performance-Fairness Spearman Correlation (* p<0.05)", fontweight="bold", fontsize=14)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 8: LLM Scale vs Fairness
# ---------------------------------------------------------------------------

def plot_llm_scale_fairness(df: pd.DataFrame | None = None) -> plt.Figure:
    """Qwen3 参数量 vs 公平性指标, 区分 zero/few-shot."""
    if df is None:
        df = _load_tidy()

    qwen_local = ["qwen-qwen3-0.6b", "qwen-qwen3-1.7b", "qwen-qwen3-4b"]
    param_map = {"qwen-qwen3-0.6b": 600, "qwen-qwen3-1.7b": 1700, "qwen-qwen3-4b": 4000}

    qwen = df[df["model_name"].isin(qwen_local)].copy()
    if qwen.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No Qwen3 local model data", ha="center", va="center")
        ax.axis("off")
        return fig

    qwen["few_shot"] = qwen["cfg"].apply(
        lambda c: "few-shot" if (isinstance(c, str) and "true" in c.lower()) else "zero-shot"
    )
    qwen["param_count"] = qwen["model_name"].map(param_map)

    fair_metrics = ["accuracy_gap", "avg_disparate_impact"]
    metric_labels: dict[str, str] = {
        "accuracy_gap": "Accuracy Gap", "avg_disparate_impact": "Avg DI",
    }
    sens_attrs = sorted(qwen["sensitive_attr"].dropna().unique())

    fig, axes = plt.subplots(
        len(fair_metrics), len(sens_attrs),
        figsize=(5 * len(sens_attrs), 5 * len(fair_metrics)),
    )
    if len(fair_metrics) == 1 and len(sens_attrs) == 1:
        axes = np.array([[axes]])
    elif len(fair_metrics) == 1:
        axes = axes.reshape(1, -1)
    elif len(sens_attrs) == 1:
        axes = axes.reshape(-1, 1)

    colors = {"zero-shot": "#4c72b0", "few-shot": "#ff9800"}
    markers = {"zero-shot": "o", "few-shot": "s"}
    size_labels = {600: "0.6B", 1700: "1.7B", 4000: "4B"}

    for i, metric in enumerate(fair_metrics):
        for j, attr in enumerate(sens_attrs):
            ax = axes[i, j]
            metric_data = qwen[
                (qwen["metric_category"] == "fairness") &
                (qwen["metric_name"] == metric) &
                (qwen["sensitive_attr"] == attr)
            ].copy()

            if metric_data.empty:
                ax.set_title(f"{metric_labels[metric]} — {attr}\n(no data)")
                ax.axis("off")
                continue

            agg = metric_data.groupby(["param_count", "few_shot"]).agg(
                mean_val=("value", "mean"),
                std_val=("std", lambda x: x.mean() if x.notna().any() else 0.0),
            ).reset_index()

            for shot_type in ["zero-shot", "few-shot"]:
                sub = agg[agg["few_shot"] == shot_type].sort_values("param_count")
                if sub.empty:
                    continue
                ax.errorbar(
                    sub["param_count"], sub["mean_val"], yerr=sub["std_val"],
                    color=colors[shot_type], marker=markers[shot_type], markersize=10,
                    label=shot_type, linewidth=2, capsize=4,
                )
                for _, pt in sub.iterrows():
                    label = size_labels.get(int(pt["param_count"]), "")
                    ax.annotate(
                        label, (pt["param_count"], pt["mean_val"]),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=8,
                    )

            ax.set_xscale("log")
            ax.set_xlabel("Parameter Count")
            ax.set_ylabel(metric_labels[metric])
            ax.set_title(f"{metric_labels[metric]} — {attr}", fontweight="bold")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle("LLM Scale vs Fairness: Qwen3 0.6B → 4B", fontweight="bold", fontsize=14)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Robustness 图表 (xlm-roberta, whitebox textfooler)
# ---------------------------------------------------------------------------

ROBUSTNESS_DIR = Path("outputs/robustness/xlm-roberta/multilingual-customer-support")


def _load_robustness() -> pd.DataFrame:
    """加载 xlm-roberta 的 robustness CSV (priority + queue)."""
    dfs = []
    for task in ["priority", "queue"]:
        csv_path = ROBUSTNESS_DIR / f"{task}_metrics.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["task"] = task
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def plot_robustness_clean_vs_perturbed(df: pd.DataFrame | None = None) -> plt.Figure:
    """Clean vs Perturbed Accuracy, 按 task × language 分组柱状图."""
    if df is None:
        df = _load_robustness()

    lang_df = df[df["granularity"] == "language"].copy()
    if lang_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No robustness language data", ha="center", va="center")
        ax.axis("off")
        return fig

    languages = sorted(lang_df["group"].unique())
    tasks = sorted(lang_df["task"].unique())

    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 5))
    if len(tasks) == 1:
        axes = np.array([axes])

    x = np.arange(len(languages))
    width = 0.35

    for i, task in enumerate(tasks):
        ax = axes[i]
        task_data = lang_df[lang_df["task"] == task].set_index("group").reindex(languages)

        ax.bar(x - width / 2, task_data["clean_accuracy"], width,
               label="Clean", color="#4c72b0")
        ax.bar(x + width / 2, task_data["perturbed_accuracy"], width,
               label="Perturbed", color="#c44e52")
        ax.set_xticks(x)
        ax.set_xticklabels(languages, rotation=30, fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"XLM-RoBERTa — {task}", fontweight="bold")
        if i == 0:
            ax.legend(fontsize=9)

    fig.suptitle("Robustness: Clean vs Perturbed Accuracy by Language", fontweight="bold", fontsize=14)
    plt.tight_layout()
    return fig


def plot_robustness_attack_success(df: pd.DataFrame | None = None) -> plt.Figure:
    """Attack Success Rate by Language, 按 task 分组水平条形图."""
    if df is None:
        df = _load_robustness()

    lang_df = df[df["granularity"] == "language"].copy()
    if lang_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No robustness language data", ha="center", va="center")
        ax.axis("off")
        return fig

    tasks = sorted(lang_df["task"].unique())
    languages = sorted(lang_df["group"].unique())

    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 5))
    if len(tasks) == 1:
        axes = np.array([axes])

    colors = {"priority": "#4c72b0", "queue": "#c44e52"}

    for i, task in enumerate(tasks):
        ax = axes[i]
        task_data = lang_df[lang_df["task"] == task].set_index("group").reindex(languages)
        task_data = task_data.sort_values("attack_success_rate")

        ax.barh(task_data.index, task_data["attack_success_rate"],
                color=colors.get(task, "#999"), height=0.6)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Attack Success Rate")
        ax.set_title(f"XLM-RoBERTa — {task}", fontweight="bold")
        for bar in ax.patches:
            w = bar.get_width()
            ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{w:.3f}", ha="left", va="center", fontsize=9)

    fig.suptitle("Robustness: Attack Success Rate by Language", fontweight="bold", fontsize=14)
    plt.tight_layout()
    return fig


def plot_robustness_accuracy_drop(df: pd.DataFrame | None = None) -> plt.Figure:
    """Accuracy Drop by Language, 按 task 分组水平条形图."""
    if df is None:
        df = _load_robustness()

    lang_df = df[df["granularity"] == "language"].copy()
    if lang_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No robustness language data", ha="center", va="center")
        ax.axis("off")
        return fig

    tasks = sorted(lang_df["task"].unique())
    languages = sorted(lang_df["group"].unique())

    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 5))
    if len(tasks) == 1:
        axes = np.array([axes])

    colors = {"priority": "#4c72b0", "queue": "#c44e52"}

    for i, task in enumerate(tasks):
        ax = axes[i]
        task_data = lang_df[lang_df["task"] == task].set_index("group").reindex(languages)
        task_data = task_data.sort_values("accuracy_drop")

        ax.barh(task_data.index, task_data["accuracy_drop"],
                color=colors.get(task, "#999"), height=0.6)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Accuracy Drop")
        ax.set_title(f"XLM-RoBERTa — {task}", fontweight="bold")
        for bar in ax.patches:
            w = bar.get_width()
            ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{w:.3f}", ha="left", va="center", fontsize=9)

    fig.suptitle("Robustness: Accuracy Drop by Language", fontweight="bold", fontsize=14)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {RESULTS_CSV}")
    tidy_df = _load_tidy()
    print(f"Loaded {len(tidy_df)} rows, {tidy_df['model_name'].nunique()} models")

    best_df = _filter_best_configs(tidy_df)
    print(f"Best configs: {best_df['model_name'].nunique()} models")

    figures: list[tuple[str, plt.Figure]] = [
        ("fairness_01_pareto.png", plot_pareto_frontier(best_df)),
    ]
    # 02a, 02b, ... 每个范式一张语言 DI 矩阵
    figures.extend(plot_language_di_matrices(tidy_df))
    figures.extend([
        ("fairness_03_user_type_eod.png", plot_user_type_eod(tidy_df)),
        ("fairness_04_thinking_vs_nothinking.png", plot_thinking_vs_nothinking(tidy_df)),
        ("fairness_05_cross_task.png", plot_cross_task_fairness(best_df)),
        ("fairness_06_gap_errorbars.png", plot_gap_with_errorbars(best_df)),
        ("fairness_07_spearman.png", plot_spearman_correlation(best_df)),
        ("fairness_08_llm_scale.png", plot_llm_scale_fairness(tidy_df)),
    ])

    # Robustness (xlm-roberta, whitebox textfooler)
    rob_df = _load_robustness()
    if not rob_df.empty:
        figures.extend([
            ("robustness_01_clean_vs_perturbed.png", plot_robustness_clean_vs_perturbed(rob_df)),
            ("robustness_02_attack_success.png", plot_robustness_attack_success(rob_df)),
            ("robustness_03_accuracy_drop.png", plot_robustness_accuracy_drop(rob_df)),
        ])

    for filename, fig in figures:
        path = OUTPUT_DIR / filename
        fig.savefig(path, format="png")
        plt.close(fig)
        print(f"Saved: {path}")

    print("\nAll fairness figures generated successfully.")


if __name__ == "__main__":
    main()
