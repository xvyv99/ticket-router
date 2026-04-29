#!/usr/bin/env python3
"""
生成 Ticket Router 评估结果的可视化图表。
学术论文风格，输出 PNG 到 outputs/figures/。
"""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("outputs/figures")
RESULTS_XLSX = sorted(glob.glob("results/eval_multilingual-customer-support_*.xlsx"))[-1]

# ---------------------------------------------------------------------------
# 范式与模型映射
# ---------------------------------------------------------------------------
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

# (display_name, paradigm)
MODEL_RENAME = {
    # Rule-Based
    "rule-based": ("Rule-Based", "Rule-Based"),
    # Supervised Non-Encoder
    "lr": ("LR", "Supervised (Non-Encoder)"),
    "xgb": ("XGBoost", "Supervised (Non-Encoder)"),
    # Supervised Encoder
    "mbert": ("mBERT", "Supervised (Encoder)"),
    "xlm-roberta": ("XLM-RoBERTa", "Supervised (Encoder)"),
    # Goal-Based LLM (local)
    "qwen-qwen3-0.6b": ("Qwen3-0.6B", "Goal-Based (LLM)"),
    "qwen-qwen3-1.7b": ("Qwen3-1.7B", "Goal-Based (LLM)"),
    "qwen-qwen3-4b": ("Qwen3-4B", "Goal-Based (LLM)"),
    # Goal-Based API
    "qwen3.5-flash(no thinking)": ("Qwen3.5-Flash (no thinking)", "Goal-Based (API)"),
    "qwen3.5-flash(thinking)": ("Qwen3.5-Flash (thinking)", "Goal-Based (API)"),
    "qwen3.5-plus(no thinking)": ("Qwen3.5-Plus (no thinking)", "Goal-Based (API)"),
    "qwen3.5-plus(thinking)": ("Qwen3.5-Plus (thinking)", "Goal-Based (API)"),
}


import json
from typing import Optional


def _parse_cfg_suffix(cfg: str | float | None) -> str:
    """从 cfg JSON 字符串解析人类可读的简短视频。

    Examples:
        '{"encoder_type": "tfidf"}' -> 'tfidf'
        '{"encoder_type": "sentence_transformer"}' -> 'ST'
        '{"few_shot": true}' -> 'few-shot'
        '{"few_shot": false}' -> 'zero-shot'
        '{"enable_candidate_search": true}' -> 'cand-search'
        '{"enable_candidate_search": false}' -> 'no-cand'
        NaN -> ''
    """
    if cfg is None or (isinstance(cfg, float) and pd.isna(cfg)):
        return ""
    try:
        c = json.loads(cfg)
    except (json.JSONDecodeError, TypeError):
        return ""
    if "encoder_type" in c:
        if c["encoder_type"] == "tfidf":
            return "tfidf"
        elif c["encoder_type"] == "sentence_transformer":
            return "ST"
    elif "few_shot" in c:
        return "few-shot" if c["few_shot"] else "zero-shot"
    elif "enable_candidate_search" in c:
        return "cand-search" if c["enable_candidate_search"] else "no-cand"
    return ""


def _model_info(model_name: str) -> tuple[str, str]:
    """返回 (base_display_name, paradigm)。"""
    return MODEL_RENAME.get(model_name, (model_name, "Unknown"))


# ---------------------------------------------------------------------------
# 学术风格配置
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
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
    }
)
sns.set_context("paper", font_scale=1.1)
sns.set_palette("muted")


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def _load_tidy() -> pd.DataFrame:
    """加载主评估结果的 tidy sheet，并附加 display_name 和 paradigm。"""
    df = pd.read_excel(RESULTS_XLSX, sheet_name="tidy")
    base_names, paradigms = [], []
    for _, row in df.iterrows():
        base_name, paradigm = _model_info(row["model_name"])
        suffix = _parse_cfg_suffix(row["cfg"])
        display_name = f"{base_name} ({suffix})" if suffix else base_name
        base_names.append(display_name)
        paradigms.append(paradigm)
    df["display_name"] = base_names
    df["paradigm"] = paradigms
    return df


def _load_robustness() -> pd.DataFrame:
    """加载鲁棒性评估指标，从文件路径提取模型名和范式。"""
    rob_files = sorted(glob.glob("outputs/robustness/*/*/*_metrics.xlsx"))
    if not rob_files:
        return pd.DataFrame()
    dfs = []
    for f in rob_files:
        df = pd.read_excel(f)
        model_name = Path(f).parts[2]  # outputs/robustness/{model}/...
        display_name, paradigm = _model_info(model_name)
        df["model_name"] = model_name
        df["display_name"] = display_name
        df["paradigm"] = paradigm
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def _sort_by_paradigm(df: pd.DataFrame) -> pd.DataFrame:
    """按范式顺序 + 模型名排序 DataFrame。"""
    paradigm_order_idx = {p: i for i, p in enumerate(PARADIGM_ORDER)}
    df = df.copy()
    df["_paradigm_order"] = df["paradigm"].map(paradigm_order_idx)
    df = df.sort_values(["_paradigm_order", "display_name"]).drop(columns=["_paradigm_order"])
    return df


# ---------------------------------------------------------------------------
# Fig 01a: Accuracy — Priority 任务
# ---------------------------------------------------------------------------
def plot_accuracy_priority(df: pd.DataFrame | None = None) -> plt.Figure:
    """Priority 任务，按 accuracy 升序排列，按范式着色。"""
    if df is None:
        df = _load_tidy()

    perf = df[
        (df["metric_category"] == "performance")
        & (df["metric_name"] == "accuracy")
        & (df["task_name"] == "priority")
    ].copy()
    perf = _sort_by_paradigm(perf)

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
# Fig 01b: Accuracy — Queue 任务
# ---------------------------------------------------------------------------
def plot_accuracy_queue(df: pd.DataFrame | None = None) -> plt.Figure:
    """Queue 任务，按 accuracy 升序排列，按范式着色。"""
    if df is None:
        df = _load_tidy()

    perf = df[
        (df["metric_category"] == "performance")
        & (df["metric_name"] == "accuracy")
        & (df["task_name"] == "queue")
    ].copy()
    perf = _sort_by_paradigm(perf)

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
# Fig 02a: Macro-F1 — Priority 任务
# ---------------------------------------------------------------------------
def plot_macro_f1_priority(df: pd.DataFrame | None = None) -> plt.Figure:
    """Priority 任务，按 Macro-F1 降序排列，按范式着色。"""
    if df is None:
        df = _load_tidy()

    perf = df[
        (df["metric_category"] == "performance")
        & (df["metric_name"] == "macro_f1")
        & (df["task_name"] == "priority")
    ].copy()
    perf = _sort_by_paradigm(perf)

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
# Fig 02b: Macro-F1 — Queue 任务
# ---------------------------------------------------------------------------
def plot_macro_f1_queue(df: pd.DataFrame | None = None) -> plt.Figure:
    """Queue 任务，按 Macro-F1 降序排列，按范式着色。"""
    if df is None:
        df = _load_tidy()

    perf = df[
        (df["metric_category"] == "performance")
        & (df["metric_name"] == "macro_f1")
        & (df["task_name"] == "queue")
    ].copy()
    perf = _sort_by_paradigm(perf)

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
# Fig 03: Disparate Impact 热力图 — 按受保护属性分列
# ---------------------------------------------------------------------------
def plot_fairness_heatmap(df: pd.DataFrame | None = None) -> plt.Figure:
    """行 = 模型（按范式排列），列 = 受保护属性（language / tech_proficiency / user_type）。"""
    if df is None:
        df = _load_tidy()

    fairness = df[
        (df["metric_category"] == "fairness_pairwise")
        & (df["metric_name"] == "disparate_impact")
    ].copy()

    fairness = _sort_by_paradigm(fairness)

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
# Fig 04: Robustness — 攻击成功率，两任务分开，仅 xlm-roberta
# ---------------------------------------------------------------------------
def plot_robustness_attack_success(df: pd.DataFrame | None = None) -> plt.Figure:
    """分面 priority / queue，x 轴 attack_type，y 轴 attack success rate。"""
    if df is None:
        df = _load_robustness()

    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No robustness data available", ha="center", va="center")
        ax.axis("off")
        return fig

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    tasks = ["priority", "queue"]
    task_labels = ["Priority", "Queue"]

    for ax, task, label in zip(axes, tasks, task_labels):
        sub = df[df["task_name"] == task]
        if sub.empty:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center")
            ax.axis("off")
            ax.set_title(label)
            continue
        sns.barplot(
            data=sub,
            x="recipe",
            y="attack_success_rate",
            hue="recipe",
            ax=ax,
            palette=["#4c72b0"],
            width=0.5,
            legend=False,
        )
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("")
        ax.set_ylabel("Attack Success Rate")
        ax.set_title(label)
        for p in ax.patches:
            h = p.get_height()
            if h > 0.001:
                ax.text(
                    p.get_x() + p.get_width() / 2,
                    h + 0.02,
                    f"{h:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    fig.suptitle("Robustness: Attack Success Rate", fontweight="bold")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 05: Clean vs Perturbed Accuracy — xlm-roberta only
# ---------------------------------------------------------------------------
def plot_clean_vs_perturbed(df: pd.DataFrame | None = None) -> plt.Figure:
    """分面 priority / queue，每个任务两组（Clean vs Perturbed）。"""
    if df is None:
        df = _load_robustness()

    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No robustness data available", ha="center", va="center")
        ax.axis("off")
        return fig

    melted = df.melt(
        id_vars=["task_name", "display_name"],
        value_vars=["clean_accuracy", "perturbed_accuracy"],
        var_name="condition",
        value_name="accuracy",
    )
    melted["condition"] = melted["condition"].map(
        {"clean_accuracy": "Clean", "perturbed_accuracy": "Perturbed"}
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    tasks = ["priority", "queue"]
    task_labels = ["Priority", "Queue"]

    for ax, task, label in zip(axes, tasks, task_labels):
        sub = melted[melted["task_name"] == task]
        if sub.empty:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center")
            ax.axis("off")
            ax.set_title(label)
            continue
        sns.barplot(
            data=sub,
            x="condition",
            y="accuracy",
            hue="condition",
            ax=ax,
            palette=["#4c72b0", "#c44e52"],
            width=0.5,
            order=["Clean", "Perturbed"],
            legend=False,
        )
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("")
        ax.set_ylabel("Accuracy")
        ax.set_title(label)
        for p in ax.patches:
            h = p.get_height()
            if h > 0.001:
                ax.text(
                    p.get_x() + p.get_width() / 2,
                    h + 0.02,
                    f"{h:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    fig.suptitle("Robustness: Clean vs Perturbed Accuracy", fontweight="bold")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 参数量映射（近似值，单位：M）
# ---------------------------------------------------------------------------
# rule-based: 2 个配置变体，放同一范式线（无参数概念，画在同一 x 位置）
# lr/xgb (Non-Encoder): 同一条线（不同的 encoder type = 同一参数量）
# supervised encoder: mBERT + XLM-R 共用一条线
# goal-based LLM (Qwen3): few-shot / zero-shot 分两条线
# goal-based API: 不画 scaling（无参数量）

PARAM_COUNT = {
    "rule-based": 0.001,   # 极小，约等于无参数
    "lr": 0.1,             # LR 参数极少，象征性位置
    "xgb": 0.1,            # XGBoost 同上
    "mbert": 560,          # google/rembert: 560M
    "xlm-roberta": 250,    # FacebookAI/xlm-roberta-base: 250M
    "qwen-qwen3-0.6b": 600,    # Qwen3-0.6B: 600M
    "qwen-qwen3-1.7b": 1700,   # Qwen3-1.7B: 1.7B
    "qwen-qwen3-4b": 4000,     # Qwen3-4B: 4B
    # API 模型无本地参数量，不加入 scaling 图
}

# scaling 曲线分组：(group_name, 参数量)
# 同一 group 内多个 x（同一 encoding type 的不同模型）→ 画同一条线，x 轴分开
# LR tfidf/ST 各是一个 group，但每条线连两个点（x = 不同模型的参数位置）
SCALING_GROUPS = {
    # Rule-Based: 画在 x=0.001，两个配置取均值
    "rule-based": ("Rule-Based", 0.001),
    # LR tfidf: LR 层 + tfidf encoder
    "lr:tfidf": ("LR (tfidf)", 0.1),
    # LR ST: LR 层 + sentence-transformer (~100M)
    "lr:ST": ("LR (ST)", 100),
    # XGB tfidf: XGBoost + tfidf encoder
    "xgb:tfidf": ("XGBoost (tfidf)", 1),
    # XGB ST: XGBoost + sentence-transformer
    "xgb:ST": ("XGBoost (ST)", 100),
    # Supervised Encoder (mBERT + XLM-R): 共线
    "mbert": ("Supervised (Encoder)", 560),
    "xlm-roberta": ("Supervised (Encoder)", 250),
    # Goal-Based LLM - zero-shot
    "qwen3-0.6b:zero-shot": ("Goal-Based (LLM, zero-shot)", 600),
    "qwen3-1.7b:zero-shot": ("Goal-Based (LLM, zero-shot)", 1700),
    "qwen3-4b:zero-shot": ("Goal-Based (LLM, zero-shot)", 4000),
    # Goal-Based LLM - few-shot
    "qwen3-0.6b:few-shot": ("Goal-Based (LLM, few-shot)", 600),
    "qwen3-1.7b:few-shot": ("Goal-Based (LLM, few-shot)", 1700),
    "qwen3-4b:few-shot": ("Goal-Based (LLM, few-shot)", 4000),
}

# Each model is a separate line:
# TFIDF: LR(tfidf) at x=0.1, XGB(tfidf) at x=1
# ST: LR(ST) at x=100, XGB(ST) at x=100
# Encoder: mBERT at x=560, XLM-R at x=250 (shared line)


def _get_scaling_key(model_name: str, cfg: str | None) -> str | None:
    """生成 SCALING_GROUPS 的 key，没有则返回 None。"""
    key = None
    if model_name == "rule-based":
        key = "rule-based"
    elif model_name in ("lr", "xgb"):
        encoder = "ST" if cfg and "sentence_transformer" in cfg else "tfidf"
        key = f"{model_name}:{encoder}"
    elif model_name in ("mbert", "xlm-roberta"):
        key = model_name
    elif model_name.startswith("qwen-qwen3-"):
        few_shot = "few-shot" if cfg and "true" in cfg else "zero-shot"
        size = {"0.6b": "0.6b", "1.7b": "1.7b", "4b": "4b"}.get(
            next((k for k in ["0.6b", "1.7b", "4b"] if k in model_name), ""), ""
        )
        key = f"qwen3-{size}:{few_shot}"
    # API models excluded from scaling
    return key


# ---------------------------------------------------------------------------
# Fig 06: Scaling Curve — 每个模型一条独立线
# ---------------------------------------------------------------------------
def plot_scaling_curve(df: pd.DataFrame | None = None, metric: str = "accuracy") -> plt.Figure:
    """参数量 vs 性能。

    每条线 = 一个模型独立：
    - Rule-Based: x=0.001
    - LR (tfidf): x=0.1 / LR (ST): x=100
    - XGBoost (tfidf): x=1 / XGBoost (ST): x=100
    - Supervised Encoder: mBERT(x=560) + XLM-R(x=250) 共一条线（共享颜色/线型）
    - Goal-Based LLM: zero-shot / few-shot 各一条，x=600/1700/4000
    """
    if df is None:
        df = _load_tidy()

    metric_df = df[
        (df["metric_category"] == "performance")
        & (df["metric_name"] == metric)
    ].copy()

    # 构建 scaling 数据点
    # key → (group_name, param_count)
    key_to_group = {}
    for key, (name, x) in SCALING_GROUPS.items():
        key_to_group[key] = (name, x)

    raw_points: dict[str, list[tuple[float, float]]] = {}  # group_name -> [(x, y), ...]
    for _, row in metric_df.iterrows():
        key = _get_scaling_key(row["model_name"], row["cfg"])
        if key is None or key not in key_to_group:
            continue
        group_name, param_count = key_to_group[key]
        raw_points.setdefault(group_name, []).append((param_count, row["value"]))

    # 图例：每个模型一条独立线（不再合并 LR+XGB）
    # tfidf 变体：LR(tfidf) + XGB(tfidf) 两条线
    # ST 变体：LR(ST) + XGB(ST) 两条线
    # 每条线一个 x 位置（只标注自己）
    def _build_line_data(raw: dict[str, list[tuple[float, float]]]) -> dict[str, list[tuple[float, float]]]:
        """每个 group 各自一条线，多个 y 值按 task 取均值得到单一 y。"""
        result = {}
        for group, xs_vals in raw.items():
            by_x: dict[float, list[float]] = {}
            for x, v in xs_vals:
                by_x.setdefault(x, []).append(v)
            result[group] = sorted((x, sum(vs) / len(vs)) for x, vs in by_x.items())
        return result

    line_data = _build_line_data(raw_points)

    # 按参数量排 order
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
    group_colors = {
        "Rule-Based": "#4caf50",
        "LR (tfidf)": "#64b5f6",
        "LR (ST)": "#1976d2",
        "XGBoost (tfidf)": "#81c784",
        "XGBoost (ST)": "#388e3c",
        "Supervised (Encoder)": "#1565c0",
        "Goal-Based (LLM, zero-shot)": "#ff9800",
        "Goal-Based (LLM, few-shot)": "#e65100",
    }
    group_markers = {
        "Rule-Based": "o",
        "LR (tfidf)": "s",
        "LR (ST)": "s",
        "XGBoost (tfidf)": "^",
        "XGBoost (ST)": "^",
        "Supervised (Encoder)": "D",
        "Goal-Based (LLM, zero-shot)": "D",
        "Goal-Based (LLM, few-shot)": "v",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for group in group_order:
        if group not in line_data:
            continue
        xs, ys = zip(*line_data[group])

        # 图例用简短模型名
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
        ax.plot(
            xs, ys,
            color=group_colors.get(group, "#999"),
            marker=group_markers.get(group, "o"),
            markersize=8,
            label=legend_names.get(group, group),
            linewidth=2,
            zorder=3,
        )
        # marker 旁边标注具体模型名
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
                label = group  # LR(tfidf), LR(ST), XGB(tfidf), XGB(ST)
            ax.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=7,
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


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {RESULTS_XLSX}")
    tidy_df = _load_tidy()
    robust_df = _load_robustness()

    figures = [
        ("01a_accuracy_priority.png", plot_accuracy_priority(tidy_df)),
        ("01b_accuracy_queue.png", plot_accuracy_queue(tidy_df)),
        ("02a_macro_f1_priority.png", plot_macro_f1_priority(tidy_df)),
        ("02b_macro_f1_queue.png", plot_macro_f1_queue(tidy_df)),
        ("03_fairness_disparate_impact.png", plot_fairness_heatmap(tidy_df)),
        ("04_robustness_attack_success_rate.png", plot_robustness_attack_success(robust_df)),
        ("05_clean_vs_perturbed_accuracy.png", plot_clean_vs_perturbed(robust_df)),
        ("06a_scaling_accuracy.png", plot_scaling_curve(tidy_df, "accuracy")),
        ("06b_scaling_macro_f1.png", plot_scaling_curve(tidy_df, "macro_f1")),
    ]

    for filename, fig in figures:
        path = OUTPUT_DIR / filename
        fig.savefig(path, format="png")
        plt.close(fig)
        print(f"Saved: {path}")

    print("\nAll figures generated successfully.")


if __name__ == "__main__":
    main()