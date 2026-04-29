#!/usr/bin/env python3
"""
生成 Ticket Router 评估结果的可视化图表。
学术论文风格，输出 PNG 到 outputs/figures/。
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

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
    "rule-based": ("Rule", "Rule-Based"),
    # Supervised Non-Encoder
    "lr": ("LR", "Supervised (Non-Encoder)"),
    "xgb": ("XGB", "Supervised (Non-Encoder)"),
    # Supervised Encoder
    "mbert": ("mBERT", "Supervised (Encoder)"),
    "xlm-roberta": ("XLM-R", "Supervised (Encoder)"),
    # Goal-Based LLM (local)
    "qwen-qwen3-0.6b": ("Qwen3-0.6B", "Goal-Based (LLM)"),
    "qwen-qwen3-1.7b": ("Qwen3-1.7B", "Goal-Based (LLM)"),
    "qwen-qwen3-4b": ("Qwen3-4B", "Goal-Based (LLM)"),
    # Goal-Based API
    "qwen3.5-flash(no thinking)": ("Qwen3.5-F", "Goal-Based (API)"),
    "qwen3.5-flash(thinking)": ("Qwen3.5-F (T)", "Goal-Based (API)"),
    "qwen3.5-plus(no thinking)": ("Qwen3.5+", "Goal-Based (API)"),
    "qwen3.5-plus(thinking)": ("Qwen3.5+ (T)", "Goal-Based (API)"),
}


def _model_info(model_name: str) -> tuple[str, str]:
    """返回 (display_name, paradigm)。"""
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
    df["display_name"] = df["model_name"].map(lambda x: _model_info(x)[0])
    df["paradigm"] = df["model_name"].map(lambda x: _model_info(x)[1])
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


def _load_class_summary(model: str, task: str) -> dict | None:
    """加载指定模型和任务的 class_summary.json，不存在返回 None。"""
    path = Path(f"outputs/interpretability/{model}/multilingual-customer-support/{task}_class_summary.json")
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Fig 01: Accuracy — 两任务分开，按范式分组着色
# ---------------------------------------------------------------------------
def plot_accuracy_comparison(df: pd.DataFrame | None = None) -> plt.Figure:
    """两行（priority / queue），每行按 accuracy 降序排列，按范式着色。"""
    if df is None:
        df = _load_tidy()

    perf = df[
        (df["metric_category"] == "performance") & (df["metric_name"] == "accuracy")
    ].copy()
    agg = (
        perf.groupby(["task_name", "display_name", "paradigm"])["value"]
        .mean()
        .reset_index()
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    tasks = ["priority", "queue"]
    task_labels = ["Priority (3-class)", "Queue (10-class)"]

    for ax, task, label in zip(axes, tasks, task_labels):
        sub = agg[agg["task_name"] == task].sort_values("value", ascending=True)
        colors = [PARADIGM_COLOR.get(p, "#999") for p in sub["paradigm"]]
        y = sub["display_name"].values
        width = sub["value"].values
        colors_arr = [PARADIGM_COLOR.get(p, "#999") for p in sub["paradigm"]]
        bars = ax.barh(y, width, height=0.7, color=colors_arr)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Accuracy")
        ax.set_title(label, fontweight="bold")
        # 标注数值
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{w:.3f}", ha="left", va="center", fontsize=8)
        # 图例（范式颜色）
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=PARADIGM_COLOR[p], label=p)
            for p in PARADIGM_ORDER if p in sub["paradigm"].values
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    fig.suptitle("Model Accuracy Comparison", fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 02: Macro-F1 — 两任务并列，按范式分组
# ---------------------------------------------------------------------------
def plot_macro_f1_by_task(df: pd.DataFrame | None = None) -> plt.Figure:
    """两任务并排柱状图，按模型分组，同一模型两个任务相邻。"""
    if df is None:
        df = _load_tidy()

    perf = df[
        (df["metric_category"] == "performance") & (df["metric_name"] == "macro_f1")
    ].copy()
    agg = (
        perf.groupby(["task_name", "display_name", "paradigm"])["value"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(14, 6))

    tasks = ["priority", "queue"]
    n_models = len(agg["display_name"].unique())
    x = range(n_models)
    width = 0.35

    # 按 paradigm + display_name 排序
    paradigm_order_idx = {p: i for i, p in enumerate(PARADIGM_ORDER)}
    agg["_sort"] = agg["paradigm"].map(paradigm_order_idx)
    agg = agg.sort_values(["_sort", "display_name"])
    display_order = agg["display_name"].unique()

    for i, task in enumerate(tasks):
        offsets = [-width/2 + i * width]  # wrong, let me redo

    # 重新设计：x 轴为模型，priority/queue 双柱
    x_positions = []
    tick_labels = []
    for di, disp in enumerate(display_order):
        for ti, task in enumerate(tasks):
            x_positions.append(di + (ti - 0.5) * width / len(display_order))
        tick_labels.append(disp)

    # 这个方法太复杂，用 seaborn catplot 更简洁
    plt.close(fig)

    # 重来：直接用 catplot 风格
    # 构造复合 x：display_name + task
    agg["model_task"] = agg["display_name"] + " | " + agg["task_name"]

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.barplot(
        data=agg,
        x="display_name",
        y="value",
        hue="task_name",
        ax=ax,
        palette=["#4c72b0", "#dd8452"],
        width=0.7,
        order=display_order,
        hue_order=tasks,
    )
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Macro-F1 by Task", fontweight="bold")
    ax.legend(title="Task")
    plt.xticks(rotation=45, ha="right")

    # 按范式在 x 轴添加分隔线
    prev_paradigm = None
    for i, (_, row) in enumerate(agg[agg["task_name"] == "priority"].sort_values("display_name").iterrows()):
        paradigm = row["paradigm"]
        if prev_paradigm is not None and paradigm != prev_paradigm:
            ax.axvline(i - 0.5, color="gray", linestyle="--", linewidth=0.5)
        prev_paradigm = paradigm

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

    # 按 paradigm + display_name 排序
    fairness["paradigm_order"] = fairness["paradigm"].map(
        {p: i for i, p in enumerate(PARADIGM_ORDER)}
    )
    fairness = fairness.sort_values(["paradigm_order", "display_name"])

    pivot = (
        fairness.groupby(["display_name", "sensitive_attr"])["value"]
        .mean()
        .unstack()
    )

    # 受保护属性列顺序
    attr_order = ["language", "tech_proficiency", "user_type"]
    pivot = pivot[[a for a in attr_order if a in pivot.columns]]

    fig, ax = plt.subplots(figsize=(8, 10))
    # 红黄绿：< 1 不利（红），> 1 特权（绿）
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
# Fig 06: 可解释性 Token Attribution — 表格形式
# ---------------------------------------------------------------------------
def plot_interpretability_tokens() -> plt.Figure:
    """两行（mBERT / XLM-R）× 两列（Priority / Queue），表格显示每类 top-5 pos/neg tokens。"""
    models = ["mbert", "xlm-roberta"]
    tasks = ["priority", "queue"]

    fig, axes = plt.subplots(2, 2, figsize=(24, 16))
    fig.suptitle("Interpretability: Top-5 Positive / Negative Tokens by Class", fontweight="bold", y=1.01)

    for row, model in enumerate(models):
        display_model = _model_info(model)[0]
        for col, task in enumerate(tasks):
            ax = axes[row][col]
            class_summary = _load_class_summary(model, task)

            if class_summary is None:
                ax.text(0.5, 0.5, f"No Data\n({display_model} / {task})",
                        ha="center", va="center", fontsize=12)
                ax.axis("off")
                ax.set_title(f"{display_model} — {task.title()}", fontweight="bold")
                continue

            # 构建表格数据：行 = class_label，列 = Top-5 Positive | Top-5 Negative
            classes = sorted(class_summary.keys())
            cell_text = []
            for cls in classes:
                tokens_scores = class_summary[cls]["top_tokens"]
                pos_tokens = [(t, s) for t, s in tokens_scores if s > 0][:5]
                neg_tokens = [(t, s) for t, s in tokens_scores if s < 0][:5]
                pos_str = ", ".join([f"{t} ({s:.3f})" for t, s in pos_tokens]) or "—"
                neg_str = ", ".join([f"{t} ({s:.3f})" for t, s in neg_tokens]) or "—"
                cell_text.append([cls, pos_str, neg_str])

            ax.axis("off")
            ax.set_title(f"{display_model} — {task.title()}", fontweight="bold", fontsize=13)

            # 绘制表格
            table = ax.table(
                cellText=cell_text,
                colLabels=["Class", "Top-5 Positive Tokens", "Top-5 Negative Tokens"],
                cellLoc="left",
                loc="center",
                bbox=[0, 0, 1, 1],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.5)
            # 表头样式
            for j in range(3):
                table[(0, j)].set_facecolor("#d4e6f1")
                table[(0, j)].set_text_props(weight="bold")

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
        ("01_model_accuracy_comparison.png", plot_accuracy_comparison(tidy_df)),
        ("02_macro_f1_by_task.png", plot_macro_f1_by_task(tidy_df)),
        ("03_fairness_disparate_impact.png", plot_fairness_heatmap(tidy_df)),
        ("04_robustness_attack_success_rate.png", plot_robustness_attack_success(robust_df)),
        ("05_clean_vs_perturbed_accuracy.png", plot_clean_vs_perturbed(robust_df)),
        ("06_interpretability_tokens.png", plot_interpretability_tokens()),
    ]

    for filename, fig in figures:
        path = OUTPUT_DIR / filename
        fig.savefig(path, format="png")
        plt.close(fig)
        print(f"Saved: {path}")

    print("\nAll figures generated successfully.")


if __name__ == "__main__":
    main()