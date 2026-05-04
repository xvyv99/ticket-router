"""Robustness visualization functions.

Combines robustness plots from both the fairness and evaluation scripts.
Functions are disambiguated by granularity: language-level vs recipe-level.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Language-level robustness (xlm-roberta only, from fairness script)
# ---------------------------------------------------------------------------

def plot_robustness_clean_vs_perturbed_by_lang(df: pd.DataFrame) -> plt.Figure:
    """Clean vs Perturbed Accuracy grouped by task × language (xlm-roberta)."""
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


def plot_robustness_attack_success_by_lang(df: pd.DataFrame) -> plt.Figure:
    """Attack Success Rate by Language, faceted by task (xlm-roberta)."""
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
        ax.set_title(f"Attack Success Rate — {task}", fontweight="bold")

    fig.suptitle("Robustness: Attack Success Rate by Language", fontweight="bold", fontsize=14)
    plt.tight_layout()
    return fig


def plot_robustness_accuracy_drop_by_lang(df: pd.DataFrame) -> plt.Figure:
    """Accuracy Drop by Language, faceted by task (xlm-roberta)."""
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

    for i, task in enumerate(tasks):
        ax = axes[i]
        task_data = lang_df[lang_df["task"] == task].set_index("group").reindex(languages)
        task_data = task_data.sort_values("accuracy_drop")

        ax.barh(task_data.index, task_data["accuracy_drop"],
                color="#c44e52", height=0.6)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Accuracy Drop")
        ax.set_title(f"Accuracy Drop — {task}", fontweight="bold")

    fig.suptitle("Robustness: Accuracy Drop by Language", fontweight="bold", fontsize=14)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Recipe-level robustness (multi-model, from evaluation script)
# ---------------------------------------------------------------------------

def plot_robustness_recipe_attack_success(df: pd.DataFrame) -> plt.Figure:
    """Attack Success Rate by recipe, faceted by task (multi-model)."""
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


def plot_robustness_recipe_clean_vs_perturbed(df: pd.DataFrame) -> plt.Figure:
    """Clean vs Perturbed Accuracy, faceted by task (multi-model)."""
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
