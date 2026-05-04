"""Shared constants for plot styling: colour palettes, paradigm mappings, scaling configs."""

from __future__ import annotations

import seaborn as sns

# ---------------------------------------------------------------------------
# Language palette (Set2, desaturated jewel tones)
# ---------------------------------------------------------------------------
_LANG_PALETTE = sns.color_palette("Set2", n_colors=5)
LANG_COLORS: dict[str, tuple[float, float, float]] = {
    "en": _LANG_PALETTE[0],
    "de": _LANG_PALETTE[1],
    "es": _LANG_PALETTE[2],
    "fr": _LANG_PALETTE[3],
    "pt": _LANG_PALETTE[4],
}

# ---------------------------------------------------------------------------
# Priority colours (warm-cool diverging)
# ---------------------------------------------------------------------------
PRIORITY_COLORS: dict[str, str] = {
    "high": "#d95f02",
    "medium": "#f2a340",
    "low": "#5ba3cf",
}

# ---------------------------------------------------------------------------
# User type colours
# ---------------------------------------------------------------------------
USER_TYPE_COLORS: dict[str, str] = {
    "enterprise": "#276d86",
    "individual": "#cf7c46",
}

# ---------------------------------------------------------------------------
# Tech proficiency colours (three-step sequential blue)
# ---------------------------------------------------------------------------
TECH_COLORS: dict[str, str] = {
    "high": "#08519c",
    "medium": "#6baed6",
    "low": "#bdd7e7",
}

# ---------------------------------------------------------------------------
# Queue canonical order
# ---------------------------------------------------------------------------
QUEUE_ORDER: list[str] = [
    "Technical Support",
    "Product Support",
    "Customer Service",
    "IT Support",
    "Billing and Payments",
    "Returns and Exchanges",
    "Service Outages and Maintenance",
    "Sales and Pre-Sales",
    "Human Resources",
    "General Inquiry",
]

# ---------------------------------------------------------------------------
# Paradigm mappings (shared by fairness + evaluation scripts)
# ---------------------------------------------------------------------------
PARADIGM_ORDER: list[str] = [
    "Rule-Based",
    "Supervised (Non-Encoder)",
    "Supervised (Encoder)",
    "Goal-Based (LLM)",
    "Goal-Based (API)",
]

PARADIGM_COLOR: dict[str, str] = {
    "Rule-Based": "#4caf50",
    "Supervised (Non-Encoder)": "#2196f3",
    "Supervised (Encoder)": "#1565c0",
    "Goal-Based (LLM)": "#ff9800",
    "Goal-Based (API)": "#e65100",
}

# model_name -> (display_name, paradigm)
MODEL_RENAME: dict[str, tuple[str, str]] = {
    # Rule-Based
    "rule-based": ("Rule-Based", "Rule-Based"),
    # Supervised (Non-Encoder)
    "lr": ("LR", "Supervised (Non-Encoder)"),
    "xgb": ("XGBoost", "Supervised (Non-Encoder)"),
    # Supervised (Encoder)
    "mbert": ("mBERT", "Supervised (Encoder)"),
    "xlm-roberta": ("XLM-RoBERTa", "Supervised (Encoder)"),
    # Goal-Based (LLM, local)
    "qwen-qwen3-0.6b": ("Qwen3-0.6B", "Goal-Based (LLM)"),
    "qwen-qwen3-1.7b": ("Qwen3-1.7B", "Goal-Based (LLM)"),
    "qwen-qwen3-4b": ("Qwen3-4B", "Goal-Based (LLM)"),
    # Goal-Based (API)
    "qwen3.5-flash(no thinking)": ("Qwen3.5-Flash (no thinking)", "Goal-Based (API)"),
    "qwen3.5-flash(thinking)": ("Qwen3.5-Flash (thinking)", "Goal-Based (API)"),
    "qwen3.5-plus(no thinking)": ("Qwen3.5-Plus (no thinking)", "Goal-Based (API)"),
    "qwen3.5-plus(thinking)": ("Qwen3.5-Plus (thinking)", "Goal-Based (API)"),
}

# ---------------------------------------------------------------------------
# Scaling curve configs (evaluation script)
# ---------------------------------------------------------------------------
# model_name -> approximate parameter count (millions)
PARAM_COUNT: dict[str, float] = {
    "rule-based": 0.001,
    "lr": 0.1,
    "xgb": 0.1,
    "mbert": 560,
    "xlm-roberta": 250,
    "qwen-qwen3-0.6b": 600,
    "qwen-qwen3-1.7b": 1700,
    "qwen-qwen3-4b": 4000,
}

# scaling_key -> (group_name, param_count)
SCALING_GROUPS: dict[str, tuple[str, float]] = {
    "rule-based": ("Rule-Based", 0.001),
    "lr:tfidf": ("LR (tfidf)", 0.1),
    "lr:ST": ("LR (ST)", 100),
    "xgb:tfidf": ("XGBoost (tfidf)", 1),
    "xgb:ST": ("XGBoost (ST)", 100),
    "mbert": ("Supervised (Encoder)", 560),
    "xlm-roberta": ("Supervised (Encoder)", 250),
    "qwen3-0.6b:zero-shot": ("Goal-Based (LLM, zero-shot)", 600),
    "qwen3-1.7b:zero-shot": ("Goal-Based (LLM, zero-shot)", 1700),
    "qwen3-4b:zero-shot": ("Goal-Based (LLM, zero-shot)", 4000),
    "qwen3-0.6b:few-shot": ("Goal-Based (LLM, few-shot)", 600),
    "qwen3-1.7b:few-shot": ("Goal-Based (LLM, few-shot)", 1700),
    "qwen3-4b:few-shot": ("Goal-Based (LLM, few-shot)", 4000),
}

# Scaling curve visual styles (used inline in plot_scaling_curve)
SCALING_GROUP_COLORS: dict[str, str] = {
    "Rule-Based": "#4caf50",
    "LR (tfidf)": "#64b5f6",
    "LR (ST)": "#1976d2",
    "XGBoost (tfidf)": "#81c784",
    "XGBoost (ST)": "#388e3c",
    "Supervised (Encoder)": "#1565c0",
    "Goal-Based (LLM, zero-shot)": "#ff9800",
    "Goal-Based (LLM, few-shot)": "#e65100",
}

SCALING_GROUP_MARKERS: dict[str, str] = {
    "Rule-Based": "o",
    "LR (tfidf)": "s",
    "LR (ST)": "s",
    "XGBoost (tfidf)": "^",
    "XGBoost (ST)": "^",
    "Supervised (Encoder)": "D",
    "Goal-Based (LLM, zero-shot)": "D",
    "Goal-Based (LLM, few-shot)": "v",
}
