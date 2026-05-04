"""Data loading and processing utilities for plot modules."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ticket_router.plot.config import MODEL_RENAME, PARADIGM_ORDER, SCALING_GROUPS


# ---------------------------------------------------------------------------
# Config suffix parsing
# ---------------------------------------------------------------------------

def parse_cfg_suffix(cfg: str | float | None) -> str:
    """Parse a JSON cfg string into a human-readable short label.

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
        return "ST" if c["encoder_type"] == "sentence_transformer" else "tfidf"
    elif "few_shot" in c:
        return "few-shot" if c["few_shot"] else "zero-shot"
    elif "enable_candidate_search" in c:
        return "cand-search" if c["enable_candidate_search"] else "no-cand"
    return ""


# ---------------------------------------------------------------------------
# Model info lookup
# ---------------------------------------------------------------------------

def get_model_info(model_name: str) -> tuple[str, str]:
    """Return (display_name, paradigm) for the given model_name."""
    return MODEL_RENAME.get(model_name, (model_name, "Unknown"))


# ---------------------------------------------------------------------------
# Sort helpers
# ---------------------------------------------------------------------------

def sort_by_paradigm(df: pd.DataFrame) -> pd.DataFrame:
    """Sort a DataFrame by PARADIGM_ORDER then by display_name."""
    paradigm_order_idx = {p: i for i, p in enumerate(PARADIGM_ORDER)}
    df = df.copy()
    df["_paradigm_order"] = df["paradigm"].map(paradigm_order_idx)
    df = df.sort_values(["_paradigm_order", "display_name"]).drop(columns=["_paradigm_order"])
    return df


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def filter_best_configs(df: pd.DataFrame) -> pd.DataFrame:
    """Per model_name, keep only the config with highest macro_f1 on priority task."""
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


# ---------------------------------------------------------------------------
# Scaling key lookup
# ---------------------------------------------------------------------------

def get_scaling_key(model_name: str, cfg: str | None) -> str | None:
    """Generate a SCALING_GROUPS key for the given model + config.

    Returns None for API models (excluded from scaling plots).
    """
    if model_name == "rule-based":
        return "rule-based"
    elif model_name in ("lr", "xgb"):
        encoder = "ST" if cfg and "sentence_transformer" in cfg else "tfidf"
        return f"{model_name}:{encoder}"
    elif model_name in ("mbert", "xlm-roberta"):
        return model_name
    elif model_name.startswith("qwen-qwen3-"):
        few_shot = "few-shot" if cfg and "true" in cfg else "zero-shot"
        size = {"0.6b": "0.6b", "1.7b": "1.7b", "4b": "4b"}.get(
            next((k for k in ["0.6b", "1.7b", "4b"] if k in model_name), ""), ""
        )
        return f"qwen3-{size}:{few_shot}"
    return None


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_eda_data(
    parquet_path: Path | str | None = None,
    jsonl_path: Path | str | None = None,
) -> pd.DataFrame:
    """Load parquet + Qwen3-4B inferred attributes JSONL, merge into unified EDA DataFrame."""
    if parquet_path is None:
        parquet_path = Path("outputs/multilingual-customer-support_test_split.parquet")
    if jsonl_path is None:
        jsonl_path = Path("outputs/infer_multilingual-customer-support_Qwen3-4B.jsonl")

    df = pd.read_parquet(parquet_path)

    records: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
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


def load_eval_tidy(path: str | Path) -> pd.DataFrame:
    """Load evaluation tidy data from CSV or Excel, attaching display_name/paradigm/cfg_suffix."""
    path = Path(path)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, sheet_name="tidy")

    base_names: list[str] = []
    paradigms: list[str] = []
    cfg_suffixes: list[str] = []
    for _, row in df.iterrows():
        base_name, paradigm = get_model_info(row["model_name"])
        suffix = parse_cfg_suffix(row["cfg"])
        cfg_suffixes.append(suffix)
        display_name = f"{base_name} ({suffix})" if suffix else base_name
        base_names.append(display_name)
        paradigms.append(paradigm)

    df["display_name"] = base_names
    df["paradigm"] = paradigms
    df["cfg_suffix"] = cfg_suffixes
    return df


def load_robustness_lang_csv(task_dir: str | Path) -> pd.DataFrame:
    """Load xlm-roberta language-level robustness CSVs (priority + queue).

    Each CSV has columns: group, clean_accuracy, perturbed_accuracy,
    attack_success_rate, accuracy_drop, granularity.
    """
    task_dir = Path(task_dir)
    dfs = []
    for task in ["priority", "queue"]:
        csv_path = task_dir / f"{task}_metrics.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["task"] = task
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def load_robustness_recipe_xlsx(
    glob_pattern: str | None = None,
) -> pd.DataFrame:
    """Load multi-model recipe-level robustness Excel files.

    Extracts model_name from path structure and attaches display_name/paradigm.
    """
    if glob_pattern is None:
        glob_pattern = "outputs/robustness/*/*/*_metrics.xlsx"

    import glob as _glob
    rob_files = sorted(_glob.glob(glob_pattern))
    if not rob_files:
        return pd.DataFrame()

    dfs = []
    for f in rob_files:
        df = pd.read_excel(f)
        model_name = Path(f).parts[2]  # outputs/robustness/{model}/...
        display_name, paradigm = get_model_info(model_name)
        df["model_name"] = model_name
        df["display_name"] = display_name
        df["paradigm"] = paradigm
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def map_display_to_paradigm(name: str) -> str:
    """Reverse-lookup paradigm from a display_name prefix."""
    for model_name, (base_name, paradigm) in MODEL_RENAME.items():
        if name.startswith(base_name):
            return paradigm
    return "Unknown"
