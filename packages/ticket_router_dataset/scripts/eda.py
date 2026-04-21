#!/usr/bin/env python3
"""
Exploratory Data Analysis for multilingual-customer-support-tickets.
Outputs summary statistics and visualizations to outputs/eda/.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

DATA_DIR = Path("dataset/multilingual-customer-support-tickets")
OUT_DIR = Path("outputs/eda")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 1. Load datasets
# ------------------------------------------------------------------
datasets = {
    "4k": pd.read_csv(DATA_DIR / "dataset-tickets-multi-lang3-4k.csv", encoding="utf-8"),
    "20k": pd.read_csv(DATA_DIR / "dataset-tickets-multi-lang-4-20k.csv", encoding="utf-8"),
    "28k": pd.read_csv(DATA_DIR / "aa_dataset-tickets-multi-lang-5-2-50-version.csv", encoding="utf-8"),
    "german_norm_small": pd.read_csv(DATA_DIR / "dataset-tickets-german_normalized.csv", encoding="utf-8"),
    "german_norm_large": pd.read_csv(DATA_DIR / "dataset-tickets-german_normalized_50_5_2.csv", encoding="utf-8"),
}

report_lines = ["# EDA Report: Multilingual Customer Support Tickets\n"]

# ------------------------------------------------------------------
# 2. Basic stats
# ------------------------------------------------------------------
report_lines.append("## 1. Basic Statistics\n")
for name, df in datasets.items():
    report_lines.append(f"- **{name}**: {len(df):,} rows, {len(df.columns)} columns")
    missing = df.isnull().sum().sum()
    report_lines.append(f"  - Total missing values: {missing}")
report_lines.append("")

# ------------------------------------------------------------------
# 3. Language distribution
# ------------------------------------------------------------------
report_lines.append("## 2. Language Distribution\n")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for ax, (name, df) in zip(axes, datasets.items()):
    if "language" in df.columns:
        lang_counts = df["language"].value_counts().sort_index()
        lang_counts.plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title(f"{name} — Language Distribution")
        ax.set_xlabel("Language")
        ax.set_ylabel("Count")
        report_lines.append(f"- **{name}**: {lang_counts.to_dict()}")
    else:
        ax.set_title(f"{name} — No language column")
        ax.axis("off")
plt.tight_layout()
plt.savefig(OUT_DIR / "language_distribution.png")
plt.close()

# ------------------------------------------------------------------
# 4. Queue distribution
# ------------------------------------------------------------------
report_lines.append("\n## 3. Queue Distribution\n")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()
for ax, (name, df) in zip(axes, datasets.items()):
    if "queue" in df.columns:
        q_counts = df["queue"].value_counts()
        q_counts.plot(kind="barh", ax=ax, color="coral")
        ax.set_title(f"{name} — Queue Distribution")
        ax.set_xlabel("Count")
        report_lines.append(f"- **{name}** (top 5): {q_counts.head(5).to_dict()}")
    else:
        ax.set_title(f"{name} — No queue column")
        ax.axis("off")
plt.tight_layout()
plt.savefig(OUT_DIR / "queue_distribution.png")
plt.close()

# ------------------------------------------------------------------
# 5. Priority distribution
# ------------------------------------------------------------------
report_lines.append("\n## 4. Priority Distribution\n")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for ax, (name, df) in zip(axes, datasets.items()):
    if "priority" in df.columns:
        p_counts = df["priority"].value_counts().sort_index()
        p_counts.plot(kind="bar", ax=ax, color="seagreen")
        ax.set_title(f"{name} — Priority Distribution")
        ax.set_xlabel("Priority")
        ax.set_ylabel("Count")
        report_lines.append(f"- **{name}**: {p_counts.to_dict()}")
    else:
        ax.set_title(f"{name} — No priority column")
        ax.axis("off")
plt.tight_layout()
plt.savefig(OUT_DIR / "priority_distribution.png")
plt.close()

# ------------------------------------------------------------------
# 6. Text length analysis (focus on 4k as primary)
# ------------------------------------------------------------------
report_lines.append("\n## 5. Text Length Analysis (4k dataset)\n")
df4k = datasets["4k"].copy()
for col in ["subject", "body", "answer"]:
    df4k[f"{col}_len"] = df4k[col].fillna("").astype(str).str.len()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, col in zip(axes, ["subject_len", "body_len", "answer_len"]):
    hist_data = df4k[col].to_frame()
    sns.histplot(hist_data, bins=50, kde=True, ax=ax, color="mediumpurple")
    ax.set_title(f"Distribution of {col.replace('_len', '').title()} Length")
    ax.set_xlabel("Characters")
    report_lines.append(
        f"- **{col}**: mean={df4k[col].mean():.1f}, median={df4k[col].median():.1f}, max={df4k[col].max()}"
    )
plt.tight_layout()
plt.savefig(OUT_DIR / "text_length_distribution.png")
plt.close()

# ------------------------------------------------------------------
# 7. Duplicate detection
# ------------------------------------------------------------------
report_lines.append("\n## 6. Duplicate Detection\n")

# Intra-dataset duplicates
for name, df in datasets.items():
    if "subject" in df.columns and "body" in df.columns:
        dupes = df.duplicated(subset=["subject", "body"]).sum()
        report_lines.append(f"- **{name}** internal duplicates (subject+body): {dupes}")

# Cross-dataset overlaps: 4k vs 20k, 4k vs 28k, 20k vs 28k
def overlap_count(df_a, df_b):
    keys_a = set(zip(df_a["subject"].fillna(""), df_a["body"].fillna("")))
    keys_b = set(zip(df_b["subject"].fillna(""), df_b["body"].fillna("")))
    return len(keys_a & keys_b)

report_lines.append(f"- 4k vs 20k overlap: {overlap_count(datasets['4k'], datasets['20k'])}")
report_lines.append(f"- 4k vs 28k overlap: {overlap_count(datasets['4k'], datasets['28k'])}")
report_lines.append(f"- 20k vs 28k overlap: {overlap_count(datasets['20k'], datasets['28k'])}")

# ------------------------------------------------------------------
# 8. Tag analysis (4k dataset)
# ------------------------------------------------------------------
report_lines.append("\n## 7. Tag Analysis (4k dataset)\n")
tag_cols = [c for c in df4k.columns if c.startswith("tag_")]
all_tags = []
for _, row in df4k.iterrows():
    for tc in tag_cols:
        v = row.get(tc)
        if pd.notna(v) and str(v).strip():
            all_tags.append(str(v).strip())

tag_counter = Counter(all_tags)
report_lines.append(f"- Unique tags: {len(tag_counter)}")
report_lines.append(f"- Top 10 tags: {tag_counter.most_common(10)}")

# Tags per ticket
tags_per_ticket = df4k[tag_cols].apply(lambda row: sum(1 for v in row if pd.notna(v) and str(v).strip()), axis=1)
report_lines.append(f"- Tags per ticket: mean={tags_per_ticket.mean():.2f}, median={tags_per_ticket.median():.1f}")

fig, ax = plt.subplots(figsize=(10, 6))
top_tags = pd.Series(tag_counter).nlargest(20)
top_tags.plot(kind="barh", ax=ax, color="teal")
ax.set_title("Top 20 Tags (4k dataset)")
ax.set_xlabel("Frequency")
plt.tight_layout()
plt.savefig(OUT_DIR / "top_tags.png")
plt.close()

# ------------------------------------------------------------------
# 9. Imbalance & small-queue analysis (4k)
# ------------------------------------------------------------------
report_lines.append("\n## 8. Class Imbalance (4k dataset)\n")
q_counts = df4k["queue"].value_counts()
report_lines.append(f"- Largest queue: {q_counts.index[0]} ({q_counts.iloc[0]} samples)")
report_lines.append(f"- Smallest queue: {q_counts.index[-1]} ({q_counts.iloc[-1]} samples)")
report_lines.append(f"- Imbalance ratio (max/min): {q_counts.iloc[0] / q_counts.iloc[-1]:.1f}:1")

# ------------------------------------------------------------------
# 10. Version field analysis (28k dataset)
# ------------------------------------------------------------------
report_lines.append("\n## 9. Version Field Analysis (28k dataset)\n")
df28k = datasets["28k"]
if "version" in df28k.columns:
    v_counts = df28k["version"].value_counts().sort_index()
    report_lines.append(f"- Versions: {v_counts.to_dict()}")
    # Check if versions have different queue distributions
    fig, ax = plt.subplots(figsize=(10, 6))
    v_counts.plot(kind="bar", ax=ax, color="darkorange")
    ax.set_title("28k — Version Distribution")
    ax.set_xlabel("Version")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "version_distribution.png")
    plt.close()
else:
    report_lines.append("- No 'version' column found.")

# ------------------------------------------------------------------
# 11. Save report
# ------------------------------------------------------------------
report_lines.append("\n## 10. Key Takeaways\n")
report_lines.append("1. The 4k dataset is the only one covering all 5 languages (EN, DE, ES, FR, PT).")
report_lines.append("2. Queue distribution is heavily imbalanced; smallest queues have <60 samples.")
report_lines.append("3. Cross-dataset overlaps exist between 4k/20k/28k; deduplication is mandatory before mixing training data.")
report_lines.append("4. German normalized datasets have fewer fields (no tags), limiting their use to robustness testing only.")
report_lines.append("5. Version field in 28k likely marks data batches; stratification by version is advisable if used for training.")

report_path = OUT_DIR / "eda_report.md"
report_path.write_text("\n".join(report_lines), encoding="utf-8")
print(f"EDA complete. Report saved to {report_path}")
print(f"Visualizations saved to {OUT_DIR}")
