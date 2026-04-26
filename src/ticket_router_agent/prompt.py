"""System prompt builder using dataset descriptors."""

import json
import random
from typing import Dict, List

import pandas as pd

from ticket_router_base.data import BaseDataset
from ticket_router_base.types import Record
from ticket_router_base.config import SEED


# --- Task semantic descriptions by dataset name ---

QUEUE_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "multilingual-customer-support": {
        "Technical Support": "Software bugs, feature malfunctions, system errors, crashes, technical glitches",
        "Product Support": "Usage questions, how-to guides, product feature inquiries, documentation help",
        "Customer Service": "General complaints, policy questions, account issues, feedback, satisfaction concerns",
        "IT Support": "Internal infrastructure, network problems, hardware failures, VPN, email outages",
        "Billing and Payments": "Invoices, refunds, payment failures, subscription issues, charge disputes",
        "Returns and Exchanges": "Refund requests, product returns, swap orders, defective or damaged items",
        "Sales and Pre-Sales": "Pricing inquiries, demos, purchase questions, upgrades, contract negotiations",
        "Service Outages and Maintenance": "Downtime alerts, scheduled maintenance, system-wide unavailability",
        "General Inquiry": "Miscellaneous requests that do not fit any specific category above",
        "Human Resources": "Employee-related issues, payroll, benefits, hiring, internal HR policies",
    }
}

PRIORITY_DESCRIPTIONS: Dict[str, str] = {
    "low": "Non-urgent, informational, no immediate business impact. Example: feature suggestions, general questions.",
    "medium": "Moderate impact, workaround exists, respond within standard SLA. Example: minor bugs, non-critical issues.",
    "high": "Critical business impact, no workaround, requires immediate attention. Example: system down, data loss, security breach.",
}


def _build_task_definitions(dataset: BaseDataset) -> str:
    """Build human-readable definitions for all tasks."""
    lines: List[str] = []

    if dataset.classification_tasks:
        lines.append("=== Queue / Category Definitions ===")
        descs = QUEUE_DESCRIPTIONS.get(dataset.name, {})
        for task in dataset.classification_tasks:
            for label in task.labels:
                desc = descs.get(label, "")
                lines.append(f"- {label}: {desc}" if desc else f"- {label}")

    if dataset.ordinal_tasks:
        lines.append("")
        lines.append("=== Priority Definitions ===")
        for task in dataset.ordinal_tasks:
            lines.append(f"{task.name} is ordered from lowest to highest urgency:")
            for label in task.labels:
                desc = PRIORITY_DESCRIPTIONS.get(label, "")
                lines.append(f"  - {label}: {desc}" if desc else f"  - {label}")

    return "\n".join(lines)


def _format_example(example: Dict) -> str:
    """Format a single few-shot example for the prompt."""
    lang = example.get("language", "")
    title = example.get("title", "")
    body = example.get("body", "")
    labels = example.get("labels", {})
    gen = example.get("generation_target", "")

    # Truncate body for prompt brevity
    body_short = body[:200] + "..." if len(body) > 200 else body

    lines = [
        f"Input [Language: {lang} | Subject: {title} | Body: {body_short}]",
        "Output:",
    ]
    output = dict(labels)
    if gen:
        output["preliminary_answer"] = gen
    lines.append(json.dumps(output, ensure_ascii=False))
    return "\n".join(lines)


def build_system_prompt(
    dataset: BaseDataset, few_shot_examples: List[Dict] | None = None
) -> str:
    """Build a rich system prompt with task definitions and examples."""
    parts: List[str] = [
        "You are a multilingual customer support routing assistant.",
        "Your job is to:",
        "1. Read the customer's request carefully",
        "2. Classify it into the most appropriate queue/category",
        "3. Assess urgency level (priority)",
        "4. Draft a brief, polite preliminary reply in the SAME language as the customer",
        "",
        _build_task_definitions(dataset),
        "",
        "=== Output Format ===",
        "Respond ONLY with a valid JSON object. Do not include markdown formatting (no ```json code blocks).",
        "The JSON must have these exact keys:",
    ]

    for task in dataset.classification_tasks:
        parts.append(f"- {task.name}: one of [{', '.join(task.labels)}]")
    for task in dataset.ordinal_tasks:
        parts.append(
            f"- {task.name}: one of [{', '.join(task.labels)}] (ordered lowest to highest)"
        )
    if dataset.generation_task:
        parts.append(
            f"- {dataset.generation_task.name}: a polite, helpful preliminary reply "
            f"in the same language as the customer's request"
        )

    parts.extend(
        [
            "",
            "=== Fairness & Consistency ===",
            "The customer may write in any supported language.",
            "The semantic content of the request — not the language — determines the queue and priority.",
            "Please ensure your classification is consistent across languages.",
            "Requests with similar meaning should receive the same queue and priority regardless of language.",
            "Pay special attention to small queues (e.g., Human Resources, General Inquiry) — do not default to large queues.",
            "",
            "=== Examples ===",
        ]
    )

    if few_shot_examples:
        for ex in few_shot_examples:
            parts.append(_format_example(ex))
            parts.append("")
    else:
        demo = dataset._demo_record()
        parts.append(
            _format_example(
                {
                    "language": demo.sensitive_attributes.get("language", "en"),
                    "title": "Example request title",
                    "body": "Example request body describing the issue.",
                    "labels": demo.labels,
                    "generation_target": demo.generation_target,
                }
            )
        )

    return "\n".join(parts)


def build_user_prompt(record: Record) -> str:
    """Build the user-side prompt from a record."""
    lines: List[str] = []

    lang_val = record.language
    if lang_val is not None:
        # Language is a StrEnum or string
        lang_str = lang_val.value
        lines.append(f"Language: {lang_str}")

    if record.title:
        lines.append(f"Subject: {record.title}")

    lines.append(f"Body: {record.body}")

    return "\n".join(lines)


def build_conversation(
    record: Record,
    dataset: BaseDataset,
    few_shot_examples: List[Dict] | None = None,
) -> List[Dict[str, str]]:
    """Build a conversation list for vLLM llm.chat().

    Returns a list of dicts with 'role' and 'content' keys, compatible with
    vLLM's chat template API.
    """
    system_content = build_system_prompt(dataset, few_shot_examples)
    user_content = build_user_prompt(record)

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def sample_few_shot_examples(
    dataset: BaseDataset,
    train_df: pd.DataFrame,
    max_per_lang: int = 3,
    max_total: int = 12,
    seed: int = SEED,
) -> List[Dict]:
    """Sample stratified few-shot examples from training data."""
    random.seed(seed)

    lang_col = dataset.language_column
    if not lang_col or not dataset.classification_tasks:
        return []

    queue_col = dataset.classification_tasks[0].target_column
    if queue_col not in train_df.columns:
        return []

    examples: List[Dict] = []

    for lang, group in train_df.groupby(lang_col):
        queues = group[queue_col].dropna().unique().tolist()
        if not queues:
            continue

        n_sample = min(max_per_lang, len(queues))
        selected_queues = random.sample(queues, n_sample)

        for q in selected_queues:
            rows = group[group[queue_col] == q]
            if len(rows) == 0:
                continue
            row = rows.sample(1, random_state=seed).iloc[0]

            ex: Dict = {
                "language": str(lang),
                "title": str(row.get(dataset.title_column, ""))
                if dataset.title_column
                else "",
                "body": str(row.get(dataset.body_column, "")),
                "labels": {},
                "generation_target": None,
            }

            for task in dataset.classification_tasks:
                ex["labels"][task.name] = str(row.get(task.target_column, ""))
            for task in dataset.ordinal_tasks:
                ex["labels"][task.name] = str(row.get(task.target_column, ""))
            if dataset.generation_task and dataset.generation_task.target_column:
                ex["generation_target"] = str(
                    row.get(dataset.generation_task.target_column, "")
                )

            examples.append(ex)

    random.shuffle(examples)
    return examples[:max_total]
