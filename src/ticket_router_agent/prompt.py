"""System prompt builder using dataset descriptors."""

import json
import random
from typing import Dict, List

from ticket_router_base.data import BaseDataset
from ticket_router_base.types import Record
from ticket_router_base.config import SEED


def _build_task_definitions(dataset: BaseDataset) -> str:
    """Build human-readable definitions for all tasks."""
    lines: List[str] = []
    descs = dataset.prompt_descriptor.label_descriptions

    if dataset.classification_tasks:
        lines.append("=== Queue / Category Definitions ===")
        for task in dataset.classification_tasks:
            task_descs = descs.get(task.name, {})
            for label in task.labels:
                desc = task_descs.get(label, "")
                lines.append(f"- {label}: {desc}" if desc else f"- {label}")

    if dataset.ordinal_tasks:
        lines.append("")
        lines.append("=== Priority Definitions ===")
        for task in dataset.ordinal_tasks:
            lines.append(f"{task.name} is ordered from lowest to highest urgency:")
            task_descs = descs.get(task.name, {})
            for label in task.labels:
                desc = task_descs.get(label, "")
                lines.append(f"  - {label}: {desc}" if desc else f"  - {label}")

    return "\n".join(lines)


def _format_example(record: Record, dataset: BaseDataset) -> str:
    """Format a single few-shot example for the prompt."""
    lang = record.language.value if record.language is not None else ""
    title = record.title or ""
    body = record.body

    # Truncate body for prompt brevity
    body_short = body[:200] + "..." if len(body) > 200 else body

    lines = [
        f"Input [Language: {lang} | Subject: {title} | Body: {body_short}]",
        "Output:",
    ]
    output = dict(record.labels)
    if record.generation_target and dataset.generation_task:
        output[dataset.generation_task.name] = record.generation_target
    lines.append(json.dumps(output, ensure_ascii=False))
    return "\n".join(lines)


def build_system_prompt(
    dataset: BaseDataset, few_shot_examples: List[Record] | None = None
) -> str:
    """Build a rich system prompt with task definitions and examples."""
    prompt_desc = dataset.prompt_descriptor

    parts: List[str] = [
        prompt_desc.system_role,
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

    if prompt_desc.fairness_notes:
        parts.extend(
            [
                "",
                "=== Fairness & Consistency ===",
                prompt_desc.fairness_notes,
            ]
        )

    parts.extend(
        [
            "",
            "=== Examples ===",
        ]
    )

    if few_shot_examples:
        for ex in few_shot_examples:
            parts.append(_format_example(ex, dataset))
            parts.append("")
    else:
        demo = dataset._demo_record()
        demo_record = Record(
            request_id="demo",
            title="Example request title",
            body="Example request body describing the issue.",
            language=None,
            labels=demo.labels,
            discrete_features=demo.discrete_features,
            sensitive_attributes=demo.sensitive_attributes,
            generation_target=demo.generation_target,
        )
        parts.append(_format_example(demo_record, dataset))

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
    few_shot_examples: List[Record] | None = None,
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
    train_records: List[Record],
    max_per_lang: int = 3,
    max_total: int = 12,
    seed: int = SEED,
) -> List[Record]:
    """Sample stratified few-shot examples from training records.

    Groups by language, then by the first classification task's label,
    and samples up to max_per_lang examples per language.
    """
    random.seed(seed)

    if not dataset.language_column or not dataset.classification_tasks:
        return []

    queue_task_name = dataset.classification_tasks[0].name

    # Group by language -> queue label
    lang_groups: Dict[str, Dict[str, List[Record]]] = {}
    for rec in train_records:
        lang = rec.language.value if rec.language is not None else ""
        queue_label = rec.labels.get(queue_task_name, "")
        if not queue_label:
            continue
        lang_groups.setdefault(lang, {}).setdefault(queue_label, []).append(rec)

    examples: List[Record] = []

    for lang, queue_map in lang_groups.items():
        queues = list(queue_map.keys())
        if not queues:
            continue

        n_sample = min(max_per_lang, len(queues))
        selected_queues = random.sample(queues, n_sample)

        for q in selected_queues:
            candidates = queue_map[q]
            if not candidates:
                continue
            examples.append(random.choice(candidates))

    random.shuffle(examples)
    return examples[:max_total]
