"""System prompt builder using TaskDescriptor."""

import json
import random
from typing import Dict, List

from ticket_router_base.data import BaseDataset
from ticket_router_base.data.desc import TaskDescriptor
from ticket_router_base.types import Record, GroundRecord
from ticket_router_base.config import SEED


def _build_task_definitions(task_descriptor: TaskDescriptor) -> str:
    """Build human-readable definitions for all tasks."""
    lines: List[str] = []
    descs = task_descriptor.prompt_descriptor.label_descriptions

    if task_descriptor.classification_tasks:
        lines.append("=== Queue / Category Definitions ===")
        for task in task_descriptor.classification_tasks:
            task_descs = descs.get(task.name, {})
            for label in task.labels:
                desc = task_descs.get(label, "")
                lines.append(f"- {label}: {desc}" if desc else f"- {label}")

    if task_descriptor.ordinal_tasks:
        lines.append("")
        lines.append("=== Priority Definitions ===")
        for task in task_descriptor.ordinal_tasks:
            lines.append(f"{task.name} is ordered from lowest to highest urgency:")
            task_descs = descs.get(task.name, {})
            for label in task.labels:
                desc = task_descs.get(label, "")
                lines.append(f"  - {label}: {desc}" if desc else f"  - {label}")

    return "\n".join(lines)


def _format_example(record: Record, task_descriptor: TaskDescriptor) -> str:
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
    if record.generation_target and task_descriptor.generation_task:
        output[task_descriptor.generation_task.name] = record.generation_target
    lines.append(json.dumps(output, ensure_ascii=False))
    return "\n".join(lines)


def build_system_prompt(
    task_descriptor: TaskDescriptor,
    few_shot_examples: List[Record] | None = None,
    demo_record: GroundRecord | None = None,
) -> str:
    """Build a rich system prompt with task definitions and examples.

    Args:
        task_descriptor: Task definitions and prompt descriptors.
        few_shot_examples: Optional list of Record examples to include.
        demo_record: Optional demo record used when few_shot_examples is empty.
    """
    pd = task_descriptor.prompt_descriptor

    parts: List[str] = [
        pd.system_role,
        "",
        _build_task_definitions(task_descriptor),
        "",
        "=== Output Format ===",
        "Respond ONLY with a valid JSON object. Do not include markdown formatting (no ```json code blocks).",
        "The JSON must have these exact keys:",
    ]

    for task in task_descriptor.classification_tasks:
        parts.append(f"- {task.name}: one of [{', '.join(task.labels)}]")
    for task in task_descriptor.ordinal_tasks:
        parts.append(
            f"- {task.name}: one of [{', '.join(task.labels)}] (ordered lowest to highest)"
        )
    if task_descriptor.generation_task:
        parts.append(
            f"- {task_descriptor.generation_task.name}: a polite, helpful preliminary reply "
            f"in the same language as the customer's request"
        )

    if pd.fairness_notes:
        parts.extend(
            [
                "",
                "=== Fairness & Consistency ===",
                pd.fairness_notes,
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
            parts.append(_format_example(ex, task_descriptor))
            parts.append("")
    else:
        if demo_record is None:
            # Build a minimal demo from task_descriptor defaults
            labels = {task.name: task.labels[0] for task in task_descriptor.classification_tasks}
            labels.update({task.name: task.labels[0] for task in task_descriptor.ordinal_tasks})
            demo_record = GroundRecord(
                labels=labels,
                discrete_features={},
                generation_target="Thank you for your request. We will get back to you shortly.",
                sensitive_attributes={},
            )
        demo_full_record = Record(
            request_id="demo",
            title="Example request title",
            body="Example request body describing the issue.",
            language=None,
            labels=demo_record.labels,
            discrete_features=demo_record.discrete_features,
            sensitive_attributes=demo_record.sensitive_attributes,
            generation_target=demo_record.generation_target,
        )
        parts.append(_format_example(demo_full_record, task_descriptor))

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
    system_content = build_system_prompt(
        dataset.task_descriptor,
        few_shot_examples,
        dataset._demo_record(),
    )
    user_content = build_user_prompt(record)

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def sample_few_shot_examples(
    task_descriptor: TaskDescriptor,
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

    if not task_descriptor.classification_tasks:
        return []

    queue_task_name = task_descriptor.classification_tasks[0].name

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
