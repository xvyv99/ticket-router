"""System prompt builder using TaskDescriptor and PromptDescriptor."""

import json
from typing import Dict, List

from ticket_router_base.data import BaseDataset
from ticket_router_base.data.desc import TaskDescriptor, PromptDescriptor
from ticket_router_base.types import Record, GroundRecord


def _build_task_definitions(
    task_descriptor: TaskDescriptor, prompt_descriptor: PromptDescriptor
) -> str:
    """Build human-readable definitions for all tasks."""
    lines: List[str] = []
    descs = prompt_descriptor.label_descriptions

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
    prompt_descriptor: PromptDescriptor,
    few_shot_examples: List[Record] | None = None,
    demo_record: GroundRecord | None = None,
) -> str:
    """Build a rich system prompt with task definitions and examples.

    Args:
        task_descriptor: Task definitions (classification, ordinal, generation).
        prompt_descriptor: Prompt text descriptors (role, label descriptions, fairness notes).
        few_shot_examples: Optional list of Record examples to include.
        demo_record: Optional demo record used when few_shot_examples is empty.
    """
    pd = prompt_descriptor

    parts: List[str] = [
        pd.system_role,
        "",
        _build_task_definitions(task_descriptor, prompt_descriptor),
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
        dataset.prompt_descriptor,
        few_shot_examples,
        dataset._demo_record(),
    )
    user_content = build_user_prompt(record)

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]



