"""System prompt builder using dataset descriptors."""

from typing import List

from ticket_router_base.datasets.base import BaseDataset
from ticket_router_base.types import GroundRecord


def build_system_prompt(
    dataset: BaseDataset, few_shot_examples: List[GroundRecord] | None = None
) -> str:
    """Build an Agent system prompt from the dataset's task definitions."""
    lines: List[str] = [
        "You are a customer support assistant. Analyze the user's request and respond "
        "ONLY with a valid JSON object. Do not include markdown formatting outside the JSON.\n",
        "The JSON must have these exact keys:",
    ]
    for task in dataset.classification_tasks:
        lines.append(f"- {task.name}: one of {', '.join(task.labels)}")
    if dataset.generation_task:
        lines.append(
            f"- {dataset.generation_task.name}: a polite, helpful reply in the same language "
            f"as the user's request"
        )

    lines.append("\nExamples:")
    if few_shot_examples:
        for ex in few_shot_examples:
            lines.append(ex.to_json_str())
    else:
        lines.append(_demo_record(dataset).to_json_str())

    return "\n".join(lines)


def build_prompt(
    dataset: BaseDataset,
    title: str,
    body: str,
    language: str | None,
    few_shot_examples: List[GroundRecord] | None = None,
) -> str:
    """Build the full prompt (system + user) for a single request."""
    system = build_system_prompt(dataset, few_shot_examples)
    user_text = f"Title: {title}\nBody: {body}" if title else f"Body: {body}"

    if language:
        user_text = f"Language: {language}\n" + user_text

    return f"{system}\n\n{user_text}"


def _demo_record(dataset: BaseDataset) -> GroundRecord:
    """Return a minimal demo record for prompt examples."""
    labels = {task.name: task.labels[0] for task in dataset.classification_tasks}
    gen = "Thank you for your request. We will get back to you shortly."
    if dataset.generation_task:
        return GroundRecord(
            labels=labels,
            discrete_features={},
            generation_target=gen,
        )
    return GroundRecord(labels=labels, discrete_features={}, generation_target=None)
