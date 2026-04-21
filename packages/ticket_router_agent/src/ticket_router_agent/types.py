"""Dynamic JSON schema builder for Agent structured output."""

from typing import Dict, Any

from ticket_router_base.data.base import BaseDataset


def build_ticket_schema(dataset: BaseDataset) -> Dict[str, Any]:
    """Build a JSON schema dict for vLLM structured output from a dataset descriptor.

    The schema enforces string enums for every classification task and
    includes the generation task as a free-text field.
    """
    properties: Dict[str, Any] = {}
    required: list[str] = []

    for task in dataset.classification_tasks:
        properties[task.name] = {
            "type": "string",
            "enum": task.labels,
        }
        required.append(task.name)

    # tags (optional auxiliary field)
    properties["tags"] = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 0,
        "maxItems": 3,
    }

    if dataset.generation_task:
        properties[dataset.generation_task.name] = {"type": "string"}
        required.append(dataset.generation_task.name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }
