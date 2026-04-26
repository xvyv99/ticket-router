"""PromptDescriptor dataclass for dataset-agnostic prompt building.

Encapsulates all descriptive text needed to build LLM system prompts
(role, label semantics, fairness notes) without hard-coding dataset specifics.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class PromptDescriptor:
    """Descriptor for prompt text that varies by dataset.

    Attributes:
        system_role: The system-role paragraph at the top of the prompt.
        label_descriptions: Nested dict mapping task_name -> label -> description.
            Example: {"queue": {"Technical Support": "Software bugs..."}}
        fairness_notes: Optional paragraph on fairness / consistency expectations.
    """

    system_role: str
    label_descriptions: Dict[str, Dict[str, str]] = field(default_factory=dict)
    fairness_notes: str | None = None
