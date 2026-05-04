"""TaskDescriptor dataclass for dataset-agnostic task definitions."""

from dataclasses import dataclass, field
from typing import List, Dict
from logging import getLogger

import pandas as pd

from .tasks import ClassificationTask, OrdinalTask, GenerationTask

logger = getLogger(__name__)


@dataclass(frozen=True)
class PromptDescriptor:
    """Descriptor for prompt text that varies by dataset.

    Attributes:
        system_role: The system-role paragraph at the top of the prompt.
        label_descriptions: Nested dict mapping task_name -> label -> description.
            Example: {"queue": {"Technical Support": "Software bugs..."}}
        fairness_notes: Optional paragraph on fairness / consistency expectations.
    """

    system_role: str = (
        "You are a customer support assistant. Analyze the user's request and respond "
        "ONLY with a valid JSON object. Do not include markdown formatting outside the JSON."
    )
    label_descriptions: Dict[str, Dict[str, str]] = field(default_factory=dict)
    fairness_notes: str | None = None


@dataclass(frozen=True)
class TaskDescriptor:
    """Lightweight descriptor holding all task definitions for a dataset.

    Decoupled from BaseDataset so that prompt builders, schema builders,
    and output parsers can depend on it directly without pulling in data I/O.
    """

    classification_tasks: List[ClassificationTask] = field(default_factory=list)
    ordinal_tasks: List[OrdinalTask] = field(default_factory=list)
    generation_task: GenerationTask | None = None

    def get_task(self, task_name: str) -> ClassificationTask | OrdinalTask:
        """Look up a classification or ordinal task by name."""
        for task in self.classification_tasks:
            if task.name == task_name:
                return task
        for task in self.ordinal_tasks:
            if task.name == task_name:
                return task
        raise ValueError(f"Task '{task_name}' not found'")

    @property
    def all_tasks(self) -> List[ClassificationTask | OrdinalTask]:
        """Return all classification and ordinal tasks."""
        return self.classification_tasks + self.ordinal_tasks

    def get_task_names(self) -> List[str]:
        """Return all classification + ordinal task names."""
        return [t.name for t in self.all_tasks]

    def get_ord_task_names(self) -> List[str]:
        """Return all ordinal task names."""
        return [t.name for t in self.ordinal_tasks]

    def _valid_df_columns(self, df: pd.DataFrame):
        """Check that all target columns exist in the given DataFrame."""
        for task in self.classification_tasks + self.ordinal_tasks:
            assert task.target_column in df.columns, (
                f"Declared target column '{task.target_column}' not found"
            )
        if self.generation_task:
            assert self.generation_task.target_column in df.columns, (
                f"Declared generation target column '{self.generation_task.target_column}' not found"
            )

    def _init_null_labels(self, df: pd.DataFrame):
        for task in self.all_tasks:
            assert len(task.labels) != 1, (
                f"Task '{task.name}' has only one label '{task.labels[0]}'; this is likely an error. Please check your dataset and task definitions."
            )

            if len(task.labels) == 0:
                # if no labels were declared, infer them from the data
                unique_labels = df[task.target_column].dropna().unique()
                task.labels.extend(sorted(unique_labels.astype(str)))

                logger.debug(f"Inferred labels for task '{task.name}': {task.labels}")
                logger.debug(
                    f"Label counts for task '{task.name}':\n{df[task.target_column].value_counts()}"
                )
