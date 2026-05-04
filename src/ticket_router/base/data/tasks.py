"""Task definition models for dataset descriptors."""

from typing import List

from pydantic import BaseModel, model_validator


class ClassificationTask(BaseModel):
    """Definition of a single classification task."""

    name: str  # task name, e.g. "queue", "priority", "product"
    target_column: str  # column name in the raw CSV
    labels: List[str]  # ordered list of valid label values


class OrdinalTask(ClassificationTask):
    """Definition of a single ordinal (ordered) classification task.

    Labels must be listed in ascending order (lowest to highest).
    """

    @model_validator(mode="after")
    def check_ordinal_labels(self):
        if len(self.labels) < 2:
            raise ValueError(f"Ordinal task '{self.name}' must have at least 2 labels")
        return self


class GenerationTask(BaseModel):
    """Definition of a single generation (text-output) task."""

    name: str
    target_column: str | None
