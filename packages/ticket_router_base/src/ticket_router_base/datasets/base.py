"""BaseDataset abstract base class for dataset-agnostic loading and task definition."""

from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ticket_router_base.types import Record, GroundRecord


@dataclass(frozen=True)
class ClassificationTask:
    """Definition of a single classification task."""

    name: str  # task name, e.g. "queue", "priority", "product"
    target_column: str  # column name in the raw CSV
    labels: List[str]  # ordered list of valid label values


@dataclass(frozen=True)
class GenerationTask:
    """Definition of a single generation (text-output) task."""

    name: str
    target_column: str


class BaseDataset(ABC):
    """Abstract base class for datasets.

    Concrete subclasses declare their schema via class attributes.
    The base class provides generic loading, prompt building, and label-mapping logic.
    """

    # --- schema declarations (subclasses override) ---
    name: str
    csv_path: Path
    delimiter: str = ","
    encoding: str = "utf-8"

    title_column: str | None = None  # maps to Record.title; None = no title
    body_column: str  # maps to Record.body; required
    language_column: str | None = None
    id_column: str | None = None  # None = auto-generate request_id

    classification_tasks: List[ClassificationTask] = []
    generation_task: GenerationTask | None = None
    discrete_feature_columns: List[str] = []

    # --- generic methods ---

    def load(self) -> List[Record]:
        """Load the CSV and return standardized Record instances.

        Subclasses may override for special formats (e.g. large-file sampling,
        nested quotes, filtering empty rows).
        """
        df = pd.read_csv(
            self.csv_path,
            delimiter=self.delimiter,
            encoding=self.encoding,
        )
        return self._df_to_records(df)

    def _df_to_records(self, df: pd.DataFrame) -> List[Record]:
        """Convert a DataFrame to Record list using this dataset's column mapping."""
        records: List[Record] = []
        for i, row in df.iterrows():
            # id
            if self.id_column and self.id_column in row:
                req_id = str(row[self.id_column])
            else:
                req_id = f"{self.name}-{i:06d}"

            # text
            title = (
                str(row[self.title_column])
                if self.title_column and self.title_column in row
                else None
            )
            body = str(row[self.body_column]) if self.body_column in row else ""
            language = (
                str(row[self.language_column])
                if self.language_column and self.language_column in row
                else None
            )

            # classification labels
            labels: Dict[str, str] = {}
            for task in self.classification_tasks:
                val = row.get(task.target_column)
                labels[task.name] = str(val) if pd.notna(val) else ""

            # discrete features
            discrete: Dict[str, str | None] = {}
            for col in self.discrete_feature_columns:
                val = row.get(col)
                discrete[col] = str(val) if pd.notna(val) else None

            # generation target
            gen_target: str | None = None
            if self.generation_task:
                val = row.get(self.generation_task.target_column)
                gen_target = str(val) if pd.notna(val) else None

            records.append(
                Record(
                    request_id=req_id,
                    title=title,
                    body=body,
                    language=language,
                    labels=labels,
                    discrete_features=discrete,
                    generation_target=gen_target,
                )
            )
        return records

    def build_system_prompt(
        self, few_shot_examples: List[GroundRecord] | None = None
    ) -> str:
        """Build an Agent system prompt from this dataset's task definitions.

        Subclasses may override for custom prompt wording.
        """
        lines: List[str] = [
            "You are a customer support assistant. Analyze the user's request and respond "
            "ONLY with a valid JSON object. Do not include markdown formatting outside the JSON.\n",
            "The JSON must have these exact keys:",
        ]
        for task in self.classification_tasks:
            lines.append(f"- {task.name}: one of {', '.join(task.labels)}")
        if self.generation_task:
            lines.append(
                f"- {self.generation_task.name}: a polite, helpful reply in the same language "
                f"as the user's request"
            )

        lines.append("\nExamples:")
        if few_shot_examples:
            for ex in few_shot_examples:
                lines.append(ex.to_json_str())
        else:
            lines.append(self._demo_record().to_json_str())

        return "\n".join(lines)

    def _demo_record(self) -> GroundRecord:
        """Return a minimal demo record for prompt examples."""
        labels = {task.name: task.labels[0] for task in self.classification_tasks}
        return GroundRecord(
            labels=labels,
            discrete_features={},
            generation_target="Thank you for your request. We will get back to you shortly.",
        )

    def get_label2id(self, task_name: str) -> Dict[str, int]:
        """Label -> int mapping for a classification task (used by supervised training)."""
        task = self._get_task(task_name)
        return {label: i for i, label in enumerate(task.labels)}

    def get_id2label(self, task_name: str) -> Dict[int, str]:
        """Int -> label mapping for a classification task (used by supervised inference)."""
        task = self._get_task(task_name)
        return {i: label for i, label in enumerate(task.labels)}

    def get_task_names(self) -> List[str]:
        """Return all classification task names."""
        return [t.name for t in self.classification_tasks]

    def get_task(self, task_name: str) -> ClassificationTask:
        """Look up a classification task by name."""
        for task in self.classification_tasks:
            if task.name == task_name:
                return task
        raise ValueError(f"Task '{task_name}' not found in dataset '{self.name}'")

    # alias for backward compatibility in eval
    _get_task = get_task
