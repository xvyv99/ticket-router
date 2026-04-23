"""BaseDataset abstract base class for dataset-agnostic loading and task definition."""

from abc import ABC
from dataclasses import dataclass, asdict
from typing import Dict, List
from pathlib import Path
import json
from logging import getLogger

import pandas as pd

from ticket_router_base.types import Record, GroundRecord

logger = getLogger(__name__)


@dataclass(frozen=True)
class ClassificationTask:
    """Definition of a single classification task."""

    name: str  # task name, e.g. "queue", "priority", "product"
    target_column: str  # column name in the raw CSV
    labels: List[str]  # ordered list of valid label values


@dataclass(frozen=True)
class OrdinalTask(ClassificationTask):
    """Definition of a single ordinal (ordered) classification task.

    Labels must be listed in ascending order (lowest to highest).
    """

    def __post_init__(self):
        assert len(self.labels) >= 2, (
            f"Ordinal task '{self.name}' must have at least 2 labels"
        )


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

    DELIMITER: str = ","
    ENCODING: str = "utf-8"

    # --- schema declarations (subclasses override) ---
    name: str

    title_column: str | None = None  # maps to Record.title; None = no title
    body_column: str  # maps to Record.body; required
    language_column: str | None = None
    id_column: str | None = None  # None = auto-generate request_id

    classification_tasks: List[ClassificationTask] = []
    ordinal_tasks: List[OrdinalTask] = []
    generation_task: GenerationTask | None = None

    discrete_feature_columns: List[str] = []

    def load(self, dataset_path: Path | None, sample_num: int = 0) -> List[Record]:
        raise NotImplementedError("Subclasses must implement load() method")

    def _valid_df(self, df: pd.DataFrame):
        # Must call this in load() to validate the raw DataFrame before processing; it checks that all declared columns exist and that classification tasks have valid labels.

        assert self.body_column in df.columns, (
            f"Required body column '{self.body_column}' not found"
        )

        if self.title_column:
            assert self.title_column in df.columns, (
                f"Declared title column '{self.title_column}' not found"
            )
        if self.language_column:
            assert self.language_column in df.columns, (
                f"Declared language column '{self.language_column}' not found"
            )
        for task in self.classification_tasks + self.ordinal_tasks:
            assert task.target_column in df.columns, (
                f"Declared target column '{task.target_column}' not found"
            )
        if self.generation_task:
            assert self.generation_task.target_column in df.columns, (
                f"Declared generation target column '{self.generation_task.target_column}' not found"
            )

        for col in self.discrete_feature_columns:
            assert col in df.columns, (
                f"Declared discrete feature column '{col}' not found"
            )

    def _init_null_labels(self, df: pd.DataFrame):
        for task in self.classification_tasks + self.ordinal_tasks:
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

    def _df_to_records(self, df: pd.DataFrame) -> List[Record]:
        """Convert a DataFrame to Record list using this dataset's column mapping."""
        records: List[Record] = []

        for i, row in df.iterrows():
            req_id = (
                str(row[self.id_column])
                if self.id_column and self.id_column in row
                else f"{self.name}-{i:06d}"
            )

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

            # classification labels (nominal + ordinal are both stored in labels dict)
            labels: Dict[str, str] = {}
            for task in self.classification_tasks:
                val = row.get(task.target_column)
                labels[task.name] = str(val) if pd.notna(val) else ""
            for task in self.ordinal_tasks:
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

            rec = Record(
                request_id=req_id,
                title=title,
                body=body,
                language=language,
                labels=labels,
                discrete_features=discrete,
                generation_target=gen_target,
            )

            records.append(rec)
        return records

    def _demo_record(self) -> GroundRecord:
        """Return a minimal demo record for prompt examples."""
        labels = {task.name: task.labels[0] for task in self.classification_tasks}
        labels.update({task.name: task.labels[0] for task in self.ordinal_tasks})
        return GroundRecord(
            labels=labels,
            discrete_features={},
            generation_target="Thank you for your request. We will get back to you shortly.",
        )

    def _get_task(self, task_name: str) -> ClassificationTask | OrdinalTask:
        """Look up a classification or ordinal task by name."""
        for task in self.classification_tasks:
            if task.name == task_name:
                return task
        for task in self.ordinal_tasks:
            if task.name == task_name:
                return task
        raise ValueError(f"Task '{task_name}' not found in dataset '{self.name}'")

    def get_label2id(self, task_name: str) -> Dict[str, int]:
        """Label -> int mapping for a classification task (used by supervised training)."""
        task = self._get_task(task_name)
        return {label: i for i, label in enumerate(task.labels)}

    def get_id2label(self, task_name: str) -> Dict[int, str]:
        """Int -> label mapping for a classification task (used by supervised inference)."""
        task = self._get_task(task_name)
        return {i: label for i, label in enumerate(task.labels)}

    @property
    def task_names(self) -> List[str]:
        """Return all classification + ordinal task names."""
        return [t.name for t in self.classification_tasks + self.ordinal_tasks]

    @property
    def ord_task_names(self) -> List[str]:
        """Return all ordinal task names."""
        return [t.name for t in self.ordinal_tasks]

    def build_system_prompt(
        self, few_shot_examples: List[GroundRecord] | None = None
    ) -> str:
        """Build an Agent system prompt from this dataset's task definitions.

        Subclasses may override for custom prompt wording.
        """
        # TODO: improve prompt formatting, e.g. clearer instructions, more realistic examples, etc.

        lines: List[str] = [
            "You are a customer support assistant. Analyze the user's request and respond "
            "ONLY with a valid JSON object. Do not include markdown formatting outside the JSON.\n",
            "The JSON must have these exact keys:",
        ]
        for task in self.classification_tasks:
            lines.append(f"- {task.name}: one of {', '.join(task.labels)}")
        for task in self.ordinal_tasks:
            lines.append(f"- {task.name}: one of {', '.join(task.labels)} (ordered)")
        if self.generation_task:
            lines.append(
                f"- {self.generation_task.name}: a polite, helpful reply in the same language "
                f"as the user's request"
            )

        lines.append("\nExamples:")
        if few_shot_examples:
            for ex in few_shot_examples:
                ex_dic = asdict(ex)
                ex_str = json.dumps(ex_dic, ensure_ascii=False)
                lines.append(ex_str)
        else:
            ex_dic = asdict(self._demo_record())
            ex_str = json.dumps(ex_dic, ensure_ascii=False)
            lines.append(ex_str)

        return "\n".join(lines)


class DFDataset(BaseDataset):
    DEFAULT_DATASET_PATH: Path  # subclasses must override with default CSV path

    def load(
        self, dataset_path: Path | None = None, sample_num: int | None = None
    ) -> List[Record]:
        if dataset_path is None:
            dataset_path = self.DEFAULT_DATASET_PATH

        assert sample_num is None or sample_num > 0, (
            "sample_num must be positive or None"
        )
        assert dataset_path.exists(), f"Dataset file not found at {dataset_path}"

        if sample_num is None:
            logger.debug(f"Loading full dataset from {dataset_path}...")
        else:
            assert sample_num > 0, "sample_num must be positive"
            logger.debug(
                f"Loading sample of {sample_num} records from {dataset_path}..."
            )

        if dataset_path.suffix == ".csv":
            df = pd.read_csv(
                dataset_path,
                delimiter=self.DELIMITER,
                encoding=self.ENCODING,
                nrows=sample_num,
            )
        elif dataset_path.suffix == ".parquet":
            df = pd.read_parquet(
                dataset_path,
                encoding=self.ENCODING,
                nrows=sample_num,
            )
        else:
            raise ValueError(
                f"Unsupported file format '{dataset_path.suffix}' for dataset '{self.name}'"
            )

        self._valid_df(df)  # validate schema before processing

        self._init_null_labels(
            df
        )  # initialize any missing labels, e.g. by inferring from data

        return self._df_to_records(df)
