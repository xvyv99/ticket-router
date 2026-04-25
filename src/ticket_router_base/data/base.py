"""BaseDataset abstract base class for dataset-agnostic loading and task definition."""

from abc import ABC
from typing import Dict, List, Tuple, ClassVar
from pathlib import Path
from logging import getLogger

from pydantic import BaseModel, model_validator
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from ticket_router_base.config import SEED, OUTPUT_DIR
from ticket_router_base.types import Record, GroundRecord

logger = getLogger(__name__)


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


class BaseDataset(ABC):
    """Abstract base class for datasets.

    Concrete subclasses declare their schema via class attributes.
    The base class provides generic loading, prompt building, and label-mapping logic.
    """

    DELIMITER: str = ","
    ENCODING: str = "utf-8"

    TEST_RATIO: ClassVar[float]
    VALID_RATIO: ClassVar[float]

    # --- schema declarations (subclasses override) ---
    name: ClassVar[str]

    title_column: ClassVar[str | None] = None  # maps to Record.title; None = no title
    body_column: ClassVar[str]  # maps to Record.body; required

    id_column: ClassVar[str | None] = None  # None = auto-generate request_id

    classification_tasks: ClassVar[List[ClassificationTask]]
    ordinal_tasks: ClassVar[List[OrdinalTask]]
    generation_task: ClassVar[GenerationTask | None]

    discrete_feature_columns: ClassVar[List[str]]

    stratified_columns: ClassVar[
        List[str]
    ]  # columns to use for stratified train/test split
    sensitive_columns: ClassVar[List[str]]  # columns to use for fairness evaluation

    def __init_subclass__(cls, skip_check: bool = False) -> None:
        if skip_check:
            return  # skip validation for subclasses that set skip_check=True (e.g. DFDataset which doesn't declare all class vars)

        if not hasattr(cls, "name"):
            raise TypeError(f"{cls.__name__} must define 'name'")

        assert "_" not in cls.name, (
            "Model name cannot contain underscores (used for parsing save file names)"
        )

    def load_df(
        self, dataset_path: Path | None = None, sample_num: int | None = None
    ) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement load_df() method")

    def load(self, dataset_path: Path | None, sample_num: int = 0) -> List[Record]:
        raise NotImplementedError("Subclasses must implement load() method")

    @classmethod
    def _valid_df_columns(cls, df: pd.DataFrame):
        # Must call this in load() to validate the raw DataFrame before processing; it checks that all declared columns exist and that classification tasks have valid labels.

        assert cls.body_column in df.columns, (
            f"Required body column '{cls.body_column}' not found"
        )

        if cls.title_column:
            assert cls.title_column in df.columns, (
                f"Declared title column '{cls.title_column}' not found"
            )
        for task in cls.classification_tasks + cls.ordinal_tasks:
            assert task.target_column in df.columns, (
                f"Declared target column '{task.target_column}' not found"
            )
        if cls.generation_task:
            assert cls.generation_task.target_column in df.columns, (
                f"Declared generation target column '{cls.generation_task.target_column}' not found"
            )

        for col in cls.discrete_feature_columns:
            assert col in df.columns, (
                f"Declared discrete feature column '{col}' not found"
            )
        for col in cls.stratified_columns:
            assert col in df.columns, f"Declared stratified column '{col}' not found"
        for col in cls.sensitive_columns:
            assert col in df.columns, f"Declared sensitive column '{col}' not found"

    @classmethod
    def _init_null_labels(cls, df: pd.DataFrame):
        for task in cls.classification_tasks + cls.ordinal_tasks:
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

    @classmethod
    def df_to_records(cls, df: pd.DataFrame) -> List[Record]:
        """Convert a DataFrame to Record list using this dataset's column mapping."""
        records: List[Record] = []

        for i, row in df.iterrows():
            req_id = (
                str(row[cls.id_column])
                if cls.id_column and cls.id_column in row
                else f"{cls.name}-{i:06d}"
            )

            # text
            title = (
                str(row[cls.title_column])
                if cls.title_column and cls.title_column in row
                else None
            )
            body = str(row[cls.body_column]) if cls.body_column in row else ""

            # classification labels (nominal + ordinal are both stored in labels dict)
            labels: Dict[str, str] = {}
            for task in cls.classification_tasks:
                val = row.get(task.target_column)
                labels[task.name] = str(val) if pd.notna(val) else ""
            for task in cls.ordinal_tasks:
                val = row.get(task.target_column)
                labels[task.name] = str(val) if pd.notna(val) else ""

            # discrete features
            discrete: Dict[str, str | None] = {}
            for col in cls.discrete_feature_columns:
                val = row.get(col)
                discrete[col] = str(val) if pd.notna(val) else None

            # sensitive attributes (for fairness evaluation)
            sensitive: Dict[str, str] = {}
            for col in cls.sensitive_columns:
                val = row.get(col)
                assert pd.notna(val), (
                    f"Sensitive attribute '{col}' cannot be null for record {req_id}"
                )
                # FIXME: Maybe null in some cases? If so, need to decide how to handle nulls in sensitive attributes for fairness eval.
                sensitive[col] = str(val)

            # generation target
            gen_target: str | None = None
            if cls.generation_task:
                val = row.get(cls.generation_task.target_column)
                gen_target = str(val) if pd.notna(val) else None

            rec = Record(
                request_id=req_id,
                title=title,
                body=body,
                labels=labels,
                discrete_features=discrete,
                sensitive_attributes=sensitive,
                generation_target=gen_target,
            )

            records.append(rec)
        return records

    def _demo_record(self) -> GroundRecord:
        """Return a minimal demo record for prompt examples."""
        raise NotImplementedError(
            "Subclasses must implement _demo_record() method for prompt examples"
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

    @classmethod
    def _get_train_test_path(cls) -> Tuple[Path, Path, Path]:
        """Get default train/test split paths for this dataset."""
        train_path = OUTPUT_DIR / f"{cls.name}_train_split.parquet"
        test_path = OUTPUT_DIR / f"{cls.name}_test_split.parquet"
        valid_path = OUTPUT_DIR / f"{cls.name}_valid_split.parquet"
        return train_path, test_path, valid_path

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
                lines.append(ex.model_dump_json(ensure_ascii=False))
        else:
            lines.append(self._demo_record().model_dump_json(ensure_ascii=False))

        return "\n".join(lines)

    def split_train_test_set(
        self,
        df: pd.DataFrame,
        save: bool = False,
        seed: int = SEED,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Stratified split using all classification tasks + language as the stratification key.

        Args:
            records: List of Record instances.
            dataset: Dataset descriptor providing task and column definitions.
            test_num: Number of test samples.
            seed: Random seed.

        Returns:
            train_records: List of training Record instances.
            test_records: List of test Record instances.
            valid_records: List of validation Record instances.
        """
        assert (
            self.stratified_columns is not None and len(self.stratified_columns) > 0
        ), (
            "stratified_columns must be defined with at least one column for stratified splitting"
        )

        for strat_col in self.stratified_columns:
            assert strat_col in df.columns, (
                f"Stratification column '{strat_col}' not found"
            )

        strat_labels = list(
            df[self.stratified_columns].itertuples(index=False, name=None)
        )

        sss1 = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.TEST_RATIO,
            random_state=seed,
        )

        train_valid_idx, test_idx = next(sss1.split(df, strat_labels))

        train_valid_df: pd.DataFrame = df.iloc[train_valid_idx].reset_index(drop=True)
        test_df: pd.DataFrame = df.iloc[test_idx].reset_index(drop=True)

        relative_test_size = self.VALID_RATIO / (1 - self.TEST_RATIO)

        sss2 = StratifiedShuffleSplit(
            n_splits=1, test_size=relative_test_size, random_state=seed
        )
        train_idx, valid_idx = next(
            sss2.split(train_valid_df, train_valid_df[self.stratified_columns])
        )
        train_df: pd.DataFrame = train_valid_df.iloc[train_idx].reset_index(drop=True)
        valid_df: pd.DataFrame = train_valid_df.iloc[valid_idx].reset_index(drop=True)

        if save:
            save_train_path, save_test_path, save_valid_path = (
                self._get_train_test_path()
            )

            train_df.to_parquet(save_train_path, index=False)
            test_df.to_parquet(save_test_path, index=False)
            valid_df.to_parquet(save_valid_path, index=False)

        return train_df, test_df, valid_df

    def load_train_test_split(
        self,
        train_path: Path | None = None,
        test_path: Path | None = None,
        valid_path: Path | None = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load a pre-split train/test set from disk."""

        default_train_path, default_test_path, default_valid_path = (
            self._get_train_test_path()
        )

        if train_path is None:
            train_path = default_train_path
        if test_path is None:
            test_path = default_test_path
        if valid_path is None:
            valid_path = default_valid_path

        if not train_path.exists():
            raise FileNotFoundError(
                f"Train split files not found at {train_path}. Please run split_train_test_set() first to create the splits."
            )

        if not test_path.exists():
            raise FileNotFoundError(
                f"Test split files not found at {test_path}. Please run split_train_test_set() first to create the splits."
            )

        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        valid_df = pd.read_parquet(valid_path)

        return train_df, test_df, valid_df


class DFDataset(BaseDataset, skip_check=True):
    DEFAULT_DATASET_PATH: ClassVar[
        Path
    ]  # subclasses must override with default CSV path

    # df_schema: pa.DataFrameSchema

    def load_df(
        self, dataset_path: Path | None = None, sample_num: int | None = None
    ) -> pd.DataFrame:
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
            if sample_num is not None:
                logger.warning("sample_num is not supported for parquet files")
            df = pd.read_parquet(
                dataset_path,
            )
        else:
            raise ValueError(
                f"Unsupported file format '{dataset_path.suffix}' for dataset '{self.name}'"
            )

        self._valid_df_columns(df)  # validate schema before processing

        self._init_null_labels(
            df
        )  # initialize any missing labels, e.g. by inferring from data

        return df

    def load(
        self, dataset_path: Path | None = None, sample_num: int | None = None
    ) -> List[Record]:
        df = self.load_df(dataset_path, sample_num)

        return self.df_to_records(df)
