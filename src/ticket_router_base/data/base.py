"""BaseDataset abstract base class for dataset-agnostic loading and task definition."""

from abc import ABC
from typing import Dict, List, Tuple, ClassVar
from pathlib import Path
from logging import getLogger

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from ticket_router_base.config import SEED, OUTPUT_DIR
from ticket_router_base.types import Record, GroundRecord, Language
from .desc import PromptDescriptor, TaskDescriptor
from .tasks import ClassificationTask, OrdinalTask

logger = getLogger(__name__)


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
    language_column: ClassVar[str | None] = (
        None  # optional column for language (used in prompts and stratification)
    )

    str2lang: Dict[str, Language]
    DEFAULT_LANGUAGE: ClassVar[Language]  # used if language_column is not defined

    id_column: ClassVar[str | None] = None  # None = auto-generate request_id

    task_descriptor: ClassVar[TaskDescriptor]
    prompt_descriptor: ClassVar[PromptDescriptor]

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
        if cls.language_column:
            assert cls.language_column in df.columns, (
                f"Declared language column '{cls.language_column}' not found"
            )

        cls.task_descriptor._valid_df_columns(df)

        for col in cls.discrete_feature_columns:
            assert col in df.columns, (
                f"Declared discrete feature column '{col}' not found"
            )
        for col in cls.stratified_columns:
            assert col in df.columns, f"Declared stratified column '{col}' not found"
        for col in cls.sensitive_columns:
            assert col in df.columns, f"Declared sensitive column '{col}' not found"

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
            td = cls.task_descriptor
            for task in td.classification_tasks:
                val = row.get(task.target_column)
                labels[task.name] = str(val) if pd.notna(val) else ""
            for task in td.ordinal_tasks:
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
            if td.generation_task:
                val = row.get(td.generation_task.target_column)
                gen_target = str(val) if pd.notna(val) else None

            language = None

            if cls.language_column:
                lang_str = row[cls.language_column]

                if pd.notna(lang_str):
                    language = cls.str2lang.get(lang_str)
                    assert language is not None, (
                        f"Unrecognized language '{lang_str}' in record {req_id}. Valid languages are: {list(cls.str2lang.keys())}"
                    )

            rec = Record(
                request_id=req_id,
                title=title,
                body=body,
                language=language,
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
        return self.task_descriptor.get_task(task_name)

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
        return self.task_descriptor.get_task_names()

    @property
    def ord_task_names(self) -> List[str]:
        """Return all ordinal task names."""
        return self.task_descriptor.get_ord_task_names()

    @property
    def all_tasks(self) -> List[ClassificationTask | OrdinalTask]:
        """Return all classification and ordinal tasks."""
        return self.task_descriptor.all_tasks

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
        # TODO: remove it
        """Build an Agent system prompt from this dataset's task definitions.

        Subclasses may override for custom prompt wording.
        """
        # TODO: improve prompt formatting, e.g. clearer instructions, more realistic examples, etc.

        lines: List[str] = [
            "You are a customer support assistant. Analyze the user's request and respond "
            "ONLY with a valid JSON object. Do not include markdown formatting outside the JSON.\n",
            "The JSON must have these exact keys:",
        ]
        td = self.task_descriptor
        for task in td.classification_tasks:
            lines.append(f"- {task.name}: one of {', '.join(task.labels)}")
        for task in td.ordinal_tasks:
            lines.append(f"- {task.name}: one of {', '.join(task.labels)} (ordered)")
        if td.generation_task:
            lines.append(
                f"- {td.generation_task.name}: a polite, helpful reply in the same language "
                f"as the user's request"
            )

        lines.append("\nExamples:")
        if few_shot_examples:
            for ex in few_shot_examples:
                lines.append(ex.model_dump_json(ensure_ascii=False))
        else:
            lines.append(self._demo_record().model_dump_json(ensure_ascii=False))

        return "\n".join(lines)

    def sample_few_shot_examples(
        self,
        max_per_stratum: int = 3,
        max_total: int = 12,
        seed: int = SEED,
    ) -> List[Record]:
        """Sample stratified few-shot examples from training data.

        Groups training data by stratified_columns, samples up to
        max_per_stratum records from each group using pandas, then
        converts the result to Record objects.

        Subclasses may override this method to provide custom sampling strategies.
        """
        train_df, _, _ = self.load_train_test_split()

        if train_df.empty:
            return []

        # Group by stratified columns and sample from each group via pandas
        sampled_dfs: List[pd.DataFrame] = []
        for _, group in train_df.groupby(self.stratified_columns):
            n = min(max_per_stratum, len(group))
            if n > 0:
                sampled = group.sample(n=n, random_state=seed)
                sampled_dfs.append(sampled)

        if not sampled_dfs:
            logger.warning(
                "No few-shot examples sampled; all groups are empty. Check your dataset and stratified_columns."
            )
            return []

        result_df = pd.concat(sampled_dfs)
        # Shuffle and limit total count
        result_df = result_df.sample(frac=1, random_state=seed).head(max_total)

        return self.df_to_records(result_df)

    @classmethod
    def split_train_test_set(
        cls,
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
        assert cls.stratified_columns is not None and len(cls.stratified_columns) > 0, (
            "stratified_columns must be defined with at least one column for stratified splitting"
        )

        for strat_col in cls.stratified_columns:
            assert strat_col in df.columns, (
                f"Stratification column '{strat_col}' not found"
            )

        strat_labels = list(
            df[cls.stratified_columns].itertuples(index=False, name=None)
        )

        sss1 = StratifiedShuffleSplit(
            n_splits=1,
            test_size=cls.TEST_RATIO,
            random_state=seed,
        )

        train_valid_idx, test_idx = next(sss1.split(df, strat_labels))

        train_valid_df: pd.DataFrame = df.iloc[train_valid_idx].reset_index(drop=True)
        test_df: pd.DataFrame = df.iloc[test_idx].reset_index(drop=True)

        relative_test_size = cls.VALID_RATIO / (1 - cls.TEST_RATIO)

        sss2 = StratifiedShuffleSplit(
            n_splits=1, test_size=relative_test_size, random_state=seed
        )
        train_idx, valid_idx = next(
            sss2.split(train_valid_df, train_valid_df[cls.stratified_columns])
        )
        train_df: pd.DataFrame = train_valid_df.iloc[train_idx].reset_index(drop=True)
        valid_df: pd.DataFrame = train_valid_df.iloc[valid_idx].reset_index(drop=True)

        if save:
            save_train_path, save_test_path, save_valid_path = (
                cls._get_train_test_path()
            )

            train_df.to_parquet(save_train_path, index=False)
            test_df.to_parquet(save_test_path, index=False)
            valid_df.to_parquet(save_valid_path, index=False)

        return train_df, test_df, valid_df

    @classmethod
    def load_train_test_split(
        cls,
        train_path: Path | None = None,
        test_path: Path | None = None,
        valid_path: Path | None = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load a pre-split train/test set from disk."""

        default_train_path, default_test_path, default_valid_path = (
            cls._get_train_test_path()
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

        self.task_descriptor._init_null_labels(
            df
        )  # initialize any missing labels, e.g. by inferring from data

        return df

    def load(
        self, dataset_path: Path | None = None, sample_num: int | None = None
    ) -> List[Record]:
        df = self.load_df(dataset_path, sample_num)

        return self.df_to_records(df)
