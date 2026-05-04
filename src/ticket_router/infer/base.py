"""Base classes and registry for attribute inferrers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Literal, Type

from ticket_router.base.types import Record

from .schema import AttributePrediction


_INFERRER_REGISTRY: Dict[str, Type[AttributeInferrer]] = {}


def register_inferrer(dataset_name: str):
    """Decorator to register an attribute inferrer for a dataset."""

    def decorator(cls: type):
        if dataset_name in _INFERRER_REGISTRY:
            raise ValueError(
                f"Inferrer for dataset '{dataset_name}' already registered"
            )
        _INFERRER_REGISTRY[dataset_name] = cls
        return cls

    return decorator


def get_inferrer(dataset_name: str) -> AttributeInferrer:
    """Get an instance of the registered inferrer for a dataset."""
    if dataset_name not in _INFERRER_REGISTRY:
        raise ValueError(
            f"No inferrer registered for dataset '{dataset_name}'. "
            f"Available: {list(_INFERRER_REGISTRY.keys())}"
        )
    return _INFERRER_REGISTRY[dataset_name]()


class AttributeInferrer(ABC):
    """Abstract base class for attribute inference."""

    dataset_name: ClassVar[str]  # must be set by subclasses

    @property
    def output_schema(self) -> Dict[str, Any]:
        """JSON schema for structured output."""
        return AttributePrediction.model_json_schema()

    @abstractmethod
    def build_system_prompt(self) -> str:
        """Build the system prompt for inference."""
        pass

    @abstractmethod
    def build_user_prompt(self, record: Record) -> str:
        """Build the user prompt from a record."""
        pass

    def build_conversation(self, record: Record) -> List[Dict[str, str]]:
        """Build the full conversation for vLLM chat API."""
        return [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": self.build_user_prompt(record)},
        ]

    @abstractmethod
    def parse_output(self, raw: str, request_id: str) -> AttributePrediction:
        """Parse raw LLM output into AttributePrediction."""
        pass

    def load_records(
        self,
        dataset_path: Path | None = None,
        limit: int | None = None,
        split: Literal["train", "test", "valid"] = "test",
    ) -> List[Record]:
        """Load records from the dataset.

        Args:
            dataset_path: Optional path to CSV file (overrides default).
            limit: Optional limit on number of records.
            split: Which split to use (train/test/valid).

        Returns:
            List of Record objects.
        """
        from ticket_router.base.data import get_dataset

        dataset_type = get_dataset(self.dataset_name)
        dataset = dataset_type()

        if dataset_path and str(dataset_path).endswith(".csv"):
            df = dataset.load_df(dataset_path)
            records = dataset.df_to_records(df)
        else:
            df_train, df_test, df_valid = dataset.load_train_test_split()
            if split == "train":
                df = df_train
            elif split == "valid":
                df = df_valid
            else:
                df = df_test
            records = dataset.df_to_records(df)

        if limit:
            records = records[:limit]

        return records

    def save_predictions(
        self,
        predictions: List[AttributePrediction],
        output_path: Path,
    ) -> None:
        """Save predictions to JSONL file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(pred.model_dump_json(ensure_ascii=False) + "\n")
