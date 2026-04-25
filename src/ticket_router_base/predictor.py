"""Predictor and Trainer protocol definitions."""

from __future__ import annotations

from logging import getLogger
from abc import ABC
from typing import List, ClassVar, Tuple, Type, Dict, TypeVar
from pathlib import Path

from ticket_router_base.data.datasets import get_dataset

from .types import Record, Prediction, PredSave
from .data import BaseDataset
from .utils import write_pred, load_pred

logger = getLogger(__name__)

MODEL_REGISTRY: Dict[str, Type[Predictor]] = {}

T = TypeVar("T", bound=type)
# Use this to be type-preserving in subclass registration, e.g. @register_model


def register_model(cls: T) -> T:
    assert issubclass(cls, Predictor), "Can only register subclasses of Predictor"

    model_name = cls.name
    if model_name in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} already registered")
    MODEL_REGISTRY[model_name] = cls

    logger.debug(f"Registered model {model_name} with class {cls.__name__}")

    return cls


def get_model(name: str) -> Type[Predictor]:
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model {name} not found. Available models: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name]


class Predictor(ABC):
    name: ClassVar[str]
    DEFAULT_SAVE_DIR: ClassVar[Path]

    dataset: BaseDataset

    def __init_subclass__(cls) -> None:
        if not hasattr(cls, "name"):
            raise TypeError(f"{cls.__name__} must define 'name'")

        assert "_" not in cls.name, (
            "Model name cannot contain underscores (used for parsing save file names)"
        )

        if not hasattr(cls, "DEFAULT_SAVE_DIR"):
            raise TypeError(f"{cls.__name__} must define 'DEFAULT_SAVE_DIR'")

    def predict(self, records: List[Record]) -> List[Prediction]:
        raise NotImplementedError

    @classmethod
    def format_pred_savea_name(cls, dataset: BaseDataset) -> str:
        return f"{cls.name}_{dataset.name}_preds.jsonl"

    @staticmethod
    def parse_pred_save_name(
        save_name: str,
    ) -> Tuple[Type[Predictor], Type[BaseDataset]]:
        """Parse the dataset name and model name from a prediction save file name."""
        parts = save_name.split("_")
        if not save_name.endswith("_preds.jsonl"):
            raise ValueError(
                f"Invalid save name format: {save_name}. Expected format: <model_name>_<dataset_name>_preds.jsonl"
            )
        model_name = parts[0]
        dataset_name = parts[1]

        return get_model(model_name), get_dataset(dataset_name)

    @classmethod
    def get_save_path(cls, dataset: BaseDataset, save_dir: Path | None = None) -> Path:
        """Generate a prediction save path based on the dataset and model name."""
        formated_name = cls.format_pred_savea_name(dataset=dataset)

        if save_dir is None:
            save_dir = cls.DEFAULT_SAVE_DIR

        return save_dir / formated_name

    @classmethod
    def save_pred(
        cls,
        dataset: BaseDataset,
        preds: List[Prediction],
        records: List[Record],
        save_path: Path | None = None,
    ) -> None:
        if save_path is None:
            save_path = cls.get_save_path(dataset=dataset)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        write_pred(preds, records, save_path)

    @classmethod
    def load_pred(
        cls, dataset: BaseDataset, save_path: Path | None = None
    ) -> List[PredSave]:
        if save_path is None:
            save_path = cls.get_save_path(dataset=dataset)

        assert save_path.exists(), (
            f"Prediction file not found at {save_path}. Did you run inference and save predictions first?"
        )

        return load_pred(save_path)


class Trainer(ABC):
    dataset: BaseDataset

    def train(
        self,
        records: List[Record],
        val_records: List[Record] | None = None,
    ) -> Predictor:
        raise NotImplementedError
