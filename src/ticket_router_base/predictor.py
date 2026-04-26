"""Predictor and Trainer protocol definitions."""

from __future__ import annotations

from logging import getLogger
from abc import ABC
from typing import List, ClassVar, Type, Dict, TypeVar, Tuple
from pathlib import Path
from dataclasses import dataclass

from .types import Record, Prediction, PredSave
from .data import BaseDataset, get_dataset, DATASET_REGISTRY
from .utils import write_pred, load_pred

logger = getLogger(__name__)

MODEL_REGISTRY: Dict[str, Type[Predictor]] = {}


@dataclass(frozen=True, kw_only=True)
class PredictorKey:
    predictor_name: str
    dataset_name: str
    sub_name: str | None = None
    path: Path


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


def parse_pred_save_name(
    save_name: str,
) -> Tuple[str, str, str | None]:
    """Parse the dataset name and model name from a prediction save file name."""
    parts = save_name.split("_")
    if not save_name.endswith("_preds.jsonl"):
        raise ValueError(
            f"Invalid save name format: {save_name}. Expected format: <model_name>_<dataset_name>_preds.jsonl"
        )

    sub_name = None

    if len(parts) == 3:
        model_name = parts[0]
        dataset_name = parts[1]
    elif len(parts) == 4:
        model_name = parts[0]
        sub_name = parts[1]
        dataset_name = parts[2]
    else:
        raise ValueError(
            f"Invalid save name format: {save_name}. Expected format: <model_name>_<dataset_name>_preds.jsonl or <model_name>_<sub_name>_<dataset_name>_preds.jsonl"
        )

    return model_name, dataset_name, sub_name


def parse_pred_save_name_safe(
    save_name: str,
) -> Tuple[Type[Predictor], Type[BaseDataset], str | None]:
    """Parse the dataset name and model name from a prediction save file name."""
    model_name, dataset_name, sub_name = parse_pred_save_name(save_name)

    return get_model(model_name), get_dataset(dataset_name), sub_name


def scan_pred_saves(
    scan_path: Path | None = None,
) -> List[PredictorKey]:
    results = []

    for name, model_cls in MODEL_REGISTRY.items():
        model_saves = model_cls.scan_pred(save_dir=scan_path)

        logger.debug(
            f"Found {len(model_saves)} saved prediction files for model {name} at {scan_path or model_cls.DEFAULT_SAVE_DIR}"
        )

        for path in model_saves:
            _, dataset, sub_name = parse_pred_save_name(path.name)

            if not model_cls.sub_name_required:
                sub_name = None

            assert dataset in DATASET_REGISTRY.keys(), (
                f"Dataset {dataset} from save name {path.name} not found in registry"
            )

            key = PredictorKey(
                predictor_name=model_cls.name,
                dataset_name=dataset,
                sub_name=sub_name,
                path=path,
            )

            results.append(key)

    return results


class Predictor(ABC):
    name: ClassVar[str]
    DEFAULT_SAVE_DIR: ClassVar[Path]

    sub_name_required: ClassVar[bool] = False
    sub_name: str | None = None  # defined in subclasses instance

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
        # TODO: add multi-run
        raise NotImplementedError

    @classmethod
    def format_pred_save_name(cls, dataset: BaseDataset, sub_name: str | None) -> str:
        if sub_name:
            name = f"{cls.name}_{sub_name}_{dataset.name}_preds.jsonl"
        else:
            name = f"{cls.name}_{dataset.name}_preds.jsonl"

        return name

    @classmethod
    def get_save_path(
        cls, dataset: BaseDataset, sub_name: str | None, save_dir: Path | None = None
    ) -> Path:
        """Generate a prediction save path based on the dataset and model name."""
        formated_name = cls.format_pred_save_name(dataset=dataset, sub_name=sub_name)

        if save_dir is None:
            save_dir = cls.DEFAULT_SAVE_DIR

        return save_dir / formated_name

    @classmethod
    def save_pred(
        cls,
        dataset: BaseDataset,
        preds: List[Prediction],
        records: List[Record],
        sub_name: str | None = None,
        save_path: Path | None = None,
    ) -> None:
        if cls.sub_name_required:
            assert sub_name is not None, (
                "sub_name is required for this model but not set on instance"
            )

        if save_path is None:
            save_path = cls.get_save_path(dataset=dataset, sub_name=sub_name)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        write_pred(preds, records, save_path)

    def save_pred_inst(
        self,
        preds: List[Prediction],
        records: List[Record],
        save_path: Path | None = None,
    ) -> None:
        if self.sub_name_required:
            assert self.sub_name is not None, (
                "sub_name is required for this model but not set on instance"
            )

        if save_path is None:
            save_path = self.get_save_path(dataset=self.dataset, sub_name=self.sub_name)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        write_pred(preds, records, save_path)

    @classmethod
    def load_pred(
        cls, dataset: BaseDataset, sub_name: str | None, save_path: Path | None = None
    ) -> List[PredSave]:
        if cls.sub_name_required:
            assert sub_name is not None, (
                "sub_name is required for this model but not set on instance"
            )

        if save_path is None:
            save_path = cls.get_save_path(dataset=dataset, sub_name=sub_name)

        assert save_path.exists(), (
            f"Prediction file not found at {save_path}. Did you run inference and save predictions first?"
        )

        return load_pred(save_path)

    @classmethod
    def scan_pred(cls, save_dir: Path | None = None) -> List[Path]:
        """Scan a directory for saved prediction files matching this model's naming convention."""

        if save_dir is None:
            save_dir = cls.DEFAULT_SAVE_DIR

        pattern = f"{cls.name}_*_preds.jsonl"
        return list(save_dir.glob(pattern))


class Trainer(ABC):
    dataset: BaseDataset

    def train(
        self,
        records: List[Record],
        val_records: List[Record] | None = None,
    ) -> Predictor:
        raise NotImplementedError
