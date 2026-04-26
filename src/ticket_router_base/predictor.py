"""Predictor and Trainer protocol definitions."""

from __future__ import annotations

from logging import getLogger
from abc import ABC
from typing import List, ClassVar, Type, Dict, TypeVar
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
    run_id: int = 0
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


def parse_pred_path(path: Path, base_dir: Path) -> PredictorKey:
    """Parse a prediction file path into a PredictorKey.

    Expected structure under base_dir:
        {model_name}/{sub_name_or_}/{dataset_name}/preds_{run_id}.jsonl
    """
    rel = path.relative_to(base_dir)
    parts = rel.parts
    if len(parts) != 4:
        raise ValueError(
            f"Invalid prediction path structure: {rel}. Expected: model/sub/dataset/preds_N.jsonl"
        )

    model_name = parts[0]
    sub_name = parts[1] if parts[1] != "_" else None
    dataset_name = parts[2]
    filename = parts[3]

    # e.g. "preds_0.jsonl" -> run_id 0
    stem = filename.replace(".jsonl", "")
    if not stem.startswith("preds_"):
        raise ValueError(f"Invalid prediction filename: {filename}. Expected preds_N.jsonl")
    run_id = int(stem.split("_", 1)[1])

    return PredictorKey(
        predictor_name=model_name,
        dataset_name=dataset_name,
        sub_name=sub_name,
        run_id=run_id,
        path=path,
    )


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
            try:
                key = parse_pred_path(path, scan_path or model_cls.DEFAULT_SAVE_DIR)
            except ValueError as e:
                logger.warning(f"Skipping malformed prediction path {path}: {e}")
                continue

            if not model_cls.sub_name_required:
                key = PredictorKey(
                    predictor_name=key.predictor_name,
                    dataset_name=key.dataset_name,
                    sub_name=None,
                    run_id=key.run_id,
                    path=key.path,
                )

            assert key.dataset_name in DATASET_REGISTRY.keys(), (
                f"Dataset {key.dataset_name} from path {path} not found in registry"
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

        if not hasattr(cls, "DEFAULT_SAVE_DIR"):
            raise TypeError(f"{cls.__name__} must define 'DEFAULT_SAVE_DIR'")

    def predict(self, records: List[Record], run_id: int = 0) -> List[Prediction]:
        """Single-run prediction. Subclasses may use run_id to vary seeds/temperature."""
        raise NotImplementedError

    def predict_multi(
        self, records: List[Record], n_runs: int = 1
    ) -> List[List[Prediction]]:
        """Run predict() n_runs times, returning a list of prediction batches."""
        return [self.predict(records, run_id=i) for i in range(n_runs)]

    @classmethod
    def get_save_path(
        cls,
        dataset: BaseDataset,
        sub_name: str | None = None,
        run_id: int = 0,
        save_dir: Path | None = None,
    ) -> Path:
        """Generate a prediction save path based on directory hierarchy.

        Structure: {save_dir}/{model_name}/{sub_or_}/{dataset_name}/preds_{run_id}.jsonl
        """
        base = (save_dir or cls.DEFAULT_SAVE_DIR) / cls.name
        sub = sub_name if sub_name else "_"
        base = base / sub / dataset.name
        return base / f"preds_{run_id}.jsonl"

    @classmethod
    def save_pred(
        cls,
        dataset: BaseDataset,
        preds: List[Prediction],
        records: List[Record],
        sub_name: str | None = None,
        run_id: int = 0,
        save_path: Path | None = None,
    ) -> None:
        if cls.sub_name_required:
            assert sub_name is not None, (
                "sub_name is required for this model but not set on instance"
            )

        if save_path is None:
            save_path = cls.get_save_path(
                dataset=dataset, sub_name=sub_name, run_id=run_id
            )
        save_path.parent.mkdir(parents=True, exist_ok=True)

        write_pred(preds, records, save_path)

    def save_pred_inst(
        self,
        preds: List[Prediction],
        records: List[Record],
        run_id: int = 0,
        save_path: Path | None = None,
    ) -> None:
        if self.sub_name_required:
            assert self.sub_name is not None, (
                "sub_name is required for this model but not set on instance"
            )

        if save_path is None:
            save_path = self.get_save_path(
                dataset=self.dataset, sub_name=self.sub_name, run_id=run_id
            )
        save_path.parent.mkdir(parents=True, exist_ok=True)

        write_pred(preds, records, save_path)

    @classmethod
    def load_pred(
        cls,
        dataset: BaseDataset,
        sub_name: str | None = None,
        run_id: int = 0,
        save_path: Path | None = None,
    ) -> List[PredSave]:
        if cls.sub_name_required:
            assert sub_name is not None, (
                "sub_name is required for this model but not set on instance"
            )

        if save_path is None:
            save_path = cls.get_save_path(
                dataset=dataset, sub_name=sub_name, run_id=run_id
            )

        assert save_path.exists(), (
            f"Prediction file not found at {save_path}. Did you run inference and save predictions first?"
        )

        return load_pred(save_path)

    @classmethod
    def scan_pred(cls, save_dir: Path | None = None) -> List[Path]:
        """Scan a directory for saved prediction files matching this model's naming convention."""
        if save_dir is None:
            save_dir = cls.DEFAULT_SAVE_DIR

        return list((save_dir / cls.name).rglob("preds_*.jsonl"))


class Trainer(ABC):
    dataset: BaseDataset

    def train(
        self,
        records: List[Record],
        val_records: List[Record] | None = None,
    ) -> Predictor:
        raise NotImplementedError
