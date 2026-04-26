"""Predictor and Trainer protocol definitions."""

from __future__ import annotations

from logging import getLogger
from abc import ABC
from typing import List, ClassVar, Type, Dict, TypeVar, Any, Generic
from pathlib import Path
from dataclasses import dataclass
import json

from .types import Record, Prediction, PredSave
from .data import BaseDataset, DATASET_REGISTRY
from .utils import write_pred, load_pred
from .cfg import Cfg

logger = getLogger(__name__)

MODEL_REGISTRY: Dict[str, Type[Predictor]] = {}


@dataclass(frozen=True, kw_only=True)
class PredictorKey:
    predictor_name: str
    dataset_name: str
    sub_name: str | None = None
    run_id: int = 0
    cfg_id: str | None = None
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


def update_index_json(save_dir: Path, cfg: Cfg) -> None:
    """Write or update index.json in save_dir with cfg_id -> cfg mapping."""
    index_path = save_dir / "index.json"
    data: dict[str, dict[str, Any]] = {}

    cfg_id = cfg.cfg_id()

    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    if cfg_id not in data:
        data[cfg_id] = cfg.to_dict()
        with index_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def load_index_json(save_dir: Path) -> dict[str, dict[str, Any]]:
    """Load index.json from save_dir, returning empty dict if absent."""
    index_path = save_dir / "index.json"
    if not index_path.exists():
        return {}
    with index_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_pred_path(path: Path, base_dir: Path) -> PredictorKey:
    """Parse a prediction file path into a PredictorKey.

    Expected filenames:
        preds_{run_id}_{cfg_id}.jsonl  (new, with cfg_id)
        preds_{run_id}.jsonl           (legacy, no cfg_id)
    """
    rel = path.relative_to(base_dir)
    parts = rel.parts
    if len(parts) != 4:
        raise ValueError(
            f"Invalid prediction path structure: {rel}. Expected: model/sub/dataset/preds_*.jsonl"
        )

    model_name = parts[0]
    sub_name = parts[1] if parts[1] != "_" else None
    dataset_name = parts[2]
    filename = parts[3]

    stem = filename.replace(".jsonl", "")
    if not stem.startswith("preds_"):
        raise ValueError(
            f"Invalid prediction filename: {filename}. Expected preds_*.jsonl"
        )

    # Parse: preds_0  or  preds_0_a3f7c2d1
    tokens = stem.split("_")
    if len(tokens) == 2:  # preds_run_id (legacy)
        run_id = int(tokens[1])
        cfg_id = None
    elif len(tokens) == 3:  # preds_run_id_cfg_id
        run_id = int(tokens[1])
        cfg_id = tokens[2]
    else:
        raise ValueError(f"Invalid prediction filename: {filename}")

    return PredictorKey(
        predictor_name=model_name,
        dataset_name=dataset_name,
        sub_name=sub_name,
        run_id=run_id,
        cfg_id=cfg_id,
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
                    cfg_id=key.cfg_id,
                    path=key.path,
                )

            assert key.dataset_name in DATASET_REGISTRY.keys(), (
                f"Dataset {key.dataset_name} from path {path} not found in registry"
            )

            results.append(key)

    return results


CfgT = TypeVar("CfgT", bound=Cfg)


class Predictor(ABC, Generic[CfgT]):
    name: ClassVar[str]
    DEFAULT_SAVE_DIR: ClassVar[Path]

    sub_name_required: ClassVar[bool] = False
    sub_name: str | None = None  # defined in subclasses instance

    dataset: BaseDataset
    cfg: CfgT | None = None

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
        cfg_id: str | None = None,
        save_dir: Path | None = None,
    ) -> Path:
        """Generate a prediction save path based on directory hierarchy.

        Structure:
            {save_dir}/{model_name}/{sub_or_}/{dataset_name}/preds_{run_id}_{cfg_id}.jsonl
            {save_dir}/{model_name}/{sub_or_}/{dataset_name}/preds_{run_id}.jsonl  (legacy)
        """
        base = (save_dir or cls.DEFAULT_SAVE_DIR) / cls.name
        sub = sub_name if sub_name else "_"
        base = base / sub / dataset.name
        if cfg_id is not None:
            return base / f"preds_{run_id}_{cfg_id}.jsonl"
        return base / f"preds_{run_id}.jsonl"

    @classmethod
    def save_pred(
        cls,
        dataset: BaseDataset,
        preds: List[Prediction],
        records: List[Record],
        sub_name: str | None = None,
        run_id: int = 0,
        cfg: Cfg | None = None,
        save_path: Path | None = None,
    ) -> None:
        if cls.sub_name_required:
            assert sub_name is not None, (
                "sub_name is required for this model but not set on instance"
            )

        cfg_id = cfg.cfg_id() if cfg is not None else None

        if save_path is None:
            save_path = cls.get_save_path(
                dataset=dataset, sub_name=sub_name, run_id=run_id, cfg_id=cfg_id
            )
        save_path.parent.mkdir(parents=True, exist_ok=True)

        write_pred(preds, records, save_path)

        # Update index.json if cfg is provided
        if cfg is not None:
            update_index_json(save_path.parent, cfg)

    def save_pred_inst(
        self,
        preds: List[Prediction],
        records: List[Record],
        run_id: int = 0,
        save_path: Path | None = None,
    ) -> None:
        self.save_pred(
            dataset=self.dataset,
            preds=preds,
            records=records,
            sub_name=self.sub_name,
            run_id=run_id,
            cfg=self.cfg,
            save_path=save_path,
        )

    @classmethod
    def load_pred(
        cls,
        dataset: BaseDataset,
        sub_name: str | None = None,
        run_id: int = 0,
        cfg_id: str | None = None,
        save_path: Path | None = None,
    ) -> List[PredSave]:
        if cls.sub_name_required:
            assert sub_name is not None, (
                "sub_name is required for this model but not set on instance"
            )

        if save_path is None:
            save_path = cls.get_save_path(
                dataset=dataset, sub_name=sub_name, run_id=run_id, cfg_id=cfg_id
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
