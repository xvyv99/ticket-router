"""Predictor and Trainer protocol definitions."""

from abc import ABC
from typing import List
from pathlib import Path

from .types import Record, Prediction, PredSave
from .data import BaseDataset
from .utils import write_pred, load_pred


class Predictor(ABC):
    name: str  # Note: must be defined by subclass
    dataset: BaseDataset

    DEFAULT_SAVE_DIR: Path  # Note: must be defined by subclass

    def __init__(self, dataset: BaseDataset) -> None:
        assert self.name is not None, "Predictor must have a name"
        assert self.DEFAULT_SAVE_DIR is not None, (
            "Predictor must define a default save directory"
        )
        self.dataset = dataset

    def predict(self, records: List[Record]) -> List[Prediction]:
        raise NotImplementedError

    @staticmethod
    def format_pred_savea_name(model_name: str, dataset_name: str) -> str:
        return f"{model_name}_{dataset_name}_preds.jsonl"

    @classmethod
    def get_save_path(cls, save_dir: Path | None = None) -> Path:
        """Generate a prediction save path based on the dataset and model name."""
        formated_name = cls.format_pred_savea_name(
            model_name=cls.name, dataset_name=cls.dataset.name
        )

        if save_dir is None:
            save_dir = cls.DEFAULT_SAVE_DIR

        return save_dir / formated_name

    @classmethod
    def save_pred(
        cls,
        preds: List[Prediction],
        records: List[Record],
        save_path: Path | None = None,
    ) -> None:
        if save_path is None:
            save_path = cls.get_save_path()
        save_path.parent.mkdir(parents=True, exist_ok=True)

        write_pred(preds, records, save_path)

    @classmethod
    def load_pred(cls, save_path: Path | None = None) -> List[PredSave]:
        if save_path is None:
            save_path = cls.get_save_path()

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
