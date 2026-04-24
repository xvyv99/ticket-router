"""Predictor and Trainer protocol definitions."""

from abc import ABC
from typing import List

from .types import Record, Prediction
from .data import BaseDataset


class Predictor(ABC):
    dataset: BaseDataset

    def predict(self, records: List[Record]) -> List[Prediction]:
        raise NotImplementedError


class Trainer(ABC):
    dataset: BaseDataset

    def train(
        self,
        records: List[Record],
        val_records: List[Record] | None = None,
    ) -> Predictor:
        raise NotImplementedError
