"""Predictor and Trainer protocol definitions."""

from typing import Protocol, List

from .types import Record, Prediction


class Predictor(Protocol):
    supports_tags: bool
    supports_preliminary_answer: bool

    def predict(self, records: List[Record]) -> List[Prediction]:
        raise NotImplementedError


class Trainer(Protocol):
    def train(
        self,
        records: List[Record],
        val_records: List[Record] | None = None,
    ) -> Predictor:
        raise NotImplementedError
