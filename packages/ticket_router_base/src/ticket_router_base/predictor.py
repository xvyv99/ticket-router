"""Predictor and Trainer protocol definitions."""

from typing import Protocol, List

from .types import Record, PredictionBatch, RecordDF


class Predictor(Protocol):
    supports_tags: bool
    supports_preliminary_answer: bool

    def predict(self, records: List[Record] | RecordDF) -> PredictionBatch:
        raise NotImplementedError


class Trainer(Protocol):
    def train(
        self,
        records: List[Record] | RecordDF,
        val_records: List[Record] | RecordDF | None = None,
    ) -> Predictor:
        raise NotImplementedError
