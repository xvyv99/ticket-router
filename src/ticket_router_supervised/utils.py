"""Shared utilities for supervised learning models."""

from typing import List, Any, Tuple

from datasets import Dataset
import joblib
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

from ticket_router_base.config import MODEL_DIR
from ticket_router_base.data import BaseDataset
from ticket_router_base.types import Record
from ticket_router_base.utils import combine_texts

from .encoder import TextEncoder


MODEL_DIR = MODEL_DIR / "supervised"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class SKModel:
    """Wrapper that pairs a text encoder with a fitted classifier, exposing predict()."""

    mlb: MultiLabelBinarizer | None = None
    le: LabelEncoder | None = None
    encoder: TextEncoder
    clf: Any

    def __init__(
        self,
        encoder: TextEncoder,
        clf,
        mlb: MultiLabelBinarizer | None = None,
        le: LabelEncoder | None = None,
    ):
        self.encoder = encoder
        self.clf = clf
        self.mlb = mlb
        self.le = le

    def predict(self, texts: List[str]) -> List[Tuple[float, float | None]]:
        X_t = self.encoder.transform(texts)

        probs = (
            self.clf.predict_proba(X_t) if hasattr(self.clf, "predict_proba") else None
        )
        preds = self.clf.predict(X_t)

        results = []

        for i, pred in enumerate(preds):
            confidence = float(max(probs[i])) if probs is not None else None

            if self.le is not None:
                pred = self.le.inverse_transform([pred])[0]

            results.append((pred, confidence))

        return results


def save_model(name: str, model_dict: SKModel):
    path = MODEL_DIR / f"{name}.joblib"
    joblib.dump(model_dict, path)
    return path


def create_datasets(records: List[Record], dataset: BaseDataset) -> Dataset:
    """Convert Records to a HuggingFace Dataset for mBERT training.

    Args:
        records: List of Record instances.
        dataset: Dataset descriptor for label mapping.
    """
    records_lst = []
    texts = combine_texts(records)

    for i, r in enumerate(records):
        row: dict[str, Any] = {"text": texts[i]}
        for task in dataset.task_descriptor.classification_tasks + dataset.task_descriptor.ordinal_tasks:
            label = r.labels.get(task.name, "")
            label2id = dataset.get_label2id(task.name)
            row[task.name] = label2id.get(label, -1)
        records_lst.append(row)

    return Dataset.from_list(records_lst)
