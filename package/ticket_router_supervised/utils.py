"""Shared utilities for supervised learning models."""

from typing import List, Any, Tuple

import pandas as pd
from datasets import Dataset
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

from ticket_router_base.config import OUTPUT_DIR
from ticket_router_base.types import (
    Record,
    RecordDF,
    df_to_records,
    QUEUE2ID,
    PRIORITY2ID,
)


MODEL_DIR = OUTPUT_DIR / "supervised" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class SKModel:
    """Wrapper that pairs a sklearn pipeline with a fitted classifier, exposing predict()."""

    mlb: MultiLabelBinarizer | None = None
    le: LabelEncoder | None = None
    pipeline: Pipeline
    clf: Any

    def __init__(
        self,
        pipeline: Pipeline,
        clf,
        mlb: MultiLabelBinarizer | None = None,
        le: LabelEncoder | None = None,
    ):
        self.pipeline = pipeline
        self.clf = clf
        self.mlb = mlb
        self.le = le

    def predict(self, texts: List[str]) -> List[Tuple[float, float | None]]:
        X_t = self.pipeline.transform(texts)

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


def combine_text(subject: str, body: str) -> str:
    # TODO: move this to global utils and apply consistently across all models
    return f"{subject}\n{body}"


def combine_texts_df(records: RecordDF) -> List[str]:
    return [
        combine_text(s, b)
        for s, b in zip(records.subject.fillna(""), records.body.fillna(""))
    ]


def combine_texts_lst(records: List[Record]) -> List[str]:
    return [combine_text(r.subject or "", r.body or "") for r in records]


def combine_texts(records: RecordDF | List[Record]) -> List[str]:
    if isinstance(records, pd.DataFrame):
        return combine_texts_df(records)
    elif isinstance(records, list):
        return combine_texts_lst(records)
    else:
        raise ValueError("Unsupported records type for combine_texts")


def create_datasets(records: RecordDF | List[Record]) -> Dataset:
    records_lst = []

    if isinstance(records, pd.DataFrame):
        records = df_to_records(records)

    for r in records:
        text = f"{r.subject}\n{r.body}"
        tags = [r.tag_1, r.tag_2] if r.tag_1 and r.tag_2 else []
        # TODO: handle tags better, currently just taking the first 2 tags, but some records have more than 2 tags
        records_lst.append(
            {
                "text": text,
                "queue": QUEUE2ID.get(str(r.queue)),
                "priority": PRIORITY2ID.get(str(r.priority)),
                "tags": tags,
            }
        )
    return Dataset.from_list(records_lst)
