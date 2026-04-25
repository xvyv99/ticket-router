"""Logistic Regression training for arbitrary classification tasks."""

from typing import Dict, List

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from ticket_router_base.data import BaseDataset
from ticket_router_base.predictor import Trainer, Predictor, register_model
from ticket_router_base.types import (
    ErrorFlag,
    Record,
    Prediction,
)
from ticket_router_base.utils import combine_texts
from ticket_router_base.config import SEED

from ticket_router_supervised.features import build_tfidf_pipeline
from ticket_router_supervised.utils import save_model, SKModel
from ticket_router_supervised.config import SAVE_DIR


def train_lr(texts: List[str], labels: List[str], save_name: str) -> SKModel:
    """Train a single LR model for one classification task."""
    pipe = build_tfidf_pipeline()
    X_t = pipe.fit_transform(texts)

    le = LabelEncoder()
    y = le.fit_transform(labels)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    clf.fit(X_t, y)

    model = SKModel(pipe, clf, le=le)
    save_model(save_name, model)
    return model


@register_model
class LRPredictor(Predictor):
    name = "lr"
    dataset: BaseDataset

    DEFAULT_SAVE_DIR = SAVE_DIR

    _models: Dict[str, SKModel]

    def __init__(
        self,
        dataset: BaseDataset,
        models: Dict[str, SKModel],
    ):
        self.dataset = dataset

        self._models = models

    def predict(self, records: List[Record]) -> List[Prediction]:
        texts = combine_texts(records)

        # Run inference for each task
        task_preds: Dict[str, List[tuple]] = {}
        for task_name, model in self._models.items():
            task_preds[task_name] = model.predict(texts)

        predictions = []
        for i, rec in enumerate(records):
            labels: Dict[str, str] = {}
            confidences: Dict[str, float] = {}
            for task_name in self.dataset.task_names:
                preds = task_preds[task_name]
                labels[task_name] = str(preds[i][0])
                confidences[task_name] = preds[i][1]

            pred = Prediction(
                request_id=rec.request_id,
                labels=labels,
                discrete_features={},
                generation_target=None,
                sensitive_attributes=rec.sensitive_attributes,
                confidences=confidences,
                raw_output=None,
                error=ErrorFlag.SUCCESS,
            )
            predictions.append(pred)

        return predictions


class LRTrainer(Trainer):
    dataset: BaseDataset

    def __init__(self, dataset: BaseDataset):
        self.dataset = dataset

    def train(
        self,
        records: List[Record],
        val_records: List[Record] | None = None,
    ) -> LRPredictor:
        texts = combine_texts(records)

        models: Dict[str, SKModel] = {}
        for task in self.dataset.classification_tasks + self.dataset.ordinal_tasks:
            labels = [r.labels.get(task.name, "") for r in records]
            model = train_lr(texts, labels, f"lr_{task.name}")
            models[task.name] = model

        return LRPredictor(dataset=self.dataset, models=models)
