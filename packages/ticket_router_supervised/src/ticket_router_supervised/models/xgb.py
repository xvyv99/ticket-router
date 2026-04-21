"""XGBoost training for arbitrary classification tasks."""

from typing import Dict, List

from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

from ticket_router_base.config import SEED
from ticket_router_base.datasets.base import BaseDataset
from ticket_router_base.types import (
    Record,
    Prediction,
    PredictionBatch,
    ErrorFlag,
)
from ticket_router_base.utils import combine_texts
from ticket_router_base.predictor import Predictor, Trainer

from ticket_router_supervised.features import build_tfidf_pipeline
from ticket_router_supervised.utils import save_model, SKModel

XGBCfg = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "max_depth": 6,
    "n_estimators": 200,
    "learning_rate": 0.1,
    "random_state": SEED,
}


def train_xgb(texts: List[str], labels: List[str], save_name: str) -> SKModel:
    """Train a single XGB model for one classification task."""
    le = LabelEncoder()
    y = le.fit_transform(labels)
    pipe = build_tfidf_pipeline()
    X_t = pipe.fit_transform(texts)

    n_classes = len(le.classes_)
    if n_classes == 2:
        # binary classification: use binary:logistic instead of multi:softprob
        cfg = {**XGBCfg, "objective": "binary:logistic", "eval_metric": "logloss"}
        clf = xgb.XGBClassifier(**cfg)
    else:
        clf = xgb.XGBClassifier(num_class=n_classes, **XGBCfg)
    clf.fit(X_t, y)

    model = SKModel(pipe, clf, le=le)
    save_model(save_name, model)
    return model


class XGBPredictor(Predictor):
    supports_tags = False
    supports_preliminary_answer = False

    _models: Dict[str, SKModel]
    _dataset: BaseDataset

    def __init__(self, models: Dict[str, SKModel], dataset: BaseDataset):
        self._models = models
        self._dataset = dataset

    def predict(self, records: List[Record]) -> PredictionBatch:
        texts = combine_texts(records)

        task_preds: Dict[str, List[tuple]] = {}
        for task_name, model in self._models.items():
            task_preds[task_name] = model.predict(texts)

        predictions = []
        for i, rec in enumerate(records):
            labels: Dict[str, str] = {}
            confidences: Dict[str, float | None] = {}
            for task_name in self._dataset.get_task_names():
                preds = task_preds[task_name]
                labels[task_name] = str(preds[i][0])
                confidences[task_name] = preds[i][1]

            pred = Prediction(
                request_id=rec.request_id,
                labels=labels,
                discrete_features={},
                generation_target=None,
                confidences=confidences,
                raw_output=None,
                error=ErrorFlag.SUCCESS,
            )
            predictions.append(pred)

        return PredictionBatch(
            predictions=predictions, parse_err_count=0, parse_json_err_count=0
        )


class XGBTrainer(Trainer):
    def train(
        self,
        records: List[Record],
        dataset: BaseDataset,
        val_records: List[Record] | None = None,
    ) -> XGBPredictor:
        texts = combine_texts(records)

        models: Dict[str, SKModel] = {}
        for task in dataset.classification_tasks:
            labels = [r.labels.get(task.name, "") for r in records]
            model = train_xgb(texts, labels, f"xgb_{task.name}")
            models[task.name] = model

        return XGBPredictor(models=models, dataset=dataset)
