"""XGBoost training for arbitrary classification tasks."""

from typing import Dict, List

from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

from ticket_router_base.config import SEED
from ticket_router_base.data import BaseDataset
from ticket_router_base.types import (
    Record,
    Prediction,
    ErrorFlag,
)
from ticket_router_base.utils import combine_texts
from ticket_router_base.predictor import Predictor, Trainer, register_model

from ticket_router_supervised.utils import save_model, SKModel
from ticket_router_supervised.config import SAVE_DIR
from ticket_router_supervised.encoder import TextEncoder
from ticket_router_supervised.cfg import SupervisedCfg

XGBCfg = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "max_depth": 6,
    "n_estimators": 200,
    "learning_rate": 0.1,
    "random_state": SEED,
}


def train_xgb(texts: List[str], labels: List[str], save_name: str, encoder: TextEncoder) -> SKModel:
    """Train a single XGB model for one classification task."""
    le = LabelEncoder()
    y = le.fit_transform(labels)
    X_t = encoder.fit_transform(texts)

    n_classes = len(le.classes_)
    if n_classes == 2:
        # binary classification: use binary:logistic instead of multi:softprob
        cfg = {**XGBCfg, "objective": "binary:logistic", "eval_metric": "logloss"}
        clf = xgb.XGBClassifier(**cfg)
    else:
        clf = xgb.XGBClassifier(num_class=n_classes, **XGBCfg)
    clf.fit(X_t, y)

    model = SKModel(encoder, clf, le=le)
    save_model(save_name, model)
    return model


@register_model
class XGBPredictor(Predictor[SupervisedCfg]):
    name = "xgb"
    dataset: BaseDataset

    DEFAULT_SAVE_DIR = SAVE_DIR

    _models: Dict[str, SKModel]

    cfg: SupervisedCfg | None

    def __init__(self, dataset: BaseDataset, models: Dict[str, SKModel], cfg: SupervisedCfg):
        self.dataset = dataset
        self._models = models
        self.cfg = cfg 

        assert self.cfg is not None, "Predictor must be initialized with a config."

    def predict(self, records: List[Record], run_id: int = 0) -> List[Prediction]:
        texts = combine_texts(records)

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
                sensitive_attributes=rec.sensitive_attributes,
                discrete_features=rec.discrete_features,
                # results
                labels=labels,
                generation_target=None,
                confidences=confidences,
                raw_output=None,
                error=ErrorFlag.SUCCESS,
            )
            predictions.append(pred)

        return predictions


class XGBTrainer(Trainer):
    dataset: BaseDataset

    def __init__(self, dataset: BaseDataset, cfg: SupervisedCfg):
        self.dataset = dataset
        self.cfg = cfg

    def train(
        self,
        records: List[Record],
        val_records: List[Record] | None = None,
    ) -> XGBPredictor:
        texts = combine_texts(records)
        encoder = self.cfg.encoder

        models: Dict[str, SKModel] = {}
        for task in self.dataset.all_tasks:
            labels = []
            for r in records:
                if task.name not in r.labels:
                    raise ValueError(
                        f"Record {r.request_id} is missing label for task {task.name}"
                    )
                labels.append(r.labels[task.name])

            model = train_xgb(texts, labels, f"xgb_{task.name}", encoder=encoder)
            models[task.name] = model

        return XGBPredictor(models=models, dataset=self.dataset, cfg=self.cfg)
