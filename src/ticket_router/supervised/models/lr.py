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

from ticket_router.supervised.utils import save_model, SKModel
from ticket_router.supervised.config import SAVE_DIR
from ticket_router.supervised.encoder import TextEncoder
from ticket_router.supervised.cfg import SupervisedCfg


def train_lr(
    texts: List[str], labels: List[str], save_name: str, encoder: TextEncoder
) -> SKModel:
    """Train a single LR model for one classification task."""
    X_t = encoder.fit_transform(texts)

    le = LabelEncoder()
    y = le.fit_transform(labels)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    clf.fit(X_t, y)

    model = SKModel(encoder, clf, le=le)
    save_model(save_name, model)
    return model


@register_model
class LRPredictor(Predictor[SupervisedCfg]):
    name = "lr"
    dataset: BaseDataset

    DEFAULT_SAVE_DIR = SAVE_DIR

    _models: Dict[str, SKModel]

    cfg: SupervisedCfg | None

    def __init__(
        self,
        dataset: BaseDataset,
        models: Dict[str, SKModel],
        cfg: SupervisedCfg,
    ):
        self.dataset = dataset
        self._models = models
        self.cfg = cfg

        assert self.cfg is not None, "Predictor must be initialized with a config."

    def predict(self, records: List[Record], run_id: int = 0) -> List[Prediction]:
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


class LRTrainer(Trainer):
    dataset: BaseDataset

    def __init__(self, dataset: BaseDataset, cfg: SupervisedCfg):
        self.dataset = dataset
        self.cfg = cfg

    def train(
        self,
        records: List[Record],
        val_records: List[Record] | None = None,
    ) -> LRPredictor:
        texts = combine_texts(records)
        encoder = self.cfg.encoder

        models: Dict[str, SKModel] = {}
        for task in self.dataset.all_tasks:
            labels = [r.labels.get(task.name, "") for r in records]
            model = train_lr(texts, labels, f"lr_{task.name}", encoder=encoder)
            models[task.name] = model

        return LRPredictor(dataset=self.dataset, models=models, cfg=self.cfg)
