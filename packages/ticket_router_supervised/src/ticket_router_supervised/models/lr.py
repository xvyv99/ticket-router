"""Logistic Regression training for queue, priority, and tags."""

from typing import List

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from ticket_router_base.predictor import Trainer, Predictor
from ticket_router_base.types import (
    ErrorFlag,
    Queue,
    Priority,
    RecordDF,
    Record,
    Prediction,
    PredictionBatch,
    record_to_df,
)
from ticket_router_base.config import SEED

from ticket_router_supervised.features import build_tfidf_pipeline
from ticket_router_supervised.utils import combine_texts, save_model, SKModel


def train_lr(
    texts: List[str],
    labels: List[str] | List[List[str]],
    save_name: str,
    multi_label: bool = False,
) -> SKModel:
    X = texts
    y = labels
    pipe = build_tfidf_pipeline()
    X_t = pipe.fit_transform(X)

    if multi_label:
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(labels)
        clf = OneVsRestClassifier(
            LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=SEED
            )
        )
        clf.fit(X_t, y)
    else:
        clf = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=SEED
        )
        clf.fit(X_t, y)

    model = SKModel(pipe, clf)

    save_model(save_name, model)
    return model


class LRPredictor(Predictor):
    supports_tags = False  # TODO: enable after implementing tag prediction
    supports_preliminary_answer = False

    _model_queue: SKModel
    _model_priority: SKModel
    _model_tags: SKModel

    def __init__(
        self, model_queue: SKModel, model_priority: SKModel, model_tags: SKModel
    ):
        self._model_queue = model_queue
        self._model_priority = model_priority
        self._model_tags = model_tags

    def predict(self, records: List[Record] | RecordDF) -> PredictionBatch:
        texts = combine_texts(records)

        q_preds = self._model_queue.predict(texts)
        p_preds = self._model_priority.predict(texts)
        # t_preds = self._model_tags.predict(texts)

        predictions = []
        for i, rec in enumerate(records):
            q = q_preds[i]
            p = p_preds[i]
            # tags = t_preds[i] if t_preds else []
            # TODO: convert predicted tag indices back to tag names using mlb

            pred = Prediction(
                request_id=rec.request_id,  # pyright: ignore[reportAttributeAccessIssue]
                queue=Queue(q[0]),
                priority=Priority(p[0]),
                tag_1=None,
                tag_2=None,
                answer=None,
                queue_confidence=q[1],
                priority_confidence=p[1],
                raw_output=None,
                error=ErrorFlag.SUCCESS,
            )

            predictions.append(pred)
        return PredictionBatch(
            predictions=predictions, parse_err_count=0, parse_json_err_count=0
        )


class LRTrainer(Trainer):
    def train(
        self,
        records: List[Record] | RecordDF,
        val_records: List[Record] | RecordDF | None = None,
    ) -> LRPredictor:
        if isinstance(records, list):
            records = record_to_df(records)
 
        texts = combine_texts(records)

        queue_lst = records["queue"].tolist()
        priority_lst = records["priority"].tolist()
        tag_lst_1 = records["tag_1"].fillna("")
        tag_lst_2 = records["tag_2"].fillna("")

        tag_lst = []

        for i in range(len(records)):
            tags = []
            if tag_lst_1.iloc[i]:
                tags.append(tag_lst_1.iloc[i])
            if tag_lst_2.iloc[i]:
                tags.append(tag_lst_2.iloc[i])
            tag_lst.append(tags)

        model_queue = train_lr(texts, queue_lst, "lr_queue")
        model_priority = train_lr(texts, priority_lst, "lr_priority")
        model_tags = train_lr(texts, tag_lst, "lr_tags", multi_label=True)

        return LRPredictor(
            model_queue=model_queue,
            model_priority=model_priority,
            model_tags=model_tags,
        )
