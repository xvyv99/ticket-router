"""XGBoost training for queue and priority classification."""

from typing import List

from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

from ticket_router_base.config import SEED
from ticket_router_base.types import (
    Record,
    RecordDF,
    Prediction,
    PredictionBatch,
    Queue,
    Priority,
    ErrorFlag,
)
from ticket_router_base.utils import to_record_df, to_records, combine_texts
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
    X = texts
    y_raw = labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    pipe = build_tfidf_pipeline()
    X_t = pipe.fit_transform(X)

    clf = xgb.XGBClassifier(
        num_class=len(le.classes_),
        **XGBCfg,
    )
    clf.fit(X_t, y)

    model = SKModel(pipe, clf, le=le)
    save_model(save_name, model)

    return model


class XGBPredictor(Predictor):
    supports_tags = False
    supports_preliminary_answer = False

    _model_queue: SKModel
    _model_priority: SKModel

    def __init__(self, model_queue: SKModel, model_priority: SKModel):
        self._model_queue = model_queue
        self._model_priority = model_priority

    def predict(self, records: List[Record] | RecordDF) -> PredictionBatch:
        records = to_records(records)
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
                request_id=rec.request_id,
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


class XGBTrainer(Trainer):
    def train(
        self,
        records: List[Record] | RecordDF,
        val_records: List[Record] | RecordDF | None = None,
    ) -> XGBPredictor:
        records = to_record_df(records)

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

        model_queue = train_xgb(texts, queue_lst, "queue_model")
        model_priority = train_xgb(texts, priority_lst, "priority_model")
        return XGBPredictor(model_queue=model_queue, model_priority=model_priority)
