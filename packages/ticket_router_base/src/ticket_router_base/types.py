"""Domain types for unified Predictor interface."""

from enum import Enum, IntFlag, auto, StrEnum
from dataclasses import dataclass
import json
from typing import List

import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame


class ITCusomterSupportSchema(pa.DataFrameModel):
    subject: str | None = pa.Field(nullable=True)
    body: str | None = pa.Field(nullable=True)
    answer: str = pa.Field()
    type: str = pa.Field()
    queue: str = pa.Field()
    priority: str = pa.Field()
    language: str = pa.Field()
    business_type: str = pa.Field()
    tag_1: str = pa.Field()
    tag_2: str | None = pa.Field(nullable=True)


type ITCusomterSupportDF = DataFrame[ITCusomterSupportSchema]


class Queue(StrEnum):
    TECHNICAL_SUPPORT = "Technical Support"
    PRODUCT_SUPPORT = "Product Support"
    CUSTOMER_SERVICE = "Customer Service"
    IT_SUPPORT = "IT Support"
    BILLING_AND_PAYMENTS = "Billing and Payments"
    RETURNS_AND_EXCHANGES = "Returns and Exchanges"
    SALES_AND_PRE_SALES = "Sales and Pre-Sales"
    SERVICE_OUTAGES_AND_MAINTENANCE = "Service Outages and Maintenance"
    GENERAL_INQUIRY = "General Inquiry"
    HUMAN_RESOURCES = "Human Resources"


class Priority(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def level(self) -> int:
        levels = {"high": 3, "medium": 2, "low": 1}
        return levels[self.value]


class Language(StrEnum):
    ENGLISH = "en"
    GERMAN = "de"
    SPANISH = "es"
    FRENCH = "fr"
    PORTUGUESE = "pt"


class Task(Enum):
    QUEUE = "queue"
    PRIORITY = "priority"
    TAGS = "tags"
    PRELIMINARY_ANSWER = "preliminary_answer"


# Mappings for model training: label string -> int id (for model config label2id)
QUEUE2ID = {str(q): i for i, q in enumerate(Queue)}
PRIORITY2ID = {str(p): i for i, p in enumerate(Priority)}

# Reverse mappings: int id -> label string (for model config id2label)
ID2QUEUE = {i: q.value for i, q in enumerate(Queue)}
ID2PRIORITY = {i: p.value for i, p in enumerate(Priority)}


class ErrorFlag(IntFlag):
    SUCCESS = 0
    JSON_ERR = auto()
    QUEUE_REGEX_ERR = auto()
    PRIORITY_REGEX_ERR = auto()
    TAGS_REGEX_ERR = auto()
    ANSWER_REGEX_ERR = auto()
    TAGS_UNSUPPORTED = auto()
    ANSWER_UNSUPPORTED = auto()

    PARSE_ERR = (
        JSON_ERR
        | QUEUE_REGEX_ERR
        | PRIORITY_REGEX_ERR
        | TAGS_REGEX_ERR
        | ANSWER_REGEX_ERR
    )


@dataclass(frozen=True)
class GroundRecord:
    queue: Queue
    priority: Priority
    tag_1: str | None
    tag_2: str | None
    answer: str | None

    def to_json_str(self) -> str:
        return json.dumps(
            {
                "queue": self.queue.value,
                "priority": self.priority.value,
                "tag_1": self.tag_1,
                "tag_2": self.tag_2,
                "answer": self.answer,
            },
            ensure_ascii=False,
        )


@dataclass(frozen=True)
class Record(GroundRecord):
    request_id: str
    subject: str | None
    body: str | None
    language: str


class RecordSchema(pa.DataFrameModel):
    request_id: str = pa.Field()
    subject: str | None = pa.Field(nullable=True)
    body: str | None = pa.Field(nullable=True)
    language: str = pa.Field()

    # ground truth fields
    queue: str = pa.Field()
    priority: str = pa.Field()
    tag_1: str | None = pa.Field(nullable=True)
    tag_2: str | None = pa.Field(nullable=True)
    answer: str | None = pa.Field(nullable=True)


type RecordDF = DataFrame[RecordSchema]


def record_to_df(record: List[Record]) -> RecordDF:
    """Convert a Record dataclass instance to a pandas DataFrame."""
    rows = []
    for r in record:
        row = {
            "request_id": r.request_id,
            "subject": r.subject,
            "body": r.body,
            "language": r.language,
            "queue": r.queue.value,
            "priority": r.priority.value,
            "tag_1": r.tag_1,
            "tag_2": r.tag_2,
            "answer": r.answer,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    RecordSchema.validate(df)
    return RecordSchema.validate(df)


def df_to_records(df: RecordDF) -> List[Record]:
    """Convert a pandas DataFrame to a list of Record dataclass instances."""
    records = []
    for _, row in df.iterrows():
        rec = Record(
            request_id=row["request_id"],
            subject=row["subject"],
            body=row["body"],
            language=row["language"],
            queue=Queue(row["queue"]),
            priority=Priority(row["priority"]),
            tag_1=row["tag_1"],
            tag_2=row["tag_2"],
            answer=row["answer"],
        )
        records.append(rec)
    return records


@dataclass(frozen=True)
class Prediction(GroundRecord):
    request_id: str
    queue_confidence: float | None
    priority_confidence: float | None
    raw_output: str | None
    error: ErrorFlag


@dataclass(frozen=True)
class PredictionBatch:
    predictions: List[Prediction]
    parse_err_count: int
    parse_json_err_count: int


@dataclass(frozen=True)
class PredSave:
    request_id: str
    language: Language
    predicted: Prediction
    ground_truth: GroundRecord
