"""Domain types for unified Predictor interface.

All dataset-specific enums (Queue, Priority) have been removed.
Classification labels are now stored in a generic `labels: Dict[str, str]` dict,
making the type system fully decoupled from any particular dataset schema.
"""

from enum import IntFlag, auto, StrEnum
from typing import Dict

from pydantic import BaseModel


class Language(StrEnum):
    """Supported languages for tickets."""

    ENGLISH = "English"
    SPANISH = "Spanish"
    FRENCH = "French"
    GERMAN = "German"
    PORTUGUESE = "Portuguese"


class ErrorFlag(IntFlag):
    """Error flags for prediction parsing."""

    SUCCESS = 0
    JSON_ERR = auto()
    CLASSIFICATION_REGEX_ERR = auto()  # formerly QUEUE_REGEX_ERR / PRIORITY_REGEX_ERR
    GENERATION_REGEX_ERR = auto()  # formerly ANSWER_REGEX_ERR
    TAGS_REGEX_ERR = auto()
    TAGS_UNSUPPORTED = auto()
    GENERATION_UNSUPPORTED = auto()  # formerly ANSWER_UNSUPPORTED

    PARSE_ERR = (
        JSON_ERR | CLASSIFICATION_REGEX_ERR | GENERATION_REGEX_ERR | TAGS_REGEX_ERR
    )


class GroundRecord(BaseModel):
    """Ground-truth record without request identity.

    `labels` holds all classification task outputs (task_name -> label_value).
    `discrete_features` holds auxiliary categorical features.
    `generation_target` holds the expected text-generation output.
    """

    labels: Dict[str, str]
    discrete_features: Dict[str, str | None]
    generation_target: str | None
    sensitive_attributes: Dict[str, str | None]  # for fairness evaluation


class Record(GroundRecord):
    """A complete data record with identity and text content."""

    request_id: str
    title: str | None  # optional subject line
    body: str
    language: Language | None


class Prediction(GroundRecord):
    """A model prediction with confidence scores and raw output."""

    request_id: str
    confidences: Dict[str, float] | None  # task_name -> confidence
    raw_output: str | None
    error: ErrorFlag


class PredSave(BaseModel):
    """Single prediction paired with its ground truth, as stored in JSONL."""

    predicted: Prediction
    ground_truth: GroundRecord
