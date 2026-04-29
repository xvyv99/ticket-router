"""All Pydantic schemas for request/response models."""

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class PredictRequest(BaseModel):
    """POST /predict request body."""

    title: str | None = Field(default=None, description="Ticket subject line (optional)")
    body: str = Field(..., description="Ticket body text")
    model: Annotated[
        str,
        Field(description="Model identifier: lr | xgb | rule-based | qwen3 | rembert | xlm-roberta"),
    ]

    model_config = {
        "json_schema_extra": {
            "example": {
                "title": "Cannot reset my password",
                "body": "I have tried clicking the reset button but nothing happens.",
                "model": "lr",
            }
        }
    }


class Confidence(BaseModel):
    """Confidence scores for classification tasks."""

    queue: float = Field(..., ge=0.0, le=1.0)
    priority: float = Field(..., ge=0.0, le=1.0)


class PredictionResult(BaseModel):
    """Prediction output — queue, priority, optional answer."""

    queue: str
    priority: str
    answer: str | None = Field(default=None, description="Preliminary answer, only for qwen3")
    confidence: Confidence


class PredictResponse(BaseModel):
    """POST /predict response — returns immediately with req_id."""

    req_id: str
    status: TaskStatus
    cached: bool = Field(default=False, description="True if this request hit the cache")


class ResultResponse(BaseModel):
    """GET /result/{req_id} response."""

    req_id: str
    status: TaskStatus
    result: PredictionResult | None = None
    error: str | None = None


class TokenAttribution(BaseModel):
    """Single token attribution score."""

    token: str
    score: float


class TaskAttribution(BaseModel):
    """Attribution for a single classification task."""

    predicted_label: str
    confidence: float
    top_positive: list[TokenAttribution]
    top_negative: list[TokenAttribution]


class AttributionResult(BaseModel):
    """Attribution for all tasks."""

    queue: TaskAttribution | None = None
    priority: TaskAttribution | None = None


class AttributionResponse(BaseModel):
    """GET /attribution/{req_id} response."""

    req_id: str
    status: TaskStatus
    attribution: AttributionResult | None = None
    error: str | None = None


class ErrorResponse(BaseModel):
    """Generic error response."""

    error: str


class HealthResponse(BaseModel):
    """GET /health response."""

    status: str = "ok"