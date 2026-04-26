"""Pydantic schemas for attribute inference output."""

from enum import Enum
from typing import List

from pydantic import BaseModel


class UserType(str, Enum):
    INDIVIDUAL = "individual"
    ENTERPRISE = "enterprise"
    UNKNOWN = "unknown"


class TechProficiency(str, Enum):
    LOW = "low"
    HIGH = "high"
    UNKNOWN = "unknown"


class AttributePrediction(BaseModel):
    """Single ticket attribute inference result."""

    request_id: str
    user_type: UserType
    industry: str
    tech_proficiency: TechProficiency
    reason: str


class InferenceResult(BaseModel):
    """Container for a batch of inference results."""

    predictions: List[AttributePrediction]
    error_count: int = 0
    total_count: int = 0
