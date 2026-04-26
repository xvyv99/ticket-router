"""Ticket attribute inference package.

Provides tools to infer sensitive attributes (user_type, industry, tech_proficiency)
from ticket text using vLLM offline inference.
"""

from .base import AttributeInferrer, get_inferrer, register_inferrer
from .schema import (
    UserType,
    TechProficiency,
    AttributePrediction,
)
from .inferrer import infer_attributes

# Import submodules to trigger decorator registration
from . import multilingual  # noqa: F401

__all__ = [
    "AttributeInferrer",
    "get_inferrer",
    "register_inferrer",
    "UserType",
    "TechProficiency",
    "AttributePrediction",
    "infer_attributes",
]
