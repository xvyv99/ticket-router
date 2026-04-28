"""Unified evaluation API for ticket router predictions.

Usage:
    from ticket_router_eval import evaluate_model_dataset
    from ticket_router_base.data import get_dataset
    from ticket_router_supervised import LRPredictor
    dataset = get_dataset("multilingual-customer-support")
    report = evaluate_model_dataset(LRPredictor, dataset())
"""

from .evaluator import TaskEvaluator
from .report import EvaluationReport
from .utils import evaluate_model_dataset
from .aggregate import aggregate_reports

__all__ = [
    "aggregate_reports",
    "evaluate_model_dataset",
    "EvaluationReport",
    "TaskEvaluator",
]

# Interpretability (optional, heavy dependency)
from .interpretability import (
    HFInterpretabilityEvaluator,
    SampleAttribution,
    TaskAttributionReport,
    TokenAttribution,
)
from .interpretability_report import (
    print_interpretability_report,
    save_interpretability_results,
)
