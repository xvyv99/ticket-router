"""Unified evaluation API for ticket router predictions.

Usage:
    from ticket_router_eval import evaluate_model_dataset
    from ticket_router.base.data import get_dataset
    from ticket_router.supervised import LRPredictor
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
from .interpret import (
    HFInterpretabilityEvaluator,
    SampleAttribution,
    TaskAttributionReport,
    TokenAttribution,
)
from .interpret_report import (
    print_interpretability_report,
    save_interpretability_results,
)

# Robustness (optional, heavy dependency)
from .robustness import (
    AdversarialExample,
    BlackBoxRobustnessEvaluator,
    WhiteBoxRobustnessEvaluator,
    RobustnessMetrics,
    CharacterPerturbation,
    WordPerturbation,
)
from .robustness_report import (
    print_robustness_report,
    save_adversarial_examples_to_jsonl,
    save_robustness_to_csv,
    save_robustness_to_excel,
    save_robustness_to_json,
)
