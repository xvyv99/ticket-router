"""Unified evaluation API for ticket router predictions.

Usage:
    from ticket_router_base.eval import evaluate_file
    from ticket_router_base.data import get_dataset
    dataset = get_dataset("multilingual-customer-support")
    report = evaluate_file(Path("outputs/supervised/lr_predictions.jsonl"), dataset)
    report.print_summary()
    report.to_json(Path("outputs/eval/lr_report.json"))
"""

from .evaluator import TaskEvaluator
from .report import EvaluationReport
from .utils import evaluate_model

__all__ = [
    "evaluate_model",
    "EvaluationReport",
    "TaskEvaluator",
]
