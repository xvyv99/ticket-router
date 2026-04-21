"""Unified evaluation API for ticket router predictions.

Usage:
    from ticket_router_base.eval import evaluate_file
    report = evaluate_file(Path("outputs/supervised/lr_predictions.jsonl"))
    report.print_summary()
    report.to_json(Path("outputs/eval/lr_report.json"))
"""

from pathlib import Path
from typing import Dict, List

from ticket_router_base.types import Task

from .evaluator import TaskEvaluator
from .loader import load_pred_saves
from .report import EvaluationReport


def evaluate_file(pred_jsonl: Path, model_name: str | None = None) -> EvaluationReport:
    """Evaluate a single prediction JSONL file for queue and priority tasks.

    Args:
        pred_jsonl: Path to the prediction JSONL file.
        model_name: Optional model name; inferred from filename if not given.

    Returns:
        An EvaluationReport with queue and priority metrics.
    """
    if model_name is None:
        model_name = pred_jsonl.stem

    pred_saves = load_pred_saves(pred_jsonl)
    evaluator = TaskEvaluator()

    queue_result = evaluator.evaluate(pred_saves, Task.QUEUE)
    priority_result = evaluator.evaluate(pred_saves, Task.PRIORITY)

    # error summary
    error_summary: Dict[str, int] = {}
    for ps in pred_saves:
        flag_name = ps.predicted.error.name
        error_summary[flag_name] = error_summary.get(flag_name, 0) + 1

    return EvaluationReport(
        model_name=model_name,
        pred_file_path=str(pred_jsonl),
        queue_result=queue_result,
        priority_result=priority_result,
        error_summary=error_summary,
        total_samples=len(pred_saves),
    )


def evaluate_files(pred_jsonls: List[Path], model_name: str) -> List[EvaluationReport]:
    """Evaluate multiple prediction JSONL files (e.g. agent multi-run).

    Args:
        pred_jsonls: List of paths to prediction JSONL files.
        model_name: Model name to label all reports.

    Returns:
        A list of EvaluationReports, one per file.
    """
    return [evaluate_file(p, model_name=model_name) for p in pred_jsonls]


__all__ = [
    "evaluate_file",
    "evaluate_files",
    "EvaluationReport",
    "TaskEvaluator",
    "load_pred_saves",
]
