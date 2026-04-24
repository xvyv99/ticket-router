"""Unified evaluation API for ticket router predictions.

Usage:
    from ticket_router_base.eval import evaluate_file
    from ticket_router_base.data import get_dataset
    dataset = get_dataset("multilingual-customer-support")
    report = evaluate_file(Path("outputs/supervised/lr_predictions.jsonl"), dataset)
    report.print_summary()
    report.to_json(Path("outputs/eval/lr_report.json"))
"""

from pathlib import Path
from typing import Dict, List

from ticket_router_base.data import BaseDataset
from .evaluator import TaskEvaluator
from .utils import load_pred_saves
from .report import EvaluationReport


def evaluate_file(
    pred_jsonl: Path, dataset: BaseDataset, model_name: str | None = None
) -> EvaluationReport:
    """Evaluate a single prediction JSONL file against all dataset tasks.

    Args:
        pred_jsonl: Path to the prediction JSONL file.
        dataset: Dataset descriptor providing task definitions.
        model_name: Optional model name; inferred from filename if not given.
    """
    if model_name is None:
        model_name = pred_jsonl.stem

    pred_saves = load_pred_saves(pred_jsonl)
    evaluator = TaskEvaluator()
    task_results = evaluator.evaluate(pred_saves, dataset)

    # error summary
    error_summary: Dict[str, int] = {}
    for ps in pred_saves:
        flag_name = ps.predicted.error.name  # IntFlag auto-generated name
        # error_summary[flag_name] = error_summary.get(flag_name, 0) + 1
        # TODO: error count

    return EvaluationReport(
        model_name=model_name,
        pred_file_path=str(pred_jsonl),
        dataset_name=dataset.name,
        task_results=task_results,
        error_summary=error_summary,
        total_samples=len(pred_saves),
    )


def evaluate_files(
    pred_jsonls: List[Path], dataset: BaseDataset, model_name: str
) -> List[EvaluationReport]:
    """Evaluate multiple prediction JSONL files (e.g. agent multi-run).

    Args:
        pred_jsonls: List of paths to prediction JSONL files.
        dataset: Dataset descriptor.
        model_name: Model name to label all reports.

    Returns:
        A list of EvaluationReports, one per file.
    """
    return [
        evaluate_file(p, dataset=dataset, model_name=model_name) for p in pred_jsonls
    ]


__all__ = [
    "evaluate_file",
    "evaluate_files",
    "EvaluationReport",
    "TaskEvaluator",
    "load_pred_saves",
]
