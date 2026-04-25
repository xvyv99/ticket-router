"""Evaluate existing model predictions and output a rich comparison table.

Usage:
    uv run python scripts/eval_models.py --dataset multilingual-customer-support
    uv run python scripts/eval_models.py --dataset french-gov-oss --pred-files lr:xgb
"""

from pathlib import Path
from argparse import ArgumentParser
from logging import basicConfig
from typing import List

from rich.console import Console

from ticket_router_base.eval import evaluate_model_dataset, EvaluationReport
from ticket_router_base.data import get_dataset, DATASET_REGISTRY
from ticket_router_base.config import LOGGING_FORMAT
from ticket_router_base.eval.report import print_overall_report
from ticket_router_base.predictor import scan_pred_saves

console = Console()


def illustrate_metric(
    dataset_name: str,
) -> None:
    dataset_type = get_dataset(dataset_name)

    results = scan_pred_saves()

    reports: List[EvaluationReport] = []
    for model, dataset_lst in results.items():
        for dataset_type in dataset_lst:
            if dataset_type.name != dataset_name:
                continue
            dataset = dataset_type()
            report = evaluate_model_dataset(model, dataset)
            reports.append(report)

    print_overall_report(reports)


def main() -> None:
    parser = ArgumentParser(description="Evaluate model predictions")
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_REGISTRY.keys()),
        default="multilingual-customer-support",
        help="Dataset to evaluate against",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=Path("../../outputs/supervised"),
        help="Directory containing prediction JSONL files",
    )
    parser.add_argument(
        "--pred-files",
        type=str,
        default="lr:xgb:mbert",
        help="Colon-separated list of prediction file stems (e.g. lr:xgb)",
    )
    args = parser.parse_args()

    illustrate_metric(args.dataset)


if __name__ == "__main__":
    basicConfig(level="INFO", format=LOGGING_FORMAT)
    main()
