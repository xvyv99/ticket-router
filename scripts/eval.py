"""Evaluate existing model predictions and output a rich comparison table.

Usage:
    uv run python scripts/eval.py --dataset multilingual-customer-support
    uv run python scripts/eval.py --dataset multilingual-customer-support --pred-dir outputs/goal_based
"""

from pathlib import Path
from argparse import ArgumentParser
from logging import basicConfig
from typing import List
from collections import defaultdict

from rich.console import Console

from ticket_router_base.eval import (
    evaluate_model_dataset,
    EvaluationReport,
    aggregate_reports,
)
from ticket_router_base.data import get_dataset, DATASET_REGISTRY
from ticket_router_base.config import LOGGING_FORMAT
from ticket_router_base.eval.report import print_overall_report
from ticket_router_base.predictor import scan_pred_saves, get_model

console = Console()


def illustrate_metric(
    dataset_name: str,
    pred_dir: Path | None = None,
) -> None:
    dataset_type = get_dataset(dataset_name)

    results = scan_pred_saves(scan_path=pred_dir)

    # Group by (predictor_name, dataset_name, sub_name)
    groups = defaultdict(list)
    for key in results:
        if key.dataset_name != dataset_name:
            continue
        groups[(key.predictor_name, key.dataset_name, key.sub_name)].append(key)

    reports: List[EvaluationReport] = []
    for (pred_name, ds_name, sub_name), keys in groups.items():
        dataset = dataset_type()
        pred_type = get_model(pred_name)

        if len(keys) == 1:
            # Single run
            report = evaluate_model_dataset(
                pred_type, dataset, sub_name, run_id=keys[0].run_id
            )
            reports.append(report)
        else:
            # Multiple runs: evaluate each and aggregate
            run_reports = []
            for key in sorted(keys, key=lambda k: k.run_id):
                run_report = evaluate_model_dataset(
                    pred_type, dataset, sub_name, run_id=key.run_id
                )
                run_reports.append(run_report)

            aggregated = aggregate_reports(run_reports)
            reports.append(aggregated)

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
        default=None,
        help="Directory containing prediction files (scans all model subdirs)",
    )
    args = parser.parse_args()

    illustrate_metric(args.dataset, pred_dir=args.pred_dir)


if __name__ == "__main__":
    basicConfig(level="INFO", format=LOGGING_FORMAT)
    main()
