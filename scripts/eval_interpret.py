"""Run interpretability evaluation on HFPredictor models.

Usage:
    export HF_ENDPOINT=https://hf-mirror.com
    PYTHONPATH=src uv run python scripts/eval_interpretability.py \
        --model mbert --dataset multilingual-customer-support --task queue --max-samples 50
"""

from __future__ import annotations

from argparse import ArgumentParser
from logging import getLogger, basicConfig
from pathlib import Path
from typing import List

import pandas as pd

from ticket_router_base.config import LOGGING_FORMAT, OUTPUT_DIR, SEED
from ticket_router_base.data import get_dataset, DATASET_REGISTRY
from ticket_router_base.predictor import get_model

from ticket_router.supervised.models import HFPredictor

from ticket_router.eval.interpret import HFInterpretabilityEvaluator
from ticket_router.eval.interpret_report import (
    print_interpretability_report,
    save_interpretability_results,
)

logger = getLogger(__name__)


def _stratified_sample_df(
    df: pd.DataFrame, strat_col: str, n_samples: int, seed: int = SEED
) -> pd.DataFrame:
    if strat_col not in df.columns:
        raise ValueError(f"Stratification column '{strat_col}' not found in DataFrame")
    if len(df) <= n_samples:
        return df

    stratum_sizes = df[strat_col].value_counts()
    proportions = stratum_sizes / stratum_sizes.sum()

    print(stratum_sizes)

    sampled_dfs: List[pd.DataFrame] = []
    for stratum, prop in proportions.items():
        stratum_df = df[df[strat_col] == stratum]
        n_from_stratum = max(1, int(round(prop * n_samples)))
        n_from_stratum = min(n_from_stratum, len(stratum_df))
        sampled = stratum_df.sample(n=n_from_stratum, random_state=seed)
        sampled_dfs.append(sampled)

    result = (
        pd.concat(sampled_dfs).sample(frac=1, random_state=seed).reset_index(drop=True)
    )
    return result


def main() -> None:
    # Ensure HuggingFace mirror is set for model downloads
    parser = ArgumentParser(description="Interpretability evaluation for HFPredictor")
    parser.add_argument(
        "--model",
        required=True,
        help="Model name (e.g. mbert, xlm-roberta)",
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_REGISTRY.keys()),
        default="multilingual-customer-support",
        help="Dataset to evaluate against",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Specific task to evaluate (default: all tasks)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Max number of test samples to evaluate (default: 100)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top tokens to extract per sample (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for attribution results",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cuda",
        help="Device for prediction pass; attribution is always on CPU (default: cpu)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=100,
        help="Number of steps for LIG attribution (default: 20)",
    )
    parser.add_argument(
        "--internal-batch-size",
        type=int,
        default=8,
        help="Internal batch size for LIG forward passes (default: 10)",
    )
    parser.add_argument(
        "--predict-batch-size",
        type=int,
        default=8,
        help="Batch size for the prediction pass (default: 1)",
    )
    args = parser.parse_args()

    dataset_type = get_dataset(args.dataset)
    dataset = dataset_type()

    # Load HFPredictor
    predictor_cls = get_model(args.model)

    assert issubclass(predictor_cls, HFPredictor), (
        f"Model {args.model} is not a valid HFPred model"
    )
    predictor = predictor_cls.load_model(dataset)

    # Load test records
    _, test_df, _ = dataset.load_train_test_split()
    if args.max_samples > 0:
        # Stratified sampling based on task column
        task_col = args.task
        assert isinstance(task_col, str), "Task column must be specified for sampling"
        test_df = _stratified_sample_df(test_df, task_col, args.max_samples)
    test_records = dataset.df_to_records(test_df)

    logger.info(
        f"Evaluating interpretability for {args.model} on {args.dataset} "
        f"({len(test_records)} samples, device={args.device}, n_steps={args.n_steps})"
    )

    evaluator = HFInterpretabilityEvaluator(
        predictor,
        dataset,
        device=args.device,
        n_steps=args.n_steps,
        internal_batch_size=args.internal_batch_size,
        predict_batch_size=args.predict_batch_size,
    )
    task_names = [args.task] if args.task else None
    reports = evaluator.evaluate(test_records, top_k=args.top_k, task_names=task_names)

    print_interpretability_report(reports)

    if args.output_dir is None:
        args.output_dir = OUTPUT_DIR / "interpretability" / args.model / args.dataset
    save_interpretability_results(reports, args.output_dir)


if __name__ == "__main__":
    basicConfig(level="INFO", format=LOGGING_FORMAT)
    main()
