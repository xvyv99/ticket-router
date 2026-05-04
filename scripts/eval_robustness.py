"""Run robustness evaluation on ticket router models using TextAttack.

Usage:
    # Rule-based (black-box character perturbation)
    PYTHONPATH=src uv run python scripts/eval_robustness.py \
        --model rule-based --dataset multilingual-customer-support --max-samples 200

    # HF model (white-box gradient attack, default)
    PYTHONPATH=src uv run python scripts/eval_robustness.py \
        --model mbert --dataset multilingual-customer-support --max-samples 200

    # HF model forced to black-box
    PYTHONPATH=src uv run python scripts/eval_robustness.py \
        --model mbert --dataset multilingual-customer-support --attack-type blackbox
"""

from __future__ import annotations

import os
from argparse import ArgumentParser
from datetime import datetime
from logging import getLogger, basicConfig
from pathlib import Path
from typing import List

from ticket_router.base.config import LOGGING_FORMAT, OUTPUT_DIR
from ticket_router.base.data import get_dataset, DATASET_REGISTRY
from ticket_router.base.predictor import get_model

from ticket_router.supervised.models import HFPredictor

from ticket_router.eval.robustness import (
    ATTACK_RECIPE_REGISTRY,
    BlackBoxRobustnessEvaluator,
    CharacterPerturbation,
    RobustnessMetrics,
    WhiteBoxRobustnessEvaluator,
    WordPerturbation,
)
from ticket_router.eval.robustness_report import (
    print_robustness_report,
    save_adversarial_examples_to_jsonl,
    save_robustness_to_csv,
    save_robustness_to_excel,
    save_robustness_to_json,
)

logger = getLogger(__name__)


def _is_hf_predictor(predictor_cls) -> bool:
    """Check if predictor class is an HFPredictor subclass."""
    try:
        return issubclass(predictor_cls, HFPredictor)
    except TypeError:
        return False


def main() -> None:
    # Ensure HF mirror is set for model downloads
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    parser = ArgumentParser(description="Robustness evaluation using TextAttack")
    parser.add_argument(
        "--model",
        required=True,
        help="Model name (e.g. rule-based, lr, mbert, xlm-roberta)",
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_REGISTRY.keys()),
        default="multilingual-customer-support",
        help="Dataset to evaluate against",
    )
    parser.add_argument(
        "--attack-type",
        choices=["auto", "blackbox", "whitebox"],
        default="auto",
        help=(
            "Attack type: auto (HF models use whitebox, others use blackbox), "
            "blackbox (character perturbation), whitebox (gradient attack)"
        ),
    )
    parser.add_argument(
        "--strategy",
        choices=["character", "word"],
        default="character",
        help="Black-box perturbation strategy (default: character)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=3,
        help="Black-box perturbation budget (number of transformations per text, default: 3)",
    )
    parser.add_argument(
        "--attack-recipe",
        choices=list(ATTACK_RECIPE_REGISTRY.keys()),
        default="textfooler",
        help="White-box attack recipe (default: textfooler)",
    )
    parser.add_argument(
        "--query-budget",
        type=int,
        default=100,
        help="White-box query budget per sample (default: 100)",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Specific task to evaluate (default: all tasks)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Max number of test samples to evaluate (default: 200)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device for white-box model inference (default: cuda)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for robustness results",
    )
    parser.add_argument(
        "--num-perturbations",
        type=int,
        default=1,
        help="Number of perturbed variants per sample for black-box (default: 1)",
    )
    parser.add_argument(
        "--no-save-adversarial",
        action="store_true",
        help="Skip saving adversarial examples (default: save them)",
    )
    args = parser.parse_args()

    dataset_type = get_dataset(args.dataset)
    dataset = dataset_type()

    # Load predictor
    predictor_cls = get_model(args.model)
    assert issubclass(predictor_cls, HFPredictor)
    predictor = predictor_cls.load_model(dataset)

    # Determine attack type
    is_hf = _is_hf_predictor(predictor_cls)
    attack_type = args.attack_type
    if attack_type == "auto":
        attack_type = "whitebox" if is_hf else "blackbox"

    if attack_type == "whitebox" and not is_hf:
        raise ValueError(
            f"White-box attack requires an HFPredictor model, but {args.model} is not. "
            f"Use --attack-type blackbox instead."
        )

    # Load test records
    _, test_df, _ = dataset.load_train_test_split()
    test_records = dataset.df_to_records(test_df)
    if args.max_samples > 0 and len(test_records) > args.max_samples:
        test_records = test_records[: args.max_samples]

    logger.info(
        f"Evaluating robustness for {args.model} on {args.dataset} "
        f"({len(test_records)} samples, attack_type={attack_type})"
    )

    # Determine tasks to evaluate
    task_names = [args.task] if args.task else dataset.task_names

    # Run evaluation
    metrics_list: List[RobustnessMetrics] = []
    for task_name in task_names:
        logger.info(f"Running robustness evaluation for task: {task_name}")

        if attack_type == "blackbox":
            strategy = (
                CharacterPerturbation(budget=args.budget)
                if args.strategy == "character"
                else WordPerturbation(budget=args.budget)
            )
            evaluator = BlackBoxRobustnessEvaluator(
                predictor,
                dataset,
                strategy=strategy,
                num_perturbations=args.num_perturbations,
            )
        else:
            evaluator = WhiteBoxRobustnessEvaluator(
                predictor,
                dataset,
                attack_recipe=args.attack_recipe,
                query_budget=args.query_budget,
                device=args.device,
            )

        metrics = evaluator.evaluate(test_records, task_name=task_name)
        metrics_list.append(metrics)

    # Print report
    print_robustness_report(metrics_list)

    # Save results
    if args.output_dir is None:
        args.output_dir = OUTPUT_DIR / "robustness" / args.model / args.dataset
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for metrics in metrics_list:
        task_name = metrics.task_name
        save_robustness_to_json(
            [metrics], args.output_dir / f"{task_name}_metrics.json"
        )
        save_robustness_to_csv([metrics], args.output_dir / f"{task_name}_metrics.csv")
        save_robustness_to_excel(
            [metrics], args.output_dir / f"{task_name}_metrics.xlsx"
        )

    if not args.no_save_adversarial:
        save_adversarial_examples_to_jsonl(metrics_list, args.output_dir)


if __name__ == "__main__":
    basicConfig(level="INFO", format=LOGGING_FORMAT)
    main()
