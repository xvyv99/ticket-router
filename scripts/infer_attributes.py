#!/usr/bin/env python3
"""Infer ticket sensitive attributes using vLLM.

Delegates to ticket_infer package.
"""

from pathlib import Path
from argparse import ArgumentParser

from ticket_router_base.config import OUTPUT_DIR
from ticket_router.infer import infer_attributes


def main():
    parser = ArgumentParser(description="Infer ticket sensitive attributes")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B",
        help="vLLM model path or name",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="multilingual-customer-support",
        help="Dataset name",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Dataset CSV path (overrides --dataset)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of records",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test", "valid"],
        help="Data split to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL path",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path) if args.dataset_path else None

    if args.output is None:
        model_name = Path(args.model).name if "/" in args.model else args.model
        args.output = str(OUTPUT_DIR / f"infer_{args.dataset}_{model_name}.jsonl")

    result = infer_attributes(
        model=args.model,
        dataset_name=args.dataset,
        dataset_path=dataset_path,
        limit=args.limit,
        split=args.split,
        temperature=args.temperature,
        output_path=Path(args.output),
    )

    print(
        f"\nInference complete: {result.total_count - result.error_count}/{result.total_count} successful"
    )


if __name__ == "__main__":
    main()
