from argparse import ArgumentParser
from logging import getLogger, basicConfig

from ticket_router_base.data import (
    DATASET_REGISTRY,
    get_dataset,
    load_dataset,
)
from ticket_router_base.data.utils import (
    build_train_test_set,
    build_difficult_cases,
)
from ticket_router_base.config import (
    OUTPUT_DIR,
    LOGGING_FORMAT,
    SEED,
    TEST_SAMPLE_NUM,
    DIFFICULT_CASE_NUM,
)

logger = getLogger(__name__)


def main():
    parser = ArgumentParser(description="Build train/test/difficult sets")
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_REGISTRY.keys()),
        default="multilingual-customer-support",
        help="Dataset to use",
    )
    parser.add_argument(
        "--test-num",
        type=int,
        default=TEST_SAMPLE_NUM,
        help="Number of test samples",
    )
    parser.add_argument(
        "--difficult-num",
        type=int,
        default=DIFFICULT_CASE_NUM,
        help="Number of difficult cases",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="",
        help="Prefix for output filenames",
    )
    args = parser.parse_args()

    dataset = get_dataset(args.dataset)
    records = load_dataset(dataset)

    train_records, test_records = build_train_test_set(
        records, dataset, test_num=args.test_num, seed=SEED
    )
    diff_records = build_difficult_cases(
        records, dataset, n=args.difficult_num, seed=SEED
    )

    # prefix request_ids
    prefix = args.output_prefix
    sep = "-" if prefix else ""
    for i, r in enumerate(train_records):
        train_records[i] = _with_request_id(r, f"Train{sep}{prefix}{i:04d}")
    for i, r in enumerate(test_records):
        test_records[i] = _with_request_id(r, f"Test{sep}{prefix}{i:04d}")
    for i, r in enumerate(diff_records):
        diff_records[i] = _with_request_id(r, f"Difficult{sep}{prefix}{i:04d}")

    _write_jsonl(train_records, OUTPUT_DIR / f"{prefix}train_set.jsonl")
    _write_jsonl(test_records, OUTPUT_DIR / f"{prefix}test_set.jsonl")
    _write_jsonl(diff_records, OUTPUT_DIR / f"{prefix}difficult_cases.jsonl")

    logger.info(
        f"Wrote {len(test_records)} test cases, {len(train_records)} training cases "
        f"and {len(diff_records)} difficult cases to {OUTPUT_DIR}"
    )


def _with_request_id(record, request_id: str):
    from dataclasses import replace

    return replace(record, request_id=request_id)


def _write_jsonl(records, path):
    import json
    from dataclasses import asdict

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    basicConfig(level="INFO", format=LOGGING_FORMAT)
    main()
