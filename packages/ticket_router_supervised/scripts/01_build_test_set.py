from logging import getLogger, basicConfig

from ticket_router_base.datasets import MultilingualCustomerSupportDataset
from ticket_router_base.data.loader import load_dataset
from ticket_router_base.data.utils import build_train_test_set, build_difficult_cases
from ticket_router_base.config import (
    OUTPUT_DIR,
    LOGGING_FORMAT,
    SEED,
    TEST_SAMPLE_NUM,
    DIFFICULT_CASE_NUM,
)

logger = getLogger(__name__)


def main():
    dataset = MultilingualCustomerSupportDataset()
    records = load_dataset(dataset)

    train_records, test_records = build_train_test_set(
        records, dataset, test_num=TEST_SAMPLE_NUM, seed=SEED
    )
    diff_records = build_difficult_cases(
        records, dataset, n=DIFFICULT_CASE_NUM, seed=SEED
    )

    # prefix request_ids
    for i, r in enumerate(train_records):
        train_records[i] = _with_request_id(r, f"Train-{i:04d}")
    for i, r in enumerate(test_records):
        test_records[i] = _with_request_id(r, f"Test-{i:04d}")
    for i, r in enumerate(diff_records):
        diff_records[i] = _with_request_id(r, f"Difficult-{i:04d}")

    _write_jsonl(train_records, OUTPUT_DIR / "train_set.jsonl")
    _write_jsonl(test_records, OUTPUT_DIR / "test_set.jsonl")
    _write_jsonl(diff_records, OUTPUT_DIR / "difficult_cases.jsonl")

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
