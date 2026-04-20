from logging import getLogger, basicConfig

import pandas as pd

from ticket_router_base.data.loader import load_4k
from ticket_router_base.data.utils import build_train_test_set, build_difficult_cases
from ticket_router_base.config import (
    OUTPUT_DIR,
    LOGGING_FORMAT,
    SEED,
    TEST_SAMPLE_NUM,
    DIFFICULT_CASE_NUM,
)

logger = getLogger(__name__)


def add_request_id(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()
    df["request_id"] = [f"{prefix}-{i:04d}" for i in df.index]
    return df


def main():
    df = load_4k()

    train_df, test_df = build_train_test_set(df, test_num=TEST_SAMPLE_NUM, seed=SEED)
    diff_df = build_difficult_cases(df, n=DIFFICULT_CASE_NUM, seed=SEED)

    train_df = add_request_id(train_df, "Train")
    test_df = add_request_id(test_df, "Test")
    diff_df = add_request_id(diff_df, "Difficult")

    train_df.to_json(
        OUTPUT_DIR / "train_set.jsonl", orient="records", lines=True, force_ascii=False
    )
    test_df.to_json(
        OUTPUT_DIR / "test_set.jsonl", orient="records", lines=True, force_ascii=False
    )
    diff_df.to_json(
        OUTPUT_DIR / "difficult_cases.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )

    logger.info(
        f"Wrote {len(test_df)} test cases, {len(train_df)} training cases and {len(diff_df)} difficult cases to {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    basicConfig(level="INFO", format=LOGGING_FORMAT)
    main()
