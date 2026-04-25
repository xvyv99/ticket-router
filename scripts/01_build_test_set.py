from argparse import ArgumentParser
from logging import getLogger, basicConfig

from ticket_router_base.data import (
    DATASET_REGISTRY,
    get_dataset,
)
from ticket_router_base.config import (
    LOGGING_FORMAT,
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

    df = dataset.load_df()

    df_train, df_test = dataset.split_train_test_set(
        df, save=True, test_num=args.test_num
    )

    logger.info(f"Wrote {len(df_train)} training cases, {len(df_test)} test cases.")


if __name__ == "__main__":
    basicConfig(level="INFO", format=LOGGING_FORMAT)
    main()
