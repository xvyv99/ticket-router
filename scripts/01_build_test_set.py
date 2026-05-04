from argparse import ArgumentParser
from logging import getLogger, basicConfig

from ticket_router.base.data import (
    DATASET_REGISTRY,
    get_dataset,
)
from ticket_router.base.config import (
    LOGGING_FORMAT,
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
    args = parser.parse_args()

    dataset_type = get_dataset(args.dataset)
    dataset = dataset_type()

    df = dataset.load_df()

    df_train, df_test, df_valid = dataset.split_train_test_set(
        df,
        save=True,
    )

    logger.info(
        f"Wrote {len(df_train)} training cases, {len(df_test)} test cases, {len(df_valid)} validation cases."
    )


if __name__ == "__main__":
    basicConfig(level="INFO", format=LOGGING_FORMAT)
    main()
