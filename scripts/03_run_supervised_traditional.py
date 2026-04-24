"""Train supervised models (LR, XGBoost) avoiding test-set leakage."""

from argparse import ArgumentParser
from logging import getLogger, basicConfig

from sklearn.model_selection import train_test_split
from ticket_router_base.data import get_dataset
from ticket_router_base.utils import write_pred
from ticket_router_base.config import OUTPUT_DIR, SEED, LOGGING_FORMAT

from ticket_router_supervised.models.lr import LRTrainer
from ticket_router_supervised.models.xgb import XGBTrainer

logger = getLogger(__name__)


def main():
    parser = ArgumentParser(description="Train LR and XGBoost models")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g. multilingual-customer-support, french-gov-oss)",
    )
    parser.add_argument(
        "--train-set",
        type=str,
        default="train_set.jsonl",
        help="Train set JSONL filename (in OUTPUT_DIR)",
    )
    parser.add_argument(
        "--test-set",
        type=str,
        default="test_set.jsonl",
        help="Test set JSONL filename (in OUTPUT_DIR)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="",
        help="Prefix for output filenames",
    )
    args = parser.parse_args()

    dataset = get_dataset(args.dataset)

    df_train, df_test = dataset.load_train_test_split()

    test_size = 0.2
    test_num = int(len(df_test) * test_size)

    df_train, df_val = dataset.split_train_test_set(df_train, test_num=test_num)

    # Stratify by the first task (classification or ordinal)
    all_tasks = dataset.classification_tasks + dataset.ordinal_tasks

    train_split = dataset.df_to_records(df_train)
    val_split = dataset.df_to_records(df_val)

    # Train LR models
    logger.info("Training LR models...")
    lr_trainer = LRTrainer(dataset=dataset)
    lr_predictor = lr_trainer.train(train_split, val_split)
    logger.info("LR models trained successfully.")

    # Train XGBoost models
    logger.info("Training XGBoost models...")
    xgb_trainer = XGBTrainer(dataset=dataset)
    xgb_predictor = xgb_trainer.train(train_split, val_split)
    logger.info("XGBoost models trained successfully.")

    test_records = dataset.df_to_records(df_test)

    # LR predictions
    logger.info("Running LR inference...")
    lr_pred = lr_predictor.predict(test_records)
    prefix = args.output_prefix
    sep = "_" if prefix else ""
    write_pred(
        lr_pred,
        test_records,
        OUTPUT_DIR / "supervised" / f"{prefix}{sep}lr_predictions.jsonl",
    )
    logger.info(f"LR predictions: {len(lr_pred)} records")

    # XGBoost predictions
    logger.info("Running XGBoost inference...")
    xgb_pred = xgb_predictor.predict(test_records)
    write_pred(
        xgb_pred,
        test_records,
        OUTPUT_DIR / "supervised" / f"{prefix}{sep}xgb_predictions.jsonl",
    )
    logger.info(f"XGBoost predictions: {len(xgb_pred)} records")

    logger.info(
        f"Trained and evaluated supervised models. Outputs in {OUTPUT_DIR / 'supervised'}"
    )


if __name__ == "__main__":
    basicConfig(level="INFO", format=LOGGING_FORMAT)
    main()
