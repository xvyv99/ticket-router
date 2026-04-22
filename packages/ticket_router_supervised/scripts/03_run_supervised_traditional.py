"""Train supervised models (LR, XGBoost) avoiding test-set leakage."""

from argparse import ArgumentParser
from logging import getLogger, basicConfig

from sklearn.model_selection import train_test_split
from ticket_router_base.data.dataset import get_dataset
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

    # Load from JSONL
    def _load(path_name: str):

        path = OUTPUT_DIR / path_name
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run prepare-data first.")
        # parse JSONL into Records
        from ticket_router_base.data.loader import _load_jsonl_records
        return _load_jsonl_records(path)

    test_records = _load(args.test_set)
    train_records = _load(args.train_set)

    # Stratify by the first task (classification or ordinal)
    all_tasks = dataset.classification_tasks + dataset.ordinal_tasks
    first_task = all_tasks[0].name if all_tasks else None
    stratify = None
    if first_task:
        stratify = [r.labels.get(first_task, "") for r in train_records]

    train_split, val_split = train_test_split(
        train_records,
        test_size=0.2,
        random_state=SEED,
        stratify=stratify,
    )

    # Train LR models
    logger.info("Training LR models...")
    lr_trainer = LRTrainer()
    lr_predictor = lr_trainer.train(train_split, dataset, val_split)
    logger.info("LR models trained successfully.")

    # Train XGBoost models
    logger.info("Training XGBoost models...")
    xgb_trainer = XGBTrainer()
    xgb_predictor = xgb_trainer.train(train_split, dataset, val_split)
    logger.info("XGBoost models trained successfully.")

    # LR predictions
    logger.info("Running LR inference...")
    lr_batch = lr_predictor.predict(test_records)
    prefix = args.output_prefix
    sep = "_" if prefix else ""
    write_pred(
        lr_batch.predictions,
        test_records,
        OUTPUT_DIR / "supervised" / f"{prefix}{sep}lr_predictions.jsonl",
    )
    logger.info(f"LR predictions: {len(lr_batch.predictions)} records")

    # XGBoost predictions
    logger.info("Running XGBoost inference...")
    xgb_batch = xgb_predictor.predict(test_records)
    write_pred(
        xgb_batch.predictions,
        test_records,
        OUTPUT_DIR / "supervised" / f"{prefix}{sep}xgb_predictions.jsonl",
    )
    logger.info(f"XGBoost predictions: {len(xgb_batch.predictions)} records")

    logger.info(
        f"Trained and evaluated supervised models. Outputs in {OUTPUT_DIR / 'supervised'}"
    )


if __name__ == "__main__":
    basicConfig(level="INFO", format=LOGGING_FORMAT)
    main()
