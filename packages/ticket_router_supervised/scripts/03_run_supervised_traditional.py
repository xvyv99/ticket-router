"""Train supervised models (LR, XGBoost) avoiding test-set leakage."""

from logging import getLogger, basicConfig

from sklearn.model_selection import train_test_split
from ticket_router_base.data.loader import load_test_set, load_train_set
from ticket_router_base.datasets import MultilingualCustomerSupportDataset
from ticket_router_base.utils import write_pred
from ticket_router_base.config import OUTPUT_DIR, SEED, LOGGING_FORMAT

from ticket_router_supervised.models.lr import LRTrainer
from ticket_router_supervised.models.xgb import XGBTrainer

logger = getLogger(__name__)


def main():
    dataset = MultilingualCustomerSupportDataset()
    test_records = load_test_set()
    train_records = load_train_set()

    # Stratify by the first task (queue proxy) to maintain distribution
    first_task = (
        dataset.classification_tasks[0].name if dataset.classification_tasks else None
    )
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
    write_pred(
        lr_batch.predictions,
        test_records,
        OUTPUT_DIR / "supervised" / "lr_predictions.jsonl",
    )
    logger.info(f"LR predictions: {len(lr_batch.predictions)} records")

    # XGBoost predictions
    logger.info("Running XGBoost inference...")
    xgb_batch = xgb_predictor.predict(test_records)
    write_pred(
        xgb_batch.predictions,
        test_records,
        OUTPUT_DIR / "supervised" / "xgb_predictions.jsonl",
    )
    logger.info(f"XGBoost predictions: {len(xgb_batch.predictions)} records")

    logger.info(
        f"Trained and evaluated supervised models. Outputs in {OUTPUT_DIR / 'supervised'}"
    )


if __name__ == "__main__":
    basicConfig(level="INFO", format=LOGGING_FORMAT)
    main()
