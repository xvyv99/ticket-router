"""Train supervised models (LR, XGBoost) on 4k data, avoiding test-set leakage."""

from logging import getLogger, basicConfig

from sklearn.model_selection import train_test_split
from ticket_router_base.data.loader import load_test_set, load_train_set
from ticket_router_base.utils import write_pred
from ticket_router_base.config import OUTPUT_DIR, SEED

from ticket_router_supervised.models.lr import LRTrainer
from ticket_router_supervised.models.xgb import XGBTrainer

logger = getLogger(__name__)


def main():
    df_test = load_test_set()
    df_train = load_train_set()

    train_records, val_records = train_test_split(
        df_train,
        test_size=0.2,
        random_state=SEED,
        stratify=df_train["queue"],  # Stratify by queue to maintain distribution
    )

    # Train LR models
    logger.info("Training LR models...")
    lr_trainer = LRTrainer()
    lr_predictor = lr_trainer.train(train_records, val_records)
    logger.info("LR models trained successfully.")

    # Train XGBoost models
    logger.info("Training XGBoost models...")
    xgb_trainer = XGBTrainer()
    xgb_predictor = xgb_trainer.train(train_records, val_records)
    logger.info("XGBoost models trained successfully.")

    # LR predictions
    logger.info("Running LR inference...")
    lr_batch = lr_predictor.predict(df_test)

    write_pred(
        lr_batch.predictions,
        df_test,
        OUTPUT_DIR / "supervised" / "lr_predictions.jsonl",
    )
    logger.info(f"LR predictions: {len(lr_batch.predictions)} records")

    # XGBoost predictions
    logger.info("Running XGBoost inference...")
    xgb_batch = xgb_predictor.predict(df_test)
    write_pred(
        xgb_batch.predictions,
        df_test,
        OUTPUT_DIR / "supervised" / "xgb_predictions.jsonl",
    )
    logger.info(f"XGBoost predictions: {len(xgb_batch.predictions)} records")

    logger.info(
        f"Trained and evaluated supervised models. Outputs in {OUTPUT_DIR / 'supervised'}"
    )


if __name__ == "__main__":
    basicConfig(level="INFO")
    main()
