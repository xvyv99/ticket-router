from argparse import ArgumentParser
from logging import getLogger, basicConfig

from sklearn.model_selection import train_test_split
from ticket_router_supervised.models.mbert import (
    PRIORITY_MODEL_PATH,
    QUEUE_MODEL_PATH,
    MBERTTrainer,
    MBERTPredictor,
)
from ticket_router_base.utils import write_pred
from ticket_router_base.config import OUTPUT_DIR, LOGGING_FORMAT
from ticket_router_base.data.loader import load_test_set, load_train_set

logger = getLogger(__name__)


def run_smoke_test():
    logger.info("Smoke Test: 200 samples, 1 epoch")
    df_small = load_train_set().head(200)

    train_df, val_df = train_test_split(
        df_small, test_size=0.2, stratify=df_small["queue"], random_state=42
    )
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}")
    trainer = MBERTTrainer()
    trainer.train(train_df, val_df)
    logger.info("Smoke test passed!")


def run_full_training():
    logger.info("Full Training: 4k minus test_set")

    df = load_train_set()
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["queue"], random_state=42
    )
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}")
    trainer = MBERTTrainer()
    trainer.train(train_df, val_df)
    logger.info("Full training done!")


def run_inference():
    assert QUEUE_MODEL_PATH.exists(), f"Queue model not found at {QUEUE_MODEL_PATH}"
    assert PRIORITY_MODEL_PATH.exists(), (
        f"Priority model not found at {PRIORITY_MODEL_PATH}"
    )

    predictor = MBERTPredictor(
        queue_model_path=QUEUE_MODEL_PATH, priority_model_path=PRIORITY_MODEL_PATH
    )

    df_test = load_test_set()

    logger.info(f"Running inference on {len(df_test)} test records...")
    batch = predictor.predict(df_test)

    # TODO: make output_path configurable and easy to query
    output_path = OUTPUT_DIR / "supervised" / "mbert_predictions.jsonl"
    write_pred(batch.predictions, df_test, output_path)

    logger.info(
        f"Processed {len(batch.predictions)} records. Output: {OUTPUT_DIR / 'supervised' / 'mbert_predictions.jsonl'}"
    )


def main():
    parser = ArgumentParser(description="Training and inference pipeline")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--smoke", action="store_true", help="Run smoke test only")
    group.add_argument("--train", action="store_true", help="Run full training only")
    group.add_argument("--infer", action="store_true", help="Run inference only")

    args = parser.parse_args()

    if args.smoke:
        run_smoke_test()
    elif args.train:
        run_full_training()
    elif args.infer:
        run_inference()
    else:
        logger.info("No specific mode selected, running all steps sequentially...")
        run_full_training()
        run_inference()


if __name__ == "__main__":
    basicConfig(level="INFO", format=LOGGING_FORMAT)
    main()
