from argparse import ArgumentParser
from logging import getLogger, basicConfig
from pathlib import Path

from ticket_router_supervised.models.mbert import (
    MBERTTrainer,
    MBERTPredictor,
    MODEL_DIR,
)
from ticket_router_base.utils import write_pred
from ticket_router_base.config import OUTPUT_DIR, LOGGING_FORMAT
from ticket_router_base.data import MultilingualCustomerSupportDataset

logger = getLogger(__name__)


def run_smoke_test():
    logger.info("Smoke Test: 100 samples, 1 epoch")
    dataset = MultilingualCustomerSupportDataset()
    df_train, df_test, df_val = dataset.load_train_test_split()
    train_split = dataset.df_to_records(df_train)
    val_split = dataset.df_to_records(df_val)

    trainer = MBERTTrainer(dataset)
    trainer.train(train_split, val_split, 1)
    logger.info("Smoke test passed!")


def run_full_training():
    logger.info("Full Training: 4k minus test_set")
    dataset = MultilingualCustomerSupportDataset()
    df_train, df_test, df_val = dataset.load_train_test_split()
    train_split = dataset.df_to_records(df_train)
    val_split = dataset.df_to_records(df_val)

    logger.info(f"Train: {len(train_split)}, Val: {len(val_split)}")
    trainer = MBERTTrainer(dataset)
    trainer.train(train_split, val_split)
    logger.info("Full training done!")


def run_inference():
    dataset = MultilingualCustomerSupportDataset()
    model_paths: dict[str, Path] = {}
    for task in dataset.task_descriptor.classification_tasks + dataset.task_descriptor.ordinal_tasks:
        path = MODEL_DIR / f"{task.name}_best"
        assert path.exists(), f"Model for {task.name} not found at {path}"
        model_paths[task.name] = path

    predictor = MBERTPredictor(model_paths=model_paths, dataset=dataset)
    df_train, df_test, df_val = dataset.load_train_test_split()
    test_records = dataset.df_to_records(df_test)

    logger.info(f"Running inference on {len(test_records)} test records...")
    preds = predictor.predict(test_records)

    predictor.save_pred_inst(preds, test_records)

    logger.info(f"Processed {len(preds)} records. Output: {predictor.get_save_path(dataset=dataset)}")


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
