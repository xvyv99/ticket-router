from argparse import ArgumentParser
from logging import getLogger, basicConfig
from pathlib import Path

from sklearn.model_selection import train_test_split
from ticket_router_supervised.models.mbert import (
    MBERTTrainer,
    MBERTPredictor,
    MODEL_DIR,
)
from ticket_router_base.utils import write_pred
from ticket_router_base.config import OUTPUT_DIR, LOGGING_FORMAT
from ticket_router_base.data.loader import load_test_set, load_train_set
from ticket_router_base.data.dataset import MultilingualCustomerSupportDataset

logger = getLogger(__name__)


def run_smoke_test():
    logger.info("Smoke Test: 200 samples, 1 epoch")
    dataset = MultilingualCustomerSupportDataset()
    records = load_train_set()[:200]

    all_tasks = dataset.classification_tasks + dataset.ordinal_tasks
    first_task = all_tasks[0].name if all_tasks else None
    stratify = None
    if first_task:
        stratify = [r.labels.get(first_task, "") for r in records]

    train_split, val_split = train_test_split(
        records, test_size=0.2, stratify=stratify, random_state=42
    )
    logger.info(f"Train: {len(train_split)}, Val: {len(val_split)}")
    trainer = MBERTTrainer()
    trainer.train(train_split, dataset, val_split)
    logger.info("Smoke test passed!")


def run_full_training():
    logger.info("Full Training: 4k minus test_set")
    dataset = MultilingualCustomerSupportDataset()
    records = load_train_set()

    all_tasks = dataset.classification_tasks + dataset.ordinal_tasks
    first_task = all_tasks[0].name if all_tasks else None
    stratify = None
    if first_task:
        stratify = [r.labels.get(first_task, "") for r in records]

    train_split, val_split = train_test_split(
        records, test_size=0.2, stratify=stratify, random_state=42
    )
    logger.info(f"Train: {len(train_split)}, Val: {len(val_split)}")
    trainer = MBERTTrainer()
    trainer.train(train_split, dataset, val_split)
    logger.info("Full training done!")


def run_inference():
    dataset = MultilingualCustomerSupportDataset()
    model_paths: dict[str, Path] = {}
    for task in dataset.classification_tasks + dataset.ordinal_tasks:
        path = MODEL_DIR / f"{task.name}_best"
        assert path.exists(), f"Model for {task.name} not found at {path}"
        model_paths[task.name] = path

    predictor = MBERTPredictor(model_paths=model_paths, dataset=dataset)
    test_records = load_test_set()

    logger.info(f"Running inference on {len(test_records)} test records...")
    batch = predictor.predict(test_records)

    output_path = OUTPUT_DIR / "supervised" / "mbert_predictions.jsonl"
    write_pred(batch.predictions, test_records, output_path)

    logger.info(f"Processed {len(batch.predictions)} records. Output: {output_path}")


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
