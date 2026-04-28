from argparse import ArgumentParser
from logging import getLogger, basicConfig

from ticket_router_base.data.base import BaseDataset
from ticket_router_base.data.datasets import DATASET_REGISTRY, get_dataset
from ticket_router_supervised.models.xlm_roberta import (
    XLMRoBERTaTrainer,
    XLMRoBERTaPredictor,
)
from ticket_router_base.config import LOGGING_FORMAT

logger = getLogger(__name__)


def run_smoke_test(dataset: BaseDataset):
    logger.info("Smoke Test: 100 samples, 1 epoch")
    df_train, _, df_val = dataset.load_train_test_split()
    train_split = dataset.df_to_records(df_train)
    val_split = dataset.df_to_records(df_val)

    trainer = XLMRoBERTaTrainer(dataset)
    trainer.train(train_split, val_split, 1)
    logger.info("Smoke test passed!")


def run_full_training(dataset: BaseDataset):
    logger.info("Full Training: 4k minus test_set")
    df_train, _, df_val = dataset.load_train_test_split()
    train_split = dataset.df_to_records(df_train)
    val_split = dataset.df_to_records(df_val)

    logger.info(f"Train: {len(train_split)}, Val: {len(val_split)}")
    trainer = XLMRoBERTaTrainer(dataset)
    trainer.train(train_split, val_split)
    logger.info("Full training done!")


def run_inference(dataset: BaseDataset):
    predictor = XLMRoBERTaPredictor.load_model(dataset)
    _, df_test, _ = dataset.load_train_test_split()
    test_records = dataset.df_to_records(df_test)

    logger.info(f"Running inference on {len(test_records)} test records...")
    preds = predictor.predict(test_records)

    predictor.save_pred_inst(preds, test_records)

    logger.info(
        f"Processed {len(preds)} records. Output: {predictor.get_save_path(dataset=dataset)}"
    )


def main():
    parser = ArgumentParser(description="XLM-RoBERTa training and inference pipeline")
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_REGISTRY.keys()),
        default="multilingual-customer-support",
        help="Dataset to use",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--smoke", action="store_true", help="Run smoke test only")
    group.add_argument("--train", action="store_true", help="Run full training only")
    group.add_argument("--infer", action="store_true", help="Run inference only")

    args = parser.parse_args()

    dataset = get_dataset(args.dataset)()

    if args.smoke:
        run_smoke_test(dataset)
    elif args.train:
        run_full_training(dataset)
    elif args.infer:
        run_inference(dataset)
    else:
        raise ValueError("Must specify one of --smoke, --train, or --infer")


if __name__ == "__main__":
    basicConfig(level="INFO", format=LOGGING_FORMAT)
    main()
