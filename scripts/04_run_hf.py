from argparse import ArgumentParser
from logging import getLogger, basicConfig

from ticket_router_base.data.base import BaseDataset
from ticket_router_base.data.datasets import DATASET_REGISTRY, get_dataset
from ticket_router_supervised.models.mbert import MBERTTrainer, MBERTPredictor
from ticket_router_supervised.models.xlm_roberta import (
    XLMRoBERTaTrainer,
    XLMRoBERTaPredictor,
)
from ticket_router_base.config import LOGGING_FORMAT

logger = getLogger(__name__)

MODELS = {
    MBERTPredictor.name: (MBERTTrainer, MBERTPredictor),
    XLMRoBERTaPredictor.name: (XLMRoBERTaTrainer, XLMRoBERTaPredictor),
}


def run_smoke_test(dataset: BaseDataset, trainer_cls):
    logger.info(f"Smoke Test: 100 samples, 1 epoch ({trainer_cls.predictor_cls.name})")
    df_train, _, df_val = dataset.load_train_test_split()
    train_split = dataset.df_to_records(df_train)
    val_split = dataset.df_to_records(df_val)

    trainer = trainer_cls(dataset)
    trainer.train(train_split, val_split, 1)
    logger.info("Smoke test passed!")


def run_full_training(dataset: BaseDataset, trainer_cls, epochs: int = 3):
    logger.info(f"Full Training: 4k minus test_set ({trainer_cls.predictor_cls.name}), epochs={epochs}")
    df_train, _, df_val = dataset.load_train_test_split()
    train_split = dataset.df_to_records(df_train)
    val_split = dataset.df_to_records(df_val)

    logger.info(f"Train: {len(train_split)}, Val: {len(val_split)}")
    trainer = trainer_cls(dataset)
    trainer.train(train_split, val_split, epochs=epochs)
    logger.info("Full training done!")


def run_inference(dataset: BaseDataset, predictor_cls):
    predictor = predictor_cls.load_model(dataset)
    _, df_test, _ = dataset.load_train_test_split()
    test_records = dataset.df_to_records(df_test, need_inject_inferred=True)

    logger.info(f"Running inference on {len(test_records)} test records...")
    preds = predictor.predict(test_records)

    predictor.save_pred_inst(preds, test_records)

    logger.info(
        f"Processed {len(preds)} records. Output: {predictor.get_save_path(dataset=dataset)}"
    )


def main():
    parser = ArgumentParser(description="HF model training and inference pipeline")
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_REGISTRY.keys()),
        default="multilingual-customer-support",
        help="Dataset to use",
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default="xlm-roberta",
        help="Model to run",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--smoke", action="store_true", help="Run smoke test only")
    group.add_argument("--train", action="store_true", help="Run full training only")
    group.add_argument("--infer", action="store_true", help="Run inference only")
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3, use 8 for xlm-roberta)",
    )

    args = parser.parse_args()

    dataset = get_dataset(args.dataset)()
    trainer_cls, predictor_cls = MODELS[args.model]

    if args.smoke:
        run_smoke_test(dataset, trainer_cls)
    elif args.train:
        run_full_training(dataset, trainer_cls, epochs=args.epochs)
    elif args.infer:
        run_inference(dataset, predictor_cls)
    else:
        raise ValueError("Must specify one of --smoke, --train, or --infer")


if __name__ == "__main__":
    basicConfig(level="INFO", format=LOGGING_FORMAT)
    main()
