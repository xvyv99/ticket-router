"""CLI script to train and run the rule-based predictor."""

from argparse import ArgumentParser
from logging import getLogger, basicConfig

from ticket_router_base.config import LOGGING_FORMAT
from ticket_router_base.data import get_dataset
from ticket_router_base.data.datasets import DATASET_REGISTRY

from ticket_router_rule.cfg import RuleBasedCfg
from ticket_router_rule.predictor import RuleBasedTrainer

logger = getLogger(__name__)


def main():
    parser = ArgumentParser(description="Train and run rule-based predictor")
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_REGISTRY.keys()),
        default="multilingual-customer-support",
        help="Dataset to use",
    )
    parser.add_argument(
        "--no-candidate-search",
        dest="candidate_search",
        action="store_false",
        default=True,
        help="Disable candidate search (use fixed hyperparameters)",
    )
    parser.add_argument(
        "--sample-num",
        type=int,
        default=0,
        help="Number of test samples to run on (0 = all)",
    )
    args = parser.parse_args()

    dataset_type = get_dataset(args.dataset)
    dataset = dataset_type()

    assert dataset.name == "multilingual-customer-support", (
        "Rule-based predictor only supports multilingual-customer-support dataset!"
    )  # FIXME

    logger.info(f"Loading train/test split for {args.dataset}...")
    df_train, df_test, df_valid = dataset.load_train_test_split()

    train_records = dataset.df_to_records(df_train)
    valid_records = dataset.df_to_records(df_valid)
    test_records = dataset.df_to_records(df_test, need_inject_inferred=True)

    if args.sample_num > 0:
        test_records = test_records[: args.sample_num]

    logger.info(
        f"Loaded {len(train_records)} train, {len(valid_records)} valid, "
        f"{len(test_records)} test records"
    )

    cfg = RuleBasedCfg(enable_candidate_search=args.candidate_search)
    trainer = RuleBasedTrainer(dataset=dataset, cfg=cfg)

    logger.info("Training rule-based models...")
    predictor = trainer.train(train_records, val_records=valid_records)
    logger.info("Training complete.")

    logger.info(f"Running prediction on {len(test_records)} test records...")
    predictions = predictor.predict(test_records)
    predictor.save_pred_inst(predictions, test_records, run_id=0)
    logger.info("Prediction complete. Saved to rule_based output directory.")


if __name__ == "__main__":
    basicConfig(level="INFO", format=LOGGING_FORMAT)
    main()
