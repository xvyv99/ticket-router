from argparse import ArgumentParser
from logging import getLogger, basicConfig

from ticket_router_base.data.loader import load_test_set
from ticket_router_base.config import LOGGING_FORMAT, OUTPUT_DIR
from ticket_router_base.utils import write_pred
from ticket_router_base.datasets import MultilingualCustomerSupportDataset

from ticket_router_agent.config import MODEL_CHOICES
from ticket_router_agent.infer import vLLMPredictor
from ticket_router_agent.utils import save_prefix_from_model_choice

logger = getLogger(__name__)


def run_infer(sample_num: int, model_choice: str, few_shot: bool):
    test_records = load_test_set()[:sample_num]
    dataset = MultilingualCustomerSupportDataset()

    predictor = vLLMPredictor(
        model_name_or_path=model_choice,
        dataset=dataset,
        few_shot=few_shot,
    )

    llm_batch = predictor.predict(test_records)

    save_prefix = save_prefix_from_model_choice(model_choice)
    save_path = (
        OUTPUT_DIR
        / "goal_based"
        / f"{save_prefix}_{'few_shot' if few_shot else 'zero_shot'}_predictions.jsonl"
    )

    write_pred(
        llm_batch.predictions,
        test_records,
        save_path,
    )


def main():
    parser = ArgumentParser(description="Local vLLM inference")
    parser.add_argument(
        "model_choice",
        nargs="+",
        choices=MODEL_CHOICES,
        help="Model choices to run",
    )
    parser.add_argument(
        "--no-few-shot",
        dest="few_shot",
        action="store_false",
        default=True,
        help="Enable few-shot prompting (default: true)",
    )
    parser.add_argument(
        "--sample-num",
        type=int,
        default=1200,
        help="Number of samples to run inference on",
    )
    args = parser.parse_args()

    for model_choice in args.model_choice:
        logger.info(f"Running inference with {model_choice}...")
        run_infer(args.sample_num, model_choice, args.few_shot)
        logger.info(f"Inference complete for {model_choice}")


if __name__ == "__main__":
    basicConfig(level="INFO", format=LOGGING_FORMAT)
    main()
