from argparse import ArgumentParser
from logging import getLogger, basicConfig

from ticket_router_base.data import get_dataset
from ticket_router_base.config import LOGGING_FORMAT
from ticket_router_base.data.datasets import DATASET_REGISTRY

from ticket_router_agent.config import MODEL_CHOICES
from ticket_router_agent.infer import vLLMPredictor

logger = getLogger(__name__)


def run_infer(dataset_name: str, sample_num: int, model_choice: str, few_shot: bool):
    dataset_type = get_dataset(dataset_name)
    dataset = dataset_type()

    _, df_test, _ = dataset.load_train_test_split()
    test_records = dataset.df_to_records(df_test)[:sample_num]

    vllm_predictor = vLLMPredictor(
        model_name_or_path=model_choice,
        dataset=dataset,
        few_shot=few_shot,
    )

    llm_batch = vllm_predictor.predict(test_records)

    vllm_predictor.save_pred(
        llm_batch,
        test_records,
    )


def main():
    parser = ArgumentParser(description="Local vLLM inference")
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_REGISTRY.keys()),
        default="multilingual-customer-support",
        help="Dataset to use",
    )
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
        run_infer(args.dataset, args.sample_num, model_choice, args.few_shot)
        logger.info(f"Inference complete for {model_choice}")


if __name__ == "__main__":
    basicConfig(level="INFO", format=LOGGING_FORMAT)
    main()
