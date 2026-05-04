"""Batch API CLI: generate requests (--gen) or parse results (--parse)."""

import re
from argparse import ArgumentParser
from logging import basicConfig, getLogger
from pathlib import Path

from ticket_router.base.config import LOGGING_FORMAT
from ticket_router.base.data import DATASET_REGISTRY, get_dataset

from ticket_router.llm.batch_api import BatchAPIPredictor, ModelConfig

logger = getLogger(__name__)

DEFAULT_THINKING_BUDGET = 256
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7

MODELS_CFG = [
    ModelConfig(
        name="DeepSeek-R1",
        model_id="deepseek-ai/DeepSeek-R1",
        thinking_budget=DEFAULT_THINKING_BUDGET,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
    ),
    ModelConfig(
        name="QwQ-32B",
        model_id="Qwen/QwQ-32B",
        thinking_budget=DEFAULT_THINKING_BUDGET,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
    ),
    ModelConfig(
        name="V3.1-Terminus",
        model_id="deepseek-ai/DeepSeek-V3.1-Terminus",
        thinking_budget=None,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
    ),
    ModelConfig(
        name="qwen3.5-flash(no thinking)",
        model_id="qwen3.5-flash",
        thinking_budget=None,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
    ),
    ModelConfig(
        name="qwen3.5-flash(thinking)",
        model_id="qwen3.5-flash",
        thinking_budget=DEFAULT_THINKING_BUDGET,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
    ),
    ModelConfig(
        name="qwen3.5-plus(no thinking)",
        model_id="qwen3.5-plus",
        thinking_budget=None,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
    ),
    ModelConfig(
        name="qwen3.5-plus(thinking)",
        model_id="qwen3.5-plus",
        thinking_budget=DEFAULT_THINKING_BUDGET,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
    ),
]

# name -> ModelConfig lookup
MODEL_CFG_MAP = {cfg.name: cfg for cfg in MODELS_CFG}


def _build_cfg_by_name(name: str) -> ModelConfig | None:
    """Look up a ModelConfig by its display name."""
    return MODEL_CFG_MAP.get(name)


def _scan_result_files(
    result_dir: Path, few_shot: bool = True
) -> list[tuple[str, Path]]:
    """Scan result_dir for batch result files and return (model_name, path) pairs."""
    suffix = "fewshot" if few_shot else "zeroshot"
    pattern = re.compile(rf"batch_(.+?)_{suffix}_result\.jsonl$")
    found: list[tuple[str, Path]] = []
    for path in sorted(result_dir.glob(f"batch_*_{suffix}_result.jsonl")):
        match = pattern.match(path.name)
        if match:
            model_name = match.group(1)
            found.append((model_name, path))
    return found


def do_gen(dataset_name: str, few_shot: bool = True) -> None:
    """Generate batch request JSONL files for all configured models."""
    dataset_type = get_dataset(dataset_name)
    dataset = dataset_type()
    _, df_test, _ = dataset.load_train_test_split()
    test_records = dataset.df_to_records(df_test)

    suffix = "fewshot" if few_shot else "zeroshot"
    for cfg in MODELS_CFG:
        predictor = BatchAPIPredictor(dataset, cfg, few_shot=few_shot)
        save_path = (
            BatchAPIPredictor.DEFAULT_SAVE_DIR
            / "batch_file"
            / f"batch_{predictor.sub_name}_{suffix}.jsonl"
        )
        predictor.gen_batch(test_records, save_path=save_path)

    logger.info("Batch generation complete.")


def do_parse(dataset_name: str, result_dir: Path, few_shot: bool = True) -> None:
    """Parse batch result JSONL files and save predictions."""
    dataset_type = get_dataset(dataset_name)
    dataset = dataset_type()
    _, df_test, _ = dataset.load_train_test_split()
    test_records = dataset.df_to_records(df_test, need_inject_inferred=True)

    result_files = _scan_result_files(result_dir, few_shot)
    if not result_files:
        logger.warning(f"No result files found in {result_dir}")
        return

    for model_name, result_path in result_files:
        cfg = _build_cfg_by_name(model_name)
        if cfg is None:
            logger.warning(
                f"No ModelConfig found for '{model_name}' (from {result_path.name}), skipping"
            )
            continue

        predictor = BatchAPIPredictor(dataset, cfg, few_shot=few_shot)
        predictor.parse_result(result_path, test_records, run_id=0)
        logger.info(f"Parsed {result_path.name}")

    logger.info("Batch parsing complete.")


def main() -> None:
    parser = ArgumentParser(description="Batch API: generate requests or parse results")
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_REGISTRY.keys()),
        default="multilingual-customer-support",
        help="Dataset to use",
    )
    parser.add_argument(
        "--no-few-shot",
        dest="few_shot",
        action="store_false",
        default=True,
        help="Disable few-shot prompting (default: enabled)",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--gen",
        action="store_true",
        help="Generate batch request JSONL files",
    )
    group.add_argument(
        "--parse",
        nargs="?",
        const="outputs_bak/goal_based/batch_result",
        default=None,
        help="Parse batch result JSONL files. Optional path to result directory (default: outputs_bak/goal_based/batch_result)",
    )

    args = parser.parse_args()

    if args.gen:
        do_gen(args.dataset, args.few_shot)
    elif args.parse is not None:
        result_dir = Path(args.parse)
        if not result_dir.exists():
            raise FileNotFoundError(f"Result directory not found: {result_dir}")
        do_parse(args.dataset, result_dir, args.few_shot)


if __name__ == "__main__":
    basicConfig(level="INFO", format=LOGGING_FORMAT)
    main()
