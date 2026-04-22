from logging import getLogger, basicConfig
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Dict

from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

from ticket_router_base.config import OUTPUT_DIR, SEED, LOGGING_FORMAT
from ticket_router_base.utils import combine_text, JSONLLogger
from ticket_router_base.data import load_train_set

CALIB_SIZE = 350
CALIB_OUTPUT = OUTPUT_DIR / "goal_based" / "calibration.jsonl"
MODEL_SIZES = ["0.6B", "1.7B", "4B"]
MODEL_DIR = Path.cwd() / "models"

logger = getLogger(__name__)


def build_calibration_dataset(records: list, size: int = 350) -> List[Dict[str, str]]:
    # Simple stratified sampling by language (if available)
    from collections import Counter

    langs = Counter(r.language or "unknown" for r in records)
    per_lang = max(1, size // max(len(langs), 1))

    sampled = []
    for lang in langs:
        lang_records = [r for r in records if (r.language or "unknown") == lang]
        import random

        random.seed(SEED)
        sampled.extend(random.sample(lang_records, min(per_lang, len(lang_records))))

    if len(sampled) < size:
        remaining = [r for r in records if r not in sampled]
        import random

        random.seed(SEED)
        sampled.extend(
            random.sample(remaining, min(size - len(sampled), len(remaining)))
        )

    records_out = []
    for r in sampled[:size]:
        text = combine_text(r.title or "", r.body)
        records_out.append({"text": text})
    return records_out


def save_calibration(records: List[Dict[str, str]], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with JSONLLogger(output_path) as jsonl:
        for rec in records:
            jsonl.write(rec)

    logger.info(f"Saved {len(records)} calibration samples to {output_path}")


def quantize_model(model_size: str, calibration_dataset: Dataset):
    model_name = f"Qwen/Qwen3-{model_size}-AWQ"
    model_path = MODEL_DIR / f"qwen3-{model_size}-awq"

    logger.info(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info("Starting quantization...")
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=calibration_dataset,
        recipe=[
            SmoothQuantModifier(smoothing_strength=0.8),
            GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
        ],
        max_seq_length=2048,
        num_calibration_samples=len(calibration_dataset),
    )

    logger.info(f"Saving quantized model to {model_path}...")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    logger.info(f"Quantization complete for {model_size}!")


def main():
    parser = ArgumentParser(description="Quantize Qwen3 models")
    parser.add_argument(
        "--sizes",
        nargs="+",
        choices=MODEL_SIZES,
        default=MODEL_SIZES,
        help="Model sizes to quantize",
    )
    args = parser.parse_args()

    train_records = load_train_set()
    calib_records = build_calibration_dataset(train_records, size=CALIB_SIZE)
    save_calibration(calib_records, CALIB_OUTPUT)

    calibration_dataset = Dataset.from_list(calib_records)

    for size in args.sizes:
        logger.info(f"Processing Qwen3-{size}...")
        quantize_model(size, calibration_dataset)

    logger.info("All models quantized successfully!")


if __name__ == "__main__":
    basicConfig(level="INFO", format=LOGGING_FORMAT)
    main()
