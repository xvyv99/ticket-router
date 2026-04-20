from logging import getLogger, basicConfig
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Dict

import pandas as pd
from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

from ticket_router_base.types import RecordDF
from ticket_router_base.config import OUTPUT_DIR, SEED, LOGGING_FORMAT
from ticket_router_base.utils import combine_text, JSONLLogger
from ticket_router_base.data.loader import load_train_set

CALIB_SIZE = 350
CALIB_OUTPUT = OUTPUT_DIR / "goal_based" / "calibration.jsonl"
MODEL_SIZES = ["0.6B", "1.7B", "4B"]
MODEL_DIR = Path.cwd() / "models"

logger = getLogger(__name__)


def build_calibration_dataset(df: RecordDF, size: int = 350) -> List[Dict[str, str]]:
    langs = df["language"].unique().tolist()
    queues = df["queue"].unique().tolist()
    per_cell = max(1, size // (len(langs) * len(queues)))

    sampled = df.groupby(["language", "queue"], group_keys=False).apply(
        lambda g: g.sample(n=min(per_cell, len(g)), random_state=SEED)
    )
    if len(sampled) < size:
        extra = df.sample(n=size - len(sampled), random_state=SEED)
        sampled = pd.concat([sampled, extra])

    records = []
    for _, row in sampled.iterrows():
        subject = row["subject"] or ""
        body = row["body"] or ""
        text = combine_text(subject, body)
        records.append({"text": text})
    return records[:size]


def save_calibration(records: List[Dict[str, str]], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with JSONLLogger(output_path) as jsonl:
        for rec in records:
            jsonl.write(rec)

    logger.info(f"Calibration: {len(records)} records -> {output_path}")


def quantize_model(size: str, calib_texts: List[Dict[str, str]]):
    model_name = f"Qwen/Qwen3-{size}"
    quant_dir = MODEL_DIR / f"qwen3-{size}-awq"

    if quant_dir.exists() and (quant_dir / "config.json").exists():
        logger.info(f"Already quantized: {quant_dir}, skipping.")
        return

    logger.info(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # W8A8
    recipe = [
        SmoothQuantModifier(smoothing_strength=0.8),
        GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
    ]
    calib_ds = Dataset.from_list(calib_texts)

    logger.info(f"Quantizing {size}...")
    oneshot(
        model=model,
        dataset=calib_ds,
        recipe=recipe,
        max_seq_length=512,
        num_calibration_samples=len(calib_ds),
    )
    model.save_pretrained(quant_dir, save_compressed=True)
    tokenizer.save_pretrained(quant_dir)
    logger.info(f"Done: {size} -> {quant_dir}")


def main():
    parser = ArgumentParser(description="Quantize Qwen models with calibration data")
    parser.add_argument(
        "--calib", action="store_true", help="Only prepare calibration dataset"
    )
    parser.add_argument(
        "--quantize",
        nargs="?",
        const="all",
        choices=MODEL_SIZES,
        help="Quantize specified model size (default: all)",
    )
    args = parser.parse_args()

    df = load_train_set()
    cal_texts = build_calibration_dataset(df, size=CALIB_SIZE)

    if args.quantize:
        target = args.quantize

        if target == "all":
            for size in MODEL_SIZES:
                logger.info(f"Quantizing {size}...")
                quantize_model(size, cal_texts)
        else:
            logger.info(f"Quantizing {target}...")
            quantize_model(target, cal_texts)
        return

    if args.calib:
        save_calibration(cal_texts, CALIB_OUTPUT)

        logger.info(f"Calibration dataset ready: {len(cal_texts)} records")
        return


if __name__ == "__main__":
    basicConfig(level="INFO", format=LOGGING_FORMAT)
    main()
