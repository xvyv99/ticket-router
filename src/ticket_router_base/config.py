"""Shared constants and paths."""

from pathlib import Path


PROJECT_ROOT = Path.cwd()

DATASET_DIR = PROJECT_ROOT / "dataset"
assert DATASET_DIR.exists(), f"Dataset directory not found at {DATASET_DIR}"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TRAIN_SET_PATH = OUTPUT_DIR / "train_set.jsonl"
TEST_SET_PATH = OUTPUT_DIR / "test_set.jsonl"

DIFFICULT_CASE_NUM = 100

SEED = 42

LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
