"""Shared constants and paths."""

from pathlib import Path

PROJECT_ROOT = Path.cwd()
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TEST_SAMPLE_NUM = 1200
DIFFICULT_CASE_NUM = 100

SEED = 42

LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
