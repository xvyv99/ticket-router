"""Shared constants and paths."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path.cwd()

# Load .env into environment variables
load_dotenv()

INFERRED_MODEL_CHOICE = os.getenv("INFERRED_MODEL_CHOICE")
assert INFERRED_MODEL_CHOICE, "INFERRED_MODEL_CHOICE not set in environment variables"

DATASET_DIR = PROJECT_ROOT / "dataset"
assert DATASET_DIR.exists(), f"Dataset directory not found at {DATASET_DIR}"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DIFFICULT_CASE_NUM = 100

SEED = 42

LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
