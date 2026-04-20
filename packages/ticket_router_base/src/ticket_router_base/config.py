from pathlib import Path

from .types import Queue, Priority, Language

PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "dataset" / "multilingual-customer-support-tickets"

DATASET_4K_PATH = DATA_DIR / "dataset-tickets-multi-lang3-4k.csv"
DATASET_20K_PATH = DATA_DIR / "dataset-tickets-multi-lang-4-20k.csv"
DATASET_28K_PATH = DATA_DIR / "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
DATASET_GERMAN_NORM_PATH = DATA_DIR / "dataset-tickets-german_normalized.csv"
DATASET_GERMAN_NORM_50_PATH = DATA_DIR / "dataset-tickets-german_normalized_50_5_2.csv"

assert DATASET_4K_PATH.exists(), f"4k dataset not found at {DATASET_4K_PATH}"
assert DATASET_20K_PATH.exists(), f"20k dataset not found at {DATASET_20K_PATH}"
assert DATASET_28K_PATH.exists(), f"28k dataset not found at {DATASET_28K_PATH}"
assert DATASET_GERMAN_NORM_PATH.exists(), (
    f"German normalized dataset not found at {DATASET_GERMAN_NORM_PATH}"
)
assert DATASET_GERMAN_NORM_50_PATH.exists(), (
    f"German normalized dataset not found at {DATASET_GERMAN_NORM_50_PATH}"
)

QUEUES = [m.value for m in Queue]
PRIORITIES = [m.value for m in Priority]
LANGUAGES = [m.value for m in Language]

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TEST_SAMPLE_NUM = 1200
DIFFICULT_CASE_NUM = 100

SEED = 42

LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
