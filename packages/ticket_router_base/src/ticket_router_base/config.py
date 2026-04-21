"""Shared constants and paths."""

from pathlib import Path


def _find_project_root() -> Path:
    """Locate project root by searching upward for the dataset directory."""
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents]:
        if (parent / "dataset").is_dir():
            return parent
    return Path.cwd()


PROJECT_ROOT = _find_project_root()
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TEST_SAMPLE_NUM = 1200
DIFFICULT_CASE_NUM = 100

SEED = 42

LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
