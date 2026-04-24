"""Train/test split and difficult-case extraction utilities."""

from typing import List
from logging import getLogger

from ticket_router_base.data.base import BaseDataset
from ticket_router_base.types import Record

logger = getLogger(__name__)


def load_dataset(dataset: BaseDataset) -> List[Record]:
    """Load a dataset via its descriptor and return standardized Records."""
    return dataset.load(None)


def build_difficult_cases(
    records: List[Record],
    dataset: BaseDataset,
    n: int = 100,
    seed: int = 42,
) -> List[Record]:
    """Heuristic extraction of difficult cases based on small classes, high first-label, long body.

    Args:
        records: List of Record instances.
        dataset: Dataset descriptor.
        n: Number of difficult cases to extract.
        seed: Random seed.
    """
    # TODO: fix it

    return []
