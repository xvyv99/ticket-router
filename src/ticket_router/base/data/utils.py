"""Train/test split and difficult-case extraction utilities."""

from typing import List
from logging import getLogger

from ticket_router.base.data.base import BaseDataset
from ticket_router.base.types import Record

logger = getLogger(__name__)


def load_dataset(dataset: BaseDataset) -> List[Record]:
    """Load a dataset via its descriptor and return standardized Records."""
    return dataset.load(None)


# TODO: difficult case extraction utilities
