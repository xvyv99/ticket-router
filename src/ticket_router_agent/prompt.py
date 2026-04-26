"""System prompt builder using dataset descriptors."""

from typing import List

from ticket_router_base.data import BaseDataset
from ticket_router_base.types import GroundRecord


def build_system_prompt(
    dataset: BaseDataset, few_shot_examples: List[GroundRecord] | None = None
) -> str:
    # TODO
    ...
