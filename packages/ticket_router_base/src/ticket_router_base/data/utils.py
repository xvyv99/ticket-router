"""Train/test split and difficult-case extraction utilities."""

from typing import List, Tuple
from logging import getLogger

from sklearn.model_selection import StratifiedShuffleSplit

from ticket_router_base.data.base import BaseDataset
from ticket_router_base.types import Record

logger = getLogger(__name__)


def build_train_test_set(
    records: List[Record],
    dataset: BaseDataset,
    test_num: int = 1200,
    seed: int = 42,
) -> Tuple[List[Record], List[Record]]:
    """Stratified split using all classification tasks + language as the stratification key.

    Args:
        records: List of Record instances.
        dataset: Dataset descriptor providing task and column definitions.
        test_num: Number of test samples.
        seed: Random seed.
    """
    # build stratification string from all tasks + language
    all_tasks = dataset.classification_tasks + dataset.ordinal_tasks
    parts: List[List[str]] = [[] for _ in range(len(records))]
    for task in all_tasks:
        for i, rec in enumerate(records):
            parts[i].append(rec.labels.get(task.name, "unknown"))
    if dataset.language_column:
        for i, rec in enumerate(records):
            parts[i].append(rec.language or "unknown")

    strat_labels = ["|".join(p) for p in parts]

    # filter out strata with < 2 samples
    from collections import Counter

    counts = Counter(strat_labels)
    valid_indices = [i for i, label in enumerate(strat_labels) if counts[label] >= 2]
    valid_records = [records[i] for i in valid_indices]
    valid_strats = [strat_labels[i] for i in valid_indices]

    assert len(valid_records) > test_num, (
        f"Not enough valid samples ({len(valid_records)}) for test_num={test_num}."
    )
    train_num = len(valid_records) - test_num

    sss = StratifiedShuffleSplit(
        n_splits=1, train_size=train_num, test_size=test_num, random_state=seed
    )

    train_idx, test_idx = next(sss.split(valid_records, valid_strats))
    train_records = [valid_records[i] for i in train_idx]
    test_records = [valid_records[i] for i in test_idx]

    return train_records, test_records


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
    import random

    # body length
    body_lens = [len(r.body) for r in records]

    # identify small classes (uses the first task)
    all_tasks = dataset.classification_tasks + dataset.ordinal_tasks
    first_task = all_tasks[0] if all_tasks else None
    if first_task:
        from collections import Counter

        task_counts = Counter(r.labels.get(first_task.name, "") for r in records)
        small_classes = {q for q, c in task_counts.items() if c < 200}
        is_small = [
            1 if r.labels.get(first_task.name, "") in small_classes else 0
            for r in records
        ]
    else:
        is_small = [0] * len(records)

    # high-label proxy (uses the second task)
    second_task = all_tasks[1] if len(all_tasks) > 1 else None
    if second_task:
        high_label = sorted(second_task.labels)[0]
        is_high = [
            1 if r.labels.get(second_task.name, "") == high_label else 0
            for r in records
        ]
    else:
        is_high = [0] * len(records)

    scores = [
        is_small[i] * 3 + is_high[i] * 2 + (1 if body_lens[i] > 500 else 0)
        for i in range(len(records))
    ]

    # sort by score descending and take top n*3
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    top = indexed[: n * 3]

    # sample from top
    sample_size = min(n, len(top))
    random.seed(seed)
    chosen = random.sample(top, sample_size)
    chosen_indices = [i for i, _ in chosen]
    return [records[i] for i in chosen_indices]
