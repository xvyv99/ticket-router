from typing import Tuple
from logging import getLogger

from sklearn.model_selection import StratifiedShuffleSplit

from ticket_router_base.types import ITCusomterSupportDF

logger = getLogger(__name__)


def build_train_test_set(
    df: ITCusomterSupportDF, test_num: int = 1200, seed: int = 42
) -> Tuple[ITCusomterSupportDF, ITCusomterSupportDF]:
    df = df.copy()
    df["_strat"] = (
        df["queue"].astype(str)
        + "|"
        + df["priority"].astype(str)
        + "|"
        + df["language"].astype(str)
    )
    # filter out strata with < 2 samples to avoid split error
    counts = df["_strat"].value_counts()
    valid = df[df["_strat"].isin(counts[counts >= 2].index)].copy()

    assert len(valid) > test_num, (
        f"Not enough valid samples ({len(valid)}) for the requested test_num={test_num}."
    )
    train_num = len(valid) - test_num

    sss = StratifiedShuffleSplit(
        n_splits=1, train_size=train_num, test_size=test_num, random_state=seed
    )

    train_idx, test_idx = next(sss.split(valid, valid["_strat"]))
    train_df: ITCusomterSupportDF = valid.iloc[train_idx].copy()
    test_df: ITCusomterSupportDF = valid.iloc[test_idx].copy()

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_difficult_cases(
    df: ITCusomterSupportDF, n: int = 100, seed: int = 42
) -> ITCusomterSupportDF:
    # heuristic: small queue, high priority, long body, ambiguous keywords
    df = df.copy()
    df["body_len"] = df["body"].fillna("").astype(str).str.len()
    queue_counts = df["queue"].value_counts()
    small_queues = queue_counts[queue_counts < 200].index.tolist()
    df["is_small"] = df["queue"].isin(small_queues).astype(int)
    df["is_high"] = (df["priority"] == "high").astype(int)
    df["score"] = (
        df["is_small"] * 3 + df["is_high"] * 2 + (df["body_len"] > 500).astype(int)
    )
    top = df.sort_values("score", ascending=False).head(n * 3)
    # sample from high-score cases to avoid overly homogeneous patterns
    sample = top.sample(n=min(n, len(top)), random_state=seed)
    return sample.reset_index(drop=True)
