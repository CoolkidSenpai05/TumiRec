"""Deterministic temporal and item-level splits."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataSplits:
    """Disjoint target tables and the selected held-out item IDs."""
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    cold_items: frozenset[int]


def temporal_split(events: pd.DataFrame, cold_fraction: float = 0.0,
                   seed: int = 42) -> DataSplits:
    """Leave last event for test, second-last for validation; ties keep input order.

    Cold items are sampled from observed IDs and removed only from train.
    Users with one/two events have no train targets. Histories must still filter
    timestamps strictly, regardless of this stable event ordering.
    """
    if not 0 <= cold_fraction <= 1:
        raise ValueError("cold_fraction must be in [0, 1]")
    ordered = events.sort_values(["user_id", "timestamp"], kind="stable").copy()
    reverse_index = ordered.groupby("user_id").cumcount(ascending=False)
    ids = np.sort(events.item_id.unique())
    count = int(np.floor(len(ids) * cold_fraction))
    cold = frozenset(np.random.default_rng(seed).choice(ids, count, replace=False).tolist())
    return DataSplits(
        ordered.loc[(reverse_index >= 2) & ~ordered.item_id.isin(cold)].reset_index(drop=True),
        ordered.loc[reverse_index == 1].reset_index(drop=True),
        ordered.loc[reverse_index == 0].reset_index(drop=True), cold,
    )
