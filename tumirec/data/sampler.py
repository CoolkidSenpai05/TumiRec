"""CPU sliding windows with strict time boundaries and reproducible memory lag."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


@dataclass(frozen=True)
class HistorySample:
    """Variable-length event frames (H rows / R rows), plus one target record."""
    target: dict
    memory: pd.DataFrame
    recent: pd.DataFrame
    lag_seconds: int


class SlidingWindowSampler(Dataset):
    """One sample per target; random lag is deterministic by seed, epoch and index.

    Pass train events as history_events for the default frozen-history evaluation
    protocol. Passing online history is explicit and still strictly time filtered.
    Memory and recent may overlap; no padding/collation is provided in M1.
    """
    def __init__(self, targets: pd.DataFrame, history_events: pd.DataFrame,
                 max_history: int = 1000, recent_window: int = 50,
                 p_stale: float = 0.0, lag_seconds: tuple[int, ...] = (0, 3600, 86400),
                 fixed_lag_seconds: int | None = None, seed: int = 42) -> None:
        if max_history < 0 or recent_window < 0 or not 0 <= p_stale <= 1:
            raise ValueError("Invalid window length or stale probability")
        if not lag_seconds or any(x < 0 for x in lag_seconds):
            raise ValueError("Lag choices must be nonempty and nonnegative")
        if fixed_lag_seconds is not None and fixed_lag_seconds < 0:
            raise ValueError("Fixed lag must be nonnegative")
        if seed < 0:
            raise ValueError("Seed must be nonnegative")
        self.targets = targets.reset_index(drop=True).copy()
        self.empty = history_events.iloc[:0].copy()
        self.histories = {u: g.sort_values("timestamp", kind="stable").copy()
                          for u, g in history_events.groupby("user_id")}
        self.max_history, self.recent_window = max_history, recent_window
        self.p_stale, self.lag_seconds = p_stale, lag_seconds
        self.fixed_lag_seconds, self.seed, self.epoch = fixed_lag_seconds, seed, 0

    def set_epoch(self, epoch: int) -> None:
        """Select a reproducible epoch-specific lag stream before worker creation."""
        if epoch < 0:
            raise ValueError("Epoch must be nonnegative")
        self.epoch = epoch

    def __len__(self) -> int:
        """Number of target events."""
        return len(self.targets)

    def __getitem__(self, index: int) -> HistorySample:
        """Return bounded windows; memory excludes times >= target_time - lag."""
        if index < 0 or index >= len(self):
            raise IndexError(index)
        target = self.targets.iloc[index].to_dict()
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        lag = self.fixed_lag_seconds
        if lag is None:
            lag = int(rng.choice(self.lag_seconds)) if rng.random() < self.p_stale else 0
        history = self.histories.get(target["user_id"], self.empty)
        recent = history.loc[history.timestamp < target["timestamp"]].tail(self.recent_window)
        memory = history.loc[history.timestamp < target["timestamp"] - lag].tail(self.max_history)
        return HistorySample(target, memory.copy(), recent.copy(), lag)
