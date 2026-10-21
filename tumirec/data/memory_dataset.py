"""Reproducible lagged memory windows for the M4 H2 experiment."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class LagConfig:
    """Discrete staleness distribution in seconds; empty weights mean uniform."""
    p_stale: float = 0.5
    choices: tuple[int, ...] = (0, 3600, 86400)
    weights: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        """Validate a nonnegative integer lag distribution."""
        if not 0 <= self.p_stale <= 1 or not self.choices or any(int(x) != x or x < 0 for x in self.choices):
            raise ValueError('Invalid stale probability or lag choices')
        if self.weights and (len(self.weights) != len(self.choices) or
                not np.isfinite(self.weights).all() or min(self.weights) < 0 or sum(self.weights) <= 0):
            raise ValueError('Invalid lag weights')


class MemoryDataset(Dataset):
    """TRAIN-only long/recent histories, independent of loader iteration order."""

    def __init__(self, targets: pd.DataFrame, train: pd.DataFrame, item_ids: list[int],
                 max_history: int = 1000, recent_window: int = 50, seed: int = 42,
                 lag: LagConfig = LagConfig(), fixed_lag: int | None = None) -> None:
        if min(max_history, recent_window) < 1 or seed < 0 or (fixed_lag is not None and (fixed_lag < 0 or int(fixed_lag) != fixed_lag)):
            raise ValueError('Invalid history, seed or fixed lag')
        mapping = {item: row for row, item in enumerate(item_ids)}
        if len(mapping) != len(item_ids) or not (set(train.item_id) | set(targets.item_id)) <= set(mapping):
            raise ValueError('Catalog IDs must be unique and cover all events')
        self.targets = targets.reset_index(drop=True)
        self.users, self.times = self.targets.user_id.to_numpy(), self.targets.timestamp.to_numpy(dtype=np.int64)
        self.rows = self.targets.item_id.map(mapping).to_numpy(dtype=np.int64)
        self.max_history, self.recent_window = max_history, recent_window
        self.seed, self.epoch, self.lag, self.fixed_lag = seed, 0, lag, fixed_lag
        self.histories = {}
        for user, frame in train.groupby('user_id', sort=False):
            frame = frame.sort_values('timestamp', kind='stable')
            self.histories[user] = (frame.timestamp.to_numpy(dtype=np.int64),
                frame.item_id.map(mapping).to_numpy(dtype=np.int64), frame.action.to_numpy(dtype=np.int64))

    def set_epoch(self, epoch: int) -> None:
        """Set the deterministic lag epoch before creating loader workers."""
        if epoch < 0:
            raise ValueError('Epoch cannot be negative')
        self.epoch = epoch

    def __len__(self) -> int:
        """Number of targets."""
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return bounded 1-D row/action/time/mask tensors plus scalar target/lag."""
        if index < 0 or index >= len(self):
            raise IndexError(index)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        lag = self.fixed_lag
        if lag is None:
            weights = np.asarray(self.lag.weights, dtype=float) if self.lag.weights else None
            weights = weights / weights.sum() if weights is not None else None
            lag = int(rng.choice(self.lag.choices, p=weights)) if rng.random() < self.lag.p_stale else 0
        times, rows, actions = self.histories.get(self.users[index], (np.array([], dtype=np.int64),) * 3)
        end = int(np.searchsorted(times, self.times[index], side='left'))
        memory_end = int(np.searchsorted(times, self.times[index] - lag, side='left'))
        recent = rows[max(0, end - self.recent_window):end]
        start = max(0, memory_end - self.max_history)
        length = memory_end - start
        padded = np.full(self.recent_window, -1, dtype=np.int64)
        padded[:len(recent)] = recent
        memory_rows = np.full(self.max_history, -1, dtype=np.int64)
        memory_rows[:length] = rows[start:memory_end]
        memory_times = np.zeros(self.max_history, dtype=np.int64)
        memory_times[:length] = times[start:memory_end]
        memory_actions = np.zeros(self.max_history, dtype=np.int64)
        memory_actions[:length] = actions[start:memory_end]
        result = {'recent': padded, 'recent_length': len(recent), 'memory_rows': memory_rows,
                  'memory_times': memory_times, 'memory_actions': memory_actions,
                  'memory_valid': np.arange(self.max_history) < length,
                  'target_time': self.times[index], 'lag': lag, 'target': self.rows[index],
                  'user_id': int(self.users[index])}
        return {key: torch.as_tensor(value) for key, value in result.items()}
