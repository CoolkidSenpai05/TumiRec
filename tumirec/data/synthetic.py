"""Offline synthetic unified-schema tables; timestamps are Unix seconds."""
from pathlib import Path
import numpy as np
import pandas as pd


def generate_synthetic(n_users: int = 500, n_items: int = 300,
                       events_per_user: int = 20, seed: int = 42
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return interactions (U*E rows) and items (I rows), without external data."""
    if min(n_users, n_items, events_per_user) < 1:
        raise ValueError("All dataset sizes must be positive")
    rng = np.random.default_rng(seed)
    size = n_users * events_per_user
    item_ids = rng.integers(n_items, size=size)
    tasks = np.arange(size) % 3
    timestamps = 1_700_000_000 + np.tile(np.arange(events_per_user), n_users) * 3600
    interactions = pd.DataFrame({
        "user_id": np.repeat(np.arange(n_users), events_per_user),
        "item_id": item_ids, "timestamp": timestamps,
        "action": rng.integers(0, 3, size=size),
        "domain": rng.integers(0, 2, size=size), "task": tasks,
        "query_text": [f"Item {i}" if t == 1 else None for i, t in zip(item_ids, tasks)],
        "source_item_id": pd.array([int(rng.integers(n_items)) if t == 2 else None
                                    for t in tasks], dtype="Int64"),
    })
    items = pd.DataFrame({
        "item_id": np.arange(n_items), "title": [f"Item {i}" for i in range(n_items)],
        "category": [f"Category {i % 5}" for i in range(n_items)],
        "description": [f"Synthetic item {i}" for i in range(n_items)],
        "brand": [f"Brand {i % 7}" for i in range(n_items)],
        "price": rng.uniform(1, 100, n_items), "first_seen_ts": 1_699_000_000,
    })
    return interactions, items


def write_synthetic(directory: str | Path, **kwargs: int) -> None:
    """Write interactions.parquet and items.parquet to a local directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    interactions, items = generate_synthetic(**kwargs)
    interactions.to_parquet(directory / "interactions.parquet", index=False)
    items.to_parquet(directory / "items.parquet", index=False)
