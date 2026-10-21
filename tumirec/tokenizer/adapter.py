"""Two-layer content adapter and training-only behavioral positive pairs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def cointeraction_pairs(train: pd.DataFrame, window_seconds: int = 86400,
                        max_neighbors: int = 20) -> np.ndarray:
    """Return unique item-ID pairs (P, 2) from strictly ordered same-user events.

    Only the supplied TRAIN interactions are used. A positive must have a later
    timestamp within the window and a distinct item ID. At most max_neighbors
    subsequent events are considered per anchor to bound work on long histories.
    """
    if window_seconds <= 0 or max_neighbors < 1:
        raise ValueError("Window and max_neighbors must be positive")
    pairs: set[tuple[int, int]] = set()
    for _, group in train.groupby("user_id", sort=True):
        group = group.sort_values("timestamp", kind="stable")
        ids, times = group.item_id.to_numpy(), group.timestamp.to_numpy()
        for i in range(len(group)):
            for j in range(i + 1, min(len(group), i + max_neighbors + 1)):
                if times[j] - times[i] > window_seconds:
                    break
                if times[j] > times[i] and ids[i] != ids[j]:
                    pairs.add((int(ids[i]), int(ids[j])))
    return np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)


class ContrastiveAdapter(nn.Module):
    """Two-layer MLP mapping content (B, input_dim) to (B, output_dim)."""

    def __init__(self, input_dim: int, output_dim: int = 128,
                 hidden_dim: int = 256, normalize: bool = True) -> None:
        super().__init__()
        if min(input_dim, output_dim, hidden_dim) < 1:
            raise ValueError("Adapter dimensions must be positive")
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, output_dim))
        self.normalize = normalize

    def forward(self, x: Tensor) -> Tensor:
        """Adapt floating point features (B, D_in) into (B, D_out)."""
        output = self.net(x)
        return F.normalize(output, dim=-1) if self.normalize else output


def info_nce(left: Tensor, right: Tensor, temperature: float = 0.07,
             item_pairs: Tensor | None = None) -> Tensor:
    """Symmetric in-batch InfoNCE on (B, D) pairs; return a scalar.

    Repeated target IDs are additional positives; copies of the anchor item are
    excluded as negatives. Other co-interacted items can still be false negatives.
    A singleton batch has no negatives and returns zero loss.
    """
    if left.ndim != 2 or left.shape != right.shape or len(left) == 0 or temperature <= 0:
        raise ValueError("Need matching nonempty (B,D) tensors and positive temperature")
    logits = F.normalize(left.float(), dim=-1) @ F.normalize(right.float(), dim=-1).T
    logits = logits / temperature
    if item_pairs is None:
        labels = torch.arange(len(left), device=left.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    if item_pairs.shape != (len(left), 2):
        raise ValueError("item_pairs must have shape (B,2)")

    def directional(scores: Tensor, anchors: Tensor, targets: Tensor) -> Tensor:
        """Multi-positive cross entropy for one direction of a (B,B) matrix."""
        positives = targets[:, None] == targets[None, :]
        allowed = (anchors[:, None] != targets[None, :]) | positives
        numerator = scores.masked_fill(~positives, -torch.inf).logsumexp(dim=1)
        denominator = scores.masked_fill(~allowed, -torch.inf).logsumexp(dim=1)
        return (denominator - numerator).mean()

    return (directional(logits, item_pairs[:, 0], item_pairs[:, 1]) +
            directional(logits.T, item_pairs[:, 1], item_pairs[:, 0])) / 2
