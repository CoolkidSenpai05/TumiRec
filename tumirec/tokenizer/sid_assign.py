"""Deterministic semantic IDs with catalog-local collision resolution."""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from tumirec.tokenizer.rqvae import RQVAE


def assign_sids(item_ids: Sequence[int], codes: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Return item_id -> code_0..code_L table, including one collision suffix.

    Codes have shape (N,L); suffixes count up within a tuple in sorted item-ID
    order. Collision rate is (N - number of distinct base tuples) / N.
    """
    item_ids = list(item_ids)
    codes = np.asarray(codes)
    if (codes.ndim != 2 or codes.shape[0] != len(item_ids) or codes.shape[1] < 1
            or len(set(item_ids)) != len(item_ids) or len(item_ids) == 0
            or not np.issubdtype(codes.dtype, np.integer) or (codes < 0).any()):
        raise ValueError("Need unique IDs and aligned nonnegative integer codes (N,L)")
    tuples = [tuple(row) for row in codes.tolist()]
    counts: Counter = Counter()
    suffix = np.zeros(len(item_ids), dtype=np.int64)
    for index in sorted(range(len(item_ids)), key=lambda i: item_ids[i]):
        suffix[index] = counts[tuples[index]]
        counts[tuples[index]] += 1
    table = pd.DataFrame({"item_id": item_ids})
    for level in range(codes.shape[1]):
        table[f"code_{level}"] = codes[:, level]
    table[f"code_{codes.shape[1]}"] = suffix
    diagnostics = {"n_items": len(item_ids), "unique_base_sids": len(counts),
                   "collision_rate": (len(item_ids) - len(counts)) / len(item_ids),
                   "collision_suffix_size": max(counts.values())}
    return table, diagnostics


@torch.no_grad()
def tokenize_catalog(model: RQVAE, features: Tensor, item_ids: Sequence[int],
                     batch_size: int = 256) -> tuple[pd.DataFrame, dict]:
    """Assign (N,L+1) codes in batches; report full-catalog usage and MSE."""
    if batch_size < 1 or len(features) != len(item_ids) or len(features) == 0:
        raise ValueError("Need positive batch size and nonempty aligned features/IDs")
    was_training = model.training
    model.eval()
    codes, squared_error = [], 0.0
    try:
        for batch in features.split(batch_size):
            output = model(batch)
            codes.append(output.codes.cpu().numpy())
            squared_error += float(output.reconstruction_loss) * batch.numel()
    finally:
        model.train(was_training)
    codes_array = np.concatenate(codes)
    table, diagnostics = assign_sids(item_ids, codes_array)
    diagnostics["reconstruction_loss"] = squared_error / features.numel()
    diagnostics["codebook_usage"] = [len(np.unique(codes_array[:, level])) / model.codebook_size
                                     for level in range(model.n_levels)]
    return table, diagnostics
