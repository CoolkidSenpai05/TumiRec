"""Macro-averaged binary ranking metrics on explicit ranked item IDs."""
from collections.abc import Iterable, Sequence
import math
import numpy as np


def ranking_metrics(rankings: Sequence[Sequence[int]], relevant: Sequence[Iterable[int]],
                    ks: Sequence[int] = (10, 20, 50)) -> dict[str, float]:
    """Compute Recall@k, NDCG@k, MRR@k and untruncated MRR for N queries.

    Rankings are ordered, unique item IDs, shape (N, variable catalog size).
    Empty truth contributes zero; an empty batch is invalid. k above catalog
    size is naturally clipped by slicing. Callers control candidate protocol.
    """
    if len(rankings) == 0 or len(rankings) != len(relevant):
        raise ValueError("Need equal nonzero numbers of rankings and truth sets")
    if not ks or any(not isinstance(k, (int, np.integer)) or k <= 0 for k in ks):
        raise ValueError("ks must contain positive integers")
    values = {f"{name}@{k}": [] for k in ks for name in ("Recall", "NDCG", "MRR")}
    values["MRR"] = []
    for ranking, truth in zip(rankings, relevant):
        ranking, truth = list(ranking), set(truth)
        if len(set(ranking)) != len(ranking):
            raise ValueError("Rankings must not contain duplicate item IDs")
        hits = [int(item in truth) for item in ranking]
        first = next((i + 1 for i, hit in enumerate(hits) if hit), None)
        values["MRR"].append(1 / first if first else 0.0)
        for k in dict.fromkeys(ks):
            dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits[:k]))
            ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(truth))))
            values[f"Recall@{k}"].append(sum(hits[:k]) / len(truth) if truth else 0.0)
            values[f"NDCG@{k}"].append(dcg / ideal if ideal else 0.0)
            values[f"MRR@{k}"].append(1 / first if first and first <= k else 0.0)
    return {key: float(np.mean(value)) for key, value in values.items()}
