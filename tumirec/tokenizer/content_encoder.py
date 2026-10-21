"""Frozen local text encoding with content-addressed, validated NumPy caches."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import pandas as pd
import torch


class TextEncoder(Protocol):
    """Minimal encoder interface, injectable for offline synthetic tests."""

    def encode(self, sentences: list[str], **kwargs: object) -> np.ndarray:
        """Return embeddings shaped (N, D)."""
        ...


def item_texts(items: pd.DataFrame) -> list[str]:
    """Format N catalog rows as title | category | brand | description."""
    columns = ["title", "category", "brand", "description"]
    frame = items.reindex(columns=columns).fillna("").astype(str)
    return [" | ".join(row) for row in frame.itertuples(index=False, name=None)]


class ContentEncoder:
    """Load a frozen sentence-transformer only from local files, never the network.

    Injected encoders must be deterministic and identified by a unique model_id
    (including revision). They are intended for testing or local custom models.
    """

    def __init__(self, model_id: str = "BAAI/bge-small-en-v1.5",
                 device: str = "cpu", batch_size: int = 64,
                 normalize: bool = True, encoder: TextEncoder | None = None) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model_id, self.device = model_id, device
        self.batch_size, self.normalize, self._encoder = batch_size, normalize, encoder

    def _load(self) -> TextEncoder:
        """Lazily load local weights so a cache hit needs no encoder dependency."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.model_id, device=self.device,
                                                local_files_only=True)
        if isinstance(self._encoder, torch.nn.Module):
            self._encoder.eval()
            self._encoder.requires_grad_(False)
        return self._encoder

    def encode_cached(self, texts: Sequence[str], cache_path: str | Path,
                      keys: Sequence[int | str] | None = None) -> np.ndarray:
        """Encode N texts once to float32 (N, D); reject mismatched/corrupt caches.

        Keys bind item IDs to row order. Queries can use implicit positional keys.
        Changing text, order, model identity or normalization needs a new cache.
        """
        texts = list(texts)
        keys = list(range(len(texts))) if keys is None else list(keys)
        if not texts or len(keys) != len(texts) or len(set(keys)) != len(keys):
            raise ValueError("Need nonempty texts and unique, aligned keys")
        path = Path(cache_path)
        if path.suffix != ".npy":
            raise ValueError("Cache path must end in .npy")
        manifest = path.with_suffix(".json")
        identity = {"version": 1, "model": self.model_id, "normalize": self.normalize,
                    "texts": texts, "keys": keys}
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if path.exists() or manifest.exists():
            if not path.exists() or not manifest.exists():
                raise ValueError("Incomplete embedding cache; use a fresh cache path")
            metadata = json.loads(manifest.read_text())
            if metadata["fingerprint"] != fingerprint:
                raise ValueError("Embedding cache identity mismatch; use a fresh cache path")
            if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
                raise ValueError("Embedding cache checksum mismatch")
            embeddings = np.load(path, allow_pickle=False)
        else:
            with torch.inference_mode():
                embeddings = np.asarray(self._load().encode(
                    texts, batch_size=self.batch_size, convert_to_numpy=True,
                    normalize_embeddings=self.normalize, show_progress_bar=False), dtype=np.float32)
            self._validate(embeddings, len(texts))
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, embeddings, allow_pickle=False)
            manifest.write_text(json.dumps({"fingerprint": fingerprint,
                "shape": list(embeddings.shape),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2))
        self._validate(embeddings, len(texts))
        return embeddings

    @staticmethod
    def _validate(embeddings: np.ndarray, n_rows: int) -> None:
        """Validate finite float32 embeddings of shape (N, D), D > 0."""
        if (embeddings.ndim != 2 or embeddings.shape[0] != n_rows or
                embeddings.shape[1] == 0 or embeddings.dtype != np.float32 or
                not np.isfinite(embeddings).all()):
            raise ValueError("Invalid cached embedding array")

    def encode_items(self, items: pd.DataFrame, cache_path: str | Path) -> np.ndarray:
        """Embed N catalog items in their input row order, returning (N, D)."""
        return self.encode_cached(item_texts(items), cache_path, items.item_id.tolist())
