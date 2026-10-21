"""Bounded, versioned in-process nearline memory cache; no background worker."""
from dataclasses import dataclass
from collections import OrderedDict
import math
import torch
from torch import Tensor


@dataclass(frozen=True)
class CacheResult:
    """Detached memory (K,D), hit/freshness flags, and exclusive event cutoff."""
    memory: Tensor
    hit: bool
    needs_refresh: bool
    cutoff: int | None


class MemoryCache:
    """LRU cache keyed by user/version, with explicit caller-driven refresh.

    A snapshot covers events strictly before cutoff. Future snapshots are never
    returned to an earlier request. Expired snapshots may be served while refresh
    is queued. Version keys must change with weights/catalog/memory configuration.
    """

    def __init__(self, n_latents: int, d_model: int, capacity: int = 10000,
                 ttl_seconds: int = 3600) -> None:
        if min(n_latents, d_model, capacity) < 1 or ttl_seconds < 0:
            raise ValueError('Invalid cache dimensions/capacity/TTL')
        self.shape, self.capacity, self.ttl_seconds = (n_latents, d_model), capacity, ttl_seconds
        self.entries: OrderedDict[tuple[int, str], tuple[int, Tensor]] = OrderedDict()
        self.pending: dict[tuple[int, str], int] = {}

    def get(self, user_id: int, cutoff: int, version: str, *, device: str | torch.device = 'cpu') -> CacheResult:
        """Get safe previous memory or zero (K,D); queue missing/expired snapshots."""
        if not math.isfinite(cutoff):
            raise ValueError('Cutoff must be finite')
        key = (user_id, version)
        entry = self.entries.get(key)
        if entry is None or entry[0] > cutoff:
            self.pending[key] = max(cutoff, self.pending.get(key, cutoff))
            return CacheResult(torch.zeros(self.shape, device=device), False, True, None)
        self.entries.move_to_end(key)
        stale = cutoff - entry[0] > self.ttl_seconds
        if stale:
            self.pending[key] = max(cutoff, self.pending.get(key, cutoff))
        return CacheResult(entry[1].to(device).clone(), True, stale, entry[0])

    def put(self, user_id: int, cutoff: int, version: str, memory: Tensor) -> bool:
        """Publish detached CPU tokens (K,D); reject late updates older than stored state."""
        if memory.shape != self.shape or not torch.isfinite(memory).all() or not math.isfinite(cutoff):
            raise ValueError('Invalid cache snapshot')
        key = (user_id, version)
        old = self.entries.get(key)
        if old is not None and old[0] > cutoff:
            return False
        self.entries[key] = (cutoff, memory.detach().cpu().float().clone())
        self.entries.move_to_end(key)
        if self.pending.get(key, cutoff) <= cutoff:
            self.pending.pop(key, None)
        while len(self.entries) > self.capacity:
            evicted, _ = self.entries.popitem(last=False)
            self.pending.pop(evicted, None)
        return True

    def refresh_requests(self) -> dict[tuple[int, str], int]:
        """Return a copy of queued user/version -> requested-cutoff work."""
        return dict(self.pending)

    def clear(self) -> None:
        """Discard snapshots and outstanding refresh requests."""
        self.entries.clear()
        self.pending.clear()
