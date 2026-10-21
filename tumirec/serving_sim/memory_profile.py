"""M4 scoped cache simulation/profiling; unified-backbone benchmarks await M6."""
from time import perf_counter
import numpy as np
import torch
from torch import Tensor
from tumirec.model.h2_baseline import MemoryBaseline
from tumirec.serving_sim.memory_cache import MemoryCache


@torch.no_grad()
def refresh_batch(model: MemoryBaseline, batch: dict[str, Tensor], cache: MemoryCache,
                  version: str) -> Tensor:
    """Simulate explicit nearline refresh for one batch and return (B,K,D) memory.

    Callers must supply correctly truncated TRAIN-only windows. Cache cutoff is
    target_time-lag, never the newest observed event time. Model must be in eval.
    """
    if model.training:
        raise ValueError('Nearline cache refresh requires an evaluation model')
    memory = model.encode_memory(batch)
    for i, user in enumerate(batch['user_id'].tolist()):
        cache.put(user, int(batch['target_time'][i] - batch['lag'][i]), version, memory[i])
    return memory


@torch.no_grad()
def profile_cache(model: MemoryBaseline, batch: dict[str, Tensor], candidates: Tensor,
                  repeats: int = 10, warmup: int = 2, bf16: bool = True) -> dict:
    """Measure CPU/CUDA batch forwards in ms with/without exact precomputed memory.

    Times exclude data loading, host transfers and cache lookup/refresh. Cache is
    prepopulated at the exact requested cutoff; this is not a production latency SLA.
    """
    if repeats < 1 or warmup < 0 or len(set(batch['user_id'].tolist())) != len(batch['user_id']):
        raise ValueError('Profiling needs positive repeats and distinct users in the batch')
    device = candidates.device
    was_training = model.training
    model.eval()
    cache = MemoryCache(model.memory.n_latents, model.baseline.d_model)

    def synchronize() -> None:
        """Synchronize the selected CUDA device when measuring GPU execution."""
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    try:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
            refresh_batch(model, batch, cache, 'profile-frozen-model')
            cached = torch.stack([cache.get(int(user), int(time - lag), 'profile-frozen-model', device=device).memory
                for user, time, lag in zip(batch['user_id'], batch['target_time'], batch['lag'])])
            direct = model(batch, candidates)
            reused = model(batch, candidates, cached)
            if not torch.allclose(direct.float(), reused.float(), rtol=1e-3, atol=1e-3):
                raise RuntimeError('Cached and fresh scores disagree at the same cutoff')
            measurements = {}
            for name, memory in [('uncached', None), ('cached', cached)]:
                for _ in range(warmup):
                    model(batch, candidates, memory)
                durations = []
                for _ in range(repeats):
                    synchronize()
                    start = perf_counter()
                    model(batch, candidates, memory)
                    synchronize()
                    durations.append((perf_counter() - start) * 1000)
                measurements[name] = {'p50_ms': float(np.percentile(durations, 50)),
                                      'p99_ms': float(np.percentile(durations, 99)), 'samples': repeats}
        return {'device': str(device), 'batch_size': len(batch['target']), 'n_candidates': len(candidates),
                'scope': 'M4 baseline forward; cache lookup/refresh and data transfer excluded',
                'timings': measurements}
    finally:
        model.train(was_training)
