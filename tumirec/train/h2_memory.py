"""M4 training/evaluation of the memory side branch on the M3 causal control."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tumirec.data.memory_dataset import LagConfig, MemoryDataset
from tumirec.data.splits import DataSplits
from tumirec.eval.metrics import ranking_metrics
from tumirec.model.memory import InterestMemory, MEMORY_MODES
from tumirec.model.h2_baseline import MemoryBaseline
from tumirec.train.h1_baseline import BaselineConfig, build_model
from tumirec.train.stage1_tokenizer import _seed, _schedule
from tumirec.serving_sim.memory_profile import profile_cache


@dataclass(frozen=True)
class MemoryConfig:
    """Memory architecture and mode; stochastic lag has its own LagConfig."""
    mode: str = 'behavior_memory'
    n_latents: int = 8
    n_actions: int = 3
    max_history: int = 1000
    time_buckets: int = 32
    time_unit_seconds: int = 1
    session_segmentation: bool = False
    session_gap_seconds: int = 1800
    dropout: float = 0.0


def to_device(batch: dict[str, Tensor], device: str) -> dict[str, Tensor]:
    """Move collated tensor fields to the configured device."""
    return {key: value.to(device) for key, value in batch.items()}


def make_model(items: pd.DataFrame, sids: pd.DataFrame, train: pd.DataFrame,
               features: np.ndarray, base: BaselineConfig, memory: MemoryConfig) -> MemoryBaseline:
    """Construct H2 with aligned content (N,C); no changes to the M3 model."""
    causal = build_model(items, sids, train, base)
    encoder = InterestMemory(d_model=base.d_model, n_heads=base.n_heads, recent_window=base.max_history,
                            content_dim=features.shape[1], gradient_checkpointing=base.gradient_checkpointing,
                            **asdict(memory))
    return MemoryBaseline(causal, encoder, torch.as_tensor(features, dtype=torch.float32)).to(base.device)


def make_dataset(targets: pd.DataFrame, train: pd.DataFrame, items: pd.DataFrame,
                 base: BaselineConfig, memory: MemoryConfig, lag: LagConfig,
                 fixed_lag: int | None = None) -> MemoryDataset:
    """Create matching training or fixed-lag evaluation windows."""
    return MemoryDataset(targets, train, items.item_id.tolist(), memory.max_history,
                         base.max_history, base.seed, lag, fixed_lag)


@torch.no_grad()
def evaluate_memory(model: MemoryBaseline, dataset: MemoryDataset, train: pd.DataFrame,
                    cold_items: frozenset[int], config: BaselineConfig) -> dict:
    """Full-catalog fixed-lag metrics, retained consumed items and ID tie breaks."""
    model.eval()
    candidates = torch.argsort(model.baseline.item_ids, stable=True)
    ids = model.baseline.item_ids.cpu().tolist()
    candidate_ids = model.baseline.item_ids[candidates].cpu().tolist()
    per_query, event_counts = [], []
    for raw in DataLoader(dataset, batch_size=config.batch_size):
        batch = to_device(raw, config.device)
        with torch.autocast(device_type=torch.device(config.device).type, dtype=torch.bfloat16, enabled=config.bf16):
            scores = model(batch, candidates)
        order = scores.float().argsort(dim=-1, descending=True, stable=True).cpu().tolist()
        for ranking, target in zip(order, raw['target'].tolist()):
            per_query.append(ranking_metrics([[candidate_ids[i] for i in ranking]], [{ids[target]}]))
        counts = raw['memory_valid'].sum(1)
        if model.memory.mode == 'none':
            counts = counts * 0
        elif model.memory.mode == 'recent_only':
            counts = counts.clamp_max(model.memory.recent_window)
        event_counts.extend(counts.tolist())
    targets = dataset.targets
    counts = train.item_id.value_counts()
    percentile = counts.rank(pct=True)
    n = targets.item_id.map(counts).fillna(0).to_numpy()
    p = targets.item_id.map(percentile).fillna(0).to_numpy()
    masks = {'all': np.ones(len(targets), dtype=bool), 'head': (n > 0) & (p > .8),
             'torso': (n > 0) & (p > .5) & (p <= .8), 'tail': (n > 0) & (p <= .5),
             'cold': n == 0, 'heldout_cold': targets.item_id.isin(cold_items).to_numpy()}
    slices = {}
    for name, mask in masks.items():
        selected = np.flatnonzero(mask).tolist()
        slices[name] = {'n_targets': len(selected), 'n_items': int(targets.loc[mask, 'item_id'].nunique()),
            'metrics': {key: float(np.mean([per_query[i][key] for i in selected]))
                        for key in per_query[selected[0]]} if selected else None}
    return {'lag_seconds': dataset.fixed_lag, 'mode': model.memory.mode, 'slices': slices,
            'mean_memory_events': float(np.mean(event_counts)) if event_counts else None,
            'protocol': 'full_catalog; frozen_train_history; consumed_items_retained; ties_by_item_id'}


def train_memory(items: pd.DataFrame, sids: pd.DataFrame, features: np.ndarray, splits: DataSplits,
                 output_dir: str | Path, base: BaselineConfig = BaselineConfig(),
                 memory: MemoryConfig = MemoryConfig(), lag: LagConfig = LagConfig(),
                 eval_lags: tuple[int, ...] = (0, 3600, 86400), resume: bool = False,
                 stop_after_epochs: int | None = None, profile_repeats: int = 0) -> dict:
    """Co-train recent baseline and memory, select once on fresh validation NDCG@10.

    Evaluate the same selected weights at every lag. Save exact epoch resume,
    actual losses/metrics and optionally measured cached/uncached forward timings.
    """
    features = np.asarray(features, dtype=np.float32)
    if (features.ndim != 2 or len(features) != len(items) or features.shape[1] == 0 or
        not np.isfinite(features).all() or any(frame.empty for frame in (splits.train, splits.validation, splits.test))):
        raise ValueError('Need finite aligned features and nonempty splits')
    if not eval_lags or len(set(eval_lags)) != len(eval_lags) or any(x < 0 or int(x) != x for x in eval_lags):
        raise ValueError('Evaluation lags must be unique nonnegative integer seconds')
    if profile_repeats < 0 or (stop_after_epochs is not None and stop_after_epochs < 1):
        raise ValueError('Invalid profiling/interruption limits')
    order = np.argsort(items.item_id.to_numpy(), kind='stable')
    items, features = items.iloc[order].reset_index(drop=True), features[order]
    _seed(base.seed)
    model = make_model(items, sids, splits.train, features, base, memory)
    data = make_dataset(splits.train, splits.train, items, base, memory, lag)
    validation = make_dataset(splits.validation, splits.train, items, base, memory, lag, 0)
    seen = torch.where(model.baseline.statistics[:, 0] > 0)[0]
    if len(seen) < 2:
        raise ValueError('At least two observed items required')
    labels = torch.full((len(items),), -1, device=base.device, dtype=torch.long)
    labels[seen] = torch.arange(len(seen), device=base.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=base.learning_rate)
    scheduler = _schedule(optimizer, base.epochs * math.ceil(len(data) / base.batch_size), base.warmup_steps)
    config = {'baseline': asdict(base), 'memory': asdict(memory), 'lag': asdict(lag)}
    digest = hashlib.sha256(features.tobytes())
    for frame in (items, sids, splits.train, splits.validation, splits.test):
        digest.update(pd.util.hash_pandas_object(frame.astype(str), index=True).values.tobytes())
    digest.update(str(sorted(splits.cold_items)).encode())
    identity = digest.hexdigest()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / 'checkpoint.pt'
    start, best_score, best_epoch, best, logs = 0, -float('inf'), 0, None, []
    if checkpoint_path.exists() and not resume:
        raise FileExistsError('Use a fresh directory or resume')
    if resume:
        state = torch.load(checkpoint_path, weights_only=True, map_location=base.device)
        if state['config'] != config or state['fingerprint'] != identity:
            raise ValueError('Resume configuration/input mismatch')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start, best_score, best_epoch, best, logs = (state[key] for key in ('epoch', 'best_score', 'best_epoch', 'best', 'logs'))
        torch.set_rng_state(state['rng'].cpu())
        if torch.device(base.device).type == 'cuda':
            torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda_rng']])
    for epoch in range(start, base.epochs):
        data.set_epoch(epoch)
        model.train()
        loss_sum = 0.0
        loader = DataLoader(data, batch_size=base.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(base.seed + epoch))
        for raw in loader:
            batch = to_device(raw, base.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=torch.device(base.device).type, dtype=torch.bfloat16, enabled=base.bf16):
                loss = F.cross_entropy(model(batch, seen).float(), labels[batch['target']])
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite H2 loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), base.gradient_clip)
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.detach()) * len(batch['target'])
        score = evaluate_memory(model, validation, splits.train, splits.cold_items, base)['slices']['all']['metrics']['NDCG@10']
        if score > best_score:
            best_score, best_epoch, best = score, epoch + 1, deepcopy(model.state_dict())
        logs.append({'epoch': epoch + 1, 'loss': loss_sum / len(data), 'validation_ndcg10_lag0': score})
        state = {'config': config, 'fingerprint': identity, 'epoch': epoch + 1, 'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'best': best,
                 'best_score': best_score, 'best_epoch': best_epoch, 'logs': logs, 'rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all() if torch.device(base.device).type == 'cuda' else []}
        temp = checkpoint_path.with_suffix('.tmp')
        torch.save(state, temp)
        temp.replace(checkpoint_path)
        pd.DataFrame(logs).to_csv(output / 'training.csv', index=False)
        if stop_after_epochs is not None and epoch - start + 1 >= stop_after_epochs:
            return {'status': 'paused', 'epoch': epoch + 1}
    model.load_state_dict(best)
    model.eval()
    evaluations = []
    for fixed in eval_lags:
        test = make_dataset(splits.test, splits.train, items, base, memory, lag, fixed)
        evaluations.append(evaluate_memory(model, test, splits.train, splits.cold_items, base))
    result = {'config': config, 'best_epoch': best_epoch, 'validation_ndcg10_lag0': best_score,
              'evaluations': evaluations, 'catalog_size': len(items), 'n_train_targets': len(data)}
    if profile_repeats:
        test = make_dataset(splits.test, splits.train, items, base, memory, lag, 0)
        batch = to_device(next(iter(DataLoader(test, batch_size=base.batch_size))), base.device)
        result['cache_profile'] = profile_cache(model, batch, torch.arange(len(items), device=base.device),
                                                repeats=profile_repeats, bf16=base.bf16)
    (output / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
