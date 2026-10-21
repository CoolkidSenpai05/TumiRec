"""Train and evaluate the M3 causal baseline under one H1 token configuration."""
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
from torch.utils.data import Dataset, DataLoader

from tumirec.data.splits import DataSplits
from tumirec.data.preprocess import item_statistics
from tumirec.eval.metrics import ranking_metrics
from tumirec.model.item_token import GatedItemToken, MODES
from tumirec.model.sasrec import SASRec
from tumirec.train.stage1_tokenizer import _seed, _schedule


@dataclass(frozen=True)
class BaselineConfig:
    """M3 training/configuration controls; small enough for a single 24GB GPU."""
    seed: int = 42
    mode: str = 'gated'
    device: str = 'cpu'
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    max_history: int = 50
    dropout: float = 0.1
    hash_buckets: int = 10000
    dropout_min: float = 0.0
    dropout_max: float = 0.8
    dropout_scale: float = 10.0
    dropout_function: str = 'inverse'
    include_collision_suffix: bool = True
    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 0.001
    gradient_clip: float = 1.0
    warmup_steps: int = 100
    bf16: bool = True
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        """Reject invalid optimization settings early."""
        if (self.mode not in MODES or min(self.epochs, self.batch_size, self.max_history) < 1
                or self.learning_rate <= 0 or self.gradient_clip <= 0 or self.warmup_steps < 0):
            raise ValueError('Invalid baseline configuration')


class HistoryDataset(Dataset):
    """Target rows with bounded strictly earlier TRAIN history; no pandas per access."""

    def __init__(self, targets: pd.DataFrame, train: pd.DataFrame,
                 item_ids: list[int], max_history: int) -> None:
        self.mapping = {item: row for row, item in enumerate(item_ids)}
        self.targets = targets.reset_index(drop=True)
        self.users = self.targets.user_id.to_numpy()
        self.times = self.targets.timestamp.to_numpy()
        self.rows = self.targets.item_id.map(self.mapping).to_numpy(dtype=np.int64)
        self.max_history = max_history
        self.histories = {}
        for user, group in train.groupby('user_id', sort=False):
            ordered = group.sort_values('timestamp', kind='stable')
            self.histories[user] = (ordered.timestamp.to_numpy(),
                                   ordered.item_id.map(self.mapping).to_numpy(dtype=np.int64))

    def __len__(self) -> int:
        """Number of prediction targets."""
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[Tensor, int, int]:
        """Return right-padded row indices (W,), length and target row index."""
        times, rows = self.histories.get(self.users[index], (np.array([]), np.array([], dtype=np.int64)))
        end = int(np.searchsorted(times, self.times[index], side='left'))
        history = rows[max(0, end - self.max_history):end]
        padded = np.full(self.max_history, -1, dtype=np.int64)
        padded[:len(history)] = history
        return torch.from_numpy(padded), len(history), int(self.rows[index])


def build_model(items: pd.DataFrame, sids: pd.DataFrame, train: pd.DataFrame,
                config: BaselineConfig) -> SASRec:
    """Align catalog, SID and train-stat rows and construct the causal baseline."""
    if items.item_id.duplicated().any() or sids.item_id.duplicated().any() or set(items.item_id) != set(sids.item_id):
        raise ValueError('Catalog and SID IDs must be unique and match exactly')
    columns = sorted([c for c in sids if c.startswith('code_')], key=lambda x: int(x.split('_')[1]))
    if not config.include_collision_suffix:
        columns = columns[:-1]
    if not columns:
        raise ValueError('No SID levels selected')
    aligned = sids.set_index('item_id').loc[items.item_id, columns].to_numpy()
    if not np.issubdtype(aligned.dtype, np.integer) or (aligned < 0).any():
        raise ValueError('SID codes must be nonnegative integers')
    sizes = (aligned.max(axis=0) + 1).tolist()
    token = GatedItemToken(sizes, config.d_model, config.hash_buckets, config.mode,
                           config.dropout_min, config.dropout_max, config.dropout_scale,
                           config.dropout_function)
    stats = item_statistics(train, items)[['count', 'age_days_normalized', 'popularity_percentile']]
    return SASRec(token, torch.tensor(items.item_id.to_numpy(), dtype=torch.long),
                  torch.tensor(aligned, dtype=torch.long), torch.tensor(stats.to_numpy(), dtype=torch.float32),
                  config.d_model, config.n_heads, config.n_layers, config.max_history,
                  config.dropout, config.gradient_checkpointing).to(config.device)


@torch.no_grad()
def evaluate(model: SASRec, targets: pd.DataFrame, train: pd.DataFrame,
             cold_items: frozenset[int], config: BaselineConfig) -> dict:
    """Exact full-catalog ranking with deterministic ID tie breaks and slice counts.

    Repeated consumed items remain eligible. No target/future events enter history.
    Empty slices have null metrics rather than fabricated zero measurements.
    """
    model.eval()
    ids = model.item_ids.cpu().tolist()
    dataset = HistoryDataset(targets, train, ids, config.max_history)
    per_query_metrics = []
    candidates = torch.argsort(model.item_ids, stable=True)
    candidate_ids = model.item_ids[candidates].cpu().tolist()
    for history, lengths, target_rows in DataLoader(dataset, batch_size=config.batch_size):
        with torch.autocast(device_type=torch.device(config.device).type, dtype=torch.bfloat16, enabled=config.bf16):
            scores = model(history.to(config.device), lengths.to(config.device), candidates)
        order = torch.argsort(scores.float(), dim=-1, descending=True, stable=True).cpu().numpy()
        for query, target_row in zip(order, target_rows.tolist()):
            ranking = [candidate_ids[row] for row in query]
            per_query_metrics.append(ranking_metrics([ranking], [{ids[target_row]}]))
    counts = train.item_id.value_counts()
    percentiles = counts.rank(pct=True)
    target_count = targets.item_id.map(counts).fillna(0).to_numpy()
    target_percentile = targets.item_id.map(percentiles).fillna(0).to_numpy()
    masks = {'all': np.ones(len(targets), dtype=bool),
             'head': (target_count > 0) & (target_percentile > 0.8),
             'torso': (target_count > 0) & (target_percentile > 0.5) & (target_percentile <= 0.8),
             'tail': (target_count > 0) & (target_percentile <= 0.5),
             'cold': target_count == 0,
             'heldout_cold': targets.item_id.isin(cold_items).to_numpy(),
             'cold_users': targets.user_id.map(train.user_id.value_counts()).fillna(0).to_numpy() < 5}
    for column in ('task', 'domain'):
        for value in sorted(targets[column].unique()):
            masks[f'{column}_{value}'] = (targets[column] == value).to_numpy()
    slices = {}
    for name, mask in masks.items():
        selected = np.flatnonzero(mask).tolist()
        slices[name] = {'n_targets': len(selected),
                        'n_items': int(targets.loc[mask, 'item_id'].nunique()),
                        'metrics': {metric: float(np.mean([per_query_metrics[i][metric] for i in selected]))
                                    for metric in per_query_metrics[selected[0]]} if selected else None}
    return {'protocol': 'full_catalog; frozen_train_history; consumed_items_retained; ties_by_item_id',
            'catalog_size': len(ids), 'slices': slices}


def train_baseline(items: pd.DataFrame, sids: pd.DataFrame, splits: DataSplits,
                   output_dir: str | Path, config: BaselineConfig = BaselineConfig(),
                   resume: bool = False, stop_after_epochs: int | None = None) -> dict:
    """Fit next-item softmax over TRAIN-observed items; select by validation NDCG@10.

    Persist epoch-boundary resume, best validation weights and computed test JSON.
    The same fixed split/SIDs should be supplied to every H1 mode for a given seed.
    """
    if splits.train.empty or splits.validation.empty or splits.test.empty:
        raise ValueError('H1 needs nonempty train, validation and test targets')
    if stop_after_epochs is not None and stop_after_epochs < 1:
        raise ValueError('stop_after_epochs must be positive')
    items = items.sort_values('item_id').reset_index(drop=True)
    _seed(config.seed)
    model = build_model(items, sids, splits.train, config)
    ids = items.item_id.tolist()
    dataset = HistoryDataset(splits.train, splits.train, ids, config.max_history)
    seen = torch.where(model.statistics[:, 0] > 0)[0]
    if len(seen) < 2:
        raise ValueError('Training needs at least two observed items for negatives')
    labels = torch.full((len(items),), -1, dtype=torch.long, device=config.device)
    labels[seen] = torch.arange(len(seen), device=config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = _schedule(optimizer, config.epochs * math.ceil(len(dataset) / config.batch_size), config.warmup_steps)
    digest = hashlib.sha256()
    for frame in (items, sids, splits.train, splits.validation, splits.test):
        digest.update(pd.util.hash_pandas_object(frame.astype(str), index=True).values.tobytes())
    digest.update(str(sorted(splits.cold_items)).encode())
    identity = digest.hexdigest()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / 'checkpoint.pt'
    start, best_score, best_epoch, logs, best_weights = 0, -float('inf'), 0, [], None
    if checkpoint_path.exists() and not resume:
        raise FileExistsError('Checkpoint exists; choose resume or a fresh directory')
    if resume:
        state = torch.load(checkpoint_path, weights_only=True, map_location=config.device)
        if state['config'] != asdict(config) or state['fingerprint'] != identity:
            raise ValueError('Resume configuration/input mismatch')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start, best_score, best_epoch, logs = state['epoch'], state['best_score'], state['best_epoch'], state['logs']
        best_weights = state['best_weights']
        torch.set_rng_state(state['rng'].cpu())
        if torch.device(config.device).type == 'cuda':
            torch.cuda.set_rng_state_all([rng.cpu() for rng in state['cuda_rng']])
    for epoch in range(start, config.epochs):
        model.train()
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(config.seed + epoch))
        loss_sum = 0.0
        for history, lengths, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=torch.device(config.device).type, dtype=torch.bfloat16, enabled=config.bf16):
                logits = model(history.to(config.device), lengths.to(config.device), seen)
                loss = F.cross_entropy(logits.float(), labels[targets.to(config.device)])
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite baseline training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.detach()) * len(targets)
        validation = evaluate(model, splits.validation, splits.train, splits.cold_items, config)
        score = validation['slices']['all']['metrics']['NDCG@10']
        if score > best_score:
            best_score, best_epoch = score, epoch + 1
            best_weights = deepcopy(model.state_dict())
        logs.append({'epoch': epoch + 1, 'loss': loss_sum / len(dataset), 'validation_ndcg10': score})
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(), 'epoch': epoch + 1, 'config': asdict(config),
                 'fingerprint': identity, 'best_weights': best_weights, 'best_score': best_score,
                 'best_epoch': best_epoch, 'logs': logs, 'rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all() if torch.device(config.device).type == 'cuda' else []}
        temporary = checkpoint_path.with_suffix('.tmp')
        torch.save(state, temporary)
        temporary.replace(checkpoint_path)
        pd.DataFrame(logs).to_csv(output / 'training.csv', index=False)
        if stop_after_epochs is not None and epoch - start + 1 >= stop_after_epochs:
            return {'status': 'paused', 'epoch': epoch + 1}
    model.load_state_dict(best_weights)
    result = evaluate(model, splits.test, splits.train, splits.cold_items, config)
    result.update(config=asdict(config), best_epoch=best_epoch, validation_ndcg10=best_score,
                  n_train_targets=len(dataset), n_heldout_items=len(splits.cold_items))
    (output / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
