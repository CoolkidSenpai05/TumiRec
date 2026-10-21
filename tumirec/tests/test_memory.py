"""M4 synthetic CPU tests for strict temporal memory, cache and H2."""
from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader
from tumirec.data.synthetic import generate_synthetic
from tumirec.data.splits import temporal_split
from tumirec.data.memory_dataset import MemoryDataset, LagConfig
from tumirec.model.memory import InterestMemory, MEMORY_MODES, pool_sessions
from tumirec.serving_sim.memory_cache import MemoryCache
from tumirec.serving_sim.memory_profile import profile_cache, refresh_batch
from tumirec.train.h1_baseline import BaselineConfig
from tumirec.train.h2_memory import MemoryConfig, train_memory, make_model, make_dataset
from tumirec.train.stage1_tokenizer import Stage1Config
from tumirec.scripts.run_h2 import run_h2


@pytest.fixture(autouse=True)
def cpu_settings():
    """Keep tiny attention tests fast and restore global settings afterward."""
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)


def memory_inputs():
    """Return (B=2,L=5,D=8) features with padding, future/tied and empty histories."""
    torch.manual_seed(2)
    tokens = torch.randn(2, 5, 8, requires_grad=True)
    actions = torch.tensor([[0, 1, 2, 0, 1], [0, 0, 0, 0, 0]])
    times = torch.tensor([[10, 20, 30, 40, 50], [0, 0, 0, 0, 0]])
    targets = torch.tensor([40, 40])
    valid = torch.tensor([[True] * 5, [False] * 5])
    return tokens, actions, times, targets, valid


@pytest.mark.parametrize('mode', [*MEMORY_MODES, 'no_memory'])
def test_memory_shape_finite_empty_and_no_memory(mode):
    model = InterestMemory(8, 2, n_latents=3, content_dim=4, mode=mode)
    inputs = memory_inputs()
    result = model(*inputs, content=torch.randn(2, 5, 4))
    assert result.shape == (2, 3, 8) and torch.isfinite(result).all()
    assert torch.count_nonzero(result[1]) == 0
    if mode in ('none', 'no_memory'):
        assert torch.count_nonzero(result) == 0
    else:
        assert result[0].abs().sum() > 0
        result[0].square().sum().backward()
        assert inputs[0].grad is not None
        assert model.latents.grad is not None


@pytest.mark.parametrize('lag', [0, 10, 30, 100])
def test_memory_lag_strict_exclusion_and_gradients(lag):
    model = InterestMemory(8, 2, n_latents=2, max_history=3).eval()
    tokens, actions, times, targets, valid = memory_inputs()
    result = model(tokens, actions, times, targets, valid, lag)
    excluded = ~(valid & (times < targets[:, None] - lag))
    changed_tokens, changed_actions = tokens.detach().clone(), actions.clone()
    changed_tokens[excluded] = 1e6
    changed_actions[excluded] = 999  # must never reach an embedding lookup
    other = model(changed_tokens, changed_actions, times, targets, valid, lag)
    torch.testing.assert_close(result, other, rtol=0, atol=0)
    result.sum().backward()
    assert (tokens.grad[excluded] == 0).all()


def test_window_selection_unsorted_padding_and_recent_only():
    tokens, actions, times, targets, valid = memory_inputs()
    model = InterestMemory(8, 2, n_latents=2, max_history=2).eval()
    a = model(tokens, actions, times, targets, valid)
    order = torch.tensor([4, 2, 0, 3, 1])
    b = model(tokens[:, order], actions[:, order], times[:, order], targets, valid[:, order])
    torch.testing.assert_close(a, b)
    altered = tokens.detach().clone()
    altered[:, 0] += 100  # oldest eligible event is outside max_history=2
    torch.testing.assert_close(a, model(altered, actions, times, targets, valid))
    model.mode, model.recent_window = 'recent_only', 1
    a = model(tokens, actions, times, targets, valid)
    altered[:, 1] += 200
    torch.testing.assert_close(a, model(altered, actions, times, targets, valid))


def test_empty_zero_width_history():
    model = InterestMemory(8, 2)
    output = model(torch.empty(2, 0, 8), torch.empty(2, 0, dtype=torch.long),
                   torch.empty(2, 0, dtype=torch.long), torch.tensor([1, 2]))
    assert output.shape == (2, 8, 8) and output.count_nonzero() == 0


def test_session_pool_means_and_memory_checkpointing():
    tokens = torch.tensor([[[1.], [3.], [8.], [99.]]], requires_grad=True)
    times = torch.tensor([[0, 10, 100, 200]])
    valid = torch.tensor([[True, True, True, False]])
    pooled, mask = pool_sessions(tokens, times, valid, 20)
    assert pooled.flatten().tolist() == [2., 8.] and mask.all()
    pooled.sum().backward()
    assert tokens.grad.flatten().tolist() == [.5, .5, 1., 0.]
    model = InterestMemory(8, 2, session_segmentation=True, session_gap_seconds=15, gradient_checkpointing=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        result = model(*memory_inputs())
    result.square().mean().backward()
    assert model.cross[0].query.weight.grad is not None


def test_content_pool_is_frozen_and_excludes_future_content():
    model = InterestMemory(8, 2, content_dim=4, mode='behavior_memory+content_pool').eval()
    inputs = memory_inputs()
    content = torch.randn(2, 5, 4, requires_grad=True)
    result = model(*inputs, content=content)
    changed = content.detach().clone()
    changed[0, 3:] = 10000
    torch.testing.assert_close(result, model(*inputs, content=changed), rtol=0, atol=0)
    changed[0, 0] += 10
    assert not torch.allclose(result[0], model(*inputs, content=changed)[0])
    result.square().mean().backward()
    assert content.grad is None and model.content_projection.weight.grad is not None


def test_lag_dataset_reproducibility_boundary_and_weighted_distribution():
    events, items = generate_synthetic(3, 8, 30)
    events.loc[1, 'timestamp'] = events.loc[2, 'timestamp']
    data = MemoryDataset(events, events, items.item_id.tolist(), max_history=4, recent_window=2,
                         lag=LagConfig(1, (0, 3600, 86400), (0, 1, 0)))
    for example in data:
        assert example['lag'] == 3600
        assert (example['memory_times'][example['memory_valid']] < example['target_time'] - example['lag']).all()
        assert example['recent_length'] <= 2 and example['memory_valid'].sum() <= 4
    data = MemoryDataset(events, events, items.item_id.tolist(), lag=LagConfig(1))
    before = [int(data[i]['lag']) for i in range(len(data))]
    assert before == [int(data[i]['lag']) for i in range(len(data))]
    assert before[::-1] == [int(data[i]['lag']) for i in reversed(range(len(data)))]
    data.set_epoch(1)
    assert before != [int(data[i]['lag']) for i in range(len(data))]
    fixed = MemoryDataset(events, events, items.item_id.tolist(), lag=LagConfig(1), fixed_lag=0)
    assert all(example['lag'] == 0 for example in fixed)
    long_lag = MemoryDataset(events, events, items.item_id.tolist(), fixed_lag=10**9)
    assert not long_lag[5]['memory_valid'].any() and long_lag[5]['recent_length'] > 0


def test_cache_miss_refresh_stale_version_future_and_lru():
    cache = MemoryCache(2, 4, capacity=2, ttl_seconds=10)
    miss = cache.get(1, 100, 'v1')
    assert not miss.hit and miss.needs_refresh and miss.memory.count_nonzero() == 0
    assert cache.refresh_requests() == {(1, 'v1'): 100}
    tokens = torch.ones(2, 4, requires_grad=True)
    assert cache.put(1, 100, 'v1', tokens)
    assert cache.refresh_requests() == {}
    tokens.data.fill_(9)
    hit = cache.get(1, 110, 'v1')
    assert hit.hit and not hit.needs_refresh and (hit.memory == 1).all()
    hit.memory.zero_()
    assert cache.get(1, 111, 'v1').needs_refresh
    assert (cache.get(1, 111, 'v1').memory == 1).all()
    assert not cache.put(1, 90, 'v1', tokens)
    assert not cache.get(1, 99, 'v1').hit
    assert not cache.get(1, 111, 'v2').hit
    cache.put(2, 100, 'v1', tokens)
    cache.put(3, 100, 'v1', tokens)
    assert not cache.get(1, 100, 'v1').hit
    cache.clear()
    assert not cache.entries and not cache.refresh_requests()


def h2_inputs():
    """Small shared SIDs, split and content fixture for H2 training."""
    events, items = generate_synthetic(4, 10, 12, seed=3)
    splits = temporal_split(events, cold_fraction=.2)
    sids = pd.DataFrame({'item_id': items.item_id, 'code_0': items.item_id % 3, 'code_1': items.item_id // 3})
    features = np.random.default_rng(3).normal(size=(len(items), 6)).astype(np.float32)
    base = BaselineConfig(d_model=8, n_heads=2, n_layers=1, max_history=3, epochs=2,
                          batch_size=16, hash_buckets=8, gradient_checkpointing=True)
    memory = MemoryConfig(n_latents=2, max_history=8)
    return events, items, splits, sids, features, base, memory


def test_cached_scores_and_measured_profile():
    _, items, splits, sids, features, base, memory = h2_inputs()
    model = make_model(items, sids, splits.train, features, base, memory).eval()
    data = make_dataset(splits.test, splits.train, items, base, memory, LagConfig(), 0)
    batch = next(iter(DataLoader(data, batch_size=4)))
    cache = MemoryCache(2, 8)
    encoded = refresh_batch(model, batch, cache, 'weights-v1')
    cached = torch.stack([cache.get(int(user), int(time), 'weights-v1').memory for user, time in zip(batch['user_id'], batch['target_time'])])
    candidates = torch.arange(len(items))
    torch.testing.assert_close(encoded, cached)
    torch.testing.assert_close(model(batch, candidates), model(batch, candidates, cached))
    profile = profile_cache(model, batch, candidates, repeats=2, warmup=1)
    assert all(part['samples'] == 2 and part['p99_ms'] >= part['p50_ms'] > 0 for part in profile['timings'].values())
    assert not model.training


def test_training_resume_and_memory_gradients(tmp_path):
    _, items, splits, sids, features, base, memory = h2_inputs()
    memory = replace(memory, mode='behavior_memory+content_pool', session_segmentation=True)
    expected = train_memory(items, sids, features, splits, tmp_path / 'full', base, memory)
    paused = train_memory(items, sids, features, splits, tmp_path / 'part', base, memory, stop_after_epochs=1)
    assert paused['status'] == 'paused'
    actual = train_memory(items, sids, features, splits, tmp_path / 'part', base, memory, resume=True)
    assert expected == actual
    a, b = (torch.load(tmp_path / name / 'checkpoint.pt', weights_only=True) for name in ('full', 'part'))
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
    assert [e['lag_seconds'] for e in actual['evaluations']] == [0, 3600, 86400]
    assert actual['evaluations'][-1]['mean_memory_events'] == 0
    with pytest.raises(ValueError, match='mismatch'):
        train_memory(items, sids, features + 1, splits, tmp_path / 'part', base, memory, resume=True)


def test_h2_all_modes_lags_and_aggregation(tmp_path):
    events, items, _, _, features, base, memory = h2_inputs()
    tokenizer = Stage1Config(adapter_epochs=1, rq_epochs=1, hidden_dim=8, latent_dim=3,
                              codebook_size=3, kmeans_iters=1, batch_size=32)
    summary = run_h2(events, items, features, tmp_path, seeds=[1, 2], base=replace(base, epochs=1),
                     memory=memory, tokenizer=tokenizer)
    assert set(summary['modes']) == set(MEMORY_MODES)
    assert len(list(tmp_path.glob('seed_*/*/results.json'))) == 8
    for mode in MEMORY_MODES:
        runs = [json.loads((tmp_path / f'seed_{seed}' / mode / 'results.json').read_text()) for seed in (1, 2)]
        values = [run['evaluations'][0]['slices']['all']['metrics']['NDCG@10'] for run in runs]
        row = next(row for row in summary['rows'] if row['mode'] == mode and row['lag_seconds'] == 0 and row['slice'] == 'all')
        assert row['NDCG@10_mean'] == pytest.approx(np.mean(values))
        assert row['NDCG@10_std'] == pytest.approx(np.std(values))
        if mode == 'none':
            for run in runs:
                assert run['evaluations'][0]['slices'] == run['evaluations'][2]['slices']
    assert (tmp_path / 'summary.csv').exists()


def test_action_time_signals_long_history_and_no_memory_baseline():
    torch.manual_seed(3)
    model = InterestMemory(8, 2, time_unit_seconds=1).eval()
    tokens = torch.randn(1, 1000, 8)
    times = torch.arange(1000)[None]
    actions = torch.zeros(1, 1000, dtype=torch.long)
    output = model(tokens, actions, times, torch.tensor([1000]))
    assert output.shape == (1, 8, 8)
    assert not torch.allclose(output, model(tokens, actions + 1, times, torch.tensor([1000])))
    assert not torch.allclose(output, model(tokens, actions, times, torch.tensor([100000])))
    _, items, splits, sids, features, base, memory = h2_inputs()
    control = make_model(items, sids, splits.train, features, base, replace(memory, mode='none')).eval()
    data = make_dataset(splits.test, splits.train, items, base, memory, LagConfig(), 0)
    batch = next(iter(DataLoader(data, batch_size=4)))
    candidates = torch.arange(len(items))
    torch.testing.assert_close(control(batch, candidates),
        control.baseline(batch['recent'], batch['recent_length'], candidates), rtol=0, atol=0)


@pytest.mark.parametrize('lag', [-1, 0.5, float('nan')])
def test_invalid_memory_lag_rejected(lag):
    with pytest.raises(ValueError, match='Lag'):
        InterestMemory(8, 2)(*memory_inputs(), lag_seconds=lag)
