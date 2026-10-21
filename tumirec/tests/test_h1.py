"""CPU/synthetic regression tests for M3 tokens, causal baseline and H1 pipeline."""
from dataclasses import replace
import gzip
import json
import numpy as np
import pandas as pd
import pytest
import torch
from tumirec.data.loaders import load_amazon, load_movielens, load_kuaisar
from tumirec.data.splits import temporal_split
from tumirec.data.synthetic import generate_synthetic
from tumirec.model.item_token import GatedItemToken, MODES
from tumirec.model.sasrec import SASRec
from tumirec.train.h1_baseline import BaselineConfig, HistoryDataset, build_model, evaluate, train_baseline
from tumirec.train.stage1_tokenizer import Stage1Config
from tumirec.scripts.run_ablation import run_h1


@pytest.fixture(autouse=True)
def cpu_settings():
    """Bound test threads and restore deterministic-algorithm settings afterward."""
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)


@pytest.mark.parametrize('mode', MODES)
def test_token_mode_algebra_and_cold_without_id_branch(mode):
    token = GatedItemToken([2, 3], d_model=4, hash_buckets=5, mode=mode)
    with torch.no_grad():
        for embedding in token.sid_embeddings:
            embedding.weight.fill_(1)
        token.id_embedding.weight.fill_(4)
        for param in token.gate.parameters():
            param.zero_()
    token.eval()
    codes = torch.tensor([[0, 1], [1, 2]])
    stats = torch.tensor([[10., 0.5, 1.], [0., 0., 0.]])
    result = token(codes, torch.tensor([3, 4]), stats)
    expected = {'sid_only': 2, 'id_only': 4, 'fixed_sum': 6, 'gated': 4, 'gated_no_id_dropout': 4}[mode]
    torch.testing.assert_close(result[0], torch.full((4,), float(expected)))
    torch.testing.assert_close(result[1], torch.full((4,), 2.))
    token.id_embedding = None
    torch.testing.assert_close(token(codes[1:], None, stats[1:]), result[1:])


@pytest.mark.parametrize('function', ['inverse', 'exponential'])
def test_dropout_function_and_observed_rates(function):
    torch.manual_seed(3)
    token = GatedItemToken([1], d_model=1, hash_buckets=1, dropout_min=0.1,
                           dropout_max=0.9, dropout_scale=10, dropout_function=function)
    count = torch.tensor([0., 1., 10., 1000.])
    probabilities = token.dropout_probability(count)
    ratio = count / 10
    expected = 0.1 + 0.8 * (1 / (1 + ratio) if function == 'inverse' else torch.exp(-ratio))
    torch.testing.assert_close(probabilities, expected)
    assert (probabilities[:-1] > probabilities[1:]).all()
    with torch.no_grad():
        token.sid_embeddings[0].weight.zero_()
        token.id_embedding.weight.fill_(1)
        for p in token.gate.parameters():
            p.zero_()
    codes = torch.zeros(20000, 1, dtype=torch.long)
    stats = torch.zeros(20000, 3)
    stats[:, 0] = 1
    result = token(codes, torch.zeros(20000, dtype=torch.long), stats)
    assert float((result == 0).float().mean()) == pytest.approx(float(probabilities[1]), abs=0.015)
    token.eval()
    assert (token(codes, torch.zeros(20000, dtype=torch.long), stats) == 0.5).all()
    token.train()
    token.mode = 'gated_no_id_dropout'
    assert (token(codes, torch.zeros(20000, dtype=torch.long), stats) == 0.5).all()


def test_id_dropout_seed_and_cold_no_id_gradients():
    token = GatedItemToken([3], d_model=4, hash_buckets=3)
    codes = torch.zeros(30, 1, dtype=torch.long)
    stats = torch.ones(30, 3)
    ids = torch.arange(30)
    torch.manual_seed(7)
    a = token(codes, ids, stats)
    torch.manual_seed(7)
    torch.testing.assert_close(a, token(codes, ids, stats), rtol=0, atol=0)
    token(codes, None, torch.zeros_like(stats)).sum().backward()
    assert token.id_embedding.weight.grad is None
    assert token.sid_embeddings[0].weight.grad is not None
    assert token.hash_ids(torch.tensor([0, 2147483647])).tolist() == [0, 0]


def tiny_model() -> SASRec:
    """Create a deterministic dropout-free baseline with four catalog entries."""
    torch.manual_seed(2)
    token = GatedItemToken([4], d_model=8, hash_buckets=8)
    return SASRec(token, torch.arange(4), torch.arange(4)[:, None], torch.ones(4, 3),
                  d_model=8, n_heads=2, n_layers=2, max_history=4, dropout=0)


def test_causal_attention_padding_empty_history_and_gradients():
    model = tiny_model().eval()
    a = model.sequence(torch.tensor([[0, 1, 2]]), torch.tensor([3]))
    b = model.sequence(torch.tensor([[0, 1, 3]]), torch.tensor([3]))
    torch.testing.assert_close(a[:, :3], b[:, :3], rtol=0, atol=0)
    short = model(torch.tensor([[1]]), torch.tensor([1]), torch.arange(4))
    padded = model(torch.tensor([[1, -1, -1, -1]]), torch.tensor([1]), torch.arange(4))
    torch.testing.assert_close(short, padded)
    empty = model(torch.full((2, 4), -1), torch.zeros(2, dtype=torch.long), torch.arange(4))
    assert torch.isfinite(empty).all()
    model.train()
    model.gradient_checkpointing = True
    with torch.autocast('cpu', dtype=torch.bfloat16):
        loss = model(torch.tensor([[0, 1, -1]]), torch.tensor([2]), torch.arange(4)).float().sum()
    loss.backward()
    assert model.blocks[0].qkv.weight.grad is not None


def test_strict_history_ties_and_frozen_evaluation():
    train = pd.DataFrame({'user_id': [0, 0, 0, 1], 'item_id': [10, 11, 12, 13], 'timestamp': [1, 2, 2, 0]})
    targets = pd.DataFrame({'user_id': [0, 0, 0, 2], 'item_id': [12, 13, 10, 10], 'timestamp': [2, 3, 0, 5]})
    dataset = HistoryDataset(targets, train, [10, 11, 12, 13], 2)
    assert dataset[0][0].tolist() == [0, -1] and dataset[0][1] == 1
    assert dataset[1][0].tolist() == [1, 2]
    assert dataset[2][1] == dataset[3][1] == 0


def h1_inputs():
    """Tiny prepared synthetic catalog, split and numeric SID fixture."""
    events, items = generate_synthetic(5, 12, 8, seed=8)
    splits = temporal_split(events, cold_fraction=0.25, seed=3)
    sids = pd.DataFrame({'item_id': items.item_id, 'code_0': items.item_id % 3, 'code_1': items.item_id // 3})
    config = BaselineConfig(d_model=8, n_heads=2, n_layers=1, max_history=4, epochs=2,
                            batch_size=16, hash_buckets=8, gradient_checkpointing=True)
    return events, items, splits, sids, config


def test_full_ranking_hand_metrics_and_slice_counts():
    _, items, splits, sids, config = h1_inputs()
    model = build_model(items, sids, splits.train, config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    test = splits.test.copy()
    test['item_id'] = 0
    result = evaluate(model, test, splits.train, splits.cold_items, config)
    assert result['slices']['all']['metrics']['MRR'] == 1
    assert result['slices']['all']['metrics']['NDCG@10'] == 1
    assert result['catalog_size'] == 12
    assert sum(result['slices'][s]['n_targets'] for s in ('head', 'torso', 'tail', 'cold')) == len(test)
    empty = evaluate(model, test.iloc[:0], splits.train, splits.cold_items, config)
    assert empty['slices']['all']['metrics'] is None


def test_baseline_learns_and_exact_resume(tmp_path):
    _, items, splits, sids, config = h1_inputs()
    expected = train_baseline(items, sids, splits, tmp_path / 'full', config)
    paused = train_baseline(items, sids, splits, tmp_path / 'partial', config, stop_after_epochs=1)
    assert paused['status'] == 'paused'
    actual = train_baseline(items, sids, splits, tmp_path / 'partial', config, resume=True)
    assert expected == actual
    a = torch.load(tmp_path / 'full/checkpoint.pt', weights_only=True)
    b = torch.load(tmp_path / 'partial/checkpoint.pt', weights_only=True)
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
    assert a['logs'] == b['logs']
    assert np.isfinite(pd.read_csv(tmp_path / 'full/training.csv').loss).all()
    assert 1 <= actual['best_epoch'] <= 2
    with pytest.raises(ValueError, match='mismatch'):
        train_baseline(items, sids, splits, tmp_path / 'partial', replace(config, dropout=0), resume=True)


def test_h1_all_modes_two_seeds_and_aggregation(tmp_path):
    events, items, _, _, config = h1_inputs()
    features = np.random.default_rng(2).normal(size=(len(items), 8)).astype(np.float32)
    tokenizer = Stage1Config(adapter_epochs=1, rq_epochs=1, hidden_dim=8, latent_dim=3,
                              codebook_size=3, kmeans_iters=1, batch_size=32)
    summary = run_h1(events, items, features, tmp_path, seeds=[2, 3], cold_fraction=0.25,
                     baseline=replace(config, epochs=1), tokenizer=tokenizer)
    assert set(summary['modes']) == set(MODES)
    for mode in MODES:
        results = [json.loads((tmp_path / f'seed_{seed}' / mode / 'results.json').read_text()) for seed in [2, 3]]
        values = [r['slices']['all']['metrics']['NDCG@10'] for r in results]
        row = next(row for row in summary['rows'] if row['mode'] == mode and row['slice'] == 'all')
        assert row['NDCG@10_mean'] == pytest.approx(np.mean(values))
        assert row['NDCG@10_std'] == pytest.approx(np.std(values))
        assert row['n_nonempty_seeds'] == 2
    assert (tmp_path / 'summary.csv').exists()
    assert (tmp_path / 'seed_2/tokenizer/sids.parquet').exists()


def test_movielens_local_fixture(tmp_path):
    (tmp_path / 'ratings.dat').write_text('1::10::5::100\n1::11::2::101\n2::11::4::102\n')
    (tmp_path / 'movies.dat').write_text('10::Amélie::Comedy|Drama\n11::Other::Action\n', encoding='latin-1')
    events, items = load_movielens(tmp_path)
    assert events.item_id.tolist() == [10, 11]
    assert items.title.iloc[0] == 'Amélie'
    assert events.task.eq(0).all() and events.query_text.isna().all()
    assert events.timestamp.tolist() == [100, 102]


@pytest.mark.parametrize('schema', ['2018', '2023'])
def test_amazon_local_fixture(tmp_path, schema):
    rows = ([{'reviewerID': 'z', 'asin': 'B', 'overall': 5, 'unixReviewTime': 100},
             {'reviewerID': 'a', 'asin': 'A', 'overall': 4, 'unixReviewTime': 101}]
            if schema == '2018' else
            [{'user_id': 'z', 'parent_asin': 'B', 'rating': 5, 'timestamp': 100000},
             {'user_id': 'a', 'parent_asin': 'A', 'rating': 4, 'timestamp': 101000}])
    key = 'asin' if schema == '2018' else 'parent_asin'
    metadata = [{key: 'B', 'title': 'Beta', 'description': ['line one', 'line two']}]
    review_path, meta_path = tmp_path / 'reviews.jsonl.gz', tmp_path / 'meta.jsonl'
    with gzip.open(review_path, 'wt') as handle:
        handle.write('\n'.join(json.dumps(row) for row in rows))
    meta_path.write_text('\n'.join(json.dumps(row) for row in metadata))
    events, items = load_amazon(review_path, meta_path, schema=schema)
    assert events.item_id.tolist() == [1, 0]
    assert events.user_id.tolist() == [1, 0]
    assert events.timestamp.tolist() == [100, 101]
    assert items.title.tolist() == ['', 'Beta']
    assert items.description.iloc[1] == 'line one line two'
    assert events.domain.eq(1).all()


def test_kuaisar_is_explicit_stub():
    with pytest.raises(NotImplementedError, match='verified'):
        load_kuaisar('unused')


def test_baseline_training_loss_decreases(tmp_path):
    _, items, splits, sids, config = h1_inputs()
    config = replace(config, epochs=8, learning_rate=0.01, warmup_steps=0,
                     dropout=0, mode='gated_no_id_dropout', bf16=False)
    train_baseline(items, sids, splits, tmp_path, config)
    logs = pd.read_csv(tmp_path / 'training.csv')
    assert logs.loss.iloc[-1] < logs.loss.iloc[0]


@pytest.mark.parametrize('source', ['movielens', 'amazon'])
def test_local_loader_to_h1_pipeline(tmp_path, source):
    rows = [(user, (user + step) % 8, 100 + step) for user in range(4) for step in range(8)]
    if source == 'movielens':
        (tmp_path / 'ratings.dat').write_text(''.join(f'{u}::{i}::5::{t}\n' for u, i, t in rows))
        (tmp_path / 'movies.dat').write_text(''.join(f'{i}::Item {i}::Genre\n' for i in range(8)))
        events, items = load_movielens(tmp_path)
    else:
        reviews, metadata = tmp_path / 'reviews.jsonl', tmp_path / 'meta.jsonl'
        reviews.write_text('\n'.join(json.dumps({'reviewerID': str(u), 'asin': str(i),
            'overall': 5, 'unixReviewTime': t}) for u, i, t in rows))
        metadata.write_text('\n'.join(json.dumps({'asin': str(i), 'title': f'Item {i}'}) for i in range(8)))
        events, items = load_amazon(reviews, metadata)
    features = np.random.default_rng(1).normal(size=(len(items), 8)).astype(np.float32)
    summary = run_h1(events, items, features, tmp_path / 'h1', seeds=[1], modes=['gated'],
        cold_fraction=0.25, baseline=BaselineConfig(d_model=8, n_heads=2, n_layers=1,
            max_history=4, epochs=1, batch_size=16),
        tokenizer=Stage1Config(adapter_epochs=1, rq_epochs=1, hidden_dim=8, latent_dim=3,
            codebook_size=3, kmeans_iters=1, batch_size=16))
    assert summary['n_events'] == 32
    assert (tmp_path / 'h1/seed_1/gated/results.json').exists()
