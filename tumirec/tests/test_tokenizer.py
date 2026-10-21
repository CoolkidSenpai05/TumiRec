"""Synthetic CPU tests for M2; never load or download real text weights."""
from dataclasses import replace
import json
import math
import sys
import types

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from tumirec.data.synthetic import generate_synthetic
from tumirec.data.splits import temporal_split
from tumirec.tokenizer.content_encoder import ContentEncoder, item_texts
from tumirec.tokenizer.adapter import ContrastiveAdapter, cointeraction_pairs, info_nce
from tumirec.tokenizer.rqvae import RQVAE
from tumirec.tokenizer.sid_assign import assign_sids, tokenize_catalog
from tumirec.train.stage1_tokenizer import Stage1Config, train_tokenizer


@pytest.fixture(autouse=True)
def cpu_threads():
    """Keep tiny MLP tests fast without altering global settings after the module."""
    previous = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)
    torch.use_deterministic_algorithms(deterministic)


class FakeEncoder(nn.Module):
    """Deterministic synthetic encoder with an observable frozen parameter."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.calls = 0

    def encode(self, sentences, **kwargs):
        assert not self.training and not self.weight.requires_grad
        assert not torch.is_grad_enabled()
        self.calls += 1
        return np.array([[len(s), sum(map(ord, s)) % 17] for s in sentences], dtype=np.float32)


def test_content_cache_freeze_identity_and_corruption(tmp_path):
    _, items = generate_synthetic(2, 5, 4)
    items.loc[0, "brand"] = None
    assert item_texts(items)[0] == "Item 0 | Category 0 |  | Synthetic item 0"
    encoder = FakeEncoder()
    content = ContentEncoder("synthetic-test-v1", encoder=encoder)
    path = tmp_path / "features.npy"
    first = content.encode_items(items, path)
    assert first.shape == (5, 2) and first.dtype == np.float32
    # A fresh wrapper must hit disk without importing/loading sentence-transformers.
    np.testing.assert_array_equal(ContentEncoder("synthetic-test-v1").encode_items(items, path), first)
    assert encoder.calls == 1
    for changed in [items.iloc[::-1], items.assign(title="changed"), items.assign(item_id=items.item_id + 10)]:
        with pytest.raises(ValueError, match="identity"):
            content.encode_items(changed, path)
    with pytest.raises(ValueError, match="identity"):
        ContentEncoder("different-model").encode_items(items, path)
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        content.encode_items(items, path)


def test_local_only_model_loading(monkeypatch, tmp_path):
    calls = []

    def factory(name, **kwargs):
        calls.append((name, kwargs))
        return FakeEncoder()

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=factory))
    ContentEncoder().encode_cached(["test query"], tmp_path / "query.npy")
    assert calls == [("BAAI/bge-small-en-v1.5", {"device": "cpu", "local_files_only": True})]


def test_incomplete_cache_and_invalid_embeddings(tmp_path):
    path = tmp_path / "incomplete.npy"
    np.save(path, np.ones((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="Incomplete"):
        ContentEncoder().encode_cached(["a", "b"], path)
    with pytest.raises(ValueError):
        ContentEncoder().encode_cached([], tmp_path / "empty.npy")
    with pytest.raises(ValueError):
        ContentEncoder._validate(np.array([[np.nan]], dtype=np.float32), 1)


def test_pair_window_ties_user_boundaries_and_train_only():
    train = pd.DataFrame({"user_id": [0, 0, 0, 0, 1, 1],
                          "item_id": [10, 11, 12, 13, 20, 21],
                          "timestamp": [0, 0, 2, 10, 1, 3]})
    assert cointeraction_pairs(train, window_seconds=3).tolist() == [[10, 12], [11, 12], [20, 21]]
    assert cointeraction_pairs(train, window_seconds=3, max_neighbors=1).tolist() == [[11, 12], [20, 21]]
    events, _ = generate_synthetic(4, 12, 8)
    split = temporal_split(events, cold_fraction=0.25)
    assert not set(cointeraction_pairs(split.train).flatten()) & split.cold_items
    assert cointeraction_pairs(train.iloc[:1]).shape == (0, 2)


def test_infonce_hand_value_and_duplicate_negatives():
    x = torch.eye(3, requires_grad=True)
    loss = info_nce(x, x, temperature=1)
    assert float(loss.detach()) == pytest.approx(math.log(math.e + 2) - 1)
    loss.backward()
    assert torch.isfinite(x.grad).all()
    pairs = torch.tensor([[0, 1], [2, 1]])
    assert info_nce(torch.randn(2, 4), torch.randn(2, 4), item_pairs=pairs).isfinite()
    # Every target is an additional positive in this direction, no false negative penalty.
    repeated = torch.tensor([[0, 1], [0, 1]])
    assert float(info_nce(torch.ones(2, 4), torch.ones(2, 4), item_pairs=repeated)) == pytest.approx(0)


def test_adapter_shape_and_learns_pairs():
    torch.manual_seed(7)
    features = torch.randn(12, 10)
    adapter = ContrastiveAdapter(10, output_dim=128, hidden_dim=32)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=0.01)
    losses = []
    for _ in range(40):
        optimizer.zero_grad()
        loss = info_nce(adapter(features[:6]), adapter(features[6:]), temperature=0.2)
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    assert adapter(features).shape == (12, 128)
    assert losses[-1] < losses[0] * 0.5


@pytest.mark.parametrize("kmeans", [False, True])
def test_rq_shapes_usage_gradients_and_learning(kmeans):
    torch.manual_seed(4)
    x = torch.randn(24, 8) * 0.3 + 0.8
    model = RQVAE(8, latent_dim=6, hidden_dim=24, n_levels=3, codebook_size=8,
                  kmeans_init=kmeans, kmeans_iters=3, dead_code_reset=False)
    if kmeans:
        with pytest.raises(RuntimeError, match="initialize"):
            model(x)
        model.initialize(x, seed=4)
    model.eval()
    initial = float(model(x).reconstruction_loss.detach())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for _ in range(80):
        optimizer.zero_grad()
        result = model(x)
        result.loss.backward()
        assert model.codebooks.grad is not None
        optimizer.step()
    model.eval()
    result = model(x)
    assert result.reconstruction.shape == x.shape
    assert result.codes.shape == (24, 3)
    assert result.codes.min() >= 0 and result.codes.max() < 8
    assert ((result.usage > 0) & (result.usage <= 1)).all()
    assert float(result.reconstruction_loss.detach()) < initial * 0.5
    assert torch.isfinite(result.loss)


def test_kmeans_deterministic_small_batch_and_dead_codes():
    torch.manual_seed(3)
    x = torch.randn(3, 4)
    model = RQVAE(4, latent_dim=3, hidden_dim=8, codebook_size=8, kmeans_iters=2,
                  dead_code_patience=2)
    model.initialize(x, seed=6)
    before = model.codebooks.detach().clone()
    model.initialize(x, seed=6)
    torch.testing.assert_close(model.codebooks, before)
    model.unused_steps.fill_(2)
    changed = model.reset_dead_codes(x, seed=9)
    assert len(changed) == 3 and not model.unused_steps.any()
    model.dead_code_reset = False
    model.unused_steps.fill_(3)
    assert model.reset_dead_codes(x) == []
    model.eval()
    steps = model.unused_steps.clone()
    model(x)
    torch.testing.assert_close(steps, model.unused_steps)


def test_sid_collisions_stable_under_reordering():
    ids = [30, 10, 20, 40]
    codes = np.array([[1, 2], [1, 2], [0, 2], [1, 2]])
    table, diagnostics = assign_sids(ids, codes)
    assert table.code_2.tolist() == [1, 0, 0, 2]
    assert diagnostics["collision_rate"] == 0.5
    assert diagnostics["collision_suffix_size"] == 3
    assert not table.drop(columns="item_id").duplicated().any()
    other, _ = assign_sids(ids[::-1], codes[::-1])
    pd.testing.assert_frame_equal(table.sort_values("item_id").reset_index(drop=True),
                                  other.sort_values("item_id").reset_index(drop=True))
    with pytest.raises(ValueError):
        assign_sids([1, 1], codes[:2])


def test_catalog_diagnostics_and_state_restore():
    torch.manual_seed(1)
    x = torch.randn(7, 4)
    model = RQVAE(4, latent_dim=3, hidden_dim=8, codebook_size=4, kmeans_init=False)
    steps = model.unused_steps.clone()
    table, diagnostics = tokenize_catalog(model, x, list(range(7)), batch_size=3)
    assert model.training
    torch.testing.assert_close(steps, model.unused_steps)
    assert len(table) == 7
    model.eval()
    expected = model(x)
    assert diagnostics["reconstruction_loss"] == pytest.approx(float(expected.reconstruction_loss.detach()))
    assert diagnostics["codebook_usage"] == expected.usage.tolist()


def stage_inputs():
    """Return a tiny train/cold split and explicit random content features (16,8)."""
    events, items = generate_synthetic(5, 16, 10, seed=3)
    train = temporal_split(events, cold_fraction=0.25, seed=3).train
    features = np.random.default_rng(3).normal(size=(len(items), 8)).astype(np.float32)
    config = Stage1Config(seed=3, adapter_dim=128, hidden_dim=16, latent_dim=4,
                          n_levels=2, codebook_size=4, kmeans_iters=2,
                          adapter_epochs=2, rq_epochs=3, batch_size=32,
                          dead_code_patience=1, gradient_checkpointing=True)
    return features, items, train, config


@pytest.mark.parametrize("pause_epochs", [1, 3])
def test_stage1_smoke_and_exact_epoch_resume(tmp_path, pause_epochs):
    features, items, train, config = stage_inputs()
    full = tmp_path / "full"
    partial = tmp_path / "partial"
    expected = train_tokenizer(features, items, train, full, config)
    paused = train_tokenizer(features, items, train, partial, config, stop_after_epochs=pause_epochs)
    assert paused["status"] == "paused"
    actual = train_tokenizer(features, items, train, partial, config, resume=True)
    assert expected == actual
    assert actual["n_train_items"] == train.item_id.nunique() < len(items)
    assert all(value > 0 for value in actual["codebook_usage"])
    pd.testing.assert_frame_equal(pd.read_parquet(full / "sids.parquet"),
                                  pd.read_parquet(partial / "sids.parquet"))
    full_state = torch.load(full / "checkpoint.pt", weights_only=True)
    partial_state = torch.load(partial / "checkpoint.pt", weights_only=True)
    for section in ("adapter", "rq"):
        for key in full_state[section]:
            torch.testing.assert_close(full_state[section][key], partial_state[section][key], rtol=0, atol=0)
    assert json.loads((full / "diagnostics.json").read_text()) == expected
    assert len(pd.read_csv(full / "training.csv")) == 5
    with pytest.raises(ValueError, match="identity"):
        train_tokenizer(features + 1, items, train, partial, config, resume=True)


@pytest.mark.parametrize("adapter_enabled", [False, True])
def test_stage1_ablation_and_cold_features_not_fitted(tmp_path, adapter_enabled):
    features, items, train, config = stage_inputs()
    config = replace(config, adapter_enabled=adapter_enabled, kmeans_init=False,
                     dead_code_reset=False, bf16=False, gradient_checkpointing=False)
    first = train_tokenizer(features, items, train, tmp_path / "a", config)
    altered = features.copy()
    altered[~items.item_id.isin(train.item_id)] += 100
    second = train_tokenizer(altered, items, train, tmp_path / "b", config)
    assert first["train_reconstruction_loss"] == second["train_reconstruction_loss"]
    a = torch.load(tmp_path / "a/checkpoint.pt", weights_only=True)["rq"]
    b = torch.load(tmp_path / "b/checkpoint.pt", weights_only=True)["rq"]
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
