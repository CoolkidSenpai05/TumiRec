import numpy as np
import pandas as pd
import pytest
from tumirec.data.synthetic import generate_synthetic, write_synthetic
from tumirec.data.splits import temporal_split
from tumirec.data.sampler import SlidingWindowSampler
from tumirec.data.preprocess import item_statistics


def test_synthetic_reproducible_and_parquet(tmp_path):
    events, items = generate_synthetic(4, 12, 8, seed=7)
    other, _ = generate_synthetic(4, 12, 8, seed=7)
    pd.testing.assert_frame_equal(events, other)
    assert set(events.task) == {0, 1, 2}
    assert events.loc[events.task == 1, "query_text"].notna().all()
    assert events.loc[events.task == 2, "source_item_id"].notna().all()
    assert set(events.item_id) <= set(items.item_id)
    write_synthetic(tmp_path, n_users=4, n_items=12, events_per_user=8, seed=7)
    pd.testing.assert_frame_equal(events, pd.read_parquet(tmp_path / "interactions.parquet"))


def test_splits_and_cold_items():
    events, _ = generate_synthetic(5, 10, 10)
    splits = temporal_split(events)
    assert (len(splits.train), len(splits.validation), len(splits.test)) == (40, 5, 5)
    assert splits.train.timestamp.max() < splits.validation.timestamp.min()
    assert splits.validation.timestamp.max() < splits.test.timestamp.min()
    cold = temporal_split(events, cold_fraction=0.3, seed=9)
    assert len(cold.cold_items) == 3
    assert not set(cold.train.item_id) & cold.cold_items
    assert cold.cold_items == temporal_split(events, 0.3, 9).cold_items
    pd.testing.assert_frame_equal(cold.test, splits.test)
    pd.testing.assert_frame_equal(cold.validation, splits.validation)


def test_short_users_and_empty_splits():
    events, _ = generate_synthetic(2, 3, 1)
    splits = temporal_split(events)
    assert len(splits.test) == 2 and splits.train.empty and splits.validation.empty
    assert temporal_split(events.iloc[:0]).test.empty
    events, _ = generate_synthetic(2, 3, 2)
    splits = temporal_split(events)
    assert splits.train.empty and len(splits.validation) == 2


@pytest.mark.parametrize("lag", [0, 3600, 86400])
def test_strict_no_leakage_with_ties_and_future(lag):
    events, _ = generate_synthetic(3, 8, 12)
    events.loc[1, "timestamp"] = events.loc[2, "timestamp"]
    shuffled = events.sample(frac=1, random_state=5)
    sampler = SlidingWindowSampler(events, shuffled, max_history=4, recent_window=2,
                                   fixed_lag_seconds=lag)
    for sample in sampler:
        assert (sample.memory.timestamp < sample.target["timestamp"] - lag).all()
        assert (sample.recent.timestamp < sample.target["timestamp"]).all()
        assert (sample.memory.user_id == sample.target["user_id"]).all()
        expected = shuffled.loc[(shuffled.user_id == sample.target["user_id"]) &
                                (shuffled.timestamp < sample.target["timestamp"] - lag)]
        pd.testing.assert_frame_equal(sample.memory, expected.sort_values("timestamp", kind="stable").tail(4))
        assert len(sample.memory) <= 4 and len(sample.recent) <= 2


def test_lag_randomness_and_zero_windows():
    events, _ = generate_synthetic(3, 8, 20)
    sampler = SlidingWindowSampler(events, events, p_stale=1, seed=8)
    lags = [sample.lag_seconds for sample in sampler]
    assert len(set(lags)) > 1
    assert lags == [sample.lag_seconds for sample in sampler]
    sampler.set_epoch(1)
    assert lags != [sample.lag_seconds for sample in sampler]
    fixed = SlidingWindowSampler(events, events, p_stale=1, fixed_lag_seconds=0)
    assert all(sample.lag_seconds == 0 for sample in fixed)
    zero = SlidingWindowSampler(events, events, max_history=0, recent_window=0)
    assert zero[10].memory.empty and zero[10].recent.empty
    unseen = events.iloc[[0]].assign(user_id=99)
    assert SlidingWindowSampler(unseen, events)[0].memory.empty


def test_statistics_train_only():
    events, items = generate_synthetic(3, 8, 12)
    splits = temporal_split(events, cold_fraction=0.25)
    stats = item_statistics(splits.train, items).set_index("item_id")
    assert stats["count"].sum() == len(splits.train)
    assert (stats.loc[list(splits.cold_items), "count"] == 0).all()
    changed_items = items.assign(first_seen_ts=-1)
    pd.testing.assert_frame_equal(item_statistics(splits.train, items),
                                  item_statistics(splits.train, changed_items))
    for item_id, group in splits.train.groupby("item_id"):
        assert stats.loc[item_id, "age_days"] == pytest.approx(
            (splits.train.timestamp.max() - group.timestamp.min()) / 86400)
    assert item_statistics(events.iloc[:0], items)["count"].sum() == 0


@pytest.mark.parametrize("kwargs", [{"p_stale": 2}, {"max_history": -1},
                                      {"lag_seconds": ()}, {"fixed_lag_seconds": -1}])
def test_invalid_sampler_config(kwargs):
    events, _ = generate_synthetic(1, 2, 3)
    with pytest.raises(ValueError):
        SlidingWindowSampler(events, events, **kwargs)


def test_frozen_evaluation_excludes_validation_and_cold_history():
    events, _ = generate_synthetic(4, 10, 10)
    splits = temporal_split(events, cold_fraction=0.3)
    sampler = SlidingWindowSampler(splits.test, splits.train)
    for sample in sampler:
        assert not set(sample.memory.item_id) & splits.cold_items
        expected = splits.train.loc[splits.train.user_id == sample.target["user_id"]]
        pd.testing.assert_frame_equal(sample.memory, expected)
        assert (sample.memory.timestamp < splits.validation.timestamp.min()).all()
