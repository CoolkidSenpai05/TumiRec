"""Training-only statistics for the future item gate and evaluation slices."""
import pandas as pd


def item_statistics(train: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Return one row per item: count, age_days, normalized age, popularity rank.

    Age uses first TRAIN observation and the maximum TRAIN timestamp, never
    catalog first_seen_ts or held-out interactions. Unobserved items are zero.
    """
    if items.item_id.duplicated().any():
        raise ValueError("Item IDs must be unique")
    aggregate = train.groupby("item_id").agg(count=("item_id", "size"),
                                             first_train_ts=("timestamp", "min"))
    result = items[["item_id"]].merge(aggregate, on="item_id", how="left")
    result["count"] = result["count"].fillna(0).astype("int64")
    cutoff = train.timestamp.max() if len(train) else 0
    result["age_days"] = ((cutoff - result.first_train_ts) / 86400).fillna(0).clip(lower=0)
    result["age_days_normalized"] = result.age_days / max(float(result.age_days.max()), 1.0)
    result["popularity_percentile"] = 0.0
    seen = result["count"] > 0
    result.loc[seen, "popularity_percentile"] = result.loc[seen, "count"].rank(pct=True)
    return result.drop(columns="first_train_ts")
