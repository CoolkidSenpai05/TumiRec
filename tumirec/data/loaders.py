"""Local-only MovieLens-1M and Amazon 2018/2023 JSONL adapters."""
from __future__ import annotations
import gzip
import json
from pathlib import Path
from collections.abc import Iterator
import pandas as pd


def _interactions(frame: pd.DataFrame, domain: int) -> pd.DataFrame:
    """Complete a local implicit-positive frame with the unified schema."""
    result = frame[['user_id', 'item_id', 'timestamp']].copy()
    result['action'], result['domain'], result['task'] = 1, domain, 0
    result['query_text'] = None
    result['source_item_id'] = pd.array([None] * len(result), dtype='Int64')
    return result.reset_index(drop=True)


def load_movielens(directory: str | Path, min_rating: float = 4.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read extracted ML-1M ratings.dat/movies.dat (:: separators, Latin-1 titles)."""
    directory = Path(directory)
    ratings = pd.read_csv(directory / 'ratings.dat', sep='::', engine='python',
                          names=['user_id', 'item_id', 'rating', 'timestamp'])
    movies = pd.read_csv(directory / 'movies.dat', sep='::', engine='python', encoding='latin-1',
                        names=['item_id', 'title', 'category'])
    events = _interactions(ratings.loc[ratings.rating >= min_rating], domain=0)
    items = movies.assign(description='', brand=None, price=None)
    items['first_seen_ts'] = items.item_id.map(events.groupby('item_id').timestamp.min())
    if movies.item_id.duplicated().any() or not set(events.item_id) <= set(items.item_id):
        raise ValueError('Duplicate/missing MovieLens catalog IDs')
    return events, items


def _jsonl(path: str | Path) -> Iterator[dict]:
    """Stream JSON objects from plain or gzip JSONL; no eval of legacy literals."""
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_amazon(reviews: str | Path, metadata: str | Path,
                min_rating: float = 4.0, schema: str = '2018') -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read one local category; 2018 uses asin/reviewerID/unixReviewTime/overall.

    2023 uses parent_asin/user_id/timestamp(milliseconds)/rating. Metadata IDs
    must use the same schema; IDs are mapped deterministically from sorted strings.
    Only reviewed items passing the rating threshold form the candidate catalog.
    """
    if schema not in ('2018', '2023'):
        raise ValueError('Amazon schema must be 2018 or 2023')
    item_key, user_key, time_key, rating_key = (('asin', 'reviewerID', 'unixReviewTime', 'overall')
        if schema == '2018' else ('parent_asin', 'user_id', 'timestamp', 'rating'))
    rows = []
    for review in _jsonl(reviews):
        if float(review[rating_key]) >= min_rating:
            rows.append((str(review[user_key]), str(review[item_key]),
                         int(review[time_key]) // (1000 if schema == '2023' else 1)))
    if not rows:
        raise ValueError('No Amazon interactions after rating filter')
    frame = pd.DataFrame(rows, columns=['user_id', 'item_id', 'timestamp'])
    users = {key: index for index, key in enumerate(sorted(frame.user_id.unique()))}
    ids = {key: index for index, key in enumerate(sorted(frame.item_id.unique()))}
    catalog = {}
    for record in _jsonl(metadata):
        key = str(record[item_key])
        if key in ids:
            catalog[key] = record
    items = []
    for key, index in ids.items():
        record = catalog.get(key, {})
        description = record.get('description', '')
        categories = record.get('category', record.get('categories', record.get('main_category', '')))
        items.append({'item_id': index, 'source_item_id': key, 'title': record.get('title', ''),
                      'category': str(categories), 'description': ' '.join(map(str, description))
                      if isinstance(description, list) else str(description),
                      'brand': record.get('brand', record.get('store')), 'price': None})
    frame['user_id'], frame['item_id'] = frame.user_id.map(users), frame.item_id.map(ids)
    events = _interactions(frame, domain=1)
    items = pd.DataFrame(items)
    items['first_seen_ts'] = items.item_id.map(events.groupby('item_id').timestamp.min())
    return events, items


def load_kuaisar(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """TODO verify real release format; expected output is SPEC section 2 schema."""
    raise NotImplementedError('KuaiSAR format must be verified against the supplied release; see SPEC section 2')
