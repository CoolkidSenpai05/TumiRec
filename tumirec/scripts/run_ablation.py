"""M3 H1 runner: local data -> frozen content -> S1 -> five causal baseline modes."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from collections.abc import Sequence
import numpy as np
import pandas as pd
from tumirec.data.loaders import load_amazon, load_movielens
from tumirec.data.synthetic import generate_synthetic
from tumirec.data.splits import temporal_split
from tumirec.model.item_token import MODES
from tumirec.tokenizer.content_encoder import ContentEncoder
from tumirec.train.stage1_tokenizer import Stage1Config, train_tokenizer
from tumirec.train.h1_baseline import BaselineConfig, train_baseline


def run_h1(events: pd.DataFrame, items: pd.DataFrame, features: np.ndarray,
           output_dir: str | Path, seeds: Sequence[int] = (42, 43, 44),
           modes: Sequence[str] = MODES, cold_fraction: float = 0.1,
           split_seed: int = 42, baseline: BaselineConfig = BaselineConfig(),
           tokenizer: Stage1Config = Stage1Config(), resume: bool = False) -> dict:
    """Run H1 with fixed split, per-seed shared SIDs and validation-selected models.

    Features (N,D) align with input catalog rows. Each seed retrains S1 once, shared
    by all modes. Aggregate population std (ddof=0) over nonempty slices only.
    """
    if not seeds or len(set(seeds)) != len(seeds) or not modes or len(set(modes)) != len(modes) or not set(modes) <= set(MODES):
        raise ValueError('Need unique seeds and valid unique H1 modes')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    splits = temporal_split(events, cold_fraction=cold_fraction, seed=split_seed)
    results = []
    for seed in seeds:
        seed_dir = output / f'seed_{seed}'
        tokenizer_dir = seed_dir / 'tokenizer'
        train_tokenizer(features, items, splits.train, tokenizer_dir,
                        replace(tokenizer, seed=seed), resume=resume and (tokenizer_dir / 'checkpoint.pt').exists())
        sids = pd.read_parquet(tokenizer_dir / 'sids.parquet')
        for mode in modes:
            run_dir = seed_dir / mode
            result = train_baseline(items, sids, splits, run_dir,
                                    replace(baseline, mode=mode, seed=seed),
                                    resume=resume and (run_dir / 'checkpoint.pt').exists())
            results.append(result)
    rows = []
    for mode in modes:
        runs = [result for result in results if result['config']['mode'] == mode]
        slice_names = sorted(set().union(*(run['slices'] for run in runs)))
        for name in slice_names:
            usable = [run['slices'][name] for run in runs if name in run['slices'] and run['slices'][name]['metrics'] is not None]
            row = {'mode': mode, 'slice': name, 'n_seeds': len(runs), 'n_nonempty_seeds': len(usable),
                   'mean_n_targets': float(np.mean([run['slices'].get(name, {}).get('n_targets', 0) for run in runs]))}
            if usable:
                for metric in usable[0]['metrics']:
                    values = [part['metrics'][metric] for part in usable]
                    row[f'{metric}_mean'], row[f'{metric}_std'] = float(np.mean(values)), float(np.std(values))
            rows.append(row)
    summary = {'ablation': 'H1', 'seeds': list(seeds), 'modes': list(modes),
               'split_seed': split_seed, 'cold_fraction': cold_fraction, 'std_ddof': 0,
               'n_catalog_items': len(items), 'n_events': len(events),
               'baseline': asdict(baseline), 'tokenizer': asdict(tokenizer), 'rows': rows}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    pd.DataFrame(rows).to_csv(output / 'summary.csv', index=False)
    return summary


def main() -> None:
    """Run H1 on synthetic, extracted ML-1M or local Amazon category files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ablation', choices=['H1'], default='H1')
    parser.add_argument('--dataset', choices=['synthetic', 'movielens', 'amazon'], required=True)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--reviews', type=Path)
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--amazon-schema', choices=['2018', '2023'], default='2018')
    parser.add_argument('--min-rating', type=float, default=4.0)
    parser.add_argument('--encoder-model', default='BAAI/bge-small-en-v1.5')
    parser.add_argument('--output', type=Path, default=Path('results/h1'))
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    parser.add_argument('--cold-fraction', type=float, default=0.1)
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--config', type=Path, help='YAML with baseline/tokenizer settings')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.dataset == 'synthetic':
        events, items = generate_synthetic(8, 24, 12, seed=42)
        features = np.random.default_rng(42).normal(size=(len(items), 16)).astype(np.float32)
        baseline = BaselineConfig(d_model=16, n_heads=2, n_layers=1, max_history=8, epochs=2,
                                  batch_size=32, hash_buckets=32, device=args.device)
        tokenizer = Stage1Config(adapter_epochs=1, rq_epochs=2, hidden_dim=16, latent_dim=4,
                                  codebook_size=4, kmeans_iters=2, batch_size=64, device=args.device)
    else:
        if args.dataset == 'movielens':
            if args.data_dir is None:
                parser.error('MovieLens requires --data-dir with extracted ML-1M files')
            events, items = load_movielens(args.data_dir, min_rating=args.min_rating)
        else:
            if args.reviews is None or args.metadata is None:
                parser.error('Amazon requires --reviews and --metadata')
            events, items = load_amazon(args.reviews, args.metadata, args.min_rating, args.amazon_schema)
        features = ContentEncoder(args.encoder_model).encode_items(items, args.output / 'content.npy')
        baseline, tokenizer = BaselineConfig(device=args.device), Stage1Config(device=args.device)
    if args.config:
        from omegaconf import OmegaConf
        loaded = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
        baseline = replace(baseline, **loaded.get('baseline', {}))
        tokenizer = replace(tokenizer, **loaded.get('tokenizer', {}))
    summary = run_h1(events, items, features, args.output, args.seeds, args.modes,
                     args.cold_fraction, args.split_seed, baseline, tokenizer, args.resume)
    manifest = {'dataset': args.dataset, 'data_dir': str(args.data_dir), 'reviews': str(args.reviews),
                'metadata': str(args.metadata), 'amazon_schema': args.amazon_schema, 'min_rating': args.min_rating,
                'features': 'seeded_random_test_features' if args.dataset == 'synthetic' else args.encoder_model}
    (args.output / 'data_manifest.json').write_text(json.dumps(manifest, indent=2))
    items.to_parquet(args.output / 'items.parquet', index=False)
    print(json.dumps({'summary': str(args.output / 'summary.json'), 'runs': len(summary['seeds']) * len(summary['modes'])}))


if __name__ == '__main__':
    main()
