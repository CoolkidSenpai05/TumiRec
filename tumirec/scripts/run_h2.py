"""M4 H2: train four memory variants and evaluate fixed staleness across seeds."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from collections.abc import Sequence
import numpy as np
import pandas as pd
from tumirec.data.loaders import load_movielens, load_amazon
from tumirec.data.synthetic import generate_synthetic
from tumirec.data.splits import temporal_split
from tumirec.data.memory_dataset import LagConfig
from tumirec.model.memory import MEMORY_MODES
from tumirec.tokenizer.content_encoder import ContentEncoder
from tumirec.train.stage1_tokenizer import Stage1Config, train_tokenizer
from tumirec.train.h1_baseline import BaselineConfig
from tumirec.train.h2_memory import MemoryConfig, train_memory


def run_h2(events: pd.DataFrame, items: pd.DataFrame, features: np.ndarray,
           output_dir: str | Path, seeds: Sequence[int] = (42, 43, 44),
           modes: Sequence[str] = MEMORY_MODES, eval_lags: tuple[int, ...] = (0, 3600, 86400),
           base: BaselineConfig = BaselineConfig(), memory: MemoryConfig = MemoryConfig(),
           lag: LagConfig = LagConfig(), tokenizer: Stage1Config = Stage1Config(),
           cold_fraction: float = 0.1, split_seed: int = 42, resume: bool = False,
           profile_repeats: int = 0) -> dict:
    """Fit S1 once per seed, share SIDs across modes, then aggregate real metrics."""
    if not seeds or len(set(seeds)) != len(seeds) or not modes or len(set(modes)) != len(modes) or not set(modes) <= set(MEMORY_MODES):
        raise ValueError('Need unique seeds and valid unique H2 modes')
    splits = temporal_split(events, cold_fraction, split_seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in seeds:
        s1_dir = output / f'seed_{seed}' / 'tokenizer'
        train_tokenizer(features, items, splits.train, s1_dir, replace(tokenizer, seed=seed),
                        resume=resume and (s1_dir / 'checkpoint.pt').exists())
        sids = pd.read_parquet(s1_dir / 'sids.parquet')
        for mode in modes:
            destination = output / f'seed_{seed}' / mode
            runs.append(train_memory(items, sids, features, splits, destination,
                replace(base, seed=seed), replace(memory, mode=mode), lag, eval_lags,
                resume=resume and (destination / 'checkpoint.pt').exists(), profile_repeats=profile_repeats))
    rows = []
    for mode in modes:
        selected_runs = [run for run in runs if run['config']['memory']['mode'] == mode]
        for seconds in eval_lags:
            evaluations = [evaluation for run in selected_runs for evaluation in run['evaluations'] if evaluation['lag_seconds'] == seconds]
            for name in evaluations[0]['slices']:
                values = [evaluation['slices'][name] for evaluation in evaluations if evaluation['slices'][name]['metrics'] is not None]
                row = {'mode': mode, 'lag_seconds': seconds, 'slice': name, 'n_seeds': len(evaluations),
                       'n_nonempty_seeds': len(values),
                       'mean_n_targets': float(np.mean([evaluation['slices'][name]['n_targets'] for evaluation in evaluations])),
                       'mean_memory_events': float(np.mean([evaluation['mean_memory_events'] for evaluation in evaluations]))}
                if values:
                    for metric in values[0]['metrics']:
                        scores = [value['metrics'][metric] for value in values]
                        row[f'{metric}_mean'], row[f'{metric}_std'] = float(np.mean(scores)), float(np.std(scores))
                rows.append(row)
    summary = {'ablation': 'H2', 'seeds': list(seeds), 'modes': list(modes), 'eval_lags': list(eval_lags),
               'cold_fraction': cold_fraction, 'split_seed': split_seed, 'std_ddof': 0,
               'baseline': asdict(base), 'memory': asdict(memory), 'lag': asdict(lag),
               'tokenizer': asdict(tokenizer), 'rows': rows}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    pd.DataFrame(rows).to_csv(output / 'summary.csv', index=False)
    return summary


def main() -> None:
    """CLI for synthetic or local-data H2; no downloads, no M5 implementation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['synthetic', 'movielens', 'amazon'], required=True)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--reviews', type=Path)
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--amazon-schema', choices=['2018', '2023'], default='2018')
    parser.add_argument('--encoder-model', default='BAAI/bge-small-en-v1.5')
    parser.add_argument('--min-rating', type=float, default=4.0)
    parser.add_argument('--output', type=Path, default=Path('results/h2'))
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--modes', choices=MEMORY_MODES, nargs='+', default=list(MEMORY_MODES))
    parser.add_argument('--eval-lags', type=int, nargs='+', default=[0, 3600, 86400])
    parser.add_argument('--cold-fraction', type=float, default=0.1)
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--profile-repeats', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    memory, lag = MemoryConfig(), LagConfig()
    if args.dataset == 'synthetic':
        events, items = generate_synthetic(8, 24, 40, seed=42)
        features = np.random.default_rng(42).normal(size=(len(items), 16)).astype(np.float32)
        base = BaselineConfig(d_model=16, n_heads=2, n_layers=1, max_history=5, epochs=2,
                              batch_size=32, hash_buckets=32, device=args.device)
        memory = replace(memory, n_latents=4, max_history=32)
        tokenizer = Stage1Config(adapter_epochs=1, rq_epochs=2, hidden_dim=16, latent_dim=4,
                                  codebook_size=4, kmeans_iters=2, batch_size=64, device=args.device)
    else:
        if args.dataset == 'movielens':
            if args.data_dir is None:
                parser.error('MovieLens requires --data-dir')
            events, items = load_movielens(args.data_dir, args.min_rating)
        else:
            if args.reviews is None or args.metadata is None:
                parser.error('Amazon requires --reviews and --metadata')
            events, items = load_amazon(args.reviews, args.metadata, args.min_rating, args.amazon_schema)
        features = ContentEncoder(args.encoder_model).encode_items(items, args.output / 'content.npy')
        base, tokenizer = BaselineConfig(device=args.device), Stage1Config(device=args.device)
    if args.config:
        from omegaconf import OmegaConf
        config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
        base, memory = replace(base, **config.get('baseline', {})), replace(memory, **config.get('memory', {}))
        lag, tokenizer = LagConfig(**config.get('lag', {})), replace(tokenizer, **config.get('tokenizer', {}))
    summary = run_h2(events, items, features, args.output, args.seeds, args.modes, tuple(args.eval_lags),
                     base, memory, lag, tokenizer, args.cold_fraction, args.split_seed, args.resume,
                     args.profile_repeats)
    items.to_parquet(args.output / 'items.parquet', index=False)
    manifest = {'dataset': args.dataset, 'data_dir': str(args.data_dir), 'reviews': str(args.reviews),
                'metadata': str(args.metadata), 'amazon_schema': args.amazon_schema, 'min_rating': args.min_rating,
                'features': 'seeded_random_test_features' if args.dataset == 'synthetic' else args.encoder_model}
    (args.output / 'data_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'summary': str(args.output / 'summary.json'),
                      'trained_runs': len(summary['seeds']) * len(summary['modes']),
                      'lag_evaluations': len(summary['seeds']) * len(summary['modes']) * len(summary['eval_lags'])}))


if __name__ == '__main__':
    main()
