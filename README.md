# TUMI-Rec

A research prototype for unified search and recommendation. Milestones M1–M3
are implemented. See SPEC.md for the architecture and PROGRESS.md for status.

## Run locally

Python 3.10+ is required. The M1 runtime uses NumPy, pandas, PyArrow and PyTorch;
tests use pytest. `pyproject.toml` declares dependencies, including OmegaConf
for consuming YAML configs. No installation or downloads happen automatically.
From the repository root:

```sh
pytest -q
python -c 'from tumirec.data.synthetic import write_synthetic; write_synthetic("artifacts/synthetic")'
```

Example in memory:

```python
from tumirec.data.synthetic import generate_synthetic
from tumirec.data.splits import temporal_split
from tumirec.data.sampler import SlidingWindowSampler
from tumirec.data.preprocess import item_statistics
from tumirec.eval.metrics import ranking_metrics

interactions, items = generate_synthetic(seed=42)
splits = temporal_split(interactions, cold_fraction=0.1, seed=42)
stats = item_statistics(splits.train, items)
train = SlidingWindowSampler(splits.train, splits.train, seed=42)
test = SlidingWindowSampler(splits.test, splits.train, fixed_lag_seconds=0)
example = test[0]
# Supply model-ranked unique item IDs and relevance sets to ranking_metrics.
```

`configs/base.yaml` is a standalone configuration. Other YAML files are explicit
presets; they are not automatically merged. Data and sampler keys map to their
function/constructor arguments (cold_fraction belongs to temporal_split).
Model/train presets reserve defaults for later milestones and do not run training.
M3 implements item tokens and a causal baseline. Unified modeling and serving
remain reserved for M4–M6. S1 is implemented in M2.

## Design decisions

- All timestamps and lag values are integer Unix seconds. Synthetic data has
  HOME, SEARCH and RELATED events; synthetic query text uses target titles and is
  suitable only for testing. It is not evidence of real search performance.
- Per-user temporal splitting reserves the last event for test and second-last
  for validation. Ties preserve input order, but no equal-time event can enter a
  target's history. A user with one event has only test; two events have validation
  and test. Empty inputs retain their schema.
- Cold IDs are sampled uniformly without replacement from sorted observed item
  IDs, with floor(fraction * number of IDs). Their training events are discarded;
  validation/test remain unchanged. Small fractions may select zero items, and
  selected items are not guaranteed to occur in held-out targets. Future cold
  metrics must report evaluated item/target counts.
- Default evaluation uses training interactions as the history source for both
  validation and test. This frozen-history protocol excludes validation events
  from test history. An online protocol can explicitly pass a different source,
  but protocols must not be mixed when comparing models. Callers must pass the
  filtered train table for training to preserve the cold-item holdout.
- The sampler returns variable-length pandas event frames (no padding yet).
  Memory and recent windows may overlap. Each is independently truncated to its
  newest eligible events; defaults are 1000 memory events and 50 recent events.
  Recent history uses timestamp < target time; memory uses timestamp < target
  time - lag, including at lag zero. Window lengths of zero yield empty frames.
- Random lag is uniform over configured choices when the stale-probability draw
  succeeds. Fixed evaluation lag overrides randomness. Seed, epoch and target
  index determine each draw, independent of access order or worker scheduling.
  Set epoch before creating data-loader workers; persistent workers need explicit
  epoch synchronization in later training code.
- Statistics use only training observations: age is measured from the first
  training observation to the maximum training timestamp. Catalog first_seen_ts
  is deliberately ignored. Age is divided by max(max age, 1 day); percentile uses
  average tied count ranks among observed items. Unseen items have all zeros.
  These are static training-cutoff features. Later strict point-in-time training
  must recompute them per cutoff if future-within-train popularity is disallowed.
- Metrics macro-average over queries, with binary relevance and unique ranked
  item IDs. Empty relevance contributes zero; empty batches and duplicate ranked
  IDs raise errors. MRR and MRR@k are both available. Recall divides by the full
  relevant set; NDCG's ideal ranking contains min(k, relevant count) items.
  Full-catalog ranking is the intended protocol; the metric function accepts
  ranked lists and cannot verify candidate generation or full-catalog coverage.
- Unified modeling, S2–S5 training and H2–H5 studies remain deferred beyond M3.
  No datasets/models have been downloaded, and no experimental results claimed.


### M2 tokenizer decisions

- `ContentEncoder` lazily loads a frozen local sentence-transformer, defaulting
  to BAAI/bge-small-en-v1.5. It uses `local_files_only=True`; missing weights or
  the optional `text` dependency cause an error, never a fallback download.
  Item text is exactly `title | category | brand | description`, with missing
  fields replaced by empty strings. Queries use `encode_cached` directly.
- `.npy` files contain float32 features; companion `.json` manifests bind model
  identity, normalization, texts and ordered IDs to a SHA-256 fingerprint and
  verify the array checksum. A mismatch/incomplete cache raises an error instead
  of silently reusing stale features. Use immutable model paths/revisions and
  new cache paths when weights change; weights modified in place cannot be
  detected from the supplied model name. Concurrent writers are not supported.
- Co-interaction positives come exclusively from the caller-supplied TRAIN table,
  with strictly increasing timestamps within a configurable seconds window.
  Only the next `max_neighbors` events are considered, self-item pairs are
  excluded, and identical directed pairs are deduplicated. Tied times are never
  positives. Temporal validation/test rows must not be supplied to S1.
- The two-layer adapter defaults to normalized 128-dimensional outputs (256 is
  configurable). Symmetric InfoNCE uses in-batch negatives; duplicate target
  IDs count as additional positives and copies of the anchor ID are masked from
  negatives. Other known behavioral positives can still be false negatives.
  Singleton final training batches are merged with the preceding batch.
- RQ-VAE means a deterministic residual vector-quantized autoencoder here,
  without a Gaussian prior/KL term. It uses a straight-through estimator,
  reconstruction MSE, gradient-trained codebooks, and mean per-level commitment
  and codebook MSE. Residual k-means initialization is the chosen SPEC option;
  EMA is not implemented. Initialization uses only training-observed features.
  Disabling k-means uses seeded normal initialization. Nearest-code assignment
  and quantization losses use float32 even under bf16 autocast.
- Dead codes are those unused for a configurable number of training forwards.
  After optimizer updates they are reset to sampled current TRAIN residuals,
  with the corresponding Adam momentum cleared. This is switchable. Tiny data
  can repeat centers when the codebook exceeds sample count; usage is measured,
  not guaranteed to approach 100 percent. Evaluation does not update counters.
- Adapter fitting, RQ initialization, and RQ fitting exclude cold/unobserved items.
  All catalog items are encoded and assigned SIDs for export. This permits
  inference from cold item content without training on it. Catalog metadata must
  itself be the snapshot appropriate to the intended evaluation cutoff.
- SID suffixes start at zero within each base tuple, ordered by numeric item ID.
  The suffix vocabulary is the largest collision group size, not constrained to
  the RQ codebook size. Assignments are stable under row reordering for the same
  catalog; rebuilding after catalog changes can renumber suffixes. Persist the
  whole SID table for a run. Collision rate is `(N - distinct base tuples) / N`.
- Diagnostics report per-level used-code fraction, reconstruction MSE (training
  items and full catalog separately), collision rate and suffix size. CSV loss
  logs and JSON diagnostics are computed, not experimental recommendation results.
- S1 supports seed control, bf16, gradient clipping, warmup/cosine scheduling,
  activation checkpointing, and exact epoch-boundary resume with unchanged config
  and inputs. Scheduler state and model/optimizer state are saved after every
  epoch. Random batch order/reset sampling derive from seed/epoch, so restarting
  does not depend on worker iteration order. Mid-epoch resume is not supported.
  The S1 checkpoint contains both adapter and RQ weights and their configuration;
  use frozen adapter outputs with RQ-VAE when assigning subsequent item codes.

### M3 token and H1 decisions

- Item tokens sum one embedding per SID level, including the collision suffix by
  default (`include_collision_suffix: false` removes that level). Each level has
  its own table sized from the supplied SID catalog. The deterministic integer
  hash is `((item_id mod 2147483647) * 48271 mod 2147483647) mod hash_buckets`.
  Hash collisions share parameters. Python's process-dependent hash is not used.
- Warm-item modes are: `sid_only = SID`, `id_only = ID`, `fixed_sum = SID + ID`,
  and gated modes use `SID + sigmoid(MLP(stats)) * ID`. Gate statistics are
  log1p(train count), normalized train age and train popularity percentile.
  All modes fall back to SID for count-zero items; ID lookup can be absent
  entirely for an all-cold batch. Thus the `id_only` control is ID-only on warm
  items with an explicit SID cold fallback, not a pure-ID cold baseline.
- Only `gated` applies ID dropout; `gated_no_id_dropout`, `id_only` and `fixed_sum`
  are controls without it. Default dropout is
  `p_min + (p_max - p_min) / (1 + count / scale)`; an exponential alternative
  replaces the reciprocal with `exp(-count / scale)`. Whole ID vectors are
  independently masked per occurrence, without inverted-dropout rescaling.
  Evaluation never drops IDs. Global Torch seeds control these draws.
- This is a SASRec-style control, not a reproduction of a published SASRec
  benchmark recipe. It uses two pre-norm causal transformer blocks by default,
  learned positions, SDPA attention, a shared item-token scoring tower and a
  learned start token. It predicts each target from up to 50 strictly earlier
  TRAIN events, including a start-only context for empty histories. Right-padding
  is masked as an attention key. This does not implement the M5 unified backbone.
- Training uses exact softmax over TRAIN-observed catalog items. Cold items are
  excluded even as negatives so their dedicated codes are not fitted using
  withheld items. SID levels can share weights with warm items; codes never seen
  in training retain their initialization. Frozen S1 assignments are shared by
  all H1 modes within each seed; each seed independently fits S1 on the same split.
- Validation NDCG@10 selects an epoch for each run; ties select the earlier epoch.
  Test is evaluated only after selection. All modes use equal training budgets,
  architecture, hash sizes, SIDs and split. Hyperparameters are configurable but
  no systematic tuning or real-data baseline comparison has been performed.
  Do not interpret synthetic results as evidence for H1 or claim state-of-the-art.
- Full-catalog evaluation retains consumed items (repeated next items are valid),
  includes cold items, and breaks score ties by ascending item ID. Histories for
  validation/test remain frozen TRAIN-only; validation events do not enter test
  context. The supplied catalog is a static snapshot, not time-filtered per query;
  use snapshot-appropriate metadata to avoid future-item availability leakage.
- Popularity slices use average tied percentile ranks among train-observed items:
  head > 0.8, torso > 0.5 and <= 0.8, tail <= 0.5. Cold means train count zero;
  `heldout_cold` is the explicitly held-out subset. Cold users have fewer than five
  train events. Task/domain slices describe labels only: this rec baseline does
  not consume query text, source context, task tokens or domain adapters.
  Every slice reports target/item counts; empty slices have null metrics. Summary
  means and population std (ddof=0) use nonempty seeds and state their count.
- Per-user splitting is not a single global time cutoff. Training statistics and
  S1 codes are fitted at the train-partition level, not re-fit before each target.
  Strict event-time histories do not eliminate those broader protocol limits.
- MovieLens ratings >= 4 become implicit positives, using all movies.dat items as
  the catalog. Amazon uses only items present in positive reviews from the chosen
  category. Metadata gaps become empty fields; duplicate reviews remain separate
  events. The 2018 and 2023 JSONL schemas are explicit options, with milliseconds
  converted to seconds only for 2023. No synthetic search queries are generated.
- H1 checkpoints preserve optimizer/scheduler, Torch RNG and validation-best
  weights for exact epoch-boundary resume. Input/config fingerprints reject a
  changed resume. Use a fresh output directory for changed configurations. The
  implementation is intended for ML-1M/a small category, with full-catalog logits
  per batch; larger catalogs require a separate scalable retrieval protocol.

## Run M3 / H1

Run all five modes and three seeds on tiny synthetic data, without downloads:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m tumirec.scripts.run_ablation --dataset synthetic --output results/h1-synthetic
```

In this workspace prepend the `LD_LIBRARY_PATH` setting below. Synthetic features
are seeded random test features, not text embeddings. Outputs include per-seed S1
artifacts, per-mode checkpoints/training.csv/results.json, data_manifest.json,
items.parquet and aggregate summary.json/summary.csv. Add `--resume` to continue;
use `--seeds 42` or `--modes gated sid_only` for explicitly smaller runs.

After manually obtaining local data and encoder weights:

```sh
python -m tumirec.scripts.run_ablation --dataset movielens --data-dir artifacts/ml-1m --encoder-model artifacts/bge-small-en-v1.5 --config tumirec/configs/ablation/h1.yaml --device cuda --output results/h1-ml1m
python -m tumirec.scripts.run_ablation --dataset amazon --reviews artifacts/category.jsonl.gz --metadata artifacts/meta-category.jsonl.gz --amazon-schema 2018 --encoder-model artifacts/bge-small-en-v1.5 --config tumirec/configs/ablation/h1.yaml --device cuda --output results/h1-amazon
```

The runner accepts only H1 in M3. YAML config sections are `baseline` and
`tokenizer`, consumed through OmegaConf; data/split/seeds/modes are CLI options.
Config values override corresponding defaults, including a device if specified.
Real text encoding requires sentence-transformers and local weights. See
`tumirec/scripts/download_data.md` and `download_encoder.md` for local schemas and
manual preparation. Full real datasets, real encoder inference and CUDA execution
have not been tested in this workspace; local schema fixtures and CPU synthetic
pipelines are covered by tests.

## Run M2 locally

A CPU plumbing run uses seeded random features, explicitly **not semantic text
embeddings**, and requires no model download:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m tumirec.train.stage1_tokenizer --synthetic --output artifacts/s1
```

In the supplied workspace prepend the `LD_LIBRARY_PATH` setting documented below.
Outputs are `checkpoint.pt`, `training.csv`, `sids.parquet`, and `diagnostics.json`.
Repeat the same command with `--resume` to resume an interrupted run. Existing
checkpoints require `--resume` or a new directory; inputs/config must match.

For real local inputs, use TRAIN-only interactions and an immutable local encoder:

```sh
python -m tumirec.train.stage1_tokenizer --items artifacts/items.parquet --train artifacts/train.parquet --model artifacts/bge-small-en-v1.5 --config tumirec/configs/train/stage1.yaml --output artifacts/s1-real
```

The YAML config uses OmegaConf (already declared as a dependency). Real text
encoding additionally requires the `text` optional dependency. Neither was
installed during M2. See `tumirec/scripts/download_encoder.md` for a manual,
opt-in weight preparation command; no network action occurs in S1.

## Environment troubleshooting

In the supplied workspace, plain `pytest -q` failed during pandas import because
system libstdc++ lacks GLIBCXX_3.4.29. Use the already-installed Conda runtime:

```sh
LD_LIBRARY_PATH=/workspace/quynm/anaconda3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} pytest -q
```

This is a command-local library selection, not a system modification. On a
normally configured Python environment, use the plain pytest command above.
