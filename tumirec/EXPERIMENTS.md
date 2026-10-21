# Experiments

M3 makes H1 runnable on local MovieLens-1M or an Amazon category. Only synthetic
plumbing runs and local schema fixtures have been exercised; no real-data
recommendation benchmark results or superiority claims are available.

Use `python -m tumirec.scripts.run_ablation --dataset synthetic --output results/h1`
for all five token modes with default seeds 42, 43 and 44. See the root README for
real local-data commands and protocol decisions. A run produces per-seed/per-mode
results.json and aggregate summary.json/summary.csv with mean and population std.
Validation NDCG@10 selects checkpoints; tests are full-catalog, retain consumed
items, and use frozen train-only histories. Popularity/cold/task/domain slices
include counts; empty slice metrics are null and excluded from aggregation.

The common defaults have not been tuned independently per control. Before making
an H1 claim, use real frozen content embeddings and validation-only tuning with
comparable budgets for all modes. Record catalog snapshots, rating threshold,
metadata gaps, SID settings, hash collisions, seeds, held-out item counts, and
whether the ID-only control's SID cold fallback fits the intended comparison.
M4 adds H2 through `python -m tumirec.scripts.run_h2 --dataset synthetic --output
results/h2`. Four memory variants share SIDs and recent-context architecture within
each seed. Fresh validation selects one checkpoint, then the same weights are
measured at 0, 1h and 1d memory lag. Current recent history remains available at
all lags. Synthetic smoke results validate the pipeline, not the hypothesis.

Optional `--profile-repeats N` reports actual cached/uncached forward p50/p99 for
the M4 baseline; it excludes cache refresh/lookup, loading and transfer. Cache
quality experiments must additionally disclose TTL, refresh policy and version
provenance. Small smoke samples do not support deployment latency claims.
H3–H5, unified-model evaluation, general latency and S1→S3 tests remain deferred.
