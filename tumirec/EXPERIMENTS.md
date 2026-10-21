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
H2–H5, unified-model evaluation, latency and S1→S3 smoke tests remain deferred.
