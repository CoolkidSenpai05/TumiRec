# Package layout

See ../README.md for usage and design decisions. M1 supplies data and metrics;
M2 supplies tokenizer/S1; M3 supplies hybrid tokens and the H1 causal control.
M4 adds InterestMemory, lagged memory data, nearline cache simulation and a separate
scripts/run_h2.py memory experiment. The M3 runner remains unchanged for H1.
Unified modeling, S3 training and later stages remain reserved for M5/M6.
