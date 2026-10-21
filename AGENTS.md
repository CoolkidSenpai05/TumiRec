# AGENTS.md — Standing rules for the TUMI-Rec repository

Read this file, `SPEC.md` and `PROGRESS.md` fully before doing anything.
This is a **research prototype for ablation studies**, not a production system.
Correctness, reproducibility and clean ablation switches matter more than speed or polish.

## 1. Hard constraints
- Hardware: a single NVIDIA RTX 3090 (24GB). Use bf16 autocast. No FP8.
- Use `torch.nn.functional.scaled_dot_product_attention` for attention.
- **No LLM-generated user summaries anywhere.** Frozen small text encoders
  (e.g. bge-small / sentence-transformers) ARE allowed, only to embed item/query text once and cache it to disk.
- Python 3.10+, PyTorch 2.x.
- Dependencies: torch, numpy, pandas, pyarrow, sentence-transformers, faiss-cpu (or faiss-gpu if available),
  hydra-core or OmegaConf, pytest, tqdm. Logging: CSV/TensorBoard by default (wandb optional).
- Every experimental component must be switchable through config flags (for ablations).
- Seeds must be fully controlled and reproducible.
- **Do not fabricate results.** If you cannot run something, say so. Never hardcode metrics.
  Never print metrics that were not actually computed.
- Do not download anything unless the user asks. Write download scripts and document them.
  All code and tests must work on a tiny synthetic dataset.

## 2. Testing rules
- All tests run on CPU in under 2 minutes using synthetic data (`pytest`).
- Required tests are listed in `SPEC.md` section 8. A milestone is not done until its tests pass.
- Always run the tests before reporting a milestone as complete.

## 3. Working rules
- Implement **ONE milestone at a time** (see `SPEC.md` section 9). Stop after each milestone and wait for the user's confirmation.
- Do not modify code from earlier milestones unless a bug requires it; list any such change in `PROGRESS.md`.
- Make small, reviewable commits with clear messages.
- Keep functions typed and documented (docstrings with tensor shapes).
- Prefer simple, readable code over clever optimizations. Add profiling hooks instead of premature optimization.
- If a spec is ambiguous, choose the simplest reasonable option, document it in `README.md`
  under "Design decisions", and continue.
- Flag any place where results could be affected by:
  - data leakage (memory/history must only use events strictly before the target event),
  - unfair baselines (e.g. un-tuned or un-finetuned baselines),
  - inconsistent evaluation protocols.

## 4. End-of-milestone checklist
1. Run the tests and report the outcome honestly.
2. Update `PROGRESS.md`: what was done, assumptions/deviations, open questions.
3. Stop and wait for confirmation.
