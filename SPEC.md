# SPEC.md — TUMI-Rec (no-LLM variant)

## 0. Context and goal
Build a PyTorch research codebase for **TUMI-Rec** (Tiered Unified Model with Interest memory).
It unifies search and recommendation in one small transformer backbone, with:
1. a **Gated Hybrid Item Token** (Semantic ID + hashed ID),
2. an asynchronous **Interest Memory** (K latent tokens compressed from long user history, NO LLM text generation),
3. a task-conditioned **unified backbone** with candidate-in-context scoring,
4. a **retrieval head** + **multi-task ranking head** (+ optional SID decoder head),
5. **multi-stage training**: contrastive ID pretraining, domain LoRA, auxiliary distillation.

Standing rules and constraints are in `AGENTS.md`. Milestone progress is tracked in `PROGRESS.md`.

### Architecture overview
```
Offline (once)                    Nearline (per session)         Online (backbone)
Content encoder (frozen)          Memory Encoder                 [TASK][DOM][CTX][MEM x K]
 -> Adapter (contrastive)         K latent queries cross-attend  [RECENT <=50 items]
 -> RQ-VAE -> SID                 long history (500-1000)        [CAND_1..CAND_n] (own mask)
Item token = SID + gated ID       -> KV cache per user           Transformer, 8L, d=384
                                                                  |- A: retrieval embedding
                                                                  |- B: multi-task ranking
                                                                  |- C: SID decoder (optional)
                                                                 -> Policy layer (rule-based)
```

## 1. Repository layout
```
tumirec/
  configs/ (base.yaml, data/*.yaml, model/*.yaml, train/*.yaml, ablation/*.yaml)
  data/ (loaders.py, preprocess.py, splits.py, synthetic.py, sampler.py)
  tokenizer/ (content_encoder.py, adapter.py, rqvae.py, sid_assign.py)
  model/ (item_token.py, memory.py, backbone.py, heads.py, lora.py, tumirec.py)
  train/ (stage1_tokenizer.py, stage2_id_pretrain.py, stage3_unified.py,
          stage4_ranking_lora.py, stage5_distill.py)
  eval/ (metrics.py, evaluate.py, slicing.py, latency.py)
  serving_sim/ (memory_cache.py, policy.py)
  scripts/ (download_*.md or .sh, run_ablation.py)
  tests/
  README.md, EXPERIMENTS.md
```

## 2. Data layer
**Unified interaction schema (parquet):**
`user_id, item_id, timestamp, action (int), domain (int), task (int: 0=HOME, 1=SEARCH, 2=RELATED), query_text (nullable), source_item_id (nullable)`

**Item table:** `item_id, title, category, description, brand (nullable), price (nullable), first_seen_ts`

Implement:
- `synthetic.py`: tiny generator (e.g. 500 users, 300 items, with search and rec events) used by all tests.
- Loaders for MovieLens (rec only) and Amazon Reviews (rec only; synthetic search queries optional, derived from item titles).
  KuaiSAR loader as a **stub** with a clear TODO and the documented expected format (verify the real format when data is available).
- Temporal split per user: leave-last-out for test, second-last for validation. **No leakage**: memory/history for a target event
  may only use events strictly before it.
- Item-level cold-start split: hold out a configurable percentage of items from training interactions to evaluate cold items.
- Sliding-window sampler for long histories (history up to 1000, recent window 50), with support for sampling a random
  "memory lag" (see 4.3).
- Per-item count/age statistics computed from TRAIN data only.

## 3. (reserved)

## 4. Modules

### 4.1 Tokenizer (`tokenizer/`)
- `content_encoder.py`: embed `"title | category | brand | description"` with a frozen sentence-transformers model
  (default: bge-small). Cache to `.npy`.
- `adapter.py`: 2-layer MLP adapter trained with InfoNCE on item pairs co-interacted by the same user within a window
  (in-batch negatives). Output dim 128 or 256.
- `rqvae.py`: Residual-Quantization VAE. Config: `n_levels` (default 3), `codebook_size` (default 256), latent dim,
  commitment loss, codebook EMA or k-means init, dead-code reset. Report codebook usage and reconstruction loss.
- `sid_assign.py`: produce a SID tuple per item plus an extra collision-resolution code.
  Output table `item_id -> codes`. Report the collision rate.

### 4.2 Gated Hybrid Item Token (`model/item_token.py`)
```
e_sid = sum_l E_level[l](code[l])
e_id  = Embedding(hash_buckets, d)(hash(item_id))
g     = sigmoid(MLP([log1p(count), age_days_normalized, popularity_percentile]))
e_item = e_sid + g * e_id
```
- **Training-time ID dropout:** drop `e_id` with probability `p = f(count)`, higher for low-count items. Make `f` configurable.
- Modes via config: `sid_only`, `id_only`, `fixed_sum`, `gated` (default), `gated_no_id_dropout`.
- Must handle unseen/cold items using only `e_sid` (`e_id = 0` or a shared OOV bucket; document the choice).

### 4.3 Interest Memory (`model/memory.py`)
- **Input:** history of up to L = 500..1000 events. Each token = `e_item + emb(action) + emb(log time-delta bucket)`.
- **Architecture:** K learnable latent queries (default K = 8), 2 cross-attention layers (latents attend to history),
  then 1 self-attention layer among latents. Output: K tokens of dim `d_model`.
- **Option:** session segmentation (split by time gap, pool per session before cross-attention) behind a flag.
- **Staleness simulation:** during training, with probability `p_stale` sample `lag` from a configured distribution and compute memory
  from history truncated at `(t - lag)`. Evaluation must be able to fix lag to specific values (0, 1h, 1d, ...) for H2.
- Also implement `no_memory` and `recent_only` modes for ablations.
- `serving_sim/memory_cache.py`: simple in-process KV cache simulating nearline updates. On cache miss, return empty/previous
  memory and mark for refresh. Used for latency/staleness experiments.

### 4.4 Unified Backbone (`model/backbone.py`)
**Sequence layout:** `[TASK][DOM][CTX...][MEM x K][RECENT x <=50][CAND x n]`
- **CTX:** for SEARCH, a frozen query embedding (cached, same encoder as items) projected to 1-2 tokens.
  For RELATED, the source item token. For HOME or missing context, a learned NULL token.
- **Attention mask:** CAND tokens attend to all non-CAND tokens and to themselves only, **not to other CAND tokens**.
  Non-CAND tokens do not attend to CAND. Implement as an explicit boolean mask and unit-test it.
- **Config defaults:** `n_layers=8`, `d_model=384`, `n_heads=6`, FFN 4x, pre-norm, RoPE or learned positions (configurable).
  Provide presets: small (~20M), base (~60M), large (~200M).
- **LoRA (`model/lora.py`):** per-domain adapters on attention and FFN linear layers, configurable rank (8/16/32) and an
  optional per-layer rank. Domain is selected by the DOM token. Base weights are frozen in stage 4.

### 4.5 Heads (`model/heads.py`)
- **A) Retrieval head:** pool the context-side output into user/query vector `u` (dim 128); item tower = MLP over `e_item` -> `v` (dim 128).
  Loss: sampled softmax with in-batch negatives + uniform negatives + optional hard negatives (mined from the current model via ANN
  every N steps), with logQ correction. Optional "quality boost" trick (append a query-independent quality score).
- **B) Ranking head:** from CAND positions, a shared trunk + per-objective MLP heads (e.g. click, positive-action,
  long-engagement if available). BCE per task with configurable loss weights; serving-time weights adjustable.
- **C) SID decoder head (optional, off by default):** small autoregressive decoder predicting SID codes from context.
  Used as an extra retriever for cold items.

### 4.6 Policy layer (`serving_sim/policy.py`)
Rule-based: demote already-consumed items, cap consecutive same brand/category (MMR-style optional),
epsilon-greedy for new items. Off by default.

## 5. Training stages
Each stage = its own script + config + checkpoint.
- **S1 Tokenizer:** adapter -> RQ-VAE -> SID table (+ diagnostics).
- **S2 ID pretraining:** minimal dot-product model with contrastive loss, multiple epochs, uniform + in-batch negatives.
  Export ID embeddings used to initialize the hash table (document how hash collisions are handled).
- **S3 Unified pretraining:** retrieval loss + causal next-item loss; mixed tasks with temperature sampling;
  sliding-window sampler; memory encoder co-trained.
- **S4 Ranking fine-tune:** add ranking head; freeze backbone; train per-domain LoRA.
- **S5 Distillation:** teacher = same architecture at ~2x size (or self-auxiliary distillation via a flag).
  Implement **auxiliary distillation**: separate logits for hard labels and teacher soft labels, so the student inherits
  teacher knowledge without its bias. Cache teacher labels to disk (parquet).

All stages: bf16 autocast, gradient clipping, cosine LR with warmup, checkpoint/resume, deterministic seeding,
gradient-checkpointing flag.

## 6. Evaluation (`eval/`)
- **Metrics:** Recall@k, NDCG@k, MRR for k in {10, 20, 50}. Prefer full ranking over the catalog; if sampled, document the protocol.
- **Slicing:** head / torso / tail items (by train popularity), cold items (held-out), cold users (few interactions),
  per task (HOME/SEARCH/RELATED), per domain.
- **Latency microbenchmark:** p50/p99 of the backbone forward with n candidates (e.g. 100, 500), with and without cached memory.
- Report mean +/- std over seeds. Save everything to `results/*.json` and a summary CSV.

## 7. Ablations (`configs/ablation/`)
| ID | Hypothesis | Variants |
|---|---|---|
| H1 | Gated SID/ID token beats alternatives, esp. cold/tail | `sid_only`, `id_only`, `fixed_sum`, `gated`, `gated_no_id_dropout`; report by popularity slice and cold items |
| H2 | Interest Memory helps with small latency cost | `none`, `recent_only`, `behavior_memory`, `behavior_memory+content_pool`; lag in {0, 1h, 1d} |
| H3 | Unified search+rec beats separate models | separate per task; unified + TASK token; unified + domain LoRA |
| H4 | Distillation helps a small student | none; direct; auxiliary |
| H5 | Contrastive ID pretraining reduces one-epoch overfitting | BCE multi-epoch; contrastive pretrain + finetune |

Provide `scripts/run_ablation.py` to run a named ablation across seeds (default 3) and aggregate the results.

## 8. Required tests (pytest, CPU, < 2 min, synthetic data)
- **No-leakage:** memory/history for a target event contains only strictly earlier events.
- **Candidate mask:** CAND tokens cannot attend to each other; scores are invariant to the order/number of other candidates.
- **RQ-VAE:** shapes, codebook usage > 0, reconstruction loss decreases on a toy batch.
- **Gated token:** a cold item (count = 0) works with `e_id` absent; ID-dropout probability follows `f(count)`.
- **Memory:** output shape (B, K, d); lag truncation works; `no_memory` mode returns the correct placeholder.
- **Metrics:** hand-computed Recall/NDCG on a toy example.
- **End-to-end smoke test:** S1 -> S3 -> eval on synthetic data runs and produces a results JSON.

## 9. Milestones (implement ONE at a time; stop after each and report)
- **M1:** repo skeleton, configs, synthetic data, data splits, sliding-window sampler, metrics, tests for leakage/metrics.
- **M2:** tokenizer (content encoder cache, adapter, RQ-VAE, SID assignment) + diagnostics + tests.
- **M3:** Gated Hybrid Item Token with all modes + ID dropout + tests; a simple SASRec-style baseline on item tokens
  to run the H1 ablation end-to-end on MovieLens-1M / a small Amazon category.
- **M4:** Interest Memory + staleness simulation + H2 ablation scripts.
- **M5:** Unified backbone with candidate-in-context mask, retrieval + ranking heads, TASK/DOM tokens, S3 training.
- **M6:** domain LoRA (S4), contrastive ID pretraining (S2), distillation (S5), H3-H5 ablations, latency benchmark, README/EXPERIMENTS.

After each milestone: run tests, summarize what was done, list assumptions/deviations, list open questions, and wait for confirmation.
