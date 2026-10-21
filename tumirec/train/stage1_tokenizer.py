"""Offline S1: contrastive adapter -> RQ-VAE -> SID table and diagnostics.

Run `python -m tumirec.train.stage1_tokenizer --synthetic --output artifacts/s1`
for a small CPU plumbing run using explicitly synthetic random content features.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from tumirec.tokenizer.adapter import ContrastiveAdapter, cointeraction_pairs, info_nce
from tumirec.tokenizer.content_encoder import ContentEncoder
from tumirec.tokenizer.rqvae import RQVAE
from tumirec.tokenizer.sid_assign import tokenize_catalog


@dataclass(frozen=True)
class Stage1Config:
    """S1 ablation and optimization settings; all time windows use seconds."""
    seed: int = 42
    device: str = "cpu"
    adapter_enabled: bool = True
    adapter_dim: int = 128
    hidden_dim: int = 256
    normalize_adapter: bool = True
    temperature: float = 0.07
    pair_window_seconds: int = 86400
    pair_max_neighbors: int = 20
    adapter_epochs: int = 5
    rq_epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 0.001
    warmup_steps: int = 10
    gradient_clip: float = 1.0
    bf16: bool = True
    gradient_checkpointing: bool = False
    latent_dim: int = 128
    n_levels: int = 3
    codebook_size: int = 256
    commitment_weight: float = 0.25
    kmeans_init: bool = True
    kmeans_iters: int = 10
    dead_code_reset: bool = True
    dead_code_patience: int = 100

    def __post_init__(self) -> None:
        """Reject invalid training controls before creating any artifacts."""
        if (self.seed < 0 or min(self.adapter_epochs, self.rq_epochs, self.batch_size) < 1
                or self.learning_rate <= 0 or self.gradient_clip <= 0
                or self.warmup_steps < 0 or self.temperature <= 0
                or (self.adapter_enabled and self.batch_size < 2)):
            raise ValueError("Invalid S1 optimization settings")


def _seed(seed: int) -> None:
    """Seed Python, NumPy and Torch, requiring deterministic Torch kernels."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def _schedule(optimizer: torch.optim.Optimizer, steps: int,
              warmup: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Build per-update linear warmup followed by cosine learning-rate decay."""
    warmup = min(warmup, max(steps - 1, 0))

    def scale(step: int) -> float:
        """Return the multiplier for zero-indexed optimizer step."""
        if step < warmup:
            return (step + 1) / warmup
        progress = min((step - warmup) / max(steps - warmup, 1), 1.0)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _fingerprint(features: np.ndarray, items: pd.DataFrame, train: pd.DataFrame) -> str:
    """Bind resume to feature bytes, item row order, and TRAIN event rows."""
    digest = hashlib.sha256(features.tobytes())
    digest.update(str(features.shape).encode())
    digest.update(items.item_id.to_json().encode())
    digest.update(train.to_json(orient="split").encode())
    return digest.hexdigest()


def train_tokenizer(features: np.ndarray, items: pd.DataFrame, train: pd.DataFrame,
                    output_dir: str | Path, config: Stage1Config = Stage1Config(),
                    resume: bool = False, stop_after_epochs: int | None = None) -> dict:
    """Fit S1 using features (N,D), catalog IDs (N), and TRAIN interactions only.

    Adapter positives and RQ fitting use only observed TRAIN items. All catalog
    features are transformed for export, including cold items. Epoch-boundary
    checkpoint/resume requires identical config and inputs. stop_after_epochs is
    a testing/interruption hook counting epochs completed in this invocation.
    """
    features = np.asarray(features, dtype=np.float32)
    if (features.ndim != 2 or features.shape[1] == 0 or len(features) != len(items)
            or len(items) == 0 or not np.isfinite(features).all()
            or items.item_id.duplicated().any() or train.empty):
        raise ValueError("Need finite features (N,D), unique catalog IDs and nonempty train data")
    if stop_after_epochs is not None and stop_after_epochs < 1:
        raise ValueError("stop_after_epochs must be positive")
    mapping = {int(item): row for row, item in enumerate(items.item_id)}
    if not set(train.item_id).issubset(mapping):
        raise ValueError("Training item IDs missing from catalog")
    _seed(config.seed)
    device = torch.device(config.device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    if checkpoint_path.exists() and not resume:
        raise FileExistsError("Existing S1 checkpoint; use resume or a new output directory")
    x = torch.from_numpy(features).to(device)
    seen = torch.tensor([mapping[i] for i in sorted(train.item_id.unique())], device=device)
    pairs = cointeraction_pairs(train, config.pair_window_seconds, config.pair_max_neighbors)
    if config.adapter_enabled and len(pairs) < 2:
        raise ValueError("Contrastive training requires at least two distinct positive pairs")
    pair_rows = torch.tensor([[mapping[int(a)], mapping[int(b)]] for a, b in pairs],
                             dtype=torch.long, device=device).reshape(-1, 2)
    adapter: nn.Module = (ContrastiveAdapter(x.shape[1], config.adapter_dim, config.hidden_dim,
                                           config.normalize_adapter).to(device)
                          if config.adapter_enabled else nn.Identity())
    rq = RQVAE(config.adapter_dim if config.adapter_enabled else x.shape[1],
               config.latent_dim, config.hidden_dim, config.n_levels, config.codebook_size,
               config.commitment_weight, config.kmeans_init, config.kmeans_iters,
               config.dead_code_reset, config.dead_code_patience,
               config.gradient_checkpointing).to(device)
    identity = _fingerprint(features, items, train)
    state = None
    logs: list[dict] = []
    if resume:
        # Checkpoints contain only tensors and primitive containers; no arbitrary pickle loading.
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if state["config"] != asdict(config) or state["fingerprint"] != identity:
            raise ValueError("Resume config or input identity mismatch")
        adapter.load_state_dict(state["adapter"])
        rq.load_state_dict(state["rq"])
        logs = state["logs"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda_rng"]])
    completed = 0
    transformed = None
    for phase, epochs in (("adapter", config.adapter_epochs if config.adapter_enabled else 0),
                          ("rq", config.rq_epochs)):
        if epochs == 0:
            continue
        if phase == "rq":
            adapter.eval()
            with torch.no_grad():
                transformed = torch.cat([adapter(batch).float() for batch in x.split(config.batch_size)])
            if not bool(rq.initialized):
                rq.initialize(transformed[seen], seed=config.seed)
        if state is not None and state["phase"] == "rq" and phase == "adapter":
            continue
        model = adapter if phase == "adapter" else rq
        model.train()
        indices = pair_rows if phase == "adapter" else seen
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        batches = math.ceil(len(indices) / config.batch_size)
        merge_singleton = phase == "adapter" and len(indices) % config.batch_size == 1 and batches > 1
        if merge_singleton:
            batches -= 1
        scheduler = _schedule(optimizer, epochs * batches, config.warmup_steps)
        start_epoch = 0
        if state is not None and state["phase"] == phase:
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            start_epoch = state["epoch"] + 1
        for epoch in range(start_epoch, epochs):
            generator = torch.Generator().manual_seed(config.seed + epoch + (100000 if phase == "rq" else 0))
            order = torch.randperm(len(indices), generator=generator).to(device)
            loss_sum, reconstruction_sum, n = 0.0, 0.0, 0
            batch_groups = list(order.split(config.batch_size))
            if merge_singleton:
                batch_groups[-2] = torch.cat((batch_groups[-2], batch_groups[-1]))
                batch_groups.pop()
            for batch_indices in batch_groups:
                batch = indices[batch_indices]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=config.bf16):
                    if phase == "adapter":
                        def apply_adapter(values: Tensor) -> Tensor:
                            """Adapt a (B,D) batch with optional activation checkpointing."""
                            return (checkpoint(adapter, values, use_reentrant=False)
                                    if config.gradient_checkpointing else adapter(values))
                        loss = info_nce(apply_adapter(x[batch[:, 0]]), apply_adapter(x[batch[:, 1]]),
                                        config.temperature, batch)
                    else:
                        result = rq(transformed[batch])
                        loss = result.loss
                        reconstruction_sum += float(result.reconstruction_loss.detach()) * len(batch)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite S1 loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                if phase == "rq":
                    changes = rq.reset_dead_codes(transformed[batch], seed=config.seed + epoch * batches + n)
                    for level, codes in changes:
                        for name in ("exp_avg", "exp_avg_sq"):
                            optimizer.state[rq.codebooks][name][level, codes] = 0
                scheduler.step()
                loss_sum += float(loss.detach()) * len(batch)
                n += len(batch)
            logs.append({"phase": phase, "epoch": epoch + 1, "loss": loss_sum / n,
                         "reconstruction_loss": reconstruction_sum / n if phase == "rq" else None})
            saved = {"config": asdict(config), "fingerprint": identity,
                     "phase": phase, "epoch": epoch, "adapter": adapter.state_dict(),
                     "rq": rq.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                     "logs": logs}
            temporary = checkpoint_path.with_suffix(".tmp")
            torch.save(saved, temporary)
            temporary.replace(checkpoint_path)
            with (output_dir / "training.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(logs[0]))
                writer.writeheader()
                writer.writerows(logs)
            completed += 1
            if stop_after_epochs is not None and completed >= stop_after_epochs:
                return {"status": "paused", "phase": phase, "epoch": epoch + 1}
    adapter.eval()
    rq.eval()
    with torch.no_grad():
        transformed = torch.cat([adapter(batch).float() for batch in x.split(config.batch_size)])
    table, diagnostics = tokenize_catalog(rq, transformed, items.item_id.tolist(), config.batch_size)
    _, train_diagnostics = tokenize_catalog(rq, transformed[seen],
                                            [int(items.item_id.iloc[i]) for i in seen.cpu().tolist()],
                                            config.batch_size)
    diagnostics["train_reconstruction_loss"] = train_diagnostics["reconstruction_loss"]
    diagnostics["train_codebook_usage"] = train_diagnostics["codebook_usage"]
    diagnostics["n_train_items"] = len(seen)
    diagnostics["n_positive_pairs"] = len(pairs)
    diagnostics["config"] = asdict(config)
    table.to_parquet(output_dir / "sids.parquet", index=False)
    (output_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False))
    return diagnostics


def main() -> None:
    """Run synthetic S1 or local parquet inputs with a locally cached text model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--items", type=Path)
    parser.add_argument("--train", type=Path, help="TRAIN-only interactions parquet")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.synthetic:
        from tumirec.data.synthetic import generate_synthetic
        from tumirec.data.splits import temporal_split
        events, items = generate_synthetic(8, 24, 12)
        train = temporal_split(events, cold_fraction=0.1).train
        features = np.random.default_rng(42).standard_normal((len(items), 16)).astype(np.float32)
        config = Stage1Config(adapter_epochs=2, rq_epochs=5, hidden_dim=32, latent_dim=8,
                              codebook_size=8, batch_size=64, kmeans_iters=3)
    else:
        if args.items is None or args.train is None:
            parser.error("Supply --synthetic or both --items and --train")
        items, train = pd.read_parquet(args.items), pd.read_parquet(args.train)
        features = ContentEncoder(args.model).encode_items(items, args.output / "content.npy")
        config = Stage1Config()
    if args.config:
        from omegaconf import OmegaConf
        config = Stage1Config(**OmegaConf.to_container(OmegaConf.load(args.config), resolve=True))
    diagnostics = train_tokenizer(features, items, train, args.output, config, args.resume)
    print(json.dumps(diagnostics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
