"""Residual vector-quantized autoencoder with trainable codebooks."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class RQOutput:
    """Forward result: reconstruction (B,D), codes (B,L), scalar losses, usage (L,)."""
    reconstruction: Tensor
    codes: Tensor
    loss: Tensor
    reconstruction_loss: Tensor
    commitment_loss: Tensor
    codebook_loss: Tensor
    usage: Tensor


class RQVAE(nn.Module):
    """RQ-VAE with straight-through gradients and optional residual k-means init.

    This is a deterministic VQ autoencoder (no Gaussian latent/KL term). Codebooks
    use gradient updates, not EMA. All nearest-code computations stay float32.
    """

    def __init__(self, input_dim: int, latent_dim: int = 128, hidden_dim: int = 256,
                 n_levels: int = 3, codebook_size: int = 256,
                 commitment_weight: float = 0.25, kmeans_init: bool = True,
                 kmeans_iters: int = 10, dead_code_reset: bool = True,
                 dead_code_patience: int = 100, gradient_checkpointing: bool = False) -> None:
        super().__init__()
        if min(input_dim, latent_dim, hidden_dim, n_levels, codebook_size,
               kmeans_iters, dead_code_patience) < 1 or commitment_weight < 0:
            raise ValueError("Invalid RQ-VAE dimensions or loss/reset parameters")
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(),
                                     nn.Linear(hidden_dim, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.GELU(),
                                     nn.Linear(hidden_dim, input_dim))
        self.codebooks = nn.Parameter(torch.randn(n_levels, codebook_size, latent_dim) * 0.05)
        self.n_levels, self.codebook_size = n_levels, codebook_size
        self.commitment_weight, self.kmeans_init = commitment_weight, kmeans_init
        self.kmeans_iters = kmeans_iters
        self.dead_code_reset, self.dead_code_patience = dead_code_reset, dead_code_patience
        self.gradient_checkpointing = gradient_checkpointing
        self.register_buffer("initialized", torch.tensor(not kmeans_init))
        self.register_buffer("unused_steps", torch.zeros(n_levels, codebook_size, dtype=torch.long))

    @staticmethod
    def _nearest(residual: Tensor, book: Tensor) -> Tensor:
        """Return nearest indices (B,) for residuals (B,D) and codebook (C,D)."""
        distance = (residual.square().sum(-1, keepdim=True) + book.square().sum(-1)[None]
                    - 2 * residual @ book.T)
        return distance.argmin(dim=-1)

    @torch.no_grad()
    def initialize(self, x: Tensor, seed: int = 42) -> None:
        """Initialize each residual codebook with TRAIN features (N,input_dim).

        N < C uses repeated samples. Empty clusters retain their previous center.
        Call before optimizer construction; initialization is never implicit in eval.
        """
        if x.ndim != 2 or len(x) == 0:
            raise ValueError("Initialization needs nonempty (N,D) training features")
        generator = torch.Generator(device=x.device).manual_seed(seed)
        with torch.autocast(device_type=x.device.type, enabled=False):
            residual = self.encoder(x.float())
            for level in range(self.n_levels):
                indices = torch.randperm(len(x), generator=generator, device=x.device)
                centers = residual[indices[torch.arange(self.codebook_size, device=x.device) % len(x)]].clone()
                for _ in range(self.kmeans_iters):
                    assignments = self._nearest(residual, centers)
                    for code in range(self.codebook_size):
                        members = residual[assignments == code]
                        if len(members):
                            centers[code] = members.mean(dim=0)
                self.codebooks[level].copy_(centers)
                residual = residual - centers[self._nearest(residual, centers)]
        self.initialized.fill_(True)
        self.unused_steps.zero_()

    def forward(self, x: Tensor) -> RQOutput:
        """Reconstruct inputs (B,input_dim), return codes (B,L) and loss diagnostics."""
        if x.ndim != 2 or len(x) == 0:
            raise ValueError("Expected nonempty (B,D) features")
        if not bool(self.initialized):
            raise RuntimeError("Call initialize on training features before forward")
        z = (checkpoint(self.encoder, x, use_reentrant=False)
             if self.gradient_checkpointing and self.training else self.encoder(x))
        with torch.autocast(device_type=x.device.type, enabled=False):
            residual = z.float()
            quantized = torch.zeros_like(residual)
            codes, commitment, codebook = [], [], []
            for level in range(self.n_levels):
                indices = self._nearest(residual, self.codebooks[level].float())
                selected = F.embedding(indices, self.codebooks[level]).float()
                codes.append(indices)
                commitment.append(F.mse_loss(residual, selected.detach()))
                codebook.append(F.mse_loss(selected, residual.detach()))
                quantized = quantized + selected
                residual = residual - selected.detach()
                if self.training:
                    with torch.no_grad():
                        self.unused_steps[level].add_(1)
                        self.unused_steps[level, indices.unique()] = 0
            straight_through = z.float() + (quantized - z.float()).detach()
        reconstruction = (checkpoint(self.decoder, straight_through, use_reentrant=False)
                          if self.gradient_checkpointing and self.training else self.decoder(straight_through))
        reconstruction_loss = F.mse_loss(reconstruction.float(), x.float())
        commitment_loss, codebook_loss = torch.stack(commitment).mean(), torch.stack(codebook).mean()
        loss = reconstruction_loss + self.commitment_weight * commitment_loss + codebook_loss
        code_tensor = torch.stack(codes, dim=-1)
        usage = torch.stack([code_tensor[:, level].unique().numel() / torch.tensor(
            self.codebook_size, device=x.device) for level in range(self.n_levels)])
        return RQOutput(reconstruction, code_tensor, loss, reconstruction_loss,
                        commitment_loss, codebook_loss, usage)

    @torch.no_grad()
    def reset_dead_codes(self, x: Tensor, seed: int = 42) -> list[tuple[int, Tensor]]:
        """Replace stale codes from TRAIN residuals after an optimizer step.

        Return (level, indices) for reset rows, so callers can clear optimizer
        momentum for those rows. No change when the ablation flag is disabled.
        """
        if not self.dead_code_reset:
            return []
        if not bool(self.initialized) or x.ndim != 2 or len(x) == 0:
            raise ValueError("Dead-code reset needs initialized books and nonempty features")
        generator = torch.Generator(device=x.device).manual_seed(seed)
        changed = []
        with torch.autocast(device_type=x.device.type, enabled=False):
            residual = self.encoder(x.float())
            for level in range(self.n_levels):
                dead = torch.where(self.unused_steps[level] >= self.dead_code_patience)[0]
                if len(dead):
                    indices = torch.randint(len(x), (len(dead),), generator=generator, device=x.device)
                    self.codebooks[level, dead] = residual[indices]
                    self.unused_steps[level, dead] = 0
                    changed.append((level, dead))
                residual = residual - self.codebooks[level, self._nearest(residual, self.codebooks[level])]
        return changed
