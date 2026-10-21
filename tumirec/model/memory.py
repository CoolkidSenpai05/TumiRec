"""Behavioral interest memory: two latent cross-attention blocks and one self block."""
from __future__ import annotations
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

MEMORY_MODES = ('none', 'recent_only', 'behavior_memory', 'behavior_memory+content_pool')


def pool_sessions(tokens: Tensor, times: Tensor, valid: Tensor, gap_seconds: int) -> tuple[Tensor, Tensor]:
    """Mean-pool chronologically ordered (B,L,D) events across gaps > threshold.

    Return padded session means (B,S,D) and validity (B,S); keep at least one slot.
    """
    if gap_seconds < 0:
        raise ValueError('Session gap cannot be negative')
    rows = []
    for values, timestamps, mask in zip(tokens, times, valid):
        values, timestamps = values[mask], timestamps[mask]
        if len(values) == 0:
            rows.append(values)
            continue
        boundaries = torch.where(timestamps[1:] - timestamps[:-1] > gap_seconds)[0] + 1
        bounds = [0] + boundaries.tolist() + [len(values)]
        rows.append(torch.stack([values[a:b].mean(0) for a, b in zip(bounds[:-1], bounds[1:])]))
    width = max(1, max(len(row) for row in rows))
    padded = torch.stack([F.pad(row, (0, 0, 0, width - len(row))) for row in rows])
    mask = torch.arange(width, device=tokens.device)[None] < torch.tensor([len(row) for row in rows], device=tokens.device)[:, None]
    return padded, mask


class MemoryAttention(nn.Module):
    """Pre-norm SDPA attention plus feed-forward residual, (B,Q,D) <- (B,L,D)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads, self.dropout = n_heads, dropout
        self.query_norm, self.source_norm, self.ff_norm = (nn.LayerNorm(d_model) for _ in range(3))
        self.query, self.key, self.value, self.out = (nn.Linear(d_model, d_model) for _ in range(4))
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, latents: Tensor, source: Tensor, valid: Tensor) -> Tensor:
        """Attend to valid keys (B,L); caller supplies a safe empty-history sentinel."""
        b, q, d = latents.shape
        normalized = self.source_norm(source)
        query = self.query(self.query_norm(latents)).reshape(b, q, self.n_heads, -1).transpose(1, 2)
        key = self.key(normalized).reshape(b, -1, self.n_heads, d // self.n_heads).transpose(1, 2)
        value = self.value(normalized).reshape(b, -1, self.n_heads, d // self.n_heads).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=valid[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0)
        latents = latents + F.dropout(self.out(attended.transpose(1, 2).reshape(b, q, d)), self.dropout, self.training)
        return latents + F.dropout(self.ff(self.ff_norm(latents)), self.dropout, self.training)


class InterestMemory(nn.Module):
    """Compress bounded behavior into (B,K,D) tokens, never generating user text.

    Time embeddings are relative to the memory cutoff (target - lag), so a cached
    snapshot remains well-defined. Padding, future and equal-cutoff events cannot
    affect the output. Empty/no-memory output is exactly zero, not learned latents.
    """

    def __init__(self, d_model: int = 384, n_heads: int = 6, n_latents: int = 8,
                 n_actions: int = 3, time_buckets: int = 32, time_unit_seconds: int = 1,
                 max_history: int = 1000, recent_window: int = 50,
                 mode: str = 'behavior_memory', session_segmentation: bool = False,
                 session_gap_seconds: int = 1800, content_dim: int | None = None,
                 dropout: float = 0.0, gradient_checkpointing: bool = False) -> None:
        super().__init__()
        mode = 'none' if mode == 'no_memory' else mode
        if mode not in MEMORY_MODES or min(d_model, n_heads, n_latents, n_actions, time_buckets,
            time_unit_seconds, max_history, recent_window) < 1 or d_model % n_heads or not 0 <= dropout < 1:
            raise ValueError('Invalid memory configuration')
        if session_gap_seconds < 0 or (mode.endswith('content_pool') and (content_dim is None or content_dim < 1)):
            raise ValueError('Invalid session gap or missing content dimension')
        self.d_model, self.n_latents, self.mode = d_model, n_latents, mode
        self.max_history, self.recent_window = max_history, recent_window
        self.time_unit_seconds, self.time_buckets = time_unit_seconds, time_buckets
        self.session_segmentation, self.session_gap_seconds = session_segmentation, session_gap_seconds
        self.gradient_checkpointing = gradient_checkpointing
        self.latents = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)
        self.action = nn.Embedding(n_actions, d_model)
        self.time = nn.Embedding(time_buckets, d_model)
        self.cross = nn.ModuleList([MemoryAttention(d_model, n_heads, dropout) for _ in range(2)])
        self.self_attention = MemoryAttention(d_model, n_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.content_projection = nn.Linear(content_dim, d_model) if content_dim is not None else None

    def forward(self, item_tokens: Tensor, actions: Tensor, timestamps: Tensor,
                target_times: Tensor, valid: Tensor | None = None,
                lag_seconds: int | Tensor = 0, content: Tensor | None = None) -> Tensor:
        """Encode (B,L,D) item tokens, (B,L) action/time/mask, (B,) target/lag.

        Optional frozen content (B,L,C) is detached before masked mean pooling.
        Histories can arrive unsorted; select newest eligible events before pooling.
        """
        if item_tokens.ndim != 3 or item_tokens.shape[0] == 0:
            raise ValueError('Expected nonempty batch of item tokens (B,L,D)')
        b, length, d = item_tokens.shape
        if d != self.d_model or actions.shape != (b, length) or timestamps.shape != (b, length) or target_times.shape != (b,):
            raise ValueError('Invalid memory input shapes')
        if timestamps.dtype != torch.int64 or target_times.dtype != torch.int64 or actions.dtype != torch.int64:
            raise ValueError('Actions and Unix-second times must be int64 tensors')
        lag = torch.as_tensor(lag_seconds, device=item_tokens.device)
        if lag.ndim > 1 or (lag.ndim == 1 and lag.shape != (b,)) or not torch.isfinite(lag).all() or (lag < 0).any() or (lag != lag.long()).any():
            raise ValueError('Lag must be nonnegative scalar or (B,)')
        lag = lag.long()
        if valid is None:
            valid = torch.ones((b, length), dtype=torch.bool, device=item_tokens.device)
        if valid.shape != (b, length) or valid.dtype != torch.bool:
            raise ValueError('Validity must be boolean (B,L)')
        if self.mode == 'none' or length == 0:
            return item_tokens.new_zeros((b, self.n_latents, d))
        cutoff = target_times - lag
        valid = valid & (timestamps < cutoff[:, None])
        # Invalid events sort before valid events, ensuring they cannot displace history.
        order = torch.argsort(timestamps.masked_fill(~valid, torch.iinfo(torch.int64).min), dim=1, stable=True)
        limit = self.recent_window if self.mode == 'recent_only' else self.max_history
        order = order[:, -limit:]
        valid, timestamps, actions = (value.gather(1, order) for value in (valid, timestamps, actions))
        tokens = item_tokens.gather(1, order[..., None].expand(-1, -1, d))
        tokens = tokens.masked_fill(~valid[..., None], 0)
        deltas = (cutoff[:, None] - timestamps).clamp_min(0).float()
        buckets = torch.log2(1 + deltas / self.time_unit_seconds).floor().long().clamp(0, self.time_buckets - 1)
        tokens = (tokens + self.action(actions.masked_fill(~valid, 0)) + self.time(buckets)) * valid[..., None]
        has_history = valid.any(1)
        pooled_content = None
        if self.mode.endswith('content_pool'):
            if content is None or content.shape[:2] != (b, length):
                raise ValueError('Content-pool mode requires aligned frozen content')
            chosen = content.detach().gather(1, order[..., None].expand(-1, -1, content.shape[-1]))
            chosen = chosen.masked_fill(~valid[..., None], 0)
            pooled_content = self.content_projection(chosen.sum(1) / valid.sum(1).clamp_min(1)[:, None])
        if self.session_segmentation:
            tokens, valid = pool_sessions(tokens, timestamps, valid, self.session_gap_seconds)
        # A zero key for empty rows prevents all-masked softmax across SDPA backends.
        safe_valid = valid.clone()
        safe_valid[~has_history, 0] = True
        latents = self.latents.expand(b, -1, -1)
        with torch.profiler.record_function('interest_memory'):
            for block in self.cross:
                latents = (checkpoint(block, latents, tokens, safe_valid, use_reentrant=False)
                           if self.gradient_checkpointing and self.training else block(latents, tokens, safe_valid))
            all_valid = torch.ones((b, self.n_latents), dtype=torch.bool, device=tokens.device)
            latents = (checkpoint(self.self_attention, latents, latents, all_valid, use_reentrant=False)
                       if self.gradient_checkpointing and self.training else self.self_attention(latents, latents, all_valid))
        if pooled_content is not None:
            latents = latents + pooled_content[:, None]
        return self.norm(latents) * has_history[:, None, None]
