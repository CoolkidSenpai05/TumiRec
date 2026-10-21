"""Small causal SASRec-style control for the M3 H1 study, not the unified model."""
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from tumirec.model.item_token import GatedItemToken


class CausalBlock(nn.Module):
    """Pre-norm SDPA attention and feed-forward block on (B,T,D)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads, self.dropout = n_heads, dropout
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.qkv, self.projection = nn.Linear(d_model, 3 * d_model), nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * d_model, d_model))

    def forward(self, x: Tensor, allowed: Tensor) -> Tensor:
        """Transform (B,T,D) with allowed attention mask (B,1,T,T)."""
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, t, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed,
                                                  dropout_p=self.dropout if self.training else 0.0)
        x = x + F.dropout(self.projection(attended.transpose(1, 2).reshape(b, t, d)), self.dropout, self.training)
        return x + F.dropout(self.ffn(self.norm2(x)), self.dropout, self.training)


class SASRec(nn.Module):
    """Shared item tower, causal recent history, learned start token for empty history."""

    def __init__(self, item_token: GatedItemToken, item_ids: Tensor, codes: Tensor,
                 statistics: Tensor, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, max_history: int = 50, dropout: float = 0.1,
                 gradient_checkpointing: bool = False) -> None:
        super().__init__()
        if min(d_model, n_heads, n_layers, max_history) < 1 or d_model % n_heads or not 0 <= dropout < 1:
            raise ValueError('Invalid baseline dimensions/dropout')
        self.item_token = item_token
        self.register_buffer('item_ids', item_ids)
        self.register_buffer('codes', codes)
        self.register_buffer('statistics', statistics)
        self.start = nn.Parameter(torch.randn(d_model) * 0.02)
        self.positions = nn.Embedding(max_history + 1, d_model)
        self.blocks = nn.ModuleList([CausalBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.max_history, self.gradient_checkpointing = max_history, gradient_checkpointing
        self.d_model = d_model

    def items(self, rows: Tensor) -> Tensor:
        """Embed catalog row indices (...) into (...,D)."""
        return self.item_token(self.codes[rows], self.item_ids[rows], self.statistics[rows])

    def sequence(self, history: Tensor, lengths: Tensor) -> Tensor:
        """Return causal states (B,W+1,D); right padding is -1, lengths shape (B,)."""
        b, width = history.shape
        if width > self.max_history or (lengths < 0).any() or (lengths > width).any():
            raise ValueError('History exceeds configured window or has invalid lengths')
        tokens = self.items(history.clamp_min(0)) * (history >= 0).unsqueeze(-1)
        x = torch.cat((self.start.expand(b, 1, -1), tokens), dim=1)
        positions = torch.arange(width + 1, device=x.device)
        x = x + self.positions(positions)
        valid = positions[None] <= lengths[:, None]
        allowed = (positions[:, None] >= positions[None, :])[None, None] & valid[:, None, None, :]
        for block in self.blocks:
            x = (checkpoint(block, x, allowed, use_reentrant=False)
                 if self.gradient_checkpointing and self.training else block(x, allowed))
        return self.norm(x)

    def forward(self, history: Tensor, lengths: Tensor, candidates: Tensor) -> Tensor:
        """Score shared candidates (C,) from histories (B,W), returning (B,C)."""
        states = self.sequence(history, lengths)
        context = states[torch.arange(len(history), device=history.device), lengths]
        return context @ self.items(candidates).T / math.sqrt(self.d_model)
