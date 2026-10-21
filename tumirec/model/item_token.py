"""Ablatable semantic/hash item tokens with count-dependent ID dropout."""
from collections.abc import Sequence
import torch
from torch import Tensor, nn

MODES = ('sid_only', 'id_only', 'fixed_sum', 'gated', 'gated_no_id_dropout')


class GatedItemToken(nn.Module):
    """Map SID codes (...,L), IDs (...), and train statistics (...,3) to (...,D).

    Statistics columns are count, normalized age and popularity percentile.
    Every cold item falls back to SID, including in the id_only control.
    """

    def __init__(self, codebook_sizes: Sequence[int], d_model: int = 128,
                 hash_buckets: int = 10000, mode: str = 'gated',
                 dropout_min: float = 0.0, dropout_max: float = 0.8,
                 dropout_scale: float = 10.0, dropout_function: str = 'inverse',
                 gate_hidden: int = 16) -> None:
        super().__init__()
        if mode not in MODES or not codebook_sizes or min(*codebook_sizes, d_model, hash_buckets, gate_hidden) < 1:
            raise ValueError('Invalid mode or embedding dimensions')
        if not 0 <= dropout_min <= dropout_max <= 1 or dropout_scale <= 0:
            raise ValueError('Invalid ID dropout settings')
        if dropout_function not in ('inverse', 'exponential'):
            raise ValueError('Unknown ID dropout function')
        self.sid_embeddings = nn.ModuleList([nn.Embedding(size, d_model) for size in codebook_sizes])
        self.id_embedding = nn.Embedding(hash_buckets, d_model)
        self.gate = nn.Sequential(nn.Linear(3, gate_hidden), nn.GELU(), nn.Linear(gate_hidden, 1))
        self.mode, self.hash_buckets = mode, hash_buckets
        self.dropout_min, self.dropout_max = dropout_min, dropout_max
        self.dropout_scale, self.dropout_function = dropout_scale, dropout_function

    def dropout_probability(self, count: Tensor) -> Tensor:
        """Return probabilities shaped like count, monotonically decreasing in count."""
        if (count < 0).any():
            raise ValueError('Counts cannot be negative')
        ratio = count.float() / self.dropout_scale
        decay = 1 / (1 + ratio) if self.dropout_function == 'inverse' else torch.exp(-ratio)
        return self.dropout_min + (self.dropout_max - self.dropout_min) * decay

    def hash_ids(self, item_ids: Tensor) -> Tensor:
        """Stable integer hash (...), independent of Python hash randomization."""
        return ((item_ids.long().remainder(2147483647) * 48271).remainder(2147483647)
                .remainder(self.hash_buckets))

    def forward(self, codes: Tensor, item_ids: Tensor | None, statistics: Tensor) -> Tensor:
        """Return (...,D); omit IDs entirely when all counts are zero or SID-only."""
        if codes.shape[-1] != len(self.sid_embeddings) or statistics.shape != (*codes.shape[:-1], 3):
            raise ValueError('SID/statistic shapes do not align')
        if not torch.isfinite(statistics).all() or (statistics < 0).any():
            raise ValueError('Statistics must be finite and nonnegative')
        sid = sum(embedding(codes[..., level]) for level, embedding in enumerate(self.sid_embeddings))
        seen = statistics[..., 0] > 0
        if self.mode == 'sid_only' or not seen.any():
            return sid
        if item_ids is None or item_ids.shape != codes.shape[:-1]:
            raise ValueError('Observed items require aligned integer IDs')
        identity = self.id_embedding(self.hash_ids(item_ids)) * seen.unsqueeze(-1)
        if self.mode == 'gated' and self.training:
            keep = torch.rand_like(statistics[..., 0]) >= self.dropout_probability(statistics[..., 0])
            identity = identity * keep.unsqueeze(-1)  # no inverted-dropout rescaling
        if self.mode == 'id_only':
            return torch.where(seen.unsqueeze(-1), identity, sid)
        if self.mode == 'fixed_sum':
            return sid + identity
        gate_features = torch.stack((torch.log1p(statistics[..., 0]), statistics[..., 1], statistics[..., 2]), -1)
        return sid + torch.sigmoid(self.gate(gate_features)) * identity
