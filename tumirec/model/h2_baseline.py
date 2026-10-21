"""M4 memory side branch on the unmodified M3 causal baseline, not M5."""
import math
import torch
from torch import Tensor, nn
from tumirec.model.memory import InterestMemory
from tumirec.model.sasrec import SASRec


class MemoryBaseline(nn.Module):
    """Score next items from recent context plus projected mean interest memory."""

    def __init__(self, baseline: SASRec, memory: InterestMemory, content: Tensor) -> None:
        super().__init__()
        if content.ndim != 2 or len(content) != len(baseline.item_ids):
            raise ValueError('Frozen content must align with catalog rows')
        self.baseline, self.memory = baseline, memory
        self.memory_projection = nn.Linear(baseline.d_model, baseline.d_model, bias=False)
        self.register_buffer('content', content.detach().clone())

    def encode_memory(self, batch: dict[str, Tensor]) -> Tensor:
        """Compress lag-filtered item/action/time inputs into (B,K,D)."""
        if self.memory.mode == 'none':
            return self.memory.latents.new_zeros((len(batch['target']), self.memory.n_latents, self.baseline.d_model))
        rows = batch['memory_rows'].clamp_min(0)
        return self.memory(self.baseline.items(rows), batch['memory_actions'], batch['memory_times'],
                           batch['target_time'], batch['memory_valid'], batch['lag'], self.content[rows])

    def forward(self, batch: dict[str, Tensor], candidates: Tensor,
                cached_memory: Tensor | None = None) -> Tensor:
        """Return scores (B,C); optional cached memory (B,K,D) skips nearline work."""
        states = self.baseline.sequence(batch['recent'], batch['recent_length'])
        context = states[torch.arange(len(states), device=states.device), batch['recent_length']]
        memory = self.encode_memory(batch) if cached_memory is None else cached_memory
        if memory.shape != (len(states), self.memory.n_latents, self.baseline.d_model):
            raise ValueError('Cached memory shape does not match model')
        context = context + self.memory_projection(memory.mean(1))
        return context @ self.baseline.items(candidates).T / math.sqrt(self.baseline.d_model)
