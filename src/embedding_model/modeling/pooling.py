"""Mask-aware token pooling functions."""

from __future__ import annotations

import torch
from torch import nn

from embedding_model.config import PoolingName


class Pooling(nn.Module):
    """Reduce ``(batch, sequence, hidden)`` states to ``(batch, hidden)``."""

    def __init__(self, strategy: PoolingName) -> None:
        super().__init__()
        self.strategy = strategy

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError("hidden states must have shape (batch, sequence, hidden)")
        if attention_mask.shape != hidden.shape[:2]:
            raise ValueError("attention mask shape must match the first two hidden dimensions")
        mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
        lengths = mask.sum(dim=1)
        if torch.any(lengths == 0):
            raise ValueError("fully padded sequences cannot be pooled")
        if self.strategy == "cls":
            return hidden[:, 0]
        if self.strategy == "max":
            minimum = torch.finfo(hidden.dtype).min
            return hidden.masked_fill(mask == 0, minimum).max(dim=1).values
        summed = (hidden * mask).sum(dim=1)
        if self.strategy == "mean":
            return summed / lengths.clamp_min(1)
        if self.strategy == "mean_sqrt_length":
            return summed / lengths.sqrt().clamp_min(1)
        raise ValueError(f"unsupported pooling strategy: {self.strategy}")
