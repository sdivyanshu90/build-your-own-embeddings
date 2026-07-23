"""Cosine-distance triplet objective."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn


class TripletEmbeddingLoss(nn.Module):
    """Enforce a configurable distance margin between positives and negatives."""

    def __init__(self, margin: float = 0.2) -> None:
        super().__init__()
        if margin < 0:
            raise ValueError("margin must not be negative")
        self.margin = margin

    def forward(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        if anchors.shape != positives.shape or anchors.shape != negatives.shape:
            raise ValueError("triplet tensors must share shape (batch, dimension)")
        positive_distance = 1 - functional.cosine_similarity(anchors, positives, dim=-1)
        negative_distance = 1 - functional.cosine_similarity(anchors, negatives, dim=-1)
        return functional.relu(positive_distance - negative_distance + self.margin).mean()
