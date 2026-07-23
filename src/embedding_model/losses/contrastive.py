"""In-batch multiple-negatives and general InfoNCE losses."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as functional
from torch import nn

from embedding_model.modeling.similarity import similarity_matrix


class MultipleNegativesRankingLoss(nn.Module):
    """Cross entropy over the diagonal positives of a batch similarity matrix."""

    def __init__(self, temperature: float = 0.05, *, symmetric: bool = False) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature
        self.symmetric = symmetric

    def forward(self, anchors: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
        if anchors.shape != positives.shape or anchors.ndim != 2:
            raise ValueError("anchors and positives must share shape (batch, dimension)")
        if anchors.shape[0] < 2:
            raise ValueError("multiple-negatives loss requires a batch of at least two")
        logits = similarity_matrix(anchors, positives, temperature=self.temperature)
        labels = torch.arange(anchors.shape[0], device=anchors.device)
        forward_loss = functional.cross_entropy(logits, labels)
        if not self.symmetric:
            return forward_loss
        return (forward_loss + functional.cross_entropy(logits.transpose(0, 1), labels)) / 2


class InfoNCELoss(nn.Module):
    """InfoNCE with one positive and zero or more explicit negatives per anchor."""

    def __init__(self, temperature: float = 0.05) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if anchors.shape != positives.shape or anchors.ndim != 2:
            raise ValueError("anchors and positives must share shape (batch, dimension)")
        if negatives is None:
            return cast(
                torch.Tensor,
                MultipleNegativesRankingLoss(self.temperature)(anchors, positives),
            )
        if (
            negatives.ndim != 3
            or negatives.shape[0] != anchors.shape[0]
            or negatives.shape[2] != anchors.shape[1]
        ):
            raise ValueError("negatives must have shape (batch, negatives, dimension)")
        anchors = functional.normalize(anchors, dim=-1, eps=1e-12)
        positives = functional.normalize(positives, dim=-1, eps=1e-12)
        negatives = functional.normalize(negatives, dim=-1, eps=1e-12)
        positive_logits = (anchors * positives).sum(dim=-1, keepdim=True)
        negative_logits = torch.einsum("bd,bnd->bn", anchors, negatives)
        logits = torch.cat((positive_logits, negative_logits), dim=1) / self.temperature
        labels = torch.zeros(anchors.shape[0], dtype=torch.long, device=anchors.device)
        return functional.cross_entropy(logits, labels)
