"""Continuous cosine-similarity regression."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn


class CosineRegressionLoss(nn.Module):
    """Mean squared error between cosine similarity and labels in [-1, 1]."""

    def forward(
        self, left: torch.Tensor, right: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        if left.shape != right.shape or left.ndim != 2:
            raise ValueError("left and right must share shape (batch, dimension)")
        if targets.shape != (left.shape[0],):
            raise ValueError("targets must have shape (batch,)")
        if torch.any((targets < -1) | (targets > 1)):
            raise ValueError("cosine targets must be between -1 and 1")
        predictions = functional.cosine_similarity(left, right, dim=-1, eps=1e-12)
        return functional.mse_loss(predictions, targets)
