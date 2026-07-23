"""Teacher-student embedding and similarity-distribution distillation."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn


class DistillationLoss(nn.Module):
    """Weighted embedding MSE, cosine distance, and relational KL divergence."""

    def __init__(
        self,
        *,
        mse_weight: float = 1.0,
        cosine_weight: float = 1.0,
        distribution_weight: float = 0.0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        weights = (mse_weight, cosine_weight, distribution_weight)
        if any(weight < 0 for weight in weights) or sum(weights) == 0:
            raise ValueError("distillation weights must be non-negative with a positive sum")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
        self.distribution_weight = distribution_weight
        self.temperature = temperature

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        if student.shape != teacher.shape or student.ndim != 2:
            raise ValueError("student and teacher must share shape (batch, dimension)")
        teacher = teacher.detach()
        loss = self.mse_weight * functional.mse_loss(student, teacher)
        cosine = 1 - functional.cosine_similarity(student, teacher, dim=-1).mean()
        loss = loss + self.cosine_weight * cosine
        if self.distribution_weight:
            student_logits = student @ student.transpose(0, 1) / self.temperature
            teacher_logits = teacher @ teacher.transpose(0, 1) / self.temperature
            distribution = functional.kl_div(
                functional.log_softmax(student_logits, dim=-1),
                functional.softmax(teacher_logits, dim=-1),
                reduction="batchmean",
            )
            loss = loss + self.distribution_weight * distribution * self.temperature**2
        return loss
