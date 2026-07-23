"""Numerically stable vector similarity functions."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as functional


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return pairwise row cosine similarities after shape validation."""

    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("left and right must share shape (batch, dimension)")
    return functional.cosine_similarity(left, right, dim=-1, eps=1e-12).clamp(-1.0, 1.0)


def dot_similarity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return pairwise row dot products."""

    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("left and right must share shape (batch, dimension)")
    return (left * right).sum(dim=-1)


def euclidean_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return pairwise Euclidean distances."""

    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("left and right must share shape (batch, dimension)")
    return cast(torch.Tensor, torch.linalg.vector_norm(left - right, dim=-1))


def similarity_matrix(
    queries: torch.Tensor,
    documents: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return temperature-scaled cosine similarities for all row pairs."""

    if queries.ndim != 2 or documents.ndim != 2:
        raise ValueError("queries and documents must be matrices")
    if queries.shape[1] != documents.shape[1]:
        raise ValueError("query and document dimensions do not match")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    queries = functional.normalize(queries, p=2, dim=-1, eps=1e-12)
    documents = functional.normalize(documents, p=2, dim=-1, eps=1e-12)
    return queries @ documents.transpose(0, 1) / temperature
