"""Semantic textual similarity metrics without hidden tie behavior."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def similarity_metrics(
    predictions: npt.NDArray[Any] | list[float],
    targets: npt.NDArray[Any] | list[float],
) -> dict[str, float]:
    """Return Pearson, average-rank Spearman, and mean squared error."""

    predicted = np.asarray(predictions, dtype=np.float64)
    expected = np.asarray(targets, dtype=np.float64)
    if predicted.ndim != 1 or predicted.shape != expected.shape or predicted.size < 2:
        raise ValueError(
            "predictions and targets must be equal one-dimensional arrays of size >= 2"
        )
    if not np.isfinite(predicted).all() or not np.isfinite(expected).all():
        raise ValueError("metric inputs must be finite")
    pearson = _correlation(predicted, expected)
    spearman = _correlation(_average_ranks(predicted), _average_ranks(expected))
    mse = float(np.mean(np.square(predicted - expected)))
    return {"pearson": pearson, "spearman": spearman, "mse": mse}


def _correlation(left: npt.NDArray[np.float64], right: npt.NDArray[np.float64]) -> float:
    centered_left = left - left.mean()
    centered_right = right - right.mean()
    denominator = np.linalg.norm(centered_left) * np.linalg.norm(centered_right)
    if denominator <= 1e-15:
        raise ValueError("correlation is undefined for a constant input")
    return float(np.dot(centered_left, centered_right) / denominator)


def _average_ranks(
    values: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2
        ranks[order[start:end]] = average_rank
        start = end
    return ranks
