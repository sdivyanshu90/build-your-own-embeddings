"""Low-cost embedding collapse and geometry diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def embedding_diagnostics(
    embeddings: npt.NDArray[Any],
) -> dict[str, float | int | bool]:
    """Summarize norms, variance, duplicates, collapse, and sampled anisotropy."""

    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("embeddings must have shape (samples, dimensions)")
    if not np.isfinite(values).all():
        raise ValueError("embeddings must be finite")
    norms = np.linalg.norm(values, axis=1)
    normalized = values / np.maximum(norms[:, None], 1e-12)
    if len(values) > 1:
        similarity = normalized @ normalized.T
        upper = similarity[np.triu_indices(len(values), k=1)]
        mean_pairwise_cosine = float(upper.mean())
    else:
        mean_pairwise_cosine = 1.0
    rounded_unique = np.unique(np.round(values, decimals=7), axis=0).shape[0]
    mean_variance = float(np.var(values, axis=0).mean())
    return {
        "count": int(values.shape[0]),
        "dimension": int(values.shape[1]),
        "norm_mean": float(norms.mean()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "mean_dimension_variance": mean_variance,
        "duplicate_count": int(values.shape[0] - rounded_unique),
        "mean_pairwise_cosine": mean_pairwise_cosine,
        "collapsed": bool(mean_variance < 1e-8 or mean_pairwise_cosine > 0.999),
    }


def positive_negative_separation(
    positive_scores: npt.NDArray[Any] | list[float],
    negative_scores: npt.NDArray[Any] | list[float],
) -> dict[str, float]:
    """Compare finite positive and negative similarity distributions."""

    positives = np.asarray(positive_scores, dtype=np.float64)
    negatives = np.asarray(negative_scores, dtype=np.float64)
    if positives.size == 0 or negatives.size == 0:
        raise ValueError("both positive and negative scores are required")
    if not np.isfinite(positives).all() or not np.isfinite(negatives).all():
        raise ValueError("scores must be finite")
    return {
        "positive_mean": float(positives.mean()),
        "negative_mean": float(negatives.mean()),
        "mean_gap": float(positives.mean() - negatives.mean()),
    }
