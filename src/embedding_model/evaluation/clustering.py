"""Clustering quality metrics with explicit shape checks."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def clustering_metrics(
    embeddings: npt.NDArray[Any],
    true_labels: list[int] | npt.NDArray[Any],
    predicted_labels: list[int] | npt.NDArray[Any],
) -> dict[str, float]:
    """Return silhouette, adjusted Rand, and normalized mutual information."""

    values = np.asarray(embeddings, dtype=np.float64)
    truth = np.asarray(true_labels)
    predictions = np.asarray(predicted_labels)
    if values.ndim != 2 or values.shape[0] != truth.size or truth.shape != predictions.shape:
        raise ValueError("embeddings and label arrays must have the same sample count")
    if not np.isfinite(values).all():
        raise ValueError("embeddings must be finite")
    cluster_count = len(np.unique(predictions))
    if not 1 < cluster_count < values.shape[0]:
        raise ValueError("silhouette requires between 2 and n-1 predicted clusters")
    return {
        "silhouette": float(silhouette_score(values, predictions, metric="cosine")),
        "adjusted_rand_index": float(adjusted_rand_score(truth, predictions)),
        "normalized_mutual_information": float(normalized_mutual_info_score(truth, predictions)),
    }
