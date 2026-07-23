"""Honest local baselines for retrieval experiments."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.feature_extraction.text import TfidfVectorizer


def random_baseline(texts: list[str], dimension: int, seed: int = 42) -> npt.NDArray[np.float64]:
    """Return deterministic unit random vectors; this is a quality floor."""

    if not texts or dimension <= 0:
        raise ValueError("non-empty texts and a positive dimension are required")
    values = np.random.default_rng(seed).normal(size=(len(texts), dimension))
    return np.asarray(
        values / np.linalg.norm(values, axis=1, keepdims=True),
        dtype=np.float64,
    )


def tfidf_baseline(texts: list[str]) -> npt.NDArray[np.float64]:
    """Fit TF-IDF only on the provided evaluation corpus."""

    if not texts:
        raise ValueError("at least one text is required")
    return np.asarray(
        TfidfVectorizer(lowercase=True).fit_transform(texts).toarray(),
        dtype=np.float64,
    )
