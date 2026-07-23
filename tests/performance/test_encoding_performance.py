from __future__ import annotations

import time

import numpy as np
import pytest

from embedding_model.inference.embedder import TextEmbedder

pytestmark = [pytest.mark.performance, pytest.mark.slow]


def test_encoding_throughput_measurement_is_finite(
    embedder: TextEmbedder,
    record_property: pytest.RecordProperty,
) -> None:
    texts = ["small deterministic benchmark sentence"] * 32
    started = time.perf_counter()
    vectors = np.asarray(embedder.encode(texts, batch_size=8))
    elapsed = time.perf_counter() - started
    throughput = len(texts) / max(elapsed, 1e-12)
    record_property("texts_per_second", throughput)
    assert vectors.shape == (32, embedder.dimension)
    assert np.isfinite(vectors).all()
    assert throughput > 0
