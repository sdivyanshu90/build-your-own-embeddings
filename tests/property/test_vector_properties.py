from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as strategies

from embedding_model.indexing.faiss_index import VectorIndex
from embedding_model.modeling.similarity import cosine_similarity

pytestmark = pytest.mark.unit


@given(
    strategies.lists(
        strategies.floats(
            min_value=-100,
            max_value=100,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=2,
        max_size=16,
    ).filter(lambda values: np.linalg.norm(values) > 1e-6)
)
@settings(max_examples=30, deadline=None)
def test_identical_nonzero_vectors_have_maximum_cosine(values: list[float]) -> None:
    tensor = torch.tensor([values], dtype=torch.float32)
    score = cosine_similarity(tensor, tensor)
    assert score.item() == pytest.approx(1.0, abs=1e-5)


@given(
    strategies.lists(
        strategies.tuples(
            strategies.floats(
                min_value=-10,
                max_value=10,
                allow_nan=False,
                allow_infinity=False,
            ),
            strategies.floats(
                min_value=-10,
                max_value=10,
                allow_nan=False,
                allow_infinity=False,
            ),
        ),
        min_size=2,
        max_size=10,
        unique=True,
    ).filter(lambda rows: all(np.linalg.norm(row) > 1e-6 for row in rows))
)
@settings(max_examples=20, deadline=None)
def test_index_scores_are_non_increasing(rows: list[tuple[float, float]]) -> None:
    vectors = np.asarray(rows, dtype=np.float32)
    index = VectorIndex(2)
    index.add(vectors, [{"id": str(i)} for i in range(len(rows))])
    results = index.search(vectors[0], top_k=len(rows))
    assert all(left.score >= right.score for left, right in pairwise(results))
