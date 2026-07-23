from __future__ import annotations

import numpy as np
import pytest

from embedding_model.data.samplers import (
    embedding_hard_negatives,
    lexical_hard_negatives,
    random_negatives,
    semi_hard_negatives,
)
from embedding_model.evaluation.diagnostics import embedding_diagnostics
from embedding_model.evaluation.retrieval import retrieval_metrics
from embedding_model.evaluation.similarity import similarity_metrics

pytestmark = pytest.mark.unit


def test_retrieval_metrics_match_hand_computed_multiple_relevance_example() -> None:
    metrics = retrieval_metrics(
        rankings={"q": ["a", "x", "b"]},
        relevant={"q": {"a", "b"}},
        k_values=(1, 3),
    )
    assert metrics["recall@1"] == pytest.approx(0.5)
    assert metrics["precision@3"] == pytest.approx(2 / 3)
    assert metrics["mrr"] == pytest.approx(1.0)
    assert metrics["map"] == pytest.approx((1.0 + 2 / 3) / 2)
    ideal = 1 + 1 / np.log2(3)
    actual = 1 + 1 / np.log2(4)
    assert metrics["ndcg@3"] == pytest.approx(actual / ideal)


def test_similarity_metrics_use_average_ranks_for_ties() -> None:
    metrics = similarity_metrics([1.0, 2.0, 2.0, 4.0], [1.0, 2.0, 3.0, 4.0])
    assert metrics["pearson"] > 0.9
    assert metrics["spearman"] > 0.9
    assert metrics["mse"] == pytest.approx(0.25)


def test_diagnostics_detect_duplicates_and_collapse() -> None:
    diagnostics = embedding_diagnostics(np.ones((3, 4)))
    assert diagnostics["duplicate_count"] == 2
    assert diagnostics["collapsed"] is True


def test_negative_miners_exclude_known_positives_and_are_deterministic() -> None:
    positives = {"q": {"d1"}}
    random_a = random_negatives(["q"], ["d1", "d2", "d3"], positives, seed=5)
    random_b = random_negatives(["q"], ["d1", "d2", "d3"], positives, seed=5)
    assert random_a == random_b
    assert all(item.document_id != "d1" for item in random_a)

    lexical = lexical_hard_negatives(
        {"q": "blue ocean"},
        {"d1": "blue ocean", "d2": "ocean wave", "d3": "bread recipe"},
        positives,
    )
    assert lexical[0].document_id == "d2"

    embedded = embedding_hard_negatives(
        ["q"],
        np.array([[1.0, 0.0]]),
        ["d1", "d2", "d3"],
        np.array([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]),
        positives,
        per_query=2,
    )
    assert [item.document_id for item in embedded] == ["d2", "d3"]
    semi_hard = semi_hard_negatives(embedded, {"q": 1.0}, margin=0.1)
    assert [item.document_id for item in semi_hard] == ["d2"]
