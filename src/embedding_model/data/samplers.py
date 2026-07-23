"""False-negative-aware random, lexical, embedding, and semi-hard mining."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class MinedNegative:
    query_id: str
    document_id: str
    score: float


def random_negatives(
    query_ids: list[str],
    candidate_ids: list[str],
    known_positive_ids: dict[str, set[str]],
    *,
    per_query: int = 1,
    seed: int = 42,
) -> list[MinedNegative]:
    """Sample reproducibly while excluding every known positive."""

    _validate_mining_inputs(query_ids, candidate_ids, known_positive_ids, per_query)
    generator = np.random.default_rng(seed)
    output: list[MinedNegative] = []
    for query_id in query_ids:
        candidates = [
            candidate_id
            for candidate_id in candidate_ids
            if candidate_id not in known_positive_ids[query_id]
        ]
        if len(candidates) < per_query:
            raise ValueError(f"not enough eligible negatives for query {query_id!r}")
        selected = generator.choice(len(candidates), size=per_query, replace=False)
        for index in np.atleast_1d(selected):
            candidate_id = candidates[int(index)]
            output.append(MinedNegative(query_id, candidate_id, 0.0))
    return output


def lexical_hard_negatives(
    queries: dict[str, str],
    documents: dict[str, str],
    known_positive_ids: dict[str, set[str]],
    *,
    per_query: int = 1,
) -> list[MinedNegative]:
    """Rank eligible documents by TF-IDF cosine score."""

    if not queries or not documents:
        raise ValueError("queries and documents must not be empty")
    vectorizer = TfidfVectorizer(lowercase=True)
    document_ids = list(documents)
    matrix = vectorizer.fit_transform([*documents.values(), *queries.values()])
    document_matrix = matrix[: len(documents)]
    query_matrix = matrix[len(documents) :]
    similarities = query_matrix @ document_matrix.T
    output: list[MinedNegative] = []
    for query_offset, query_id in enumerate(queries):
        scores = np.asarray(similarities[query_offset].toarray()).ravel()
        order = np.lexsort((np.arange(len(document_ids)), -scores))
        eligible = [
            index
            for index in order
            if document_ids[index] not in known_positive_ids.get(query_id, set())
        ][:per_query]
        if len(eligible) < per_query:
            raise ValueError(f"not enough eligible negatives for query {query_id!r}")
        output.extend(
            MinedNegative(query_id, document_ids[index], float(scores[index])) for index in eligible
        )
    return output


def embedding_hard_negatives(
    query_ids: list[str],
    query_embeddings: npt.NDArray[Any],
    candidate_ids: list[str],
    candidate_embeddings: npt.NDArray[Any],
    known_positive_ids: dict[str, set[str]],
    *,
    per_query: int = 1,
) -> list[MinedNegative]:
    """Mine by normalized dot product, with deterministic insertion-order ties."""

    _validate_mining_inputs(query_ids, candidate_ids, known_positive_ids, per_query)
    queries = _normalize(query_embeddings)
    candidates = _normalize(candidate_embeddings)
    if queries.shape[0] != len(query_ids) or candidates.shape[0] != len(candidate_ids):
        raise ValueError("embedding row counts must match their ID lists")
    if queries.shape[1] != candidates.shape[1]:
        raise ValueError("query and candidate embedding dimensions do not match")
    similarities = queries @ candidates.T
    output: list[MinedNegative] = []
    for query_offset, query_id in enumerate(query_ids):
        scores = similarities[query_offset]
        order = np.lexsort((np.arange(len(candidate_ids)), -scores))
        eligible = [
            index for index in order if candidate_ids[index] not in known_positive_ids[query_id]
        ][:per_query]
        if len(eligible) < per_query:
            raise ValueError(f"not enough eligible negatives for query {query_id!r}")
        output.extend(
            MinedNegative(query_id, candidate_ids[index], float(scores[index]))
            for index in eligible
        )
    return output


def semi_hard_negatives(
    candidates: list[MinedNegative],
    positive_scores: dict[str, float],
    *,
    margin: float,
) -> list[MinedNegative]:
    """Keep negatives below the positive but within the configured margin."""

    if margin <= 0:
        raise ValueError("margin must be positive")
    return [
        candidate
        for candidate in candidates
        if candidate.query_id in positive_scores
        and positive_scores[candidate.query_id] - margin
        <= candidate.score
        < positive_scores[candidate.query_id]
    ]


def _validate_mining_inputs(
    query_ids: list[str],
    candidate_ids: list[str],
    known_positive_ids: dict[str, set[str]],
    per_query: int,
) -> None:
    if not query_ids or not candidate_ids or per_query <= 0:
        raise ValueError("non-empty IDs and positive per_query are required")
    if len(query_ids) != len(set(query_ids)) or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("query and candidate IDs must be unique")
    missing = set(query_ids) - set(known_positive_ids)
    if missing:
        raise ValueError(f"known-positive mappings are missing query IDs: {sorted(missing)}")


def _normalize(values: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("embeddings must be finite matrices")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("zero embeddings cannot be mined")
    return np.asarray(matrix / norms, dtype=np.float64)
