"""Macro-averaged binary-relevance retrieval metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence, Set


def retrieval_metrics(
    rankings: Mapping[str, Sequence[str]],
    relevant: Mapping[str, Set[str]],
    *,
    k_values: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    """Compute retrieval metrics; input ranking order is the explicit tie breaker."""

    if not rankings:
        raise ValueError("at least one query ranking is required")
    if set(rankings) != set(relevant):
        raise ValueError("rankings and relevance judgments must have identical query IDs")
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("k_values must contain positive integers")
    per_metric: dict[str, list[float]] = {}
    reciprocal_ranks: list[float] = []
    average_precisions: list[float] = []
    for query_id, ranking in rankings.items():
        positives = relevant[query_id]
        if not positives:
            raise ValueError(f"query {query_id!r} has no relevant documents")
        if len(ranking) != len(set(ranking)):
            raise ValueError(f"query {query_id!r} ranking contains duplicate document IDs")
        first_relevant = next(
            (
                position
                for position, document_id in enumerate(ranking, start=1)
                if document_id in positives
            ),
            None,
        )
        reciprocal_ranks.append(0.0 if first_relevant is None else 1.0 / first_relevant)
        hits = 0
        precision_sum = 0.0
        for position, document_id in enumerate(ranking, start=1):
            if document_id in positives:
                hits += 1
                precision_sum += hits / position
        average_precisions.append(precision_sum / len(positives))
        for k in k_values:
            top = ranking[:k]
            relevant_count = sum(document_id in positives for document_id in top)
            _append(per_metric, f"recall@{k}", relevant_count / len(positives))
            _append(per_metric, f"precision@{k}", relevant_count / k)
            _append(per_metric, f"hit_rate@{k}", float(relevant_count > 0))
            gains = [1.0 if document_id in positives else 0.0 for document_id in top]
            dcg = sum(gain / math.log2(position + 1) for position, gain in enumerate(gains, 1))
            ideal_hits = min(len(positives), k)
            idcg = sum(1 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
            _append(per_metric, f"ndcg@{k}", dcg / idcg)
    output = {name: sum(values) / len(values) for name, values in per_metric.items()}
    output["mrr"] = sum(reciprocal_ranks) / len(reciprocal_ranks)
    output["map"] = sum(average_precisions) / len(average_precisions)
    return output


def _append(metrics: dict[str, list[float]], name: str, value: float) -> None:
    metrics.setdefault(name, []).append(value)
