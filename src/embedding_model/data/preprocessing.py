"""Reproducible splitting and lightweight dataset diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np

from embedding_model.exceptions import DataValidationError

RecordT = TypeVar("RecordT")


@dataclass(frozen=True)
class DatasetStatistics:
    count: int
    unique_texts: int
    duplicate_pairs: int
    maximum_characters: int
    mean_characters: float


def split_records(
    records: Sequence[RecordT],
    *,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[RecordT], list[RecordT]]:
    """Shuffle deterministically and create non-overlapping train/validation lists."""

    if len(records) < 2:
        raise DataValidationError("at least two records are required for a split")
    if not 0 < validation_fraction < 1:
        raise DataValidationError("validation_fraction must be between zero and one")
    indices = list(range(len(records)))
    np.random.default_rng(seed).shuffle(indices)
    validation_size = min(len(records) - 1, max(1, round(len(records) * validation_fraction)))
    validation_indices = set(indices[:validation_size])
    train = [record for index, record in enumerate(records) if index not in validation_indices]
    validation = [record for index, record in enumerate(records) if index in validation_indices]
    return train, validation


def pair_statistics(pairs: Sequence[Any]) -> DatasetStatistics:
    """Summarize objects exposing ``text_a`` and ``text_b``."""

    texts: list[str] = []
    keys: list[tuple[str, str]] = []
    for pair in pairs:
        text_a = pair.text_a
        text_b = pair.text_b
        texts.extend((text_a, text_b))
        keys.append((text_a, text_b))
    lengths = [len(text) for text in texts]
    return DatasetStatistics(
        count=len(pairs),
        unique_texts=len(set(texts)),
        duplicate_pairs=len(keys) - len(set(keys)),
        maximum_characters=max(lengths, default=0),
        mean_characters=sum(lengths) / len(lengths) if lengths else 0.0,
    )
