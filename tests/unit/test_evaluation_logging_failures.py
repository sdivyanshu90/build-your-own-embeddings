from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from embedding_model.data.readers import read_records
from embedding_model.data.schemas import PairRecord
from embedding_model.evaluation.baselines import random_baseline, tfidf_baseline
from embedding_model.evaluation.clustering import clustering_metrics
from embedding_model.evaluation.diagnostics import positive_negative_separation
from embedding_model.exceptions import DataValidationError
from embedding_model.logging import JsonFormatter, redact_mapping

pytestmark = pytest.mark.unit


def test_baselines_and_clustering_metrics_have_valid_contracts() -> None:
    random = random_baseline(["a", "b", "c", "d"], dimension=3, seed=2)
    np.testing.assert_allclose(np.linalg.norm(random, axis=1), 1.0)
    tfidf = tfidf_baseline(["red apple", "red berry", "blue ocean", "blue sky"])
    metrics = clustering_metrics(
        tfidf,
        true_labels=[0, 0, 1, 1],
        predicted_labels=[0, 0, 1, 1],
    )
    assert metrics["adjusted_rand_index"] == pytest.approx(1.0)
    assert metrics["normalized_mutual_information"] == pytest.approx(1.0)
    assert -1 <= metrics["silhouette"] <= 1
    with pytest.raises(ValueError, match="clusters"):
        clustering_metrics(tfidf, [0, 0, 1, 1], [0, 0, 0, 0])


def test_distribution_separation_and_logging_redaction() -> None:
    separation = positive_negative_separation([0.9, 0.7], [0.2, 0.0])
    assert separation["mean_gap"] == pytest.approx(0.7)
    redacted = redact_mapping(
        {"auth_token": "secret", "nested": {"password": "hidden"}, "count": 2}
    )
    assert redacted == {
        "auth_token": "[REDACTED]",
        "nested": {"password": "[REDACTED]"},
        "count": 2,
    }
    record = logging.LogRecord(
        "embedding_model.test",
        logging.INFO,
        __file__,
        1,
        "safe event",
        (),
        None,
    )
    record.fields = {"api_key": "never-print", "count": 1}
    payload = json.loads(JsonFormatter().format(record))
    assert payload["api_key"] == "[REDACTED]"
    assert "never-print" not in json.dumps(payload)


def test_csv_reader_and_reader_failure_paths(tmp_path: Path) -> None:
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text("text_a,text_b\nalpha,beta\ngamma,delta\n", encoding="utf-8")
    assert len(read_records(csv_path, PairRecord)) == 2
    unsupported = tmp_path / "pairs.txt"
    unsupported.write_text("anything", encoding="utf-8")
    with pytest.raises(DataValidationError, match="unsupported"):
        read_records(unsupported, PairRecord)
    with pytest.raises(DataValidationError, match="does not exist"):
        read_records(tmp_path / "missing.jsonl", PairRecord)
