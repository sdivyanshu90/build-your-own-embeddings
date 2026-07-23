from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from embedding_model.config import ModelConfig, load_config
from embedding_model.data.preprocessing import pair_statistics, split_records
from embedding_model.data.readers import read_records
from embedding_model.data.schemas import PairRecord, RetrievalRecord, TripletRecord
from embedding_model.exceptions import ConfigurationError, DataValidationError
from embedding_model.tokenization import BPETokenizer, LocalTokenizer

pytestmark = pytest.mark.unit


def test_model_configuration_rejects_incompatible_shapes_and_unknown_keys() -> None:
    with pytest.raises(ValidationError, match="divisible"):
        ModelConfig(hidden_size=10, num_attention_heads=3)
    with pytest.raises(ValidationError, match="projection"):
        ModelConfig(hidden_size=8, embedding_dimension=4, projection=False)
    with pytest.raises(ValidationError, match="extra"):
        ModelConfig.model_validate({"unexpected": True})


def test_yaml_configuration_reports_file_and_validation_error(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(
        "model:\n  hidden_size: -1\ntraining:\n  epochs: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="invalid configuration"):
        load_config(path)


def test_records_normalize_whitespace_and_prevent_false_negatives() -> None:
    pair = PairRecord(text_a="  alpha   beta ", text_b="gamma")
    assert pair.text_a == "alpha beta"
    with pytest.raises(ValidationError, match="differ"):
        TripletRecord(anchor="a", positive="same", negative="same")
    with pytest.raises(ValidationError, match="positive documents"):
        RetrievalRecord(
            query_id="q",
            query="query",
            positive_documents=["known"],
            negative_documents=["known"],
        )


def test_jsonl_reader_never_silently_drops_invalid_or_duplicate_rows(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text('{"text_a":"a","text_b":"b"}\n\n', encoding="utf-8")
    with pytest.raises(DataValidationError, match="blank JSONL"):
        read_records(invalid, PairRecord)

    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(
        '{"record_id":"1","text_a":"a","text_b":"b"}\n'
        '{"record_id":"1","text_a":"c","text_b":"d"}\n',
        encoding="utf-8",
    )
    with pytest.raises(DataValidationError, match="duplicate record_id"):
        read_records(duplicate, PairRecord)


def test_split_is_reproducible_disjoint_and_statistics_are_honest() -> None:
    records = [PairRecord(text_a=f"a{i}", text_b=f"b{i}") for i in range(5)]
    train_a, validation_a = split_records(records, seed=9)
    train_b, validation_b = split_records(records, seed=9)
    assert train_a == train_b
    assert validation_a == validation_b
    assert set(train_a).isdisjoint(validation_a)
    assert pair_statistics(records).unique_texts == 10


def test_tokenizer_round_trip_preserves_unicode_and_truncates(
    tmp_path: Path,
    tokenizer: LocalTokenizer,
) -> None:
    ids, mask = tokenizer.encode("unicode café 東京 extra tokens", max_length=4)
    assert len(ids) == len(mask) == 4
    path = tokenizer.save(tmp_path / "tokenizer")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    loaded = LocalTokenizer.load(tmp_path / "tokenizer")
    assert loaded.encode("unicode café 東京", 8) == tokenizer.encode("unicode café 東京", 8)
    with pytest.raises(DataValidationError, match="whitespace"):
        loaded.encode("  ", 8)


def test_bpe_tokenizer_learns_and_serializes_ordered_merges(tmp_path: Path) -> None:
    tokenizer = BPETokenizer.train(
        ["low lower lowest", "newer wider"],
        vocab_size=24,
    )
    assert tokenizer.merges
    before = tokenizer.encode("lower newer", max_length=12)
    path = tokenizer.save(tmp_path / "bpe")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["algorithm"] == "bpe"
    assert payload["merges"]
    loaded = LocalTokenizer.load(tmp_path / "bpe")
    assert isinstance(loaded, BPETokenizer)
    assert loaded.encode("lower newer", max_length=12) == before
