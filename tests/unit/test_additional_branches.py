from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from embedding_model.config import ModelConfig, TrainingConfig, load_config
from embedding_model.data.preprocessing import split_records
from embedding_model.data.readers import read_records
from embedding_model.data.schemas import (
    PairRecord,
    RetrievalRecord,
    ScoredPairRecord,
    TripletRecord,
)
from embedding_model.evaluation.baselines import random_baseline, tfidf_baseline
from embedding_model.evaluation.clustering import clustering_metrics
from embedding_model.evaluation.diagnostics import (
    embedding_diagnostics,
    positive_negative_separation,
)
from embedding_model.evaluation.retrieval import retrieval_metrics
from embedding_model.evaluation.similarity import similarity_metrics
from embedding_model.exceptions import (
    ArtifactValidationError,
    ConfigurationError,
    DataValidationError,
)
from embedding_model.export.exporter import export_model, load_exported_model
from embedding_model.export.manifest import validate_manifest, write_manifest
from embedding_model.indexing.faiss_index import VectorIndex
from embedding_model.inference.embedder import TextEmbedder
from embedding_model.losses import InfoNCELoss, MultipleNegativesRankingLoss
from embedding_model.modeling.embedding_model import EmbeddingModel
from embedding_model.modeling.pooling import Pooling
from embedding_model.tokenization import LocalTokenizer
from embedding_model.training.checkpointing import load_checkpoint

pytestmark = pytest.mark.unit
ARTIFACT_FILES = [
    "config.json",
    "model.safetensors",
    "tokenizer/tokenizer.json",
    "training_metadata.json",
    "evaluation.json",
    "model_card.md",
]


def test_remaining_configuration_and_split_invalid_states(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="pad_token_id"):
        ModelConfig(vocabulary_size=8, pad_token_id=8)
    with pytest.raises(ValidationError, match="CUDA"):
        TrainingConfig(device="cuda")
    with pytest.raises(ValidationError, match="mixed_precision"):
        TrainingConfig(device="cpu", mixed_precision=True)
    with pytest.raises(ConfigurationError, match="does not exist"):
        load_config(tmp_path / "missing.yaml")
    scalar = tmp_path / "scalar.yaml"
    scalar.write_text("value\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="mapping"):
        load_config(scalar)
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("model: [\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="invalid YAML"):
        load_config(malformed)
    pair = PairRecord(text_a="a", text_b="b")
    with pytest.raises(DataValidationError, match="at least two"):
        split_records([pair])
    with pytest.raises(DataValidationError, match="between"):
        split_records([pair, PairRecord(text_a="c", text_b="d")], validation_fraction=1)


def test_reader_and_schema_invalid_rows_are_actionable(tmp_path: Path) -> None:
    invalid_json = tmp_path / "invalid.jsonl"
    invalid_json.write_text("{broken}\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="invalid JSON"):
        read_records(invalid_json, PairRecord)
    non_object = tmp_path / "list.jsonl"
    non_object.write_text("[1,2]\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="object"):
        read_records(non_object, PairRecord)
    empty = tmp_path / "empty.csv"
    empty.write_text("text_a,text_b\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="empty"):
        read_records(empty, PairRecord)
    with pytest.raises(ValidationError, match="null"):
        PairRecord(text_a=None, text_b="b")  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="whitespace"):
        PairRecord(text_a=" ", text_b="b")
    assert PairRecord(text_a="a", text_b="b", record_id=None).record_id is None
    assert (
        TripletRecord(
            anchor="a",
            positive="b",
            negative="c",
            record_id=None,
        ).record_id
        is None
    )
    assert (
        ScoredPairRecord(
            text_a="a",
            text_b="b",
            score=0.5,
            record_id=None,
        ).record_id
        is None
    )
    with pytest.raises(ValidationError, match="duplicate"):
        RetrievalRecord(
            query_id="q",
            query="query",
            positive_documents=["same", "same"],
        )
    with pytest.raises(ValidationError, match="empty"):
        RetrievalRecord(query_id="q", query="query", positive_documents=[" "])
    with pytest.raises(ValidationError, match="whitespace"):
        RetrievalRecord(query_id="q", query=" ", positive_documents=["document"])


def test_evaluation_invalid_inputs_and_singleton_diagnostic() -> None:
    with pytest.raises(ValueError, match="positive dimension"):
        random_baseline(["text"], 0)
    with pytest.raises(ValueError, match="at least one"):
        tfidf_baseline([])
    embeddings = np.eye(3)
    with pytest.raises(ValueError, match="same sample"):
        clustering_metrics(embeddings, [0, 1], [0, 1])
    with pytest.raises(ValueError, match="finite"):
        clustering_metrics(np.array([[np.nan], [0.0], [1.0]]), [0, 0, 1], [0, 0, 1])
    singleton = embedding_diagnostics(np.array([[1.0, 0.0]]))
    assert singleton["mean_pairwise_cosine"] == 1.0
    with pytest.raises(ValueError, match="shape"):
        embedding_diagnostics(np.array([]))
    with pytest.raises(ValueError, match="finite"):
        embedding_diagnostics(np.array([[np.nan]]))
    with pytest.raises(ValueError, match="required"):
        positive_negative_separation([], [0.0])
    with pytest.raises(ValueError, match="finite"):
        positive_negative_separation([np.nan], [0.0])
    with pytest.raises(ValueError, match="positive integers"):
        retrieval_metrics({"q": []}, {"q": {"d"}}, k_values=(0,))
    with pytest.raises(ValueError, match="equal one-dimensional"):
        similarity_metrics([1.0], [1.0])
    with pytest.raises(ValueError, match="finite"):
        similarity_metrics([0.0, np.nan], [0.0, 1.0])


def test_symmetric_and_implicit_contrastive_paths() -> None:
    matrix = torch.eye(2)
    symmetric = MultipleNegativesRankingLoss(temperature=1, symmetric=True)(matrix, matrix)
    asymmetric = MultipleNegativesRankingLoss(temperature=1)(matrix, matrix)
    assert symmetric.item() == pytest.approx(asymmetric.item())
    assert InfoNCELoss(temperature=1)(matrix, matrix).item() == pytest.approx(asymmetric.item())
    with pytest.raises(ValueError, match="share shape"):
        InfoNCELoss()(matrix, torch.ones(2, 3))
    with pytest.raises(ValueError, match="unsupported"):
        Pooling("invalid")(torch.ones(1, 2, 2), torch.ones(1, 2))  # type: ignore[arg-type]


def test_model_projection_normalization_options_are_executable(
    tokenizer: LocalTokenizer,
) -> None:
    model = EmbeddingModel(
        ModelConfig(
            vocabulary_size=tokenizer.vocab_size,
            hidden_size=8,
            embedding_dimension=8,
            num_attention_heads=2,
            num_hidden_layers=1,
            intermediate_size=16,
            max_sequence_length=8,
            projection=False,
            layer_normalization=True,
            normalize_embeddings=False,
            dropout=0,
        )
    )
    encoded = tokenizer.batch_encode(["alpha beta"], 8)
    assert model(**encoded, normalize=False).shape == (1, 8)


def test_export_and_manifest_failure_modes(
    tmp_path: Path,
    artifact_dir: Path,
    model: EmbeddingModel,
    tokenizer: LocalTokenizer,
) -> None:
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "existing").write_text("x", encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="overwrite"):
        export_model(model, tokenizer, nonempty)
    incompatible = LocalTokenizer.train(["tiny vocabulary"], vocab_size=8)
    with pytest.raises(ArtifactValidationError, match="vocabulary"):
        export_model(model, incompatible, tmp_path / "incompatible")
    with pytest.raises(ArtifactValidationError, match="escapes"):
        write_manifest(tmp_path, ["../outside"])
    with pytest.raises(ArtifactValidationError, match="missing"):
        write_manifest(tmp_path, ["missing"])

    config = artifact_dir / "config.json"
    config.write_text("{invalid", encoding="utf-8")
    write_manifest(artifact_dir, ARTIFACT_FILES)
    with pytest.raises(ArtifactValidationError, match="config"):
        load_exported_model(artifact_dir)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "root"),
        ({"schema_version": 99, "files": {}}, "schema"),
        ({"schema_version": 1, "files": []}, "files"),
        ({"schema_version": 1, "files": {}}, "required"),
    ],
)
def test_manifest_structure_validation(tmp_path: Path, payload: object, message: str) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match=message):
        validate_manifest(tmp_path)


def test_index_manifest_schema_and_metadata_type_are_validated(tmp_path: Path) -> None:
    index = VectorIndex(2)
    index.add(np.eye(2, dtype=np.float32), [{"id": "a"}, {"id": "b"}])
    schema_dir = index.save(tmp_path / "schema")
    schema_manifest = json.loads((schema_dir / "index_manifest.json").read_text(encoding="utf-8"))
    schema_manifest["schema_version"] = 99
    (schema_dir / "index_manifest.json").write_text(json.dumps(schema_manifest), encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="schema"):
        VectorIndex.load(schema_dir)

    metadata_dir = index.save(tmp_path / "metadata")
    (metadata_dir / "metadata.json").write_text("{}", encoding="utf-8")
    manifest = json.loads((metadata_dir / "index_manifest.json").read_text(encoding="utf-8"))
    import hashlib

    manifest["files"]["metadata.json"] = hashlib.sha256(b"{}").hexdigest()
    (metadata_dir / "index_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="metadata"):
        VectorIndex.load(metadata_dir)


def test_embedder_and_checkpoint_public_failures(
    tmp_path: Path,
    embedder: TextEmbedder,
    model: EmbeddingModel,
) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        embedder.encode(["text"], batch_size=0)
    with pytest.raises(TypeError, match="string"):
        embedder.encode([1])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="CUDA"):
        TextEmbedder(model, embedder.tokenizer, device="cuda")

    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    with pytest.raises(ArtifactValidationError, match="does not exist"):
        load_checkpoint(
            tmp_path / "missing.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
        )
    corrupted = tmp_path / "corrupt.pt"
    corrupted.write_text("not a checkpoint", encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="invalid checkpoint"):
        load_checkpoint(
            corrupted,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
        )
