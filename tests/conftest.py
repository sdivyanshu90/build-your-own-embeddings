from __future__ import annotations

from pathlib import Path

import pytest

from embedding_model.config import ModelConfig
from embedding_model.export.exporter import export_model
from embedding_model.inference.embedder import TextEmbedder
from embedding_model.modeling.embedding_model import EmbeddingModel
from embedding_model.tokenization import LocalTokenizer


@pytest.fixture()
def tokenizer() -> LocalTokenizer:
    return LocalTokenizer.train(
        ["alpha beta", "gamma delta", "alpha gamma", "unicode café 東京"],
        vocab_size=32,
    )


@pytest.fixture()
def model(tokenizer: LocalTokenizer) -> EmbeddingModel:
    return EmbeddingModel(
        ModelConfig(
            vocabulary_size=tokenizer.vocab_size,
            hidden_size=8,
            embedding_dimension=4,
            num_attention_heads=2,
            num_hidden_layers=1,
            intermediate_size=16,
            max_sequence_length=12,
            dropout=0.0,
        )
    )


@pytest.fixture()
def artifact_dir(
    tmp_path: Path,
    model: EmbeddingModel,
    tokenizer: LocalTokenizer,
) -> Path:
    return export_model(model, tokenizer, tmp_path / "model")


@pytest.fixture()
def embedder(artifact_dir: Path) -> TextEmbedder:
    return TextEmbedder.from_pretrained(artifact_dir)
