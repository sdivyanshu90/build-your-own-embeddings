from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from embedding_model.config import TrainingConfig
from embedding_model.data.samplers import (
    embedding_hard_negatives,
    random_negatives,
    semi_hard_negatives,
)
from embedding_model.data.schemas import PairRecord
from embedding_model.evaluation.retrieval import retrieval_metrics
from embedding_model.evaluation.similarity import similarity_metrics
from embedding_model.exceptions import ArtifactValidationError, DataValidationError
from embedding_model.indexing.faiss_index import VectorIndex
from embedding_model.losses import (
    CosineRegressionLoss,
    DistillationLoss,
    InfoNCELoss,
    MultipleNegativesRankingLoss,
    TripletEmbeddingLoss,
)
from embedding_model.modeling.embedding_model import EmbeddingModel
from embedding_model.modeling.pooling import Pooling
from embedding_model.modeling.similarity import (
    cosine_similarity,
    dot_similarity,
    euclidean_distance,
    similarity_matrix,
)
from embedding_model.tokenization import BPETokenizer, LocalTokenizer
from embedding_model.training.trainer import Trainer

pytestmark = pytest.mark.unit


def test_loss_validation_rejects_mathematically_invalid_inputs() -> None:
    matrix = torch.eye(2)
    with pytest.raises(ValueError, match="temperature"):
        MultipleNegativesRankingLoss(0)
    with pytest.raises(ValueError, match="batch"):
        MultipleNegativesRankingLoss()(matrix[:1], matrix[:1])
    with pytest.raises(ValueError, match="share shape"):
        MultipleNegativesRankingLoss()(matrix, torch.ones(2, 3))
    with pytest.raises(ValueError, match="temperature"):
        InfoNCELoss(0)
    with pytest.raises(ValueError, match="negatives"):
        InfoNCELoss()(matrix, matrix, torch.ones(2, 2))
    with pytest.raises(ValueError, match="negative"):
        TripletEmbeddingLoss(-1)
    with pytest.raises(ValueError, match="share shape"):
        TripletEmbeddingLoss()(matrix, matrix, torch.ones(2, 3))
    with pytest.raises(ValueError, match="targets"):
        CosineRegressionLoss()(matrix, matrix, torch.ones(2, 1))
    with pytest.raises(ValueError, match="between"):
        CosineRegressionLoss()(matrix, matrix, torch.tensor([2.0, 0.0]))
    with pytest.raises(ValueError, match="weights"):
        DistillationLoss(mse_weight=0, cosine_weight=0)
    with pytest.raises(ValueError, match="temperature"):
        DistillationLoss(temperature=0)
    with pytest.raises(ValueError, match="share shape"):
        DistillationLoss()(matrix, torch.ones(2, 3))


def test_modeling_validation_rejects_shape_and_numerical_contract_violations(
    model: EmbeddingModel,
) -> None:
    with pytest.raises(ValueError, match="shape"):
        model(torch.ones(2, dtype=torch.long), torch.ones(2, dtype=torch.long))
    with pytest.raises(ValueError, match="exceeds"):
        model(
            torch.ones((1, model.config.max_sequence_length + 1), dtype=torch.long),
            torch.ones((1, model.config.max_sequence_length + 1), dtype=torch.long),
        )
    with pytest.raises(ValueError, match="hidden states"):
        Pooling("mean")(torch.ones(2, 3), torch.ones(2, 3))
    with pytest.raises(ValueError, match="attention mask"):
        Pooling("mean")(torch.ones(2, 3, 4), torch.ones(2, 2))
    left = torch.ones(2, 2)
    with pytest.raises(ValueError, match="share shape"):
        cosine_similarity(left, torch.ones(2, 3))
    with pytest.raises(ValueError, match="share shape"):
        dot_similarity(left, torch.ones(2, 3))
    with pytest.raises(ValueError, match="share shape"):
        euclidean_distance(left, torch.ones(2, 3))
    with pytest.raises(ValueError, match="matrices"):
        similarity_matrix(torch.ones(2), left)
    with pytest.raises(ValueError, match="dimensions"):
        similarity_matrix(left, torch.ones(2, 3))
    with pytest.raises(ValueError, match="temperature"):
        similarity_matrix(left, left, temperature=0)


def test_tokenizer_rejects_empty_bad_special_tokens_and_corruption(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="vocab_size"):
        LocalTokenizer.train(["text"], vocab_size=4)
    with pytest.raises(DataValidationError, match="at least one"):
        LocalTokenizer.train([], vocab_size=8)
    with pytest.raises(DataValidationError, match="strings"):
        LocalTokenizer.train([1], vocab_size=8)  # type: ignore[list-item]
    with pytest.raises(DataValidationError, match="special-token"):
        LocalTokenizer({"[PAD]": 1, "[UNK]": 0, "[CLS]": 2, "[SEP]": 3})
    tokenizer = LocalTokenizer.train(["valid text"], vocab_size=8)
    with pytest.raises(DataValidationError, match="empty batch"):
        tokenizer.batch_encode([], 8)
    with pytest.raises(DataValidationError, match="at least 2"):
        tokenizer.encode("text", 1)
    with pytest.raises(ArtifactValidationError, match="missing"):
        LocalTokenizer.load(tmp_path)
    directory = tmp_path / "bad"
    directory.mkdir()
    (directory / "tokenizer.json").write_text(
        json.dumps({"schema_version": 99, "vocabulary": {}}), encoding="utf-8"
    )
    with pytest.raises(ArtifactValidationError, match="schema"):
        LocalTokenizer.load(directory)


def test_bpe_training_and_load_failures_are_actionable(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="vocab_size"):
        BPETokenizer.train(["text"], vocab_size=4)
    with pytest.raises(DataValidationError, match="at least one"):
        BPETokenizer.train([], vocab_size=12)
    with pytest.raises(DataValidationError, match="strings"):
        BPETokenizer.train([1], vocab_size=12)  # type: ignore[list-item]
    with pytest.raises(DataValidationError, match="empty"):
        BPETokenizer.train([" "], vocab_size=12)
    with pytest.raises(DataValidationError, match="character alphabet"):
        BPETokenizer.train(["abcdefghi"], vocab_size=8)
    assert BPETokenizer.train(["a"], vocab_size=12).merges == []

    bad_merges = tmp_path / "bad-merges"
    bad_merges.mkdir()
    (bad_merges / "tokenizer.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "algorithm": "bpe",
                "vocabulary": {
                    "[PAD]": 0,
                    "[UNK]": 1,
                    "[CLS]": 2,
                    "[SEP]": 3,
                    "▁a": 4,
                },
                "merges": ["invalid"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ArtifactValidationError, match="merges"):
        LocalTokenizer.load(bad_merges)

    unsupported = tmp_path / "unsupported"
    unsupported.mkdir()
    (unsupported / "tokenizer.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "algorithm": "unrecognized",
                "vocabulary": {
                    "[PAD]": 0,
                    "[UNK]": 1,
                    "[CLS]": 2,
                    "[SEP]": 3,
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ArtifactValidationError, match="algorithm"):
        LocalTokenizer.load(unsupported)


def test_index_rejects_unsafe_or_inconsistent_state(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive"):
        VectorIndex(0)
    index = VectorIndex(2)
    with pytest.raises(ValueError, match="shape"):
        index.add(np.ones((1, 3)), [{"id": "x"}])
    with pytest.raises(ValueError, match="count"):
        index.add(np.ones((1, 2)), [])
    with pytest.raises(ValueError, match="finite"):
        index.add(np.array([[np.nan, 0.0]]), [{"id": "x"}])
    with pytest.raises(ValueError, match="string id"):
        index.add(np.ones((1, 2)), [{"id": 1}])
    with pytest.raises(ValueError, match="zero"):
        index.add(np.zeros((1, 2)), [{"id": "x"}])
    index.add(np.eye(2, dtype=np.float32), [{"id": "x"}, {"id": "y"}])
    with pytest.raises(ValueError, match="positive"):
        index.search(np.array([1.0, 0.0]), top_k=0)
    with pytest.raises(ValueError, match="shape"):
        index.search(np.ones(3))
    with pytest.raises(ValueError, match="finite"):
        index.search(np.array([np.nan, 0.0]))
    with pytest.raises(ValueError, match="zero"):
        index.search(np.zeros(2))
    directory = index.save(tmp_path / "index")
    with pytest.raises(ArtifactValidationError, match="overwrite"):
        index.save(directory)
    (directory / "metadata.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="checksum"):
        VectorIndex.load(directory)


def test_metric_and_miner_invalid_states_fail_instead_of_biasing_results() -> None:
    with pytest.raises(ValueError, match="query ranking"):
        retrieval_metrics({}, {})
    with pytest.raises(ValueError, match="identical"):
        retrieval_metrics({"q": []}, {"other": {"d"}})
    with pytest.raises(ValueError, match="no relevant"):
        retrieval_metrics({"q": []}, {"q": set()})
    with pytest.raises(ValueError, match="duplicate"):
        retrieval_metrics({"q": ["d", "d"]}, {"q": {"d"}})
    with pytest.raises(ValueError, match="constant"):
        similarity_metrics([1, 1], [0, 1])
    with pytest.raises(ValueError, match="missing query"):
        random_negatives(["q"], ["d"], {}, per_query=1)
    with pytest.raises(ValueError, match="eligible"):
        random_negatives(["q"], ["d"], {"q": {"d"}}, per_query=1)
    with pytest.raises(ValueError, match="row counts"):
        embedding_hard_negatives(
            ["q"],
            np.ones((2, 2)),
            ["d"],
            np.ones((1, 2)),
            {"q": set()},
        )
    with pytest.raises(ValueError, match="margin"):
        semi_hard_negatives([], {}, margin=0)


def test_trainer_validation_and_best_checkpoint_path(
    tmp_path: Path,
    model: EmbeddingModel,
    tokenizer: LocalTokenizer,
) -> None:
    pairs = [
        PairRecord(text_a="alpha beta", text_b="alpha gamma"),
        PairRecord(text_a="gamma delta", text_b="unicode café"),
    ]
    trainer = Trainer(
        model,
        tokenizer,
        TrainingConfig(
            epochs=1,
            batch_size=2,
            learning_rate=0.01,
            warmup_ratio=0,
            checkpoint_dir=tmp_path / "checkpoints",
            device="cpu",
        ),
    )
    result = trainer.train_pairs(pairs, validation_pairs=pairs)
    assert result.validation_loss is not None
    assert (tmp_path / "checkpoints" / "best.pt").is_file()
    incompatible = Trainer(
        model,
        tokenizer,
        TrainingConfig(
            objective="triplet",
            epochs=1,
            batch_size=2,
            checkpoint_dir=tmp_path / "other",
            device="cpu",
        ),
    )
    with pytest.raises(ValueError, match="incompatible"):
        incompatible.train_pairs(pairs)
