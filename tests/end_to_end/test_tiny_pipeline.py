from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import numpy as np
import pytest

from embedding_model.config import ModelConfig, TrainingConfig
from embedding_model.data.schemas import PairRecord
from embedding_model.evaluation.retrieval import retrieval_metrics
from embedding_model.export.exporter import export_model
from embedding_model.indexing.faiss_index import VectorIndex
from embedding_model.inference.embedder import TextEmbedder
from embedding_model.modeling.embedding_model import EmbeddingModel
from embedding_model.serving.app import create_app
from embedding_model.serving.settings import ServingSettings
from embedding_model.tokenization import LocalTokenizer
from embedding_model.training.trainer import Trainer


@pytest.mark.end_to_end
def test_tiny_cpu_pipeline_crosses_every_public_boundary(tmp_path: Path) -> None:
    pairs = [
        PairRecord(text_a="cat sits on mat", text_b="kitten rests on rug"),
        PairRecord(text_a="dog runs outside", text_b="puppy plays outdoors"),
        PairRecord(text_a="blue ocean water", text_b="sea water is blue"),
        PairRecord(text_a="fresh bread recipe", text_b="how to bake bread"),
    ]
    all_texts = [text for pair in pairs for text in (pair.text_a, pair.text_b)]
    tokenizer = LocalTokenizer.train(all_texts, vocab_size=64)
    model_config = ModelConfig(
        vocabulary_size=tokenizer.vocab_size,
        hidden_size=16,
        embedding_dimension=8,
        num_attention_heads=2,
        num_hidden_layers=1,
        intermediate_size=32,
        max_sequence_length=16,
        dropout=0.0,
    )
    training_config = TrainingConfig(
        epochs=1,
        batch_size=2,
        learning_rate=0.01,
        warmup_ratio=0.0,
        checkpoint_dir=tmp_path / "checkpoints",
        seed=7,
    )

    model = EmbeddingModel(model_config)
    trainer = Trainer(model, tokenizer, training_config)
    result = trainer.train_pairs(pairs)
    assert np.isfinite(result.training_loss)
    assert result.global_step == 2
    assert result.last_checkpoint.is_file()

    resumed_model = EmbeddingModel(model_config)
    resumed = Trainer(resumed_model, tokenizer, training_config)
    resumed.resume(result.last_checkpoint)
    assert resumed.global_step == result.global_step

    artifact_dir = tmp_path / "model"
    export_model(
        resumed_model,
        tokenizer,
        artifact_dir,
        training_metadata={"global_step": resumed.global_step},
        evaluation={"training_loss": result.training_loss},
    )
    embedder = TextEmbedder.from_pretrained(artifact_dir)
    corpus = ["kitten rests on rug", "puppy plays outdoors", "sea water is blue"]
    vectors = embedder.encode(corpus)
    assert vectors.shape == (3, 8)
    assert np.isfinite(vectors).all()
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)

    index = VectorIndex(dimension=8)
    index.add(vectors, [{"id": str(i), "text": text} for i, text in enumerate(corpus)])
    index_dir = tmp_path / "index"
    index.save(index_dir)
    loaded_index = VectorIndex.load(index_dir)
    hits = loaded_index.search(embedder.encode("kitten on mat"), top_k=2)
    assert len(hits) == 2
    assert hits[0].score >= hits[1].score

    metrics = retrieval_metrics(
        rankings={"q1": [hit.document_id for hit in hits]},
        relevant={"q1": {hits[0].document_id}},
        k_values=(1, 2),
    )
    assert metrics["recall@1"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(1.0)

    app = create_app(
        embedder=embedder,
        index=loaded_index,
        settings=ServingSettings(max_batch_size=4, max_text_length=100),
    )
    asyncio.run(_exercise_api(app))


async def _exercise_api(app: object) -> None:
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get("/health/live")).status_code == 200
        assert (await client.get("/health/ready")).json()["ready"] is True
        response = await client.post("/v1/embeddings", json={"texts": ["hello world"]})
        assert response.status_code == 200
        assert response.json()["dimension"] == 8
        search = await client.post("/v1/search", json={"query": "ocean", "top_k": 2})
        assert search.status_code == 200
        assert len(search.json()["results"]) == 2
