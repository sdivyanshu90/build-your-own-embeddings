from __future__ import annotations

import asyncio

import httpx
import numpy as np
import pytest

from embedding_model.indexing.faiss_index import VectorIndex
from embedding_model.inference.embedder import TextEmbedder
from embedding_model.serving.app import create_app
from embedding_model.serving.settings import ServingSettings

pytestmark = pytest.mark.integration


def test_api_similarity_version_metrics_and_not_ready(embedder: TextEmbedder) -> None:
    vector_index = VectorIndex(embedder.dimension)
    vector_index.add(np.asarray(embedder.encode(["alpha"])), [{"id": "document"}])
    ready_app = create_app(
        embedder=embedder,
        index=vector_index,
        settings=ServingSettings(max_text_length=10, max_top_k=2),
    )
    not_ready_app = create_app(settings=ServingSettings())
    asyncio.run(_exercise_routes(ready_app, not_ready_app))


async def _exercise_routes(ready_app: object, not_ready_app: object) -> None:
    ready_transport = httpx.ASGITransport(app=ready_app)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=ready_transport, base_url="http://test") as client:
        version = await client.get("/version")
        assert version.status_code == 200
        assert "version" in version.json()
        model_metadata = await client.get("/v1/model")
        assert model_metadata.status_code == 200
        assert model_metadata.json()["dimension"] > 0
        metrics = await client.get("/metrics")
        assert metrics.status_code == 200
        assert "embedding_model_ready" in metrics.text
        similarity = await client.post(
            "/v1/similarity",
            json={"texts_a": ["alpha beta"], "texts_b": ["alpha beta"]},
        )
        assert similarity.status_code == 200
        assert similarity.json()["similarities"][0] == pytest.approx(1.0, abs=1e-5)
        mismatch = await client.post(
            "/v1/similarity",
            json={"texts_a": ["a"], "texts_b": ["a", "b"]},
        )
        assert mismatch.status_code == 400
        blank = await client.post("/v1/embeddings", json={"texts": [" "]})
        assert blank.status_code == 400
        too_long = await client.post("/v1/embeddings", json={"texts": ["x" * 11]})
        assert too_long.status_code == 400
        too_many_hits = await client.post("/v1/search", json={"query": "alpha", "top_k": 3})
        assert too_many_hits.status_code == 400
        invalid_length = await client.get("/health/live", headers={"content-length": "invalid"})
        assert invalid_length.status_code == 400

    not_ready_transport = httpx.ASGITransport(app=not_ready_app)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=not_ready_transport, base_url="http://test") as client:
        readiness = await client.get("/health/ready")
        assert readiness.status_code == 503
        response = await client.post("/v1/embeddings", json={"texts": ["alpha"]})
        assert response.status_code == 400
