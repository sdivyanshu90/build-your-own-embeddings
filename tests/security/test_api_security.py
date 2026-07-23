from __future__ import annotations

import asyncio
import logging

import httpx
import pytest
from pydantic import SecretStr

from embedding_model.inference.embedder import TextEmbedder
from embedding_model.serving.app import create_app
from embedding_model.serving.settings import ServingSettings

pytestmark = [pytest.mark.security, pytest.mark.integration]


def test_api_enforces_auth_limits_and_safe_errors(
    embedder: TextEmbedder,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    settings = ServingSettings(
        max_batch_size=2,
        max_text_length=20,
        max_request_bytes=1024,
        auth_token=SecretStr("correct-token"),
    )
    app = create_app(embedder=embedder, settings=settings)
    asyncio.run(_assert_api_security(app))
    assert "raw-sensitive-user-text" not in caplog.text


async def _assert_api_security(app: object) -> None:
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        unauthorized = await client.post(
            "/v1/embeddings",
            json={"texts": ["raw-sensitive-user-text"]},
        )
        assert unauthorized.status_code == 401
        assert "raw-sensitive-user-text" not in unauthorized.text

        headers = {"Authorization": "Bearer correct-token"}
        invalid = await client.post(
            "/v1/embeddings",
            json={"texts": "raw-sensitive-user-text"},
            headers=headers,
        )
        assert invalid.status_code == 422
        assert "raw-sensitive-user-text" not in invalid.text

        oversized = await client.post(
            "/v1/embeddings",
            content=b"x" * 1025,
            headers={**headers, "content-type": "application/json"},
        )
        assert oversized.status_code == 413
        assert "request_id" in oversized.json()["error"]

        too_many = await client.post(
            "/v1/embeddings",
            json={"texts": ["one", "two", "three"]},
            headers=headers,
        )
        assert too_many.status_code == 400
        assert "traceback" not in too_many.text.lower()
