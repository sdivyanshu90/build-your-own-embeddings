"""Hardened FastAPI application factory."""

from __future__ import annotations

import asyncio
import hmac
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from functools import partial

import numpy as np
from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.types import Message, Receive, Scope, Send

from embedding_model.constants import __version__
from embedding_model.exceptions import EmbeddingModelError
from embedding_model.indexing.faiss_index import VectorIndex
from embedding_model.inference.embedder import TextEmbedder
from embedding_model.serving.metrics import create_metrics
from embedding_model.serving.schemas import (
    EmbeddingsRequest,
    EmbeddingsResponse,
    SearchItem,
    SearchRequest,
    SearchResponse,
    SimilarityRequest,
    SimilarityResponse,
)
from embedding_model.serving.settings import ServingSettings

logger = logging.getLogger("embedding_model.serving")

KNOWN_ROUTES = {
    "/health/live",
    "/health/ready",
    "/version",
    "/v1/model",
    "/metrics",
    "/v1/embeddings",
    "/v1/similarity",
    "/v1/search",
}


def create_app(
    *,
    embedder: TextEmbedder | None = None,
    index: VectorIndex | None = None,
    settings: ServingSettings | None = None,
) -> FastAPI:
    """Create an isolated app; dependencies may be injected for tests or startup."""

    active_settings = settings or ServingSettings()
    metrics = create_metrics()
    metrics.model_ready.set(float(embedder is not None))
    metrics.index_size.set(0 if index is None else index.size)
    semaphore = asyncio.Semaphore(active_settings.concurrency_limit)
    application = FastAPI(title="Text Embedding Service", version=__version__)
    application.state.embedder = embedder
    application.state.index = index
    application.state.settings = active_settings
    application.state.metrics = metrics
    application.state.semaphore = semaphore
    application.add_middleware(
        RequestSizeLimitMiddleware, max_bytes=active_settings.max_request_bytes
    )

    if active_settings.cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=active_settings.cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        )

    @application.middleware("http")
    async def safeguards(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id[:128]
        started = time.monotonic()
        route = request.url.path if request.url.path in KNOWN_ROUTES else "other"
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > active_settings.max_request_bytes:
                    return _error(413, "request_too_large", request.state.request_id)
            except ValueError:
                return _error(400, "invalid_content_length", request.state.request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        metrics.requests.labels(route=route, status=str(response.status_code)).inc()
        elapsed = time.monotonic() - started
        metrics.duration.labels(route=route).observe(elapsed)
        logger.info(
            "request_complete",
            extra={
                "fields": {
                    "request_id": request.state.request_id,
                    "route": route,
                    "method": request.method,
                    "status": response.status_code,
                    "duration_seconds": elapsed,
                }
            },
        )
        return response

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, _: RequestValidationError) -> JSONResponse:
        metrics.failures.labels(category="validation").inc()
        return _error(422, "request_validation_failed", request.state.request_id)

    @application.exception_handler(EmbeddingModelError)
    @application.exception_handler(ValueError)
    async def expected_error(request: Request, _: Exception) -> JSONResponse:
        metrics.failures.labels(category="input").inc()
        return _error(400, "invalid_request", request.state.request_id)

    @application.exception_handler(Exception)
    async def unexpected_error(request: Request, _: Exception) -> JSONResponse:
        metrics.failures.labels(category="internal").inc()
        return _error(500, "internal_error", request.state.request_id)

    async def authenticate(authorization: str | None = Header(default=None)) -> None:
        expected = active_settings.auth_token
        if expected is None:
            return
        presented = ""
        if authorization and authorization.startswith("Bearer "):
            presented = authorization.removeprefix("Bearer ")
        if not hmac.compare_digest(presented, expected.get_secret_value()):
            raise AuthenticationError

    @application.exception_handler(AuthenticationError)
    async def authentication_error(request: Request, _: Exception) -> JSONResponse:
        metrics.failures.labels(category="authentication").inc()
        return _error(401, "authentication_required", request.state.request_id)

    @application.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "alive"}

    @application.get("/health/ready")
    async def ready() -> JSONResponse:
        is_ready = application.state.embedder is not None
        return JSONResponse(
            status_code=200 if is_ready else 503,
            content={"ready": is_ready},
        )

    @application.get("/version")
    async def version() -> dict[str, str]:
        return {"version": __version__}

    @application.get("/v1/model")
    async def model_metadata() -> dict[str, str | int | bool]:
        runtime = _require_embedder(application)
        return {
            "name": runtime.model_name,
            "dimension": runtime.dimension,
            "max_sequence_length": runtime.model.config.max_sequence_length,
            "pooling": runtime.model.config.pooling,
            "normalized_by_default": runtime.model.config.normalize_embeddings,
        }

    @application.get("/metrics")
    async def prometheus_metrics() -> PlainTextResponse:
        return PlainTextResponse(
            generate_latest(metrics.registry).decode("utf-8"),
            media_type=CONTENT_TYPE_LATEST,
        )

    @application.post(
        "/v1/embeddings",
        response_model=EmbeddingsResponse,
        dependencies=[Depends(authenticate)],
    )
    async def embeddings(payload: EmbeddingsRequest) -> EmbeddingsResponse:
        runtime = _require_embedder(application)
        _validate_texts(payload.texts, active_settings)
        async with semaphore:
            vectors = partial(runtime.encode, payload.texts, normalize=payload.normalize)()
        metrics.texts.inc(len(payload.texts))
        return EmbeddingsResponse(
            model=runtime.model_name,
            dimension=runtime.dimension,
            count=len(payload.texts),
            embeddings=np.asarray(vectors).tolist(),
        )

    @application.post(
        "/v1/similarity",
        response_model=SimilarityResponse,
        dependencies=[Depends(authenticate)],
    )
    async def similarity(payload: SimilarityRequest) -> SimilarityResponse:
        runtime = _require_embedder(application)
        if len(payload.texts_a) != len(payload.texts_b):
            raise ValueError("similarity lists must have equal lengths")
        _validate_texts([*payload.texts_a, *payload.texts_b], active_settings)
        async with semaphore:
            left = runtime.encode(payload.texts_a)
            right = runtime.encode(payload.texts_b)
        scores = (np.asarray(left) * np.asarray(right)).sum(axis=1)
        metrics.texts.inc(len(payload.texts_a) + len(payload.texts_b))
        return SimilarityResponse(count=len(scores), similarities=scores.tolist())

    @application.post(
        "/v1/search",
        response_model=SearchResponse,
        dependencies=[Depends(authenticate)],
    )
    async def search(payload: SearchRequest) -> SearchResponse:
        runtime = _require_embedder(application)
        active_index: VectorIndex | None = application.state.index
        if active_index is None:
            raise ValueError("search index is not loaded")
        _validate_texts([payload.query], active_settings)
        if payload.top_k > active_settings.max_top_k:
            raise ValueError("top_k exceeds configured limit")
        async with semaphore:
            vector = np.asarray(runtime.encode(payload.query), dtype=np.float32)
            hits = active_index.search(vector, top_k=payload.top_k)
        metrics.texts.inc()
        return SearchResponse(
            count=len(hits),
            results=[
                SearchItem(
                    id=hit.document_id,
                    score=hit.score,
                    metadata=hit.metadata,
                )
                for hit in hits
            ],
        )

    return application


class AuthenticationError(Exception):
    """Internal signal intentionally mapped to a generic 401."""


class RequestSizeLimitMiddleware:
    """Buffer at most the configured body size, including chunked requests."""

    def __init__(
        self,
        app: Callable[
            [Scope, Receive, Send],
            Awaitable[None],
        ],
        *,
        max_bytes: int,
    ) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                await self.app(scope, _single_message_receiver(message), send)
                return
            body.extend(message.get("body", b""))
            if len(body) > self.max_bytes:
                headers = dict(scope.get("headers", []))
                request_id = headers.get(b"x-request-id", b"").decode("utf-8", errors="ignore")
                request_id = request_id[:128] or str(uuid.uuid4())
                await _error(413, "request_too_large", request_id)(scope, receive, send)
                return
            if not message.get("more_body", False):
                break
        replay = _single_message_receiver(
            {"type": "http.request", "body": bytes(body), "more_body": False}
        )
        await self.app(scope, replay, send)


def _require_embedder(application: FastAPI) -> TextEmbedder:
    embedder: TextEmbedder | None = application.state.embedder
    if embedder is None:
        raise ValueError("model is not ready")
    return embedder


def _validate_texts(texts: list[str], settings: ServingSettings) -> None:
    if len(texts) > settings.max_batch_size:
        raise ValueError("batch exceeds configured limit")
    if any(not text.strip() for text in texts):
        raise ValueError("blank text is invalid")
    if any(len(text) > settings.max_text_length for text in texts):
        raise ValueError("text exceeds configured length limit")


def _error(status: int, code: str, request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "request_id": request_id}},
        headers={"X-Request-ID": request_id},
    )


def _single_message_receiver(message: Message) -> Receive:
    delivered = False

    async def receive() -> Message:
        nonlocal delivered
        if delivered:
            await asyncio.Event().wait()
            return {"type": "http.disconnect"}
        delivered = True
        return message

    return receive
