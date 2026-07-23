"""HTTP request and response contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmbeddingsRequest(ApiModel):
    texts: list[str] = Field(min_length=1)
    normalize: bool = True


class EmbeddingsResponse(ApiModel):
    model: str
    dimension: int
    count: int
    embeddings: list[list[float]]


class SimilarityRequest(ApiModel):
    texts_a: list[str] = Field(min_length=1)
    texts_b: list[str] = Field(min_length=1)


class SimilarityResponse(ApiModel):
    count: int
    similarities: list[float]


class SearchRequest(ApiModel):
    query: str
    top_k: int = Field(default=10, ge=1)

    @field_validator("query")
    @classmethod
    def query_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class SearchItem(ApiModel):
    id: str
    score: float
    metadata: dict[str, Any]


class SearchResponse(ApiModel):
    count: int
    results: list[SearchItem]
