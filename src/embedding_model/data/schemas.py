"""Strict schemas for supported embedding objectives."""

from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)


class RecordModel(BaseModel):
    """Common strict and immutable record behavior."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("*", mode="before")
    @classmethod
    def reject_nulls(cls, value: Any, info: ValidationInfo) -> Any:
        if value is None and info.field_name != "record_id":
            raise ValueError("null values are not allowed")
        return value


class PairRecord(RecordModel):
    """Two semantically related texts for in-batch contrastive training."""

    text_a: str
    text_b: str
    record_id: str | None = None

    @field_validator("text_a", "text_b", "record_id")
    @classmethod
    def normalize_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("text fields must not be empty or whitespace")
        return normalized


class TripletRecord(RecordModel):
    """Anchor, positive, and known-negative texts."""

    anchor: str
    positive: str
    negative: str
    record_id: str | None = None

    @field_validator("anchor", "positive", "negative", "record_id")
    @classmethod
    def normalize_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("text fields must not be empty or whitespace")
        return normalized

    @model_validator(mode="after")
    def negative_must_differ(self) -> TripletRecord:
        if self.negative == self.positive:
            raise ValueError("negative text must differ from the positive text")
        return self


class ScoredPairRecord(RecordModel):
    """Text pair with a continuous cosine-similarity target."""

    text_a: str
    text_b: str
    score: float = Field(ge=-1.0, le=1.0)
    record_id: str | None = None

    @field_validator("text_a", "text_b", "record_id")
    @classmethod
    def normalize_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("text fields must not be empty or whitespace")
        return normalized


class RetrievalRecord(RecordModel):
    """Query with one or more positives and optional known negatives."""

    query_id: str
    query: str
    positive_documents: list[str] = Field(min_length=1)
    negative_documents: list[str] = Field(default_factory=list)

    @field_validator("query_id", "query")
    @classmethod
    def normalize_scalar(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("query fields must not be empty or whitespace")
        return normalized

    @field_validator("positive_documents", "negative_documents")
    @classmethod
    def normalize_documents(cls, documents: list[str]) -> list[str]:
        normalized = [" ".join(document.split()) for document in documents]
        if any(not document for document in normalized):
            raise ValueError("documents must not be empty or whitespace")
        if len(normalized) != len(set(normalized)):
            raise ValueError("duplicate documents are not allowed within a record")
        return normalized

    @model_validator(mode="after")
    def prevent_false_negatives(self) -> RetrievalRecord:
        overlap = set(self.positive_documents) & set(self.negative_documents)
        if overlap:
            raise ValueError("known positive documents cannot be used as negatives")
        return self
