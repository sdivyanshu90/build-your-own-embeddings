"""Environment-loadable serving limits and authentication configuration."""

from __future__ import annotations

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServingSettings(BaseSettings):
    """Bound every user-controlled resource before model execution."""

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        extra="forbid",
        case_sensitive=False,
    )

    max_batch_size: int = Field(default=64, ge=1, le=2048)
    max_text_length: int = Field(default=4096, ge=1, le=1_000_000)
    max_request_bytes: int = Field(default=1_000_000, ge=1024)
    max_top_k: int = Field(default=100, ge=1, le=10_000)
    concurrency_limit: int = Field(default=4, ge=1, le=1024)
    auth_token: SecretStr | None = None
    cors_origins: list[str] = Field(default_factory=list)
