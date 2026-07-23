"""Minimal structured JSON logging with sensitive-value redaction."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

SENSITIVE_KEYS = {"api_key", "auth_token", "password", "secret", "token"}


class JsonFormatter(logging.Formatter):
    """Serialize bounded operational fields; callers must not pass raw text."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        fields = getattr(record, "fields", {})
        if isinstance(fields, dict):
            payload.update(redact_mapping(fields))
        if record.exc_info and record.exc_info[0] is not None:
            payload["exception_type"] = record.exc_info[0].__name__
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def configure_logging(level: str = "INFO") -> None:
    """Configure the package root logger without changing import-time state."""

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("embedding_model")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(level.upper())
    logger.propagate = False


def redact_mapping(values: dict[str, Any]) -> dict[str, Any]:
    """Recursively redact keys that conventionally carry credentials."""

    output: dict[str, Any] = {}
    for key, value in values.items():
        if any(sensitive in key.lower() for sensitive in SENSITIVE_KEYS):
            output[key] = "[REDACTED]"
        elif isinstance(value, dict):
            output[key] = redact_mapping(value)
        else:
            output[key] = value
    return output
