"""FastAPI inference service."""

from embedding_model.serving.app import create_app

__all__ = ["create_app"]
