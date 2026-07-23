"""Public package interface for the embedding system."""

from embedding_model.constants import __version__
from embedding_model.inference.embedder import TextEmbedder

__all__ = ["TextEmbedder", "__version__"]
