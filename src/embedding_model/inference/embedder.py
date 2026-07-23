"""Stable, thread-safe public text encoding API."""

from __future__ import annotations

import threading
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch

from embedding_model.export.exporter import load_exported_model
from embedding_model.modeling.embedding_model import EmbeddingModel
from embedding_model.tokenization import LocalTokenizer


class TextEmbedder:
    """Load an exported model and encode strings in stable input order."""

    def __init__(
        self,
        model: EmbeddingModel,
        tokenizer: LocalTokenizer,
        *,
        device: str | torch.device = "cpu",
        model_name: str = "local-transformer-embedding",
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.model_name = model_name
        self._lock = threading.RLock()

    @property
    def dimension(self) -> int:
        return self.model.embedding_dimension

    @classmethod
    def from_pretrained(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> TextEmbedder:
        """Load a checksum-verified local artifact."""

        model, tokenizer = load_exported_model(artifact_dir, device=device)
        return cls(model, tokenizer, device=device)

    def encode(
        self,
        texts: str | Iterable[str],
        *,
        batch_size: int = 32,
        normalize: bool | None = None,
        convert_to_tensor: bool = False,
    ) -> npt.NDArray[np.float32] | torch.Tensor:
        """Return float32 shape ``(count, dimension)``; empty iterables are valid."""

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        values = [texts] if isinstance(texts, str) else list(texts)
        if any(not isinstance(text, str) for text in values):
            raise TypeError("every input must be a string")
        if not values:
            empty = torch.empty((0, self.dimension), dtype=torch.float32)
            return empty if convert_to_tensor else np.asarray(empty.numpy(), dtype=np.float32)
        outputs: list[torch.Tensor] = []
        with self._lock, torch.inference_mode():
            self.model.eval()
            for offset in range(0, len(values), batch_size):
                batch = values[offset : offset + batch_size]
                encoded = self.tokenizer.batch_encode(
                    batch,
                    self.model.config.max_sequence_length,
                    device=self.device,
                )
                result = self.model(**encoded, normalize=normalize)
                outputs.append(result.detach().cpu())
        tensor = torch.cat(outputs, dim=0).to(dtype=torch.float32)
        if not torch.isfinite(tensor).all():
            raise FloatingPointError("encoding produced non-finite output")
        return tensor if convert_to_tensor else np.asarray(tensor.numpy(), dtype=np.float32)
