"""A tiny, fully local transformer encoder embedding model."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as functional
from torch import nn

from embedding_model.config import ModelConfig
from embedding_model.modeling.pooling import Pooling


class EmbeddingModel(nn.Module):
    """Encode padded token IDs into fixed-width finite embeddings."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(
            config.vocabulary_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.position_embeddings = nn.Embedding(config.max_sequence_length, config.hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_hidden_layers)
        self.pooling = Pooling(config.pooling)
        self.projection: nn.Module
        if config.projection:
            self.projection = nn.Linear(config.hidden_size, config.embedding_dimension)
        else:
            self.projection = nn.Identity()
        self.output_normalization: nn.Module
        if config.layer_normalization:
            self.output_normalization = nn.LayerNorm(config.embedding_dimension)
        else:
            self.output_normalization = nn.Identity()

    @property
    def embedding_dimension(self) -> int:
        return self.config.embedding_dimension

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        normalize: bool | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must share shape (batch, sequence)")
        if input_ids.shape[1] > self.config.max_sequence_length:
            raise ValueError("input sequence exceeds configured max_sequence_length")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        hidden = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        hidden = self.encoder(hidden, src_key_padding_mask=attention_mask == 0)
        embeddings = self.output_normalization(
            self.projection(self.pooling(hidden, attention_mask))
        )
        should_normalize = self.config.normalize_embeddings if normalize is None else normalize
        if should_normalize:
            embeddings = functional.normalize(embeddings, p=2, dim=-1, eps=1e-12)
        if not torch.isfinite(embeddings).all():
            raise FloatingPointError("model produced non-finite embeddings")
        return cast(torch.Tensor, embeddings)
