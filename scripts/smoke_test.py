"""Run the public tiny lifecycle through subprocess-visible project APIs."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from embedding_model.config import ModelConfig, TrainingConfig
from embedding_model.data.readers import read_records
from embedding_model.data.schemas import PairRecord
from embedding_model.export.exporter import export_model
from embedding_model.indexing.faiss_index import VectorIndex
from embedding_model.inference.embedder import TextEmbedder
from embedding_model.modeling.embedding_model import EmbeddingModel
from embedding_model.tokenization import LocalTokenizer
from embedding_model.training.trainer import Trainer


def main() -> None:
    pairs = read_records("data/sample_pairs.jsonl", PairRecord)[:4]
    texts = [text for pair in pairs for text in (pair.text_a, pair.text_b)]
    tokenizer = LocalTokenizer.train(texts, vocab_size=64)
    model = EmbeddingModel(
        ModelConfig(
            vocabulary_size=tokenizer.vocab_size,
            hidden_size=16,
            embedding_dimension=8,
            num_attention_heads=2,
            num_hidden_layers=1,
            intermediate_size=32,
            max_sequence_length=24,
            dropout=0.0,
        )
    )
    with tempfile.TemporaryDirectory(prefix="embedding-smoke-") as temporary:
        root = Path(temporary)
        result = Trainer(
            model,
            tokenizer,
            TrainingConfig(
                epochs=1,
                batch_size=4,
                learning_rate=0.005,
                warmup_ratio=0,
                checkpoint_dir=root / "checkpoints",
                device="cpu",
            ),
        ).train_pairs(pairs)
        export_model(model, tokenizer, root / "model")
        embedder = TextEmbedder.from_pretrained(root / "model")
        vectors = np.asarray(embedder.encode(texts[:3]))
        index = VectorIndex(embedder.dimension)
        index.add(vectors, [{"id": str(index)} for index in range(len(vectors))])
        assert index.search(vectors[0], top_k=1)[0].document_id == "0"
        assert result.last_checkpoint.is_file()
    print("smoke test passed")


if __name__ == "__main__":
    main()
