from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from embedding_model.exceptions import ArtifactValidationError
from embedding_model.export.manifest import validate_manifest
from embedding_model.indexing.faiss_index import VectorIndex
from embedding_model.inference.embedder import TextEmbedder

pytestmark = pytest.mark.unit


def test_embedder_contract_for_string_generator_empty_and_tensor(
    embedder: TextEmbedder,
) -> None:
    single = embedder.encode("alpha beta")
    generated = embedder.encode(text for text in ["alpha beta", "gamma delta"])
    empty = embedder.encode([])
    tensor = embedder.encode(["alpha"], convert_to_tensor=True)
    assert single.shape == (1, embedder.dimension)
    assert generated.shape == (2, embedder.dimension)
    assert empty.shape == (0, embedder.dimension)
    assert tensor.shape == (1, embedder.dimension)
    assert isinstance(tensor, torch.Tensor)
    np.testing.assert_allclose(np.linalg.norm(generated, axis=1), 1.0, atol=1e-5)


def test_index_validates_dimensions_duplicates_empty_and_stable_ties(tmp_path: Path) -> None:
    index = VectorIndex(2)
    with pytest.raises(ValueError, match="empty"):
        index.search(np.array([1.0, 0.0]))
    index.add(
        np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        [{"id": "first"}, {"id": "second"}],
    )
    assert [hit.document_id for hit in index.search(np.array([1.0, 0.0]))] == [
        "first",
        "second",
    ]
    with pytest.raises(ValueError, match="unique"):
        index.add(np.array([[0.0, 1.0]]), [{"id": "first"}])
    directory = index.save(tmp_path / "index")
    loaded = VectorIndex.load(directory)
    assert loaded.size == 2


def test_artifact_checksum_detects_tampering(artifact_dir: Path) -> None:
    validate_manifest(artifact_dir)
    config = artifact_dir / "config.json"
    config.write_text(config.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="size mismatch"):
        validate_manifest(artifact_dir)


def test_manifest_rejects_traversal_entry(artifact_dir: Path) -> None:
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["../outside"] = {"sha256": "0", "size": 0}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="escapes"):
        validate_manifest(artifact_dir)
