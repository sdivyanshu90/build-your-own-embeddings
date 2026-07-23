"""Versioned cosine/IP vector index with FAISS acceleration when installed."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from embedding_model.constants import INDEX_SCHEMA_VERSION
from embedding_model.exceptions import ArtifactValidationError

try:
    import faiss as _faiss
except ImportError:  # pragma: no cover - exercised in minimal installations
    _faiss = None

faiss: Any = _faiss


@dataclass(frozen=True)
class SearchResult:
    document_id: str
    score: float
    metadata: dict[str, Any]


class VectorIndex:
    """Exact inner-product index for normalized embeddings."""

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("index dimension must be positive")
        self.dimension = dimension
        self._vectors: npt.NDArray[np.float32] = np.empty((0, dimension), dtype=np.float32)
        self._metadata: list[dict[str, Any]] = []
        self._faiss_index = faiss.IndexFlatIP(dimension) if faiss is not None else None

    @property
    def size(self) -> int:
        return len(self._metadata)

    def add(
        self,
        vectors: npt.NDArray[Any],
        metadata: list[dict[str, Any]],
    ) -> None:
        """Validate, normalize, and append vectors with unique string IDs."""

        values = np.asarray(vectors, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError(
                f"vectors must have shape (count, {self.dimension}), got {values.shape}"
            )
        if values.shape[0] != len(metadata):
            raise ValueError("metadata count must equal vector count")
        if not np.isfinite(values).all():
            raise ValueError("index vectors must be finite")
        incoming_ids = []
        for item in metadata:
            document_id = item.get("id")
            if not isinstance(document_id, str) or not document_id:
                raise ValueError("every metadata item requires a non-empty string id")
            incoming_ids.append(document_id)
        existing_ids = {str(item["id"]) for item in self._metadata}
        if len(incoming_ids) != len(set(incoming_ids)) or existing_ids & set(incoming_ids):
            raise ValueError("document IDs must be unique")
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise ValueError("zero vectors cannot be indexed for cosine search")
        normalized = np.ascontiguousarray(values / norms, dtype=np.float32)
        self._vectors = np.concatenate((self._vectors, normalized), axis=0)
        self._metadata.extend(dict(item) for item in metadata)
        if self._faiss_index is not None:
            self._faiss_index.add(normalized)

    def search(self, query: npt.NDArray[Any], top_k: int = 10) -> list[SearchResult]:
        """Return descending scores; insertion order breaks equal-score ties."""

        if self.size == 0:
            raise ValueError("cannot search an empty index")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        value = np.asarray(query, dtype=np.float32)
        if value.ndim == 2 and value.shape[0] == 1:
            value = value[0]
        if value.shape != (self.dimension,):
            raise ValueError(f"query must have shape ({self.dimension},)")
        if not np.isfinite(value).all():
            raise ValueError("query vector must be finite")
        norm = float(np.linalg.norm(value))
        if norm <= 1e-12:
            raise ValueError("zero query vectors cannot be searched")
        value = np.ascontiguousarray(value / norm, dtype=np.float32)
        if self._faiss_index is not None:
            raw_scores, raw_indices = self._faiss_index.search(value.reshape(1, -1), self.size)
            ranked = sorted(
                zip(raw_indices[0].tolist(), raw_scores[0].tolist(), strict=True),
                key=lambda item: (-item[1], item[0]),
            )
            selected = ranked[: min(top_k, self.size)]
        else:
            scores = self._vectors @ value
            indices = np.lexsort((np.arange(self.size), -scores))[: min(top_k, self.size)]
            selected = [(int(index), float(scores[index])) for index in indices]
        return [
            SearchResult(
                document_id=str(self._metadata[index]["id"]),
                score=score,
                metadata=dict(self._metadata[index]),
            )
            for index, score in selected
        ]

    def save(self, directory: str | Path) -> Path:
        """Save vectors without pickle plus checksummed JSON metadata."""

        output = Path(directory).expanduser().resolve()
        if output.exists() and any(output.iterdir()):
            raise ArtifactValidationError(f"refusing to overwrite non-empty index: {output}")
        output.mkdir(parents=True, exist_ok=True)
        vector_path = output / "vectors.npy"
        metadata_path = output / "metadata.json"
        np.save(vector_path, self._vectors, allow_pickle=False)
        metadata_path.write_text(
            json.dumps(self._metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "dimension": self.dimension,
            "size": self.size,
            "files": {
                "vectors.npy": _sha256(vector_path),
                "metadata.json": _sha256(metadata_path),
            },
        }
        (output / "index_manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        return output

    @classmethod
    def load(cls, directory: str | Path) -> VectorIndex:
        """Validate checksums, shapes, and metadata before rebuilding the index."""

        root = Path(directory).expanduser().resolve()
        manifest_path = root / "index_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != INDEX_SCHEMA_VERSION:
                raise ArtifactValidationError("unsupported index schema version")
            for filename, digest in manifest["files"].items():
                path = (root / filename).resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    raise ArtifactValidationError(f"invalid index file path: {filename}")
                if _sha256(path) != digest:
                    raise ArtifactValidationError(f"index checksum mismatch: {filename}")
            vectors = np.load(root / "vectors.npy", allow_pickle=False)
            metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
            if not isinstance(metadata, list):
                raise ArtifactValidationError("index metadata must be a list")
            index = cls(int(manifest["dimension"]))
            index.add(vectors, metadata)
            if index.size != int(manifest["size"]):
                raise ArtifactValidationError("index size does not match manifest")
            return index
        except ArtifactValidationError:
            raise
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ArtifactValidationError(f"invalid index artifact: {root}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
