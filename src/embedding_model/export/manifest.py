"""Checksum manifest creation and path-safe validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from embedding_model.constants import ARTIFACT_SCHEMA_VERSION
from embedding_model.exceptions import ArtifactValidationError

MANIFEST_NAME = "manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(directory: Path, relative_files: list[str]) -> Path:
    """Write checksums after rejecting paths outside the artifact root."""

    root = directory.resolve()
    files: dict[str, dict[str, str | int]] = {}
    for relative_name in sorted(relative_files):
        path = (root / relative_name).resolve()
        if not path.is_relative_to(root):
            raise ArtifactValidationError(f"artifact path escapes root: {relative_name}")
        if not path.is_file():
            raise ArtifactValidationError(f"cannot manifest missing file: {relative_name}")
        files[relative_name] = {"sha256": sha256_file(path), "size": path.stat().st_size}
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "files": files,
    }
    output = root / MANIFEST_NAME
    output.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return output


def validate_manifest(directory: str | Path) -> dict[str, Any]:
    """Validate schema, required entries, paths, byte sizes, and SHA-256 checksums."""

    root = Path(directory).expanduser().resolve()
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ArtifactValidationError(f"missing artifact manifest: {manifest_path}")
    try:
        manifest: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactValidationError("artifact manifest is not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise ArtifactValidationError("artifact manifest root must be an object")
    if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactValidationError("unsupported artifact schema version")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ArtifactValidationError("artifact manifest files must be an object")
    required = {
        "config.json",
        "model.safetensors",
        "tokenizer/tokenizer.json",
        "training_metadata.json",
        "evaluation.json",
        "model_card.md",
    }
    missing = required - set(files)
    if missing:
        raise ArtifactValidationError(f"manifest is missing required files: {sorted(missing)}")
    for relative_name, expected in files.items():
        if not isinstance(relative_name, str) or not isinstance(expected, dict):
            raise ArtifactValidationError("invalid artifact manifest file entry")
        path = (root / relative_name).resolve()
        if not path.is_relative_to(root):
            raise ArtifactValidationError(f"manifest path escapes artifact root: {relative_name}")
        if not path.is_file():
            raise ArtifactValidationError(f"artifact file is missing: {relative_name}")
        if path.stat().st_size != expected.get("size"):
            raise ArtifactValidationError(f"artifact size mismatch: {relative_name}")
        if sha256_file(path) != expected.get("sha256"):
            raise ArtifactValidationError(f"artifact checksum mismatch: {relative_name}")
    return manifest
