"""Safe model artifact export and strict reload."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from embedding_model.config import ModelConfig
from embedding_model.exceptions import ArtifactValidationError
from embedding_model.export.manifest import validate_manifest, write_manifest
from embedding_model.modeling.embedding_model import EmbeddingModel
from embedding_model.tokenization import LocalTokenizer


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def export_model(
    model: EmbeddingModel,
    tokenizer: LocalTokenizer,
    output_dir: str | Path,
    *,
    training_metadata: dict[str, Any] | None = None,
    evaluation: dict[str, Any] | None = None,
) -> Path:
    """Export inspectable metadata and non-executable tensor weights."""

    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ArtifactValidationError(f"refusing to overwrite non-empty artifact: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if tokenizer.vocab_size != model.config.vocabulary_size:
        raise ArtifactValidationError("tokenizer vocabulary size does not match model")
    _write_json(output / "config.json", model.config.model_dump(mode="json"))
    tokenizer.save(output / "tokenizer")
    state = {
        name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()
    }
    save_file(state, output / "model.safetensors")
    _write_json(output / "training_metadata.json", training_metadata or {})
    _write_json(output / "evaluation.json", evaluation or {})
    (output / "model_card.md").write_text(
        "# Local Transformer Embedding Model\n\n"
        "This artifact was produced by a tiny-capable training system. Metrics in "
        "`evaluation.json` are only as representative as the supplied data. Do not "
        "infer production semantic quality from synthetic training.\n",
        encoding="utf-8",
    )
    files = [
        "config.json",
        "model.safetensors",
        "tokenizer/tokenizer.json",
        "training_metadata.json",
        "evaluation.json",
        "model_card.md",
    ]
    write_manifest(output, files)
    return output


def load_exported_model(
    artifact_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[EmbeddingModel, LocalTokenizer]:
    """Verify an artifact fully before loading its non-executable tensor file."""

    root = Path(artifact_dir).expanduser().resolve()
    validate_manifest(root)
    try:
        config_payload: Any = json.loads((root / "config.json").read_text(encoding="utf-8"))
        config = ModelConfig.model_validate(config_payload)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ArtifactValidationError("model config is invalid") from exc
    tokenizer = LocalTokenizer.load(root / "tokenizer")
    if tokenizer.vocab_size != config.vocabulary_size:
        raise ArtifactValidationError("tokenizer and model vocabulary sizes do not match")
    model = EmbeddingModel(config)
    try:
        weights = load_file(root / "model.safetensors", device="cpu")
        model.load_state_dict(weights, strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ArtifactValidationError("model tensor names or shapes are invalid") from exc
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise ArtifactValidationError("CUDA was requested but is unavailable")
    model.to(selected_device)
    model.eval()
    return model, tokenizer
