"""Atomic, versioned local-training checkpoints."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import torch

from embedding_model.constants import CHECKPOINT_SCHEMA_VERSION
from embedding_model.exceptions import ArtifactValidationError


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler | None,
    epoch: int,
    global_step: int,
) -> Path:
    """Atomically save trusted local training state."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        },
        temporary,
    )
    os.replace(temporary, output)
    return output


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler | None,
) -> tuple[int, int]:
    """Load only tensor/basic training state from a trusted local checkpoint."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise ArtifactValidationError(f"checkpoint does not exist: {checkpoint_path}")
    try:
        payload: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ArtifactValidationError("unsupported checkpoint schema version")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        if scaler is not None and payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])
        return int(payload["epoch"]), int(payload["global_step"])
    except ArtifactValidationError:
        raise
    except (
        EOFError,
        KeyError,
        OSError,
        pickle.UnpicklingError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise ArtifactValidationError(f"invalid checkpoint: {checkpoint_path}") from exc
