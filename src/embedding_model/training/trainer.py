"""Deterministic CPU/CUDA trainer for pairwise contrastive learning."""

from __future__ import annotations

import logging
import math
import random
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch

from embedding_model.config import TrainingConfig
from embedding_model.data.schemas import PairRecord
from embedding_model.losses import InfoNCELoss, MultipleNegativesRankingLoss
from embedding_model.modeling.embedding_model import EmbeddingModel
from embedding_model.tokenization import LocalTokenizer
from embedding_model.training.checkpointing import load_checkpoint, save_checkpoint

logger = logging.getLogger("embedding_model.training")


@dataclass(frozen=True)
class TrainingResult:
    training_loss: float
    validation_loss: float | None
    global_step: int
    last_checkpoint: Path
    elapsed_seconds: float


def set_reproducible_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch without hidden module initialization."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


class Trainer:
    """Train a local model with in-batch negatives and resumable state."""

    def __init__(
        self,
        model: EmbeddingModel,
        tokenizer: LocalTokenizer,
        config: TrainingConfig,
    ) -> None:
        set_reproducible_seed(config.seed, config.deterministic)
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.run_id = str(uuid.uuid4())
        if tokenizer.vocab_size != model.config.vocabulary_size:
            raise ValueError("tokenizer vocabulary size does not match model configuration")
        self.device = self._select_device()
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda _: 1.0)
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)
        self.epoch = 0
        self.global_step = 0

    def _select_device(self) -> torch.device:
        if self.config.device == "cuda":
            return torch.device("cuda")
        if self.config.device == "auto" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _configure_scheduler(self, total_steps: int) -> None:
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        def multiplier(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                return max(1e-8, step / warmup_steps)
            remaining = max(1, total_steps - warmup_steps)
            return max(0.0, (total_steps - step) / remaining)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=multiplier)

    def resume(self, checkpoint: str | Path) -> None:
        """Restore model and optimizer state from a local checkpoint."""

        self.epoch, self.global_step = load_checkpoint(
            checkpoint,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
        )
        self.model.to(self.device)

    def train_pairs(
        self,
        pairs: Sequence[PairRecord],
        *,
        validation_pairs: Sequence[PairRecord] | None = None,
    ) -> TrainingResult:
        """Train the configured in-batch objective and checkpoint each epoch."""

        if len(pairs) < 2:
            raise ValueError("pair training requires at least two records")
        if self.config.objective not in {
            "multiple_negatives_ranking",
            "info_nce",
        }:
            raise ValueError(
                f"objective {self.config.objective!r} is incompatible with PairRecord data"
            )
        batches_per_epoch = math.ceil(len(pairs) / self.config.batch_size)
        optimizer_steps = math.ceil(batches_per_epoch / self.config.gradient_accumulation_steps)
        if self.global_step == 0:
            self._configure_scheduler(optimizer_steps * self.config.epochs)
        objective = (
            InfoNCELoss(self.config.temperature)
            if self.config.objective == "info_nce"
            else MultipleNegativesRankingLoss(self.config.temperature)
        )
        start = time.monotonic()
        final_loss = float("nan")
        validation_loss: float | None = None
        last_checkpoint = self.config.checkpoint_dir / "last.pt"
        best_validation = float("inf")
        stale_epochs = 0
        try:
            for epoch in range(self.epoch, self.config.epochs):
                final_loss = self._train_epoch(pairs, objective, epoch)
                self.epoch = epoch + 1
                if validation_pairs:
                    validation_loss = self._evaluate_pairs(validation_pairs, objective)
                    if validation_loss < best_validation:
                        best_validation = validation_loss
                        stale_epochs = 0
                        save_checkpoint(
                            self.config.checkpoint_dir / "best.pt",
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            scaler=self.scaler,
                            epoch=self.epoch,
                            global_step=self.global_step,
                        )
                    else:
                        stale_epochs += 1
                last_checkpoint = save_checkpoint(
                    last_checkpoint,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    epoch=self.epoch,
                    global_step=self.global_step,
                )
                logger.info(
                    "training_epoch_complete",
                    extra={
                        "fields": {
                            "run_id": self.run_id,
                            "epoch": self.epoch,
                            "global_step": self.global_step,
                            "training_loss": final_loss,
                            "validation_loss": validation_loss,
                            "learning_rate": self.optimizer.param_groups[0]["lr"],
                        }
                    },
                )
                patience = self.config.early_stopping_patience
                if patience is not None and stale_epochs >= patience:
                    break
        except KeyboardInterrupt:
            interrupted = save_checkpoint(
                self.config.checkpoint_dir / "interrupted.pt",
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                epoch=self.epoch,
                global_step=self.global_step,
            )
            logger.warning(
                "training_interrupted",
                extra={
                    "fields": {
                        "run_id": self.run_id,
                        "checkpoint": str(interrupted),
                        "global_step": self.global_step,
                    }
                },
            )
            raise
        return TrainingResult(
            training_loss=final_loss,
            validation_loss=validation_loss,
            global_step=self.global_step,
            last_checkpoint=last_checkpoint,
            elapsed_seconds=time.monotonic() - start,
        )

    def _train_epoch(
        self,
        pairs: Sequence[PairRecord],
        objective: torch.nn.Module,
        epoch: int,
    ) -> float:
        generator = torch.Generator().manual_seed(self.config.seed + epoch)
        order = torch.randperm(len(pairs), generator=generator).tolist()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        batches = [
            order[offset : offset + self.config.batch_size]
            for offset in range(0, len(order), self.config.batch_size)
        ]
        # A final singleton cannot provide an in-batch negative; merge it backward.
        if len(batches) > 1 and len(batches[-1]) == 1:
            batches[-2].extend(batches.pop())
        for batch_number, indices in enumerate(batches, start=1):
            batch = [pairs[index] for index in indices]
            try:
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.config.mixed_precision,
                ):
                    left = self._encode_training([pair.text_a for pair in batch])
                    right = self._encode_training([pair.text_b for pair in batch])
                    loss = objective(left, right)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    raise RuntimeError(
                        "training ran out of memory; reduce batch_size or max_sequence_length"
                    ) from exc
                raise
            if not torch.isfinite(loss):
                raise FloatingPointError("training loss became NaN or infinity")
            scaled_loss = loss / self.config.gradient_accumulation_steps
            self.scaler.scale(scaled_loss).backward()
            losses.append(float(loss.detach().cpu()))
            should_step = (
                batch_number % self.config.gradient_accumulation_steps == 0
                or batch_number == len(batches)
            )
            if should_step:
                self.scaler.unscale_(self.optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_gradient_norm
                )
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError("gradient norm became NaN or infinity")
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
        return sum(losses) / len(losses)

    @torch.no_grad()
    def _evaluate_pairs(self, pairs: Sequence[PairRecord], objective: torch.nn.Module) -> float:
        if len(pairs) < 2:
            raise ValueError("validation requires at least two pair records")
        self.model.eval()
        losses: list[float] = []
        for offset in range(0, len(pairs), self.config.batch_size):
            batch = list(pairs[offset : offset + self.config.batch_size])
            if len(batch) == 1:
                continue
            left = self._encode_training([pair.text_a for pair in batch])
            right = self._encode_training([pair.text_b for pair in batch])
            losses.append(float(objective(left, right).cpu()))
        if not losses:
            raise ValueError("validation batches must contain at least two records")
        return sum(losses) / len(losses)

    def _encode_training(self, texts: list[str]) -> torch.Tensor:
        encoded = self.tokenizer.batch_encode(
            texts,
            self.model.config.max_sequence_length,
            device=self.device,
        )
        return cast(torch.Tensor, self.model(**encoded))
