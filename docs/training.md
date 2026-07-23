# Training and reproducibility

The trainer turns validated `PairRecord` examples into optimizer updates, resumable trusted
checkpoints, and finally a safe inference artifact. The current vertical CLI path supports
multiple-negatives ranking and implicit InfoNCE on one process, with CPU fallback and optional
single-device CUDA mixed precision.

## End-to-end training sequence

```mermaid
sequenceDiagram
    autonumber
    participant CLI as CLI
    participant Data as Reader/schema
    participant Tok as BPE tokenizer
    participant Model as EmbeddingModel
    participant T as Trainer
    participant C as Checkpoint
    participant X as Exporter

    CLI->>Data: load PairRecord file
    Data-->>CLI: validated records
    CLI->>Tok: fit on both text columns
    Tok-->>CLI: vocabulary + merges
    CLI->>Model: construct with actual vocab size
    CLI->>T: model, tokenizer, TrainingConfig
    opt --resume-from
        T->>C: load trusted local state
        C-->>T: epoch, global step, optimizer/scheduler/scaler
    end
    T->>T: train epochs and validate finite state
    T->>C: atomic last/best checkpoints
    T-->>CLI: TrainingResult
    CLI->>X: model + tokenizer + metadata
    X-->>CLI: checksummed inference artifact
```

## Configuration contract

| Setting | Meaning | Validation/interaction |
|---|---|---|
| `objective` | Training loss | Pair trainer accepts only MNR or InfoNCE |
| `epochs` | Maximum passes over records | At least 1; resume starts from saved epoch |
| `batch_size` | Pair records per microbatch | At least 2 |
| `gradient_accumulation_steps` | Microbatches per optimizer step | At least 1 |
| `learning_rate`, `weight_decay` | AdamW settings | Non-negative domains |
| `warmup_ratio` | Fraction of optimizer steps warming up | `[0, 1)` |
| `max_gradient_norm` | Clip threshold | Positive |
| `temperature` | Contrastive logit scale | Positive |
| `mixed_precision` | CUDA float16 autocast/scaler | Rejected without CUDA |
| `device` | `auto`, `cpu`, or `cuda` | Explicit unavailable CUDA is rejected |
| `seed`, `deterministic` | Randomness policy | Python, NumPy, PyTorch, CUDA seeded |
| `early_stopping_patience` | Stale validation epochs | Optional positive integer |

Unknown YAML keys fail because configuration models use `extra="forbid"`. Model configuration
also validates head divisibility, padding ID, sequence bounds, and projection dimensions.

## One epoch in detail

```mermaid
flowchart TD
    Start[epoch e] --> Seed["generator seed = base seed + e"]
    Seed --> Permute[Deterministic record permutation]
    Permute --> Batch[Split into microbatches]
    Batch --> Singleton{Final batch size 1?}
    Singleton -->|yes| Merge[Merge into previous batch]
    Singleton -->|no| Forward
    Merge --> Forward[Encode anchor and positive text]
    Forward --> Loss[Contrastive loss]
    Loss --> Finite{Loss finite?}
    Finite -->|no| Stop[Raise without optimizer step]
    Finite -->|yes| Scale["loss / accumulation steps"]
    Scale --> Backward[Backward / GradScaler]
    Backward --> Step{Accumulation boundary or final batch?}
    Step -->|no| Batch
    Step -->|yes| Unscale[Unscale gradients]
    Unscale --> Clip[Clip global norm]
    Clip --> GradFinite{Gradient norm finite?}
    GradFinite -->|no| Stop
    GradFinite -->|yes| Optimizer[AdamW step]
    Optimizer --> Schedule[Scheduler step]
    Schedule --> Zero[Zero gradients]
    Zero --> Batch
```

The reported epoch loss is the mean of unscaled microbatch losses. Accumulation changes update
frequency, not the number of in-batch negatives visible to each loss call.

## Learning-rate schedule

The trainer computes total optimizer steps from batches, accumulation, and configured epochs.
The multiplier warms linearly from near zero and then decays linearly:

```mermaid
xychart-beta
    title "Conceptual warmup and linear decay"
    x-axis "optimizer step" [0, 1, 2, 3, 4, 5, 6, 7, 8]
    y-axis "LR multiplier" 0 --> 1
    line [0.01, 0.5, 1.0, 0.83, 0.67, 0.5, 0.33, 0.17, 0.0]
```

The exact warmup step count is `int(total_steps * warmup_ratio)`. Scheduler state is included
in checkpoints so resume continues the saved schedule rather than restarting it.

## Validation, best checkpoint, and early stopping

```mermaid
stateDiagram-v2
    [*] --> TrainEpoch
    TrainEpoch --> Validate: validation records supplied
    TrainEpoch --> SaveLast: no validation records
    Validate --> SaveBest: loss improves
    Validate --> Stale: loss does not improve
    SaveBest --> SaveLast
    Stale --> SaveLast
    SaveLast --> Stop: stale epochs >= patience
    SaveLast --> TrainEpoch: epochs remain
    Stop --> [*]
```

Validation uses evaluation mode and no gradients. Singleton validation fragments are skipped;
the full validation input must still yield at least one batch of two because the objective
requires an in-batch negative. `best.pt` is written on improvement and `last.pt` after each
completed epoch.

## Checkpoint state and trust

```mermaid
flowchart LR
    State["model\noptimizer\nscheduler\nGradScaler\nepoch\nglobal_step\nschema version"] --> Temp["checkpoint.pt.tmp"]
    Temp --> Replace["os.replace"]
    Replace --> Final["checkpoint.pt"]
    Final --> Trusted["torch.load(weights_only=True)\ntrusted local boundary"]
```

Atomic replacement avoids exposing a partially written final filename. `weights_only=True`
narrows PyTorch loading, but resume files still contain framework-specific optimizer state and
are not the published untrusted artifact format. Only load checkpoints produced inside the
trusted training environment.

Resume reconstructs the same model, tokenizer, optimizer, scheduler, and scaler first, then
loads state strictly. Changing architecture or tokenizer between checkpoint creation and
resume fails rather than partially applying tensors.

## Interruption and failure behavior

| Failure | Trainer behavior | Operator response |
|---|---|---|
| `KeyboardInterrupt` | Writes `interrupted.pt`, logs bounded run/step fields, re-raises | Resume explicitly after inspecting state |
| CUDA/CPU out of memory text from PyTorch | Raises actionable batch/length guidance | Reduce batch/length; consider accumulation |
| Non-finite loss | Stops before backward/update | Reproduce on CPU; inspect data/LR/objective |
| Non-finite gradient norm | Stops before optimizer step | Lower LR, inspect activations/mixed precision |
| Incompatible objective/data | Fails before training | Use supported pair objective or build typed path |
| Invalid checkpoint | Raises artifact validation error | Use earlier trusted checkpoint; never ignore corruption |

```mermaid
flowchart TD
    Failure[Training failure] --> Valid{Last completed checkpoint valid?}
    Valid -->|yes| Diagnose[Preserve config, versions, seed, logs]
    Valid -->|no| Previous[Select previous valid checkpoint]
    Diagnose --> Fix[Correct data/config/resource issue]
    Previous --> Fix
    Fix --> Resume[Resume explicitly]
    Resume --> Reevaluate[Run held-out evaluation]
```

## Reproducibility envelope

The trainer seeds Python, NumPy, PyTorch, and all CUDA devices. Epoch shuffling uses
`seed + epoch`, and deterministic algorithms are requested with warnings. Reproduction still
depends on CPU/GPU model, driver, PyTorch/NumPy versions, kernel selection, worker scheduling,
and input bytes.

Record at least:

- resolved model/training configuration and Git commit;
- tokenizer and dataset versions/checksums;
- Python/package/driver versions and hardware;
- run ID, seed, batch/accumulation schedule, and objective;
- checkpoint schema/step and exported artifact manifest;
- held-out evaluation dataset/version and report.

## Commands and outputs

```bash
make train-tiny

embedding-project train \
  --config configs/train_cpu.yaml \
  --data data/sample_pairs.jsonl \
  --output-dir artifacts/domain-model

embedding-project train \
  --config configs/train_cpu.yaml \
  --data data/sample_pairs.jsonl \
  --output-dir artifacts/resumed-model \
  --resume-from artifacts/checkpoints/last.pt
```

Export refuses to overwrite a non-empty artifact directory. Training emits JSON with artifact
path, global step, and training loss; epoch logs include run ID, epoch, step, train/validation
loss, and current learning rate without raw training text.

## Current limits

The code is single-process. It does not implement DDP, cross-device negative gathering,
gradient checkpointing, MLM pretraining, or trainer methods for triplet/regression/distillation
records. `configs/train_distributed.yaml` is sizing documentation, not executable DDP proof.
See [scaling](scaling.md) for the required state transitions and
[testing strategy](testing_strategy.md) for the verified tiny lifecycle.
