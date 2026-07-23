# System architecture

The project is a vertical reference system for building and operating a text-embedding model.
It deliberately keeps the standard path local and inspectable: inputs are validated, every
tensor transition has a shape contract, training state and published artifacts have different
trust rules, and serving receives already-constructed runtime objects.

## Architectural goals and boundaries

| Goal | Architectural response | Explicit boundary |
|---|---|---|
| End-to-end operability | One CLI owns train, evaluate, export, index, search, and serve | Semantic quality still depends on data/pretraining |
| Network-free CI | Local BPE and randomly initialized PyTorch Transformer | No Hugging Face download in the standard path |
| Safe publication | Safetensors plus JSON/Markdown and SHA-256 manifest | Resume checkpoints remain trusted-local files |
| Deterministic retrieval | Normalized vectors, exact inner product, insertion-order ties | Large-scale ANN is not implemented |
| Bounded serving | Strict schemas, byte/text/batch/K/concurrency limits | TLS, tenant authorization, and global rate limits live upstream |
| Testability | Dependency-injected app and tiny real lifecycle | Internal domain logic is not mocked |

The production-quality boundary matters: the system validates that a model can be trained and
operated correctly, but tiny random initialization and synthetic data do not establish useful
language understanding.

## Context diagram

```mermaid
flowchart LR
    Author[ML engineer] -->|config + typed data| CLI[embedding-project CLI]
    CLI --> Train[Training system]
    Train --> Registry[(Artifact storage)]
    Registry --> Runtime[Inference runtime]
    CorpusOwner[Corpus pipeline] -->|id, text, metadata| Runtime
    Client[API client] -->|bounded JSON request| Gateway[TLS / identity / rate limits]
    Gateway --> Service[FastAPI service]
    Runtime --> Service
    Service --> Client
    Service --> Prom[Prometheus]
    Service --> LogPipe[Structured log pipeline]

    classDef external fill:#f8f8f8,stroke:#555,stroke-dasharray:4 3
    class Author,CorpusOwner,Client,Gateway,Registry,Prom,LogPipe external
```

External systems are shown with dashed borders. The repository supplies the CLI, training
system, artifact validation, inference runtime, index, and application. A deployment supplies
durable artifact storage, ingress security, metrics collection, and log retention.

## Container and package view

```mermaid
flowchart TB
    subgraph Interface["Interface layer"]
        CLI[cli.py]
        API[serving/]
        Public[TextEmbedder]
    end

    subgraph Application["Application workflows"]
        Trainer[training/]
        Evaluator[evaluation/]
        Exporter[export/]
        Search[indexing/]
    end

    subgraph Domain["Domain model"]
        Config[config.py]
        Data[data/]
        Tokenizer[tokenization.py]
        Model[modeling/]
        Loss[losses/]
    end

    subgraph Infrastructure["Infrastructure"]
        Torch[PyTorch]
        Safe[safetensors]
        Faiss[FAISS / NumPy]
        FastAPI[FastAPI]
        Metrics[Prometheus]
    end

    CLI --> Trainer
    CLI --> Evaluator
    CLI --> Exporter
    CLI --> Search
    CLI --> API
    API --> Public
    Public --> Exporter
    Public --> Model
    Trainer --> Data
    Trainer --> Tokenizer
    Trainer --> Model
    Trainer --> Loss
    Evaluator --> Public
    Search --> Public
    Config --> Model
    Config --> Trainer
    Model --> Torch
    Exporter --> Safe
    Search --> Faiss
    API --> FastAPI
    API --> Metrics
```

Dependencies point inward toward domain contracts. Domain modules do not import the CLI or
HTTP layer. Runtime objects are created explicitly, so importing the package does not load a
model, configure logging, touch the network, or mutate global application state.

## Training data flow

```mermaid
flowchart TD
    File[JSONL / CSV / Parquet] --> Reader[read_records]
    Reader --> Schema[Pydantic record schema]
    Schema --> Clean[Whitespace normalization]
    Clean --> Duplicate[Duplicate-ID check]
    Duplicate --> Split[Seeded split / shuffle]
    Split --> TokFit[Fit BPE on training text]
    TokFit --> Batch[Batch PairRecord values]
    Batch --> TokenTensors["input_ids, attention_mask\n(B, L) int64"]
    TokenTensors --> Encoder["Transformer hidden states\n(B, L, H) float"]
    Encoder --> Pool["Mask-aware pooling\n(B, H)"]
    Pool --> Head["Projection / LayerNorm / L2\n(B, D)"]
    Head --> Similarity["Similarity logits\n(B, B)"]
    Similarity --> Loss[Cross-entropy loss scalar]
    Loss --> Backward[Scaled backward pass]
    Backward --> Clip[Finite gradient + norm clipping]
    Clip --> AdamW[AdamW + warmup/decay]
    AdamW --> Epoch{Epoch complete?}
    Epoch -->|no| Batch
    Epoch -->|yes| Checkpoint[Atomic trusted checkpoint]
```

No invalid row is silently dropped. Pair training requires at least two records because a
singleton cannot provide an in-batch negative; a final singleton batch is merged backward.
See [training](training.md) for optimizer state transitions and
[contrastive learning](contrastive_learning.md) for the loss matrix.

## Tensor contract through the model

```mermaid
flowchart LR
    Text["list[str]\nB items"] --> IDs["token IDs\nB × L"]
    IDs --> Lookup["token + position\nB × L × H"]
    Mask["attention mask\nB × L"] --> Encoder
    Lookup --> Encoder["N Transformer layers\nB × L × H"]
    Encoder --> Pool["pooling\nB × H"]
    Mask --> Pool
    Pool --> Project["optional projection\nB × D"]
    Project --> LN["optional LayerNorm\nB × D"]
    LN --> Norm["optional L2 normalization\nB × D float32"]
```

| Symbol | Meaning | Validity rule |
|---|---|---|
| `B` | Batch size | Positive inside tokenizer/model; public empty input returns `(0, D)` before model execution |
| `L` | Padded sequence length | `2 <= L <= max_sequence_length` after CLS/SEP |
| `H` | Transformer hidden width | Divisible by attention-head count |
| `D` | Public embedding dimension | Equals `H` if projection is disabled |
| Mask | `1` for active token, `0` for padding | Shape exactly matches token IDs; no fully padded row |
| Output | CPU NumPy or PyTorch tensor | Stable order, float32, finite; unit norm when enabled |

## Runtime request sequence

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant M as Size-limit middleware
    participant A as FastAPI route
    participant E as TextEmbedder
    participant I as VectorIndex
    participant O as Metrics / logs

    C->>M: POST JSON + optional X-Request-ID
    M->>M: Bound Content-Length and streamed bytes
    M->>A: Replay bounded body
    A->>A: Authenticate and validate schema/text/K
    A->>E: encode(texts)
    E->>E: lock, tokenize, inference_mode, finite check
    opt search route
        E->>I: normalized query vector
        I->>I: exact IP ranking + insertion-order ties
        I-->>A: IDs, scores, metadata
    end
    A->>O: bounded route/status/latency/count
    A-->>C: typed response + X-Request-ID
```

The semaphore bounds concurrent handlers, while the embedder lock prevents concurrent
mutation of model mode/state. Model compute is synchronous; multiple process replicas or a
dedicated dynamic batcher are deployment extensions, not hidden behavior.

## Artifact boundary

Training checkpoints and published inference artifacts solve different problems:

```mermaid
flowchart LR
    subgraph Trusted["Trusted local training boundary"]
        State["model + optimizer + scheduler\nscaler + epoch + step"]
        PT["last.pt / best.pt\nPyTorch weights_only load"]
        State -->|atomic replace| PT
    end

    subgraph Untrusted["Publication / untrusted-load boundary"]
        Config[config.json]
        Tok[tokenizer/tokenizer.json]
        Weights[model.safetensors]
        Meta[training_metadata.json]
        Eval[evaluation.json]
        Card[model_card.md]
        Manifest[manifest.json]
        Config --> Manifest
        Tok --> Manifest
        Weights --> Manifest
        Meta --> Manifest
        Eval --> Manifest
        Card --> Manifest
    end

    PT -->|explicit export| Weights
```

Loading resolves every manifest path under the artifact root, verifies schema, byte size, and
SHA-256, validates configuration and tokenizer compatibility, constructs the expected model,
then performs a strict safetensors name/shape load. A checksum detects corruption; it does not
authenticate the publisher. Production promotion should sign or attest the manifest.

## Index boundary

```mermaid
flowchart TD
    Docs["JSONL documents\nid + text + metadata"] --> Embed[TextEmbedder]
    Embed --> Validate["finite N × D\nunique IDs"]
    Validate --> Normalize[Row-wise L2 normalize]
    Normalize --> Memory["NumPy float32 matrix\ncanonical persistence"]
    Normalize --> Faiss["FAISS IndexFlatIP\noptional acceleration"]
    Memory --> Files["vectors.npy\nallow_pickle=False"]
    Docs --> Metadata[metadata.json]
    Files --> IndexManifest[index_manifest.json]
    Metadata --> IndexManifest
    Query[Query embedding] --> QNorm[L2 normalize]
    QNorm --> Faiss
    QNorm --> Memory
    Faiss --> Tie[Score descending, insertion-order ties]
    Memory --> Tie
```

The model and index must agree on dimension, but dimension alone is insufficient provenance.
A production registry should promote a model, tokenizer, corpus, and index as one immutable
version. The current manifest protects index file integrity but does not embed the model hash.

## State ownership and lifecycle

| State | Owner | Mutable when | Persistence |
|---|---|---|---|
| Configuration | CLI/application construction | Never after validation | YAML input; JSON in artifact |
| Model parameters | Trainer | Training only | Checkpoint, then safetensors |
| Optimizer/scheduler/scaler | Trainer | Training only | Trusted checkpoint |
| Tokenizer vocabulary/merges | Tokenizer training | Before model construction | Checksummed JSON |
| Index vectors/metadata | Index builder | Before save or explicit append | Checksummed NumPy/JSON |
| Embedder model mode | `TextEmbedder` | Forced to evaluation under lock | Reconstructed from artifact |
| Readiness | FastAPI application state | Deployment startup/reload | Process memory |
| Metrics | Per-app Prometheus registry | Request processing | Scraped externally |

## Failure propagation

```mermaid
flowchart TD
    Input[Failure source] --> Kind{Layer}
    Kind -->|Config/data| Actionable[Typed exception with file/field context]
    Kind -->|Training numeric| Stop[Stop step; preserve last valid checkpoint]
    Kind -->|Artifact/index| Reject[Reject before model/search use]
    Kind -->|HTTP validation/auth| Safe4xx[Generic 4xx + request ID]
    Kind -->|Unexpected HTTP| Safe5xx[Generic 500 + request ID]
    Actionable --> Operator[Operator fixes source]
    Stop --> Operator
    Reject --> Operator
    Safe4xx --> Client[Client corrects request]
    Safe5xx --> Incident[Correlate bounded logs and metrics]
```

Internal exceptions are useful at CLI and test boundaries. HTTP responses deliberately avoid
exception text and stack traces because those can expose paths, configuration, or input.

## Extension seams

| Desired extension | Stable seam | Work still required |
|---|---|---|
| Pretrained encoder | Implement a model adapter preserving `(B, D)` | Tokenizer compatibility, offline fixture, licensing, evaluation |
| New training record/objective | Add typed record plus a dedicated trainer method | Collator, compatibility validation, E2E path |
| Distributed training | Trainer orchestration boundary | Samplers, rank-zero writes, gathered negatives, resume tests |
| Approximate retrieval | `VectorIndex`-compatible API | Recall benchmark, training parameters, safe persistence |
| Dynamic batching | Service-to-embedder boundary | Queue, deadlines, cancellation, queue metrics |
| Signed artifacts | Manifest promotion boundary | Signature format, key rotation, verification policy |

Architecture changes that alter these boundaries should update the relevant
[ADR](index.md#architecture-decisions), end-to-end test, traceability entry, and operational
guide together.
