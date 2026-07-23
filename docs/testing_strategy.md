# Testing strategy

The test suite proves the real lifecycle with tiny deterministic inputs, then surrounds it with
focused mathematical, validation, property, security, integration, and performance checks.
Internal domain logic is not mocked; mocks are reserved for genuinely external boundaries.

## Test architecture

```mermaid
flowchart TB
    E2E["End-to-end\none complete tiny lifecycle"]
    Integration["Integration\nCLI, files, ASGI routes"]
    Unit["Unit\nmath, schemas, failures, persistence"]
    Property["Property\ninvariants across generated inputs"]
    Security["Security\nabuse and information boundaries"]
    Performance["Performance\nmeasurement, marked slow"]

    Unit --> Integration
    Property --> Integration
    Security --> Integration
    Integration --> E2E
    Performance -. separate hardware-sensitive evidence .-> E2E
```

This is not a strict count-based pyramid. A small number of high-value lifecycle tests protects
cross-module behavior that isolated tests cannot.

## Canonical end-to-end path

```mermaid
flowchart LR
    Data[Tiny synthetic pairs] --> Train[CPU train]
    Train --> Ckpt[Checkpoint]
    Ckpt --> Resume[Resume/reload]
    Resume --> Export[Safe export]
    Export --> Load[Public TextEmbedder]
    Load --> Encode[Encode corpus]
    Encode --> Index[Build/save/load index]
    Index --> Search[Ranked search]
    Search --> Eval[Retrieval evaluation]
    Load --> API[FastAPI app]
    API --> Routes[Health/embedding/similarity/search/metrics]
```

`tests/end_to_end/test_tiny_pipeline.py` crosses each boundary with real PyTorch optimization,
safetensors, NumPy/FAISS-compatible indexing, and ASGI requests. It is network-free and CPU
compatible.

## Marker matrix

| Marker | Scope | Default expectation |
|---|---|---|
| `unit` | One mathematical/domain contract | Fast, deterministic |
| `integration` | Multiple real modules or file/API boundary | CPU/network-free |
| `end_to_end` | Full lifecycle | Tiny and deterministic |
| `security` | Abuse, tampering, privacy, safe error | Required standard path |
| `performance` | Throughput/latency measurement | Hardware-sensitive |
| `slow` | Intentionally longer work | Excluded from standard quick run |
| `gpu` | CUDA-specific behavior | Explicit compatible runner |
| `network` | Remote integration | Explicit opt-in only |

Tests may carry more than one marker when the behavior crosses concerns.

## Mathematical tests

```mermaid
flowchart LR
    Formula[Documented formula] --> Tiny[Small manually computable tensors]
    Tiny --> Expected[Independent expected value]
    Tiny --> Implementation[Implementation output]
    Expected --> Approx[Approximate float comparison]
    Implementation --> Approx
    Implementation --> Backward[Backward pass]
    Backward --> Finite[Finite gradients]
```

Pooling tests include padding values that would corrupt an unmasked result. Loss tests compare
cross-entropy/cosine/triplet values and execute gradients. Retrieval tests use multiple
relevance labels with manually computed outcomes. Correlation tests cover average ranks for
ties and undefined constant inputs.

## Contract and failure tests

| Boundary | Success contract | Representative failures |
|---|---|---|
| Config | Typed valid object | Unknown keys, incompatible shapes/device/precision |
| Data | No silent dropping | Nulls, blanks, duplicate IDs, malformed rows |
| Tokenizer | Stable IDs/merges | Corrupt JSON, special IDs, tiny vocabulary |
| Model/loss | Finite expected shapes | Rank mismatch, fully padded rows, invalid target domains |
| Checkpoint/artifact | Exact reload | Missing/corrupt/schema/shape mismatch |
| Index | Stable exact ranking | Empty, duplicate ID, dimension, zero/non-finite, traversal |
| API | Typed safe response | Oversize, auth, unknown field, blank/long text, safe 4xx/5xx |

Failure tests assert behavior and safe information boundaries, not incidental full exception
formatting.

## Property tests

```mermaid
flowchart TD
    Generator[Hypothesis finite vectors] --> Normalize[Cosine implementation]
    Normalize --> Bounds["score in [-1,1] within tolerance"]
    Generator --> Same[Compare vector with itself]
    Same --> One["cosine ≈ 1 for nonzero vector"]
    Generator --> Index[Build/search index]
    Index --> Order[Scores non-increasing]
```

Properties explore more inputs than fixed examples while examples retain clear arithmetic
evidence. Strategies must avoid invalid values unless the test is specifically validating
rejection.

## Security tests

```mermaid
flowchart LR
    Attacker[Adversarial input] --> Body[Oversized declared/chunked body]
    Attacker --> Auth[Missing/wrong bearer]
    Attacker --> Path[Traversal/tampered manifest]
    Attacker --> Payload[Unknown fields/invalid text]
    Attacker --> Secret[Secret-like log fields]
    Body --> Assert[Bounded generic response]
    Auth --> Assert
    Path --> Reject[Fail before load]
    Payload --> Assert
    Secret --> Redact[No raw credential]
```

Security behavior is part of the public contract. Tests verify request IDs, status/error code,
absence of input/stack traces, path containment, checksum rejection, auth hooks, and redaction.

## Documentation tests

Every Markdown page is required to be discoverable from `docs/index.md`, substantial enough to
have navigable sections, include a diagram or comparison table, balance Mermaid fences, and
resolve relative links.

```mermaid
flowchart LR
    Docs[docs/**/*.md] --> Structure[Length + headings]
    Docs --> Visual[Mermaid or table]
    Docs --> Links[Relative targets exist]
    Index[docs/index.md] --> Discover[Every page linked]
    Structure --> Test[test_documentation.py]
    Visual --> Test
    Links --> Test
    Discover --> Test
```

This structural gate cannot determine whether prose is correct; source review and command
verification remain required.

## Determinism controls

- tiny local tokenizer and random Transformer, with no downloads;
- fixed Python/NumPy/PyTorch seeds;
- CPU baseline and small shapes;
- deterministic tie rules for vocabulary, merges, splits, and search;
- temporary directories for artifacts;
- approximate comparisons with justified tolerances;
- no dependence on wall-clock performance in portable functional tests.

Determinism should not be achieved by mocking away the behavior under test.

## Verification order

```bash
pytest tests/unit/test_documentation.py -q
pytest tests/unit/test_modeling_losses.py -q
pytest -m unit -q
pytest -m "not slow and not network and not gpu" -q
pytest -m integration -q
pytest -m end_to_end -q
pytest -m security -q
pytest --cov=embedding_model --cov-branch --cov-report=term-missing
ruff format --check .
ruff check .
mypy src
```

Run the narrowest reproducer first, then expand. Performance tests must report hardware,
software, warmup, sample count, and configurable thresholds; they should not fail portable CI
on an arbitrary latency number.

## Adding behavior

```mermaid
flowchart TD
    Requirement[User-visible requirement] --> E2E[Add/update smallest real boundary test]
    E2E --> Fail[Observe expected failure]
    Fail --> Vertical[Implement smallest vertical slice]
    Vertical --> Focused[Add math/failure/security tests]
    Focused --> Narrow[Run narrow checks]
    Narrow --> Broad[Run standard suite + lint/types]
    Broad --> Docs[Update affected docs and traceability]
```

Avoid tests that merely duplicate implementation branches. Prefer observable contracts:
outputs, saved artifacts, reload behavior, safe failures, and stable public APIs.
