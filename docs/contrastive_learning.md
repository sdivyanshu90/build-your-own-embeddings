# Contrastive learning and objectives

Contrastive learning shapes the embedding space by comparing designated positives with
candidate negatives. The default pair trainer uses a batch similarity matrix: its diagonal is
positive by construction and off-diagonal pairs become in-batch candidates.

## Multiple-negatives ranking loss

For normalized anchor and positive matrices \(Q,P\in\mathbb{R}^{B\times D}\):

```text
logits[i,j] = (Q[i] · P[j]) / temperature
labels      = [0, 1, ..., B-1]
loss        = cross_entropy(logits, labels)
```

```mermaid
flowchart LR
    Q["anchors Q\nB × D"] --> MatMul["QPᵀ / τ"]
    P["positives P\nB × D"] --> MatMul
    MatMul --> Matrix["logits\nB × B"]
    Diag["diagonal = positives"] --> CE[Cross entropy]
    Off["off-diagonal = candidates"] --> CE
    Matrix --> CE
    CE --> Scalar[Mean scalar loss]
```

For a batch of three:

| Anchor | Candidate 0 | Candidate 1 | Candidate 2 | Target |
|---|---:|---:|---:|---:|
| `q0` | positive score | negative score | negative score | 0 |
| `q1` | negative score | positive score | negative score | 1 |
| `q2` | negative score | negative score | positive score | 2 |

The optional symmetric mode also applies cross-entropy to the transposed matrix and averages
both directions. The current trainer selects the one-direction default.

## Temperature

```mermaid
flowchart TD
    Scores[Cosine scores] --> Divide["divide by τ"]
    Divide --> Small{Temperature}
    Small -->|smaller| Sharp[Sharper softmax; larger gradient differences]
    Small -->|larger| Smooth[Smoother softmax; weaker distinctions]
    Sharp --> Risk1[Greater sensitivity to mislabeled/false negatives]
    Smooth --> Risk2[Slower or under-separated learning]
```

Temperature must be positive. It changes optimization geometry, so compare it with fixed data,
batch construction, seeds, and evaluation rather than treating it as a cosmetic scaling.

## InfoNCE with explicit negatives

With one positive and \(N\) explicit negatives per anchor:

```text
anchors   shape B × D
positives shape B × D
negatives shape B × N × D
logits    shape B × (1 + N)
target    index 0 for every row
```

```mermaid
flowchart LR
    A[Anchor] --> Pos["cos(anchor, positive)"]
    A --> Neg1["cos(anchor, negative 1)"]
    A --> NegN["cos(anchor, negative N)"]
    Pos --> Cat["[positive, negatives] / τ"]
    Neg1 --> Cat
    NegN --> Cat
    Cat --> CE["cross entropy target 0"]
```

When explicit negatives are omitted, the implementation delegates to multiple-negatives
ranking. The pair trainer calls this implicit path; explicit-negative dataset collation is not
wired into its public training method.

## Triplet loss

Triplet loss receives one anchor, positive, and negative per row:

```text
d_pos = 1 - cosine(anchor, positive)
d_neg = 1 - cosine(anchor, negative)
loss  = mean(max(0, d_pos - d_neg + margin))
```

```mermaid
flowchart LR
    A[Anchor] --> DP[Positive distance]
    P[Positive] --> DP
    A --> DN[Negative distance]
    N[Negative] --> DN
    DP --> Hinge["ReLU(d_pos - d_neg + margin)"]
    DN --> Hinge
```

Once the negative is farther than the positive by at least the margin, that triplet contributes
zero. Hard and semi-hard mining determine how often triplets remain informative.

## Cosine regression

For continuous labels \(y\in[-1,1]\):

```text
prediction = cosine(left, right)
loss       = mean((prediction - y)²)
```

This objective suits semantic textual similarity labels rather than retrieval relevance IDs.
The implementation rejects target shapes other than `(B,)` and labels outside the cosine
range.

## Distillation

```mermaid
flowchart TD
    Student["student embeddings\nB × D"] --> MSE[Embedding MSE]
    Teacher["teacher embeddings, detached\nB × D"] --> MSE
    Student --> Cos[Mean cosine distance]
    Teacher --> Cos
    Student --> SRel["student self-similarity\nB × B"]
    Teacher --> TRel["teacher self-similarity\nB × B"]
    SRel --> KL[Optional relational KL]
    TRel --> KL
    MSE --> Weighted[Weighted sum]
    Cos --> Weighted
    KL --> Weighted
```

Teacher tensors are detached so gradients only update the student. Non-negative component
weights must have a positive sum; relational KL uses temperature scaling.

## Objective-to-data compatibility

| Objective | Required batch fields | Implemented loss | Wired into `Trainer.train_pairs` |
|---|---|---:|---:|
| Multiple negatives | anchor, positive | Yes | Yes |
| InfoNCE implicit | anchor, positive | Yes | Yes |
| InfoNCE explicit | anchor, positive, negative set | Yes | No |
| Triplet | anchor, positive, negative | Yes | No |
| Cosine regression | left, right, score | Yes | No |
| Distillation | student input, teacher embeddings | Yes | No |

This distinction prevents a configuration from silently training the wrong task. Selecting a
non-pair objective with `PairRecord` data fails. Adding a public path requires a typed record,
collator, trainer method, checkpoint/resume coverage, CLI contract, and end-to-end test.

## Batch size and false negatives

```mermaid
flowchart TD
    Bigger[Increase batch size] --> More[More in-batch candidates]
    Bigger --> Memory[More activation memory]
    Bigger --> False[Greater chance of unlabeled positives]
    More --> Signal[Potentially stronger ranking signal]
    False --> Damage[True semantic neighbors pushed apart]
```

Gradient accumulation does not enlarge the similarity matrix; each microbatch still sees only
its own negatives. Known-positive exclusion, multi-positive annotation, deduplication, and
mined-negative review are data responsibilities described in
[negative sampling](negative_sampling.md).

## Numerical and test contracts

Every objective checks tensor ranks/shapes and its own temperature, margin, target, or weight
domain. Unit tests compare small hand-computed cases, execute backward, and require finite
gradients. Training adds finite-loss and finite-gradient-norm checks before the optimizer step.
These checks establish mathematical execution, not model quality; held-out retrieval evidence
remains mandatory.
