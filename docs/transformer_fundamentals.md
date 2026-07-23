# Transformer fundamentals

The local encoder is a PyTorch `nn.TransformerEncoder`: token and learned positional
embeddings enter one or more encoder layers, each layer applies multi-head self-attention and
a position-wise feed-forward network, and a mask prevents padding from participating.

## Implemented encoder stack

```mermaid
flowchart TD
    IDs["input_ids\nB × L"] --> Tok["token embedding\nB × L × H"]
    PosIDs["0 ... L-1"] --> Pos["position embedding\n1 × L × H"]
    Tok --> Add[Elementwise addition]
    Pos --> Add
    Add --> Layer1[Transformer encoder layer 1]
    Mask["attention_mask\nB × L"] --> PadMask["padding mask = mask == 0"]
    PadMask --> Layer1
    Layer1 --> More["... N configured layers ..."]
    More --> States["contextual states\nB × L × H"]
    States --> Pool[Mask-aware pooling]
```

`ModelConfig` validates that `hidden_size` is divisible by `num_attention_heads`, sequence
length is bounded, padding ID is inside the vocabulary, and public dimension equals hidden
width when projection is disabled.

## Token and position embeddings

An integer token ID selects a row from a learned matrix
\(W_{token}\in\mathbb{R}^{V\times H}\). Position \(i\) selects a row from
\(W_{position}\in\mathbb{R}^{L_{max}\times H}\):

```text
h⁰[b, i] = W_token[input_ids[b, i]] + W_position[i]
```

```mermaid
flowchart LR
    ID["token id 17"] --> TokenRow["W_token[17]\nH values"]
    Position["position 3"] --> PositionRow["W_position[3]\nH values"]
    TokenRow --> Sum["initial state at position 3"]
    PositionRow --> Sum
```

Token embeddings say what symbol is present; position embeddings distinguish order. Learned
positions make the maximum sequence length an architectural parameter stored in the artifact.

## Scaled dot-product self-attention

For each head, learned projections transform hidden states into queries, keys, and values:

```text
Q = HW_Q              shape B × L × d_head
K = HW_K              shape B × L × d_head
V = HW_V              shape B × L × d_head
A = softmax(QKᵀ / sqrt(d_head) + padding_mask)
head_output = AV       shape B × L × d_head
```

```mermaid
flowchart TD
    H["hidden states H"] --> Q["Q = HWq"]
    H --> K["K = HWk"]
    H --> V["V = HWv"]
    Q --> Scores["QKᵀ / √d_head\nB × L × L"]
    K --> Scores
    Pad["padding mask"] --> Scores
    Scores --> Softmax[Row-wise softmax]
    Softmax --> Weighted[Attention weights × V]
    V --> Weighted
    Weighted --> Context["contextual token states"]
```

Scaling by \(\sqrt{d_{head}}\) keeps dot products from growing with width and saturating the
softmax. The key-padding mask marks padded key positions so active tokens cannot attend to
them. Pooling separately applies the original attention mask, ensuring padded outputs do not
affect the text vector.

## Multi-head attention

```mermaid
flowchart LR
    Input["B × L × H"] --> Split{Project into heads}
    Split --> H1["head 1\nB × L × d"]
    Split --> H2["head 2\nB × L × d"]
    Split --> Hn["head h\nB × L × d"]
    H1 --> Concat["concatenate\nB × L × H"]
    H2 --> Concat
    Hn --> Concat
    Concat --> Out[Output projection]
```

Heads have separate parameters and may learn different interaction patterns, but head
interpretations are empirical rather than guaranteed. The divisibility validation ensures
each head receives an integer width.

## One encoder layer

```mermaid
flowchart TD
    X[Input states] --> MHA[Multi-head attention]
    MHA --> Drop1[Dropout]
    X --> Add1[Residual add]
    Drop1 --> Add1
    Add1 --> Norm1[LayerNorm]
    Norm1 --> FF1["Linear H → intermediate"]
    FF1 --> GELU[GELU]
    GELU --> FF2["Linear intermediate → H"]
    FF2 --> Drop2[Dropout]
    Norm1 --> Add2[Residual add]
    Drop2 --> Add2
    Add2 --> Norm2[LayerNorm]
```

Residual paths make identity-like signal propagation possible. LayerNorm controls activation
statistics. The feed-forward block transforms each token position independently after
attention has mixed information between positions. Dropout is active in `train()` and
disabled in `eval()`; `TextEmbedder` forces evaluation mode for stable inference.

## Mask and shape invariants

| Invariant | Enforcement | Failure prevented |
|---|---|---|
| IDs and mask are rank 2 with equal shape | Model forward validation | Misaligned padding or accidental broadcasting |
| `L <= max_sequence_length` | Tokenizer truncation and model validation | Position table overrun |
| Padding mask derives from zeros | Encoder call | Padding influencing attention |
| At least one active token | Pooling validation | Undefined pooled vector |
| Output is finite | Model forward validation | NaN/Inf entering loss or index |

Special CLS and SEP tokens mean normal encoded strings contain at least two active positions.
The fully padded check still protects direct tensor callers and corrupted pipelines.

## Compute and memory

Self-attention creates an \(L\times L\) score matrix per head, so its dominant sequence cost
is \(O(BL^2H)\); feed-forward work is roughly \(O(BLH\,I)\), where \(I\) is intermediate
width.

```mermaid
flowchart LR
    Longer["increase L"] --> Quadratic["attention scores grow ~L²"]
    Wider["increase H"] --> Params["embeddings/attention parameters grow"]
    Deeper["increase layers"] --> Activations["activation memory and compute grow"]
    Batch["increase B"] --> Linear["activation memory grows roughly linearly"]
```

Reduce maximum length before reducing model correctness controls. Gradient accumulation
reduces optimizer-step frequency but does not remove per-microbatch activation cost or create
additional simultaneous in-batch negatives.

## Initialization and quality boundary

The standard workflow initializes all Transformer weights randomly. Contrastive pair training
can teach a tiny corpus-specific mapping, but it does not recreate broad linguistic knowledge.
A production quality path generally needs licensed representative pretraining or a compatible
pretrained encoder, followed by domain fine-tuning and held-out evaluation. That adapter is an
extension seam described in [architecture](architecture.md#extension-seams), not an existing
feature.
