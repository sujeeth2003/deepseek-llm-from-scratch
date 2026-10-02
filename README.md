# Building DeepSeek LLM From Scratch

A from-scratch, tested PyTorch implementation of the architecture, scaling
laws, training loop, and alignment (SFT + DPO) described in *"DeepSeek LLM:
Scaling Open-Source Language Models with Longtermism"* (Bi et al., 2024).

Every file below has been run and verified during development (not just
written from memory) — see the test snippets embedded as `__main__` blocks
or run the commands under **"Verifying it works"**.

```
deepseek_llm_scratch/
├── layers.py         # RMSNorm, RoPE, SwiGLU, Grouped-Query Attention
├── model.py          # Full pre-norm decoder-only transformer
├── scaling_laws.py   # Eq. (1), (2), (4) from Section 3
├── train.py          # Multi-step LR scheduler + training loop (Sec 2.3-2.4)
├── alignment.py       # SFT loss + DPO loss (Section 4)
└── README.md
```

---

## 1. Architecture (Section 2.2)

### RMSNorm
Instead of LayerNorm's `(x - mean) / std`, RMSNorm skips mean-centering:

```
RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight
```

Cheaper (one reduction instead of two) and empirically works as well in
this class of model. Pre-Norm placement (`x = x + Attn(Norm(x))`) is used,
not Post-Norm — this is what makes very deep networks (95 layers for the
67B model) stable to train.

### Rotary Position Embeddings (RoPE)
Rather than adding a learned/sinusoidal position vector, RoPE **rotates**
each pair of dimensions in Q and K by an angle proportional to position:

```
theta_i = base^(-2i/d),   i = 0 .. d/2-1
(x1, x2) -> (x1*cos(m*theta) - x2*sin(m*theta),  x1*sin(m*theta) + x2*cos(m*theta))
```

where `m` is the token's position. The dot product `q_m · k_n` after
rotation depends only on the *relative* position `m - n`, which is exactly
the property we want for a language model. Implemented efficiently in
`layers.py` by treating each (x1, x2) pair as a complex number and
multiplying by `e^{i*m*theta}`.

### SwiGLU Feed-Forward Network
```
FFN(x) = W2( SiLU(W1 x) ⊙ W3 x )
```
A "gated" FFN: `W3 x` is the normal transformation, `SiLU(W1 x)` acts as a
learned per-element gate. The hidden dimension is `8/3 * d_model` (not 4x
like vanilla Transformers) — this keeps parameter count roughly constant
vs. a non-gated FFN despite having 3 weight matrices instead of 2.

### Grouped-Query Attention (GQA)
Standard Multi-Head Attention gives every query head its own K/V head.
GQA shares one K/V head across a *group* of query heads:

```
n_heads = 64, n_kv_heads = 8  ->  8 query heads share each K/V head
```

This shrinks the KV-cache at inference time by `n_heads / n_kv_heads`,
which matters a lot once you're serving a 67B model. The paper uses plain
MHA for 7B (`n_kv_heads == n_heads`) but GQA for 67B.

### Macro design: depth over width
Table 2 in the paper:

| Params | Layers | d_model | Heads | KV Heads |
|---|---|---|---|---|
| 7B  | 30 | 4096 | 32 | 32 |
| 67B | 95 | 8192 | 64 | 8  |

Most GQA papers widen the FFN to compensate for GQA's capacity loss; this
paper instead **adds layers** (95 vs. the ~80 you'd expect from naive
scaling) — a deliberate choice, explained as improving performance and
also easing pipeline-parallel partitioning across many GPUs.

Verified parameter counts (see `model.py`, run with `torch.device('meta')`
so it doesn't need real memory):
```
7B config  -> 6.49B params
67B config -> 66.29B params
```
matching the paper's naming.

---

## 2. Scaling Laws (Section 3) — the paper's central contribution

### Why not just count parameters?
The classical formula `C ≈ 6ND` (compute ≈ 6 × parameters × tokens)
over- or under-estimates cost depending on model shape, because it ignores
the attention operation's cost and (optionally) the vocabulary projection's
cost. The paper introduces **non-embedding FLOPs per token, M**:

```
M = 72 * n_layer * d_model²              (dense FFN + QKVO projections)
  + 12 * n_layer * d_model * seq_len      (attention operation itself)
```

so that `C = M * D` is a **more accurate** stand-in for `C = 6ND`,
especially at small scale where the difference between `6N1`, `6N2`, and
`M` can be 50%+ (Table 3 — reproduced exactly by `scaling_laws.py`,
verified output below).

### Fitting hyperparameters to compute budget (Eq. 1)
Grid-searching batch size and LR at many small compute budgets and keeping
only near-optimal points (generalization error within 0.25% of the best),
the paper fits power laws:

```
eta_opt(C) = 0.3118 * C^-0.1250    (learning rate shrinks as compute grows)
B_opt(C)   = 0.2920 * C^0.3271     (batch size grows as compute grows)
```

This means: **don't hand-tune LR/batch size for every model size** — pick
your compute budget first, and these formulas hand you good starting
hyperparameters directly.

### Fitting the optimal model/data split (Eq. 4, IsoFLOP method)
For a fixed compute budget C, there's a tradeoff between training a bigger
model on fewer tokens vs. a smaller model on more tokens. The paper runs
~10 model/data allocations at each of 8 compute budgets (1e17 to 3e20),
and fits:

```
M_opt(C) = 0.1715 * C^0.5243
D_opt(C) = 5.8316 * C^0.4757
```

