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

