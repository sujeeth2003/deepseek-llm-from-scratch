"""
DeepSeek LLM architecture (Section 2.2):
- Pre-Norm transformer block: x = x + Attn(RMSNorm(x)); x = x + FFN(RMSNorm(x))
- SwiGLU FFN, RoPE, GQA (for large models) / MHA (small models)
- Depth-scaling: the 67B model gets MORE LAYERS (95) rather than a WIDER FFN,
  unlike most GQA-based open models (paper explicitly calls this out).
"""
from dataclasses import dataclass
import torch
import torch.nn as nn
from layers import RMSNorm, SwiGLU, GroupedQueryAttention


@dataclass
class ModelConfig:
    vocab_size: int = 102400        # paper Sec 2.1: 100000 conventional + 15 special -> rounded to 102400
    d_model: int = 4096
    n_layers: int = 30
    n_heads: int = 32
    n_kv_heads: int = 32            # == n_heads means plain MHA (7B config)
    max_seq_len: int = 4096
    norm_eps: float = 1e-6

    @classmethod
    def deepseek_7b(cls):
        # Table 2: 7B -> 30 layers, d_model=4096, 32 heads, 32 kv_heads (MHA)
        return cls(d_model=4096, n_layers=30, n_heads=32, n_kv_heads=32)

    @classmethod
    def deepseek_67b(cls):
        # Table 2: 67B -> 95 layers, d_model=8192, 64 heads, 8 kv_heads (GQA)
        return cls(d_model=8192, n_layers=95, n_heads=64, n_kv_heads=8)

    @classmethod
    def tiny(cls, vocab_size=256):
        # a tiny config for local testing / demo training on CPU
        return cls(vocab_size=vocab_size, d_model=128, n_layers=4,
                    n_heads=4, n_kv_heads=2, max_seq_len=256)


