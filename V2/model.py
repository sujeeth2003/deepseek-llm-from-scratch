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
