"""
Core building blocks used by DeepSeek LLM (Section 2.2 of the paper):
- RMSNorm            (Zhang & Sennrich, 2019)
- Rotary Embeddings   (Su et al., 2024)
- SwiGLU FFN          (Shazeer, 2020), intermediate dim = 8/3 * d_model
- Grouped-Query Attention (Ainslie et al., 2023) - used for the 67B model
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """
    RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight

    Unlike LayerNorm, there is no mean-subtraction / bias term -- only
    rescaling by the root-mean-square. Cheaper and works just as well.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # compute in fp32 for stability, then cast back
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm.to(dtype)) * self.weight


# ---------------------------------------------------------------------------
# Rotary Positional Embeddings (RoPE)
# ---------------------------------------------------------------------------
def precompute_rope_freqs(head_dim: int, max_seq_len: int, base: float = 10000.0):
    """
    theta_i = base ^ (-2i/d), i = 0 .. d/2-1
    Returns complex-exponential table of shape (max_seq_len, head_dim/2)
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq)              # (seq_len, head_dim/2)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64, e^{i*theta}


def apply_rope(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    """
    x: (batch, n_heads, seq_len, head_dim)
    Rotate pairs of dims (x1, x2) -> (x1*cos - x2*sin, x1*sin + x2*cos)
    by viewing them as complex numbers and multiplying by e^{i*theta}.
    """
    b, h, s, d = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(b, h, s, d // 2, 2))
    rope = rope_cache[:s].view(1, 1, s, d // 2)
    x_rotated = x_complex * rope
    x_out = torch.view_as_real(x_rotated).reshape(b, h, s, d)
    return x_out.type_as(x)


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------
class SwiGLU(nn.Module):
    """
    FFN(x) = W2( SiLU(W1 x) * W3 x )
    Intermediate dim is 8/3 * d_model per the paper (Sec 2.2), rounded to a
    multiple of 128 for hardware efficiency.
    """
    def __init__(self, d_model: int, multiple_of: int = 128):
        super().__init__()
        hidden = int(8 * d_model / 3)
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(d_model, hidden, bias=False)   # gate
        self.w3 = nn.Linear(d_model, hidden, bias=False)   # up
        self.w2 = nn.Linear(hidden, d_model, bias=False)   # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# Grouped-Query Attention (with RoPE)
# ---------------------------------------------------------------------------
