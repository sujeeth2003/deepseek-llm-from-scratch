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

