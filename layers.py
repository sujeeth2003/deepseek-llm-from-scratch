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
