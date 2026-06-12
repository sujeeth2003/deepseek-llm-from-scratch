"""
Scaling laws from Section 3 of the DeepSeek LLM paper.

Key equations:
  1. Non-embedding FLOPs/token M (eq. 2), replacing the old 6N approximation:
        6N1 = 72 * n_layer * d_model^2                          (non-embed params only)
        6N2 = 72 * n_layer * d_model^2 + 6 * n_vocab * d_model   (+ vocab)
        M   = 72 * n_layer * d_model^2 + 12 * n_layer * d_model * l_seq   (+ attention cost)
     Compute budget:  C = M * D   (D = number of training tokens)

  2. Hyperparameter scaling laws (eq. 1), fit from small-scale grid search:
        eta_opt (learning rate) = 0.3118 * C^-0.1250
        B_opt   (batch size)    = 0.2920 * C^0.3271

  3. Optimal model/data allocation (eq. 4), fit via IsoFLOP profiling
     (Chinchilla-style, but using M instead of N):
        M_opt = 0.1715 * C^0.5243
        D_opt = 5.8316 * C^0.4757
"""
import math


def non_embedding_flops_per_token(n_layer: int, d_model: int, seq_len: int) -> float:
    """M in the paper -- Eq. (2), third line."""
    return 72 * n_layer * d_model ** 2 + 12 * n_layer * d_model * seq_len


def six_n1(n_layer: int, d_model: int) -> float:
    """Non-embedding-parameter approximation used by Kaplan et al. (2020)."""
    return 72 * n_layer * d_model ** 2


