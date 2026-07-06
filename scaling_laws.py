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


def six_n2(n_layer: int, d_model: int, n_vocab: int) -> float:
    """Complete-parameter approximation used by Hoffmann et al. (2022) / Chinchilla."""
    return 72 * n_layer * d_model ** 2 + 6 * n_vocab * d_model


def optimal_hyperparams(compute_budget: float):
    """Eq. (1): near-optimal batch size (in tokens) and learning rate for a given C."""
    eta_opt = 0.3118 * compute_budget ** (-0.1250)
    b_opt = 0.2920 * compute_budget ** 0.3271
    return {"learning_rate": eta_opt, "batch_size_tokens": b_opt}


def optimal_model_data_allocation(compute_budget: float):
    """Eq. (4): given compute budget C = M*D, what's the optimal split?"""
    m_opt = 0.1715 * compute_budget ** 0.5243
    d_opt = 5.8316 * compute_budget ** 0.4757
    return {"M_opt_flops_per_token": m_opt, "D_opt_tokens": d_opt}


def compute_budget_from_model_data(m_flops_per_token: float, d_tokens: float) -> float:
    """C = M * D"""
    return m_flops_per_token * d_tokens


if __name__ == "__main__":
    print("=" * 70)
    print("Reproducing Table 3 (model-scale representation comparison)")
    print("=" * 70)
    # rows from Table 3 of the paper
    rows = [
        (8, 512), (12, 768), (24, 1024), (24, 2048),
        (32, 4096), (40, 5120), (80, 8192),
    ]
    n_vocab, l_seq = 102400, 4096
    print(f"{'n_layer':>8} {'d_model':>8} {'6N1':>12} {'6N2':>12} {'M':>12} {'6N1/M':>8} {'6N2/M':>8}")
    for n_layer, d_model in rows:
        N1 = six_n1(n_layer, d_model)
        N2 = six_n2(n_layer, d_model, n_vocab)
        M = non_embedding_flops_per_token(n_layer, d_model, l_seq)
        print(f"{n_layer:>8} {d_model:>8} {N1:>12.3e} {N2:>12.3e} {M:>12.3e} "
              f"{N1/M:>8.2f} {N2/M:>8.2f}")

