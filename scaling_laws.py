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

