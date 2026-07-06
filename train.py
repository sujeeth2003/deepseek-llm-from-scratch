"""
Training loop for DeepSeek LLM, following Sections 2.3 - 2.4:

- AdamW optimizer: beta1=0.9, beta2=0.95, weight_decay=0.1
- Gradient clipping = 1.0
- Multi-step LR scheduler (NOT cosine):
    * linear warmup for 2000 steps
    * full LR until 80% of tokens processed
    * decay to 31.6% of max LR from 80% -> 90% of tokens
    * decay to 10% of max LR from 90% -> 100% of tokens
  This differs from cosine decay but reaches similar final performance
  (Fig 1a), while being reusable across continual-training runs (Fig 1b).
- Checkpoints saved periodically (paper: every 5 min async; here: every N steps)
"""
import math
import os
import time
import torch
import torch.nn.functional as F
from dataclasses import dataclass

from model import DeepSeekLLM, ModelConfig
from scaling_laws import optimal_hyperparams, optimal_model_data_allocation, compute_budget_from_model_data


@dataclass
class TrainConfig:
    total_tokens: int          # D: total training tokens (target)
    seq_len: int
    micro_batch_size: int      # sequences per forward/backward pass
    max_lr: float
    warmup_steps: int = 2000
    stage1_frac: float = 0.80  # full LR until 80% of tokens
    stage2_frac: float = 0.90  # decay to 31.6% until 90% of tokens
    stage2_lr_mult: float = 0.316
    stage3_lr_mult: float = 0.10
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    log_every: int = 20
    ckpt_every: int = 200
    ckpt_dir: str = "checkpoints"

