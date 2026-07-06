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


class MultiStepLRScheduler:
    """
    Implements the exact 3-stage schedule described in Sec 2.3, driven by
    *tokens processed* (not just step count), matching the paper's framing.
    """
    def __init__(self, cfg: TrainConfig, tokens_per_step: int):
        self.cfg = cfg
        self.tokens_per_step = tokens_per_step
        self.total_steps = max(1, cfg.total_tokens // tokens_per_step)
        self.stage1_step = int(self.total_steps * cfg.stage1_frac)
        self.stage2_step = int(self.total_steps * cfg.stage2_frac)

    def lr_at_step(self, step: int) -> float:
        cfg = self.cfg
        # 1) linear warmup
        if step < cfg.warmup_steps:
            return cfg.max_lr * (step + 1) / cfg.warmup_steps
        # 2) full LR plateau (0% - 80% of tokens)
        if step < self.stage1_step:
            return cfg.max_lr
        # 3) decay to 31.6% (80% - 90% of tokens)
        if step < self.stage2_step:
            return cfg.max_lr * cfg.stage2_lr_mult
        # 4) decay to 10% (90% - 100% of tokens)
        return cfg.max_lr * cfg.stage3_lr_mult


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig):
    """AdamW with the paper's beta/weight-decay settings (Sec 2.3)."""
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.max_lr, betas=cfg.betas)


def get_batch(data: torch.Tensor, batch_size: int, seq_len: int, device: str):
    """Sample random contiguous chunks from a 1-D token tensor for next-token prediction."""
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in ix])
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix])
    return x.to(device), y.to(device)


