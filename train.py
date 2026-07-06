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

