"""
Bare-minimum text generation script.

Usage:
    python3 generate.py --prompt "Once upon a time"
    python3 generate.py --prompt "The dog" --length 300 --temperature 0.8

Loads the checkpoint saved by train_simple.py and autoregressively samples
new characters one at a time, feeding each prediction back in as input.
"""
import argparse
import torch
import torch.nn.functional as F

