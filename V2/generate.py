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

from model import DeepSeekLLM


def generate(model, stoi, itos, prompt: str, length: int, temperature: float,
             top_k: int, max_seq_len: int, device: str):
    model.eval()

    # encode prompt -> list of token ids (unknown chars are skipped)
    ids = [stoi[c] for c in prompt if c in stoi]
    if len(ids) == 0:
        raise ValueError("None of the characters in --prompt were seen during training.")
    x = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(length):
            x_cond = x[:, -max_seq_len:]  # truncate to model's context window
            logits, _ = model(x_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

