"""
Bare-minimum training script.

Usage:
    python3 train_simple.py my_book.txt
    python3 train_simple.py my_book.txt --steps 2000 --seq_len 128

What it does:
1. Reads a plain-text file.
2. Builds a character-level vocabulary (no tokenizer library needed).
3. Trains the DeepSeekLLM model (from model.py) on next-character prediction.
4. Saves a single checkpoint file "checkpoint.pt" containing the model
   weights, the vocab, and the config -- everything generate.py needs.
"""
import argparse
import torch

from model import DeepSeekLLM, ModelConfig


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_vocab(text: str):
    chars = sorted(list(set(text)))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    return stoi, itos


def get_batch(data: torch.Tensor, batch_size: int, seq_len: int, device: str):
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in ix])
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix])
    return x.to(device), y.to(device)


