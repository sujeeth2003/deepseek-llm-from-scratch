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


