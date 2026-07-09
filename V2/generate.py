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

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_id], dim=1)

    out_ids = x[0].tolist()
    return "".join(itos[i] for i in out_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoint.pt")
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--length", type=int, default=200, help="number of new characters to generate")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    stoi, itos = ckpt["stoi"], ckpt["itos"]

    model = DeepSeekLLM(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])

    text = generate(
        model, stoi, itos, args.prompt,
        length=args.length, temperature=args.temperature, top_k=args.top_k,
        max_seq_len=cfg.max_seq_len, device=device,
    )
    print(text)


if __name__ == "__main__":
    main()
