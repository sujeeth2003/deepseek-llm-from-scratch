"""
DeepSeek LLM architecture (Section 2.2):
- Pre-Norm transformer block: x = x + Attn(RMSNorm(x)); x = x + FFN(RMSNorm(x))
- SwiGLU FFN, RoPE, GQA (for large models) / MHA (small models)
- Depth-scaling: the 67B model gets MORE LAYERS (95) rather than a WIDER FFN,
  unlike most GQA-based open models (paper explicitly calls this out).
"""
from dataclasses import dataclass
import torch
import torch.nn as nn
from layers import RMSNorm, SwiGLU, GroupedQueryAttention


@dataclass
class ModelConfig:
    vocab_size: int = 102400        # paper Sec 2.1: 100000 conventional + 15 special -> rounded to 102400
    d_model: int = 4096
    n_layers: int = 30
    n_heads: int = 32
    n_kv_heads: int = 32            # == n_heads means plain MHA (7B config)
    max_seq_len: int = 4096
    norm_eps: float = 1e-6

    @classmethod
    def deepseek_7b(cls):
        # Table 2: 7B -> 30 layers, d_model=4096, 32 heads, 32 kv_heads (MHA)
        return cls(d_model=4096, n_layers=30, n_heads=32, n_kv_heads=32)

    @classmethod
    def deepseek_67b(cls):
        # Table 2: 67B -> 95 layers, d_model=8192, 64 heads, 8 kv_heads (GQA)
        return cls(d_model=8192, n_layers=95, n_heads=64, n_kv_heads=8)

    @classmethod
    def tiny(cls, vocab_size=256):
        # a tiny config for local testing / demo training on CPU
        return cls(vocab_size=vocab_size, d_model=128, n_layers=4,
                    n_heads=4, n_kv_heads=2, max_seq_len=256)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GroupedQueryAttention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.max_seq_len)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))   # pre-norm residual attention
        x = x + self.ffn(self.ffn_norm(x))     # pre-norm residual FFN
        return x


class DeepSeekLLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # weight tying (common practice, saves params)
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # paper Sec 2.3: init std = 0.006
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.006)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.006)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None):
        x = self.tok_emb(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

