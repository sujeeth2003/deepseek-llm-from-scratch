"""
Alignment pipeline (Section 4): Supervised Fine-Tuning (SFT) + Direct
Preference Optimization (DPO, Rafailov et al. 2023).

SFT: ordinary next-token cross-entropy, but the loss is masked so it is
     only computed over the "response" tokens, not the "prompt" tokens.

DPO: given a preference pair (chosen, rejected) for the same prompt, and
     a frozen reference model (the SFT model before DPO), the loss is:

        L_DPO = -log sigmoid( beta * [
                    (logpi(chosen) - logpi_ref(chosen))
                  - (logpi(rejected) - logpi_ref(rejected))
                ] )

     This directly increases the model's relative preference for the
     chosen response over the rejected one, without needing an explicit
     reward model (hence "direct" preference optimization).
"""
import torch
import torch.nn.functional as F


def sft_loss(logits: torch.Tensor, targets: torch.Tensor, response_mask: torch.Tensor):
    """
    logits: (B, T, V), targets: (B, T), response_mask: (B, T) with 1s where
    the token belongs to the model's response (loss computed there) and 0s
    over the prompt (loss ignored there).
    """
    logp = F.log_softmax(logits, dim=-1)
    token_logp = torch.gather(logp, 2, targets.unsqueeze(-1)).squeeze(-1)  # (B, T)
    masked = token_logp * response_mask
    # average over response tokens only
    loss = -(masked.sum(dim=1) / response_mask.sum(dim=1).clamp(min=1)).mean()
    return loss


