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


def sequence_logprob(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sum of log p(token) over the masked (response) span, per sequence. Shape: (B,)"""
    logp = F.log_softmax(logits, dim=-1)
    token_logp = torch.gather(logp, 2, targets.unsqueeze(-1)).squeeze(-1)
    return (token_logp * mask).sum(dim=1)


def dpo_loss(
    policy_chosen_logits, policy_rejected_logits,
    ref_chosen_logits, ref_rejected_logits,
    chosen_targets, rejected_targets,
    chosen_mask, rejected_mask,
    beta: float = 0.1,
):
    """
    Direct Preference Optimization loss (Rafailov et al., 2023), as used in
    Section 4 of the DeepSeek LLM paper to align helpfulness/harmlessness.
    """
    pi_chosen = sequence_logprob(policy_chosen_logits, chosen_targets, chosen_mask)
    pi_rejected = sequence_logprob(policy_rejected_logits, rejected_targets, rejected_mask)
    ref_chosen = sequence_logprob(ref_chosen_logits, chosen_targets, chosen_mask)
    ref_rejected = sequence_logprob(ref_rejected_logits, rejected_targets, rejected_mask)

    pi_logratios = pi_chosen - pi_rejected
    ref_logratios = ref_chosen - ref_rejected

    logits = beta * (pi_logratios - ref_logratios)
    loss = -F.logsigmoid(logits).mean()

    # useful metrics to log during training
    chosen_rewards = beta * (pi_chosen - ref_chosen).detach()
    rejected_rewards = beta * (pi_rejected - ref_rejected).detach()
    accuracy = (chosen_rewards > rejected_rewards).float().mean()

    return loss, {"accuracy": accuracy.item(),
                  "chosen_reward": chosen_rewards.mean().item(),
                  "rejected_reward": rejected_rewards.mean().item()}
