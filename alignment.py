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


