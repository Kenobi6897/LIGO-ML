"""Stage 1, Step 5 — the 1D CNN. Modest by design.

This is the Gabbard-class problem (arXiv:1712.06041), not ImageNet: ~250k parameters,
three conv blocks, one hidden dense layer. The 3070 is wildly overkill for it, and the
plan says not to let the GPU shape the architecture. It doesn't.

    Conv1d -> BatchNorm -> ReLU -> MaxPool   (x3, widening 16 -> 32 -> 64)
    -> Flatten -> Dense(128) -> Dropout -> Dense(1)

WHY THE FIRST KERNEL IS LONG — 64 samples = 31 ms at 2048 Hz.
A 1D convolution slides a kernel along the strain computing a dot product at every lag,
which is *exactly* what correlating against a template does. Layer 1 is therefore,
functionally, a learned template bank — and Step 7's payoff figure plots it. For those
kernels to be able to *look like* chirp snippets they must be long enough to hold real
oscillations: 31 ms is ~1 cycle at 30 Hz and ~11 at 350 Hz, spanning the analysis band.
(The setup gate ran exactly this shape on the 3070: (256, 1, 2048) -> (256, 16, 1985).)

WHAT IS DELIBERATELY ABSENT — both are [[Stage 1#Footguns]]:
  - NO per-segment input normalisation (no LayerNorm on the input, no x/x.std()).
    Adding a signal changes a segment's variance, so a per-segment rescale hands the CNN
    the label. `condition()` already set the scale with one global constant.
  - NO sigmoid at the output. The head emits a raw logit for `BCEWithLogitsLoss`
    (numerically stabler than sigmoid + BCELoss); eval applies sigmoid itself when it
    wants probabilities. Thresholds are set on scores, so the choice changes nothing.

Conv layers carry no bias — each is immediately followed by BatchNorm, which has its own
shift and would silently absorb (and waste) a conv bias.
"""

from __future__ import annotations

import torch
from torch import nn

N_INPUT = 2048  # samples — the 1 s crop at 2048 Hz, fixed by the Stage 1 decisions
FIRST_KERNEL = 64  # samples — long enough for Step 7's kernels to hold a chirp cycle


def _block(c_in: int, c_out: int, k: int) -> list[nn.Module]:
    return [
        nn.Conv1d(c_in, c_out, k, bias=False),
        nn.BatchNorm1d(c_out),
        nn.ReLU(),
        nn.MaxPool1d(4),
    ]


class Stage1CNN(nn.Module):
    """(N, 1, 2048) whitened strain -> (N, 1) logit. Sigmoid is the caller's job."""

    def __init__(self, n_in: int = N_INPUT):
        super().__init__()
        self.features = nn.Sequential(
            *_block(1, 16, FIRST_KERNEL),  # 2048 -> 1985 -> pool -> 496
            *_block(16, 32, 8),            #  496 ->  489 -> pool -> 122
            *_block(32, 64, 8),            #  122 ->  115 -> pool ->  28
        )
        # Measure the flatten size instead of hard-coding 64*28: a change to any kernel
        # or pool above adjusts the head automatically instead of failing at runtime.
        with torch.no_grad():
            n_flat = self.features(torch.zeros(1, 1, n_in)).numel()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    def first_layer_kernels(self) -> "np.ndarray":
        """(16, 64) — the learned template bank, for Step 7's figure."""
        return self.features[0].weight.detach().cpu().numpy()[:, 0, :]


def load_checkpoint(path) -> tuple["Stage1CNN", dict]:
    """The trained model, in eval mode, plus the checkpoint metadata.

    Used by everything downstream of training (eval, kernels, GW150914). eval() is not
    optional: BatchNorm must use its running statistics — in train mode a *batch of one
    segment* normalises by its own statistics, which is the per-segment leak reborn at
    inference time, and it produces garbage scores that look like a broken model.
    """
    ckpt = torch.load(path, map_location="cpu")
    model = Stage1CNN()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


if __name__ == "__main__":
    model = Stage1CNN()
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.zeros(4, 1, N_INPUT)
    print(model)
    print(f"\nparameters: {n_params:,}")
    print(f"forward: {tuple(x.shape)} -> {tuple(model(x).shape)}")
