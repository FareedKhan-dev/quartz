"""The four training stages, and the optimiser all of them share.

    optim     Muon on the stacked matrices, AdamW on everything else.
    pretrain  stage one: 120B packed tokens from random initialisation.
    sft       stage two: 1.2M teacher-written calls, scored on the target only.
    qapt      stage three: the same data again, with the two-bit codebook in
              the forward pass, so the weights land where Grist will round them.
    heads     stage four: Gauge, Dowser, and the synthetic curriculum that
              teaches copying before the words mean anything.

Importing this package must not require JAX. Every stage imports jax, flax and
optax inside the function that needs them, so a device-side install with numpy
and sentencepiece can still `import quartz.train` to read a constant out of it.
"""
from quartz.train import heads, optim, pretrain, qapt, sft

__all__ = ["heads", "optim", "pretrain", "qapt", "sft"]
