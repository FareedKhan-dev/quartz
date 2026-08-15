"""LoRA on the five attention projections, and the trap that looks like a bug.

The base model does not know your tools. Grafting is how a few hundred of your
own examples get in without retraining anything, and there are three decisions
in it.

**Only the attention projections are adapted.** ``q_proj``, ``k_proj``,
``v_proj``, ``gate_proj`` and ``out_proj`` are where a layer decides what to
read and what to write, which is what changes when the tools change. Spin's
diagonals, the Loom routers and the Imprint tables all carry structure that is
global to the model rather than to a task, and a rank-16 nudge to a router that
feeds a Sinkhorn iteration moves the whole residual stream.

**The adapter has the stacked layer axis.** ``nn.scan`` gave every parameter a
leading layer axis, so one adapter of shape ``(layers, in, rank)`` covers all
twenty seven layers at once. Five weight groups, not 135 tensors.

**B starts at exactly zero.** The merged weight is then exactly the base weight
at step zero, so grafting can only be an improvement on a measurement you have
already taken, and a broken adapter is visible as "nothing happened" rather
than as a model that got worse for a reason you now have to find.

Then the trap. Two hundred examples at batch 16 for three epochs is 39
optimiser steps, and 39 steps of a rank-16 adapter at 1e-4 moves the loss by
nineteen thousandths. Fine-tuning looks broken and is only barely being asked
to move::

      steps    train    val
         39   0.9810  0.9840
        130   0.7120  0.7410
        260   0.4880  0.5520
        390   0.3710  0.4890
        520   0.3020  0.4780
        650   0.2570  0.5010

So :func:`graft` reports the step count it is about to run and says so when it
is under :data:`MIN_USEFUL_STEPS`. Raise the epochs before you raise the
learning rate, and watch the validation loss for the turn, which in that table
is somewhere past 390.
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_RANK",
    "LORA_TARGETS",
    "MIN_USEFUL_STEPS",
    "GraftPlan",
    "graft",
    "init_lora",
    "lora_target_paths",
    "merge_lora",
    "recommended_epochs",
]

#: The five projections a graft touches. Nothing else in the model is adapted.
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "gate_proj", "out_proj")

DEFAULT_RANK = 16
DEFAULT_ALPHA = 32

#: Below this many optimiser steps a rank-16 adapter has not moved far enough
#: to be measurable, and the run will look like a failure of the method rather
#: than of the schedule. Read off the table in the module docstring.
MIN_USEFUL_STEPS = 260

#: Training-only subtrees. Foresight owns a second copy of every projection
#: name and is dropped at export, so adapting it would train weights that never
#: reach the device.
_TRAINING_ONLY = ("fs_", "foresight")


def _flatten(tree: Any, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    """Flatten a parameter tree to {path: leaf}, without needing Flax."""
    out: dict[tuple[str, ...], Any] = {}
    if isinstance(tree, Mapping):
        for key, value in tree.items():
            out.update(_flatten(value, (*prefix, str(key))))
    else:
        out[prefix] = tree
    return out


def _unwrap(params: Mapping[str, Any]) -> Mapping[str, Any]:
    """Accept either the params collection or a whole Flax variable dict."""
    inner = params.get("params")
    return inner if isinstance(inner, Mapping) else params


def _set_in(tree: Any, path: Sequence[str], value: Any) -> dict[str, Any]:
    """A copy of `tree` with one leaf replaced, sharing everything else."""
    node = dict(tree)
    if len(path) == 1:
        node[path[0]] = value
        return node
    node[path[0]] = _set_in(node[path[0]], path[1:], value)
    return node


def lora_target_paths(params: Mapping[str, Any], *,
                      targets: Sequence[str] = LORA_TARGETS,
                      scope: str = "layers") -> list[tuple[str, ...]]:
    """Every kernel a graft adapts, as paths into the parameter tree.

    ``scope`` keeps the search inside the scanned trunk, which is what makes
    the "five weight groups covering twenty seven layers" claim true: a match
    outside it would be a single unscanned layer wearing the same name.

    A projection the config disabled would get an adapter that trains against
    a zero weight and merges back into nothing, so anything that is uniformly
    zero is skipped rather than quietly wasting a fifth of the trainable
    parameters.
    """
    flat = _flatten(_unwrap(params))
    wanted = tuple(targets)
    paths = [
        path for path in flat
        if path[-1] == "kernel"
        and scope in path
        and any(t in path for t in wanted)
        and not any(seg.startswith(_TRAINING_ONLY) for seg in path)
    ]
    live = [p for p in sorted(paths) if float(np.max(np.abs(np.asarray(flat[p])))) > 1e-6]
    if not live:
        sample = ", ".join("/".join(p) for p in sorted(flat)[:3])
        raise ValueError(
            f"no adaptable kernel under {scope!r} matching {wanted}; the tree "
            f"starts {sample}. Pass the params collection of a QuartzNetwork, "
            "not the full variable dict of some other model.")
    return live


def init_lora(w: Any, rank: int, key: Any) -> dict[str, Any]:
    """One adapter for one weight: A random, B zero.

    ``w`` keeps its leading axes, so a scanned ``(layers, in, out)`` kernel
    gives ``A: (layers, in, rank)`` and ``B: (layers, rank, out)`` and the whole
    stack is adapted by one pair of tensors.

    A is divided by the rank rather than scaled by 1/sqrt(in), so the product
    ``A @ B`` starts at the same magnitude whatever rank you pick and the
    learning rate does not have to be retuned alongside it. The adapters are
    float32 even under a bf16 model: they are 4.6 percent of the weights and
    the only thing carrying a gradient.
    """
    import jax
    import jax.numpy as jnp

    if rank < 1:
        raise ValueError(f"rank must be positive, got {rank}")
    w = jnp.asarray(w)
    if w.ndim < 2:
        raise ValueError(f"LoRA needs a matrix, got shape {w.shape}")
    lead = w.shape[:-2]
    return {
        "A": jax.random.normal(key, (*lead, w.shape[-2], rank), jnp.float32) / rank,
        # exactly zero, so the merged weight is exactly W at step zero
        "B": jnp.zeros((*lead, rank, w.shape[-1]), jnp.float32),
    }


def merge_lora(params: Mapping[str, Any], lora: Mapping[tuple[str, ...], Any], *,
               alpha: float = DEFAULT_ALPHA, rank: int | None = None) -> dict[str, Any]:
    """Fold every adapter back into the weights it was attached to.

    ``W + (alpha / rank) * A @ B``, cast to the base weight's dtype, so what
    comes out is an ordinary parameter tree: the same shapes, the same names,
    no adapter left to carry around and nothing for the exporter or the
    quantiser to learn about. Scaling by ``alpha / rank`` is what keeps the
    update the same size when the rank changes.
    """
    import jax.numpy as jnp

    merged: dict[str, Any] = dict(params)
    flat = _flatten(merged)          # the paths are distinct, so one pass is enough
    for raw_path, adapter in lora.items():
        path = tuple(raw_path)
        if path not in flat:
            raise KeyError(f"adapter at {'/'.join(path)} has no base weight")
        a, b = jnp.asarray(adapter["A"]), jnp.asarray(adapter["B"])
        r = a.shape[-1] if rank is None else int(rank)
        base = jnp.asarray(flat[path])
        delta = jnp.einsum("...ir,...ro->...io", a, b) * (float(alpha) / r)
        if delta.shape != base.shape:
            raise ValueError(
                f"{'/'.join(path)}: adapter merges to {delta.shape}, weight is "
                f"{base.shape}")
        merged = _set_in(merged, path, base + delta.astype(base.dtype))
    return merged


def recommended_epochs(n_examples: int, batch: int = 16, *,
                       target_steps: int = 390, lo: int = 10, hi: int = 30) -> int:
    """How many epochs a few hundred examples actually need.

    Three is the default everywhere and three is wrong here. The table in the
    module docstring turns at somewhere past 390 steps, so this aims there and
    clamps to the ten-to-thirty band the post recommends.
    """
    per_epoch = max(1, math.ceil(max(1, int(n_examples)) / max(1, int(batch))))
    return int(min(hi, max(lo, math.ceil(target_steps / per_epoch))))


@dataclass
class GraftPlan:
    """The adapters, and the arithmetic of the run about to be started."""

    paths: list[tuple[str, ...]]
    lora: dict[tuple[str, ...], Any]
    rank: int
    alpha: float
    trainable: int
    base: int
    n_examples: int | None = None
    batch: int = 16
    epochs: int | None = None
    steps: int | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def groups(self) -> int:
        """Weight groups, not tensors: each one covers every scanned layer."""
        return len(self.paths)

    @property
    def share(self) -> float:
        return self.trainable / self.base if self.base else 0.0

    def is_identity(self) -> bool:
        """True while every B is still exactly zero, so a merge is a no-op.

        This is the check worth running before the first optimiser step: if it
        is ever False at step zero, the adapters were initialised wrong and
        every number after it is measuring a different model.
        """
        return all(not np.any(np.asarray(a["B"])) for a in self.lora.values())

    def merge(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Fold this plan's adapters into `params`."""
        return merge_lora(params, self.lora, alpha=self.alpha, rank=self.rank)

    def summary(self) -> str:
        lines = [
            f"  lora      rank {self.rank}  alpha {self.alpha:g}  "
            f"{self.groups} weight groups",
            f"  params    {self.trainable:,} trainable of {self.base:,}  "
            f"({self.share:.2%})",
            f"  check     merged == base at step 0: {self.is_identity()}",
        ]
        if self.steps is not None:
            lines.append(f"  schedule  {self.n_examples} examples / batch {self.batch} "
                         f"x {self.epochs} epochs = {self.steps} steps")
        lines.extend(f"  warning   {w}" for w in self.warnings)
        return "\n".join(lines)


def graft(params: Mapping[str, Any], *, rank: int = DEFAULT_RANK,
          alpha: float = DEFAULT_ALPHA, seed: int = 0,
          targets: Sequence[str] = LORA_TARGETS, n_examples: int | None = None,
          batch: int = 16, epochs: int | None = None) -> GraftPlan:
    """Attach adapters to the five projections and price the run.

    Returns the plan rather than a trained model: the optimiser loop belongs to
    :mod:`quartz.train.sft`, which takes ``plan.lora`` as the only trainable
    tree and calls ``plan.merge`` when it is done. Everything this function
    knows is what a graft costs and whether the schedule is long enough to
    move it, which is the part people get wrong.

    Pass ``n_examples`` to have the step arithmetic checked. With ``epochs``
    left at None it picks a count from :func:`recommended_epochs` rather than
    inheriting the three-epoch default that produces 39 steps and a flat curve.
    """
    import jax

    paths = lora_target_paths(params, targets=targets)
    flat = _flatten(_unwrap(params))
    keys = jax.random.split(jax.random.PRNGKey(seed), len(paths))
    lora = {p: init_lora(flat[p], rank, k) for p, k in zip(paths, keys, strict=True)}

    trainable = sum(int(np.asarray(a["A"]).size + np.asarray(a["B"]).size)
                    for a in lora.values())
    base = sum(int(np.asarray(v).size) for k, v in flat.items()
               if not any(seg.startswith(_TRAINING_ONLY) for seg in k))

    notes: list[str] = []
    steps = None
    if n_examples is not None:
        if epochs is None:
            epochs = recommended_epochs(n_examples, batch)
            notes.append(f"epochs not given, using {epochs} rather than the usual 3")
        per_epoch = max(1, math.ceil(max(1, int(n_examples)) / max(1, int(batch))))
        steps = per_epoch * int(epochs)
        if steps < MIN_USEFUL_STEPS:
            short = (
                f"{steps} optimiser steps is under {MIN_USEFUL_STEPS}: a rank-{rank} "
                f"adapter will barely move and the graft will look broken. Raise "
                f"epochs to about {recommended_epochs(n_examples, batch)} before "
                "touching the learning rate.")
            notes.append(short)
            # the only thing here worth interrupting for: everything else is a
            # default being reported, and a warning nobody needs is a warning
            # everybody learns to ignore
            warnings.warn(short, RuntimeWarning, stacklevel=2)

    return GraftPlan(paths=paths, lora=lora, rank=rank, alpha=float(alpha),
                     trainable=trainable, base=base, n_examples=n_examples,
                     batch=batch, epochs=epochs, steps=steps, warnings=notes)
