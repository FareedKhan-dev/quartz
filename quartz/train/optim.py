"""Muon on the stacked weight matrices, AdamW on everything else.

A raw gradient pushes hard along a few directions and barely along the rest.
Newton-Schulz pushes the update towards the nearest orthogonal matrix, which
evens the directions out, and it does it with matrix multiplies only: no SVD,
no QR, nothing that serialises on an accelerator.

That only makes sense for something that *is* a linear map. A bias, a Spin
diagonal, a norm gain and the embedding are not, so they keep AdamW.

This module also holds the small pieces every stage needs -- argument coercion,
the schedule, and the two calls into the checkpoint format -- because all four
stages build an optimiser and all four save a checkpoint, and one copy of each
is easier to keep honest than four.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, NamedTuple

from quartz.model.config import QuartzConfig, preset

#: The published quintic coefficients. They are tuned so the iteration expands
#: small singular values fast while staying stable near one; changing them
#: quietly changes the update rule, so they are named rather than inlined.
NS_COEFFS = (3.4445, -4.7750, 2.0315)
NS_STEPS = 5

#: Leaves whose rows are memory rather than a linear map. Orthogonalising a
#: lookup table would force every row to the same length, which is the one
#: property a table of learned facts must be free to break. Matching Imprint by
#: name is deliberately blunt: sending one of its projections to AdamW costs a
#: little, sending one of its nine million table rows to Muon costs a lot.
LOOKUP_HINTS = ("embedding", "table", "imprint")


# --- arguments -------------------------------------------------------------
@dataclass
class OptimArgs:
    """Everything `build_optimiser` reads.

    Stage-specific argument classes subclass this, so a single Namespace from
    the CLI can be poured into any of them.
    """

    lr: float = 3e-3
    warmup: int = 0            # 0 means warmup_frac of total_steps
    warmup_frac: float = 0.05
    total_steps: int = 0       # 0 means "derive it from the token budget"
    end_lr_frac: float = 0.01  # the cosine floor, as a fraction of the peak
    momentum: float = 0.95     # Muon's heavy ball
    b1: float = 0.9
    b2: float = 0.95           # lower than the 0.999 default: the loss is noisy
    weight_decay: float = 0.1
    clip: float = 1.0
    ns_steps: int = NS_STEPS
    optimiser: str = "muon"    # "adamw" sends the matrices to AdamW too: the ablation


def as_args(cls, args: Any = None, **overrides):
    """Return an instance of `cls`, whatever the caller had to hand.

    The CLI passes an argparse.Namespace, a test passes the dataclass, a
    notebook passes nothing and a few keywords. Unknown keys are dropped
    rather than raising, matching QuartzConfig, and an override of None means
    "not supplied" so a default is never clobbered by an unset CLI flag.
    """
    names = {f.name for f in fields(cls)}
    values: dict[str, Any] = {}
    if args is not None:
        source = vars(args) if hasattr(args, "__dict__") else dict(args)
        values.update({k: v for k, v in source.items() if k in names})
    values.update({k: v for k, v in overrides.items() if k in names and v is not None})
    return cls(**values)


def config_from_args(args: Any) -> QuartzConfig:
    """The geometry for a run: a yaml path if given, otherwise a named preset."""
    path = getattr(args, "config", "")
    if path:
        return QuartzConfig.from_yaml(path)
    return preset(getattr(args, "preset_name", "base"))


def learning_rate_schedule(args: Any):
    """Warmup then cosine, floored at `end_lr_frac` of the peak.

    Both arms read this same schedule, so the two halves of the model never
    drift apart in effective step size.

    `warmup` is a step count, and a value below 1 is read as a fraction of the
    schedule instead. Every stage's `--warmup` flag documents both spellings,
    but only stage one resolved the fraction before handing it over, so
    `--warmup 0.2` on stages two and three used to truncate to 0 and fall
    through to `warmup_frac` -- a different number, silently.
    """
    import optax

    a = as_args(OptimArgs, args)
    total = max(1, int(a.total_steps))
    fraction = a.warmup if 0.0 < a.warmup < 1.0 else a.warmup_frac
    warmup = int(a.warmup) or max(1, int(total * fraction))
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=a.lr, warmup_steps=warmup,
        decay_steps=total, end_value=a.lr * a.end_lr_frac)


# --- orthogonalisation -----------------------------------------------------
def newton_schulz(g, steps: int = NS_STEPS):
    """Push a matrix towards the nearest orthogonal one, using only matmuls.

    Five iterations of a fixed quintic. The result is not exactly orthogonal
    and does not need to be: what matters is that the singular values are
    squeezed into a narrow band, so no direction of the update dominates.
    """
    import jax.numpy as jnp

    a, b, c = NS_COEFFS
    x = jnp.asarray(g)
    if x.ndim != 2:
        raise ValueError(f"newton_schulz wants a matrix, got shape {x.shape}")
    orig = x.dtype
    x = x.astype(jnp.float32)
    # Normalise first, so the singular values start inside [0, 1] where the
    # iteration converges, and transpose tall matrices because the iteration
    # only ever touches the smaller of the two Gram matrices.
    x = x / (jnp.linalg.norm(x) + 1e-7)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        a_gram = x @ x.T
        x = a * x + (b * a_gram + c * a_gram @ a_gram) @ x
    if transposed:
        x = x.T
    return x.astype(orig)


def orthogonalise(g, steps: int = NS_STEPS, shape_scale: bool = True):
    """Newton-Schulz over the last two axes, whatever the leading ones are.

    Under `nn.scan` every trunk matrix carries a leading layer axis, so the
    twenty seven layers are orthogonalised independently rather than as one
    tall block, which would mix layers that share nothing.
    """
    import jax
    import jax.numpy as jnp

    x = jnp.asarray(g)
    if x.ndim < 2:
        return x
    if x.ndim == 2:
        out = newton_schulz(x, steps)
    else:
        flat = x.reshape(-1, *x.shape[-2:])
        out = jax.vmap(lambda m: newton_schulz(m, steps))(flat).reshape(x.shape)
    if shape_scale:
        # An orthogonal update has unit singular values, so its RMS depends on
        # the aspect ratio. This factor makes a wide and a tall matrix take
        # steps of comparable size under one learning rate.
        rows, cols = x.shape[-2], x.shape[-1]
        out = out * math.sqrt(max(1.0, rows / cols))
    return out


class MuonState(NamedTuple):
    """Momentum only. There is no second moment: the orthogonalisation is what
    normalises the step, and a per-coordinate scale on top would undo it."""

    mu: Any


def muon(momentum: float = 0.95, *, nesterov: bool = True,
         ns_steps: int = NS_STEPS, shape_scale: bool = True):
    """The Muon transformation, as an optax GradientTransformation.

    It emits a *direction*: no learning rate and no sign. Whoever chains it is
    responsible for both, which is the trap `build_optimiser` documents.
    """
    import jax
    import jax.numpy as jnp
    import optax

    def init_fn(params):
        return MuonState(mu=jax.tree_util.tree_map(jnp.zeros_like, params))

    def update_fn(updates, state, params=None):
        del params
        mu = jax.tree_util.tree_map(lambda m, g: momentum * m + g, state.mu, updates)
        # Nesterov looks one momentum step ahead, which on a schedule this long
        # is worth the extra tree_map and nothing else.
        direction = (jax.tree_util.tree_map(lambda m, g: g + momentum * m, mu, updates)
                     if nesterov else mu)
        new = jax.tree_util.tree_map(
            lambda g: orthogonalise(g, ns_steps, shape_scale), direction)
        return new, MuonState(mu=mu)

    return optax.GradientTransformation(init_fn, update_fn)


# --- the split -------------------------------------------------------------
def path_str(path) -> str:
    """A dotted name for a jax tree path, without importing jax to read it."""
    parts = []
    for key in path:
        name = getattr(key, "key", None)
        if name is None:
            name = getattr(key, "name", None)
        if name is None:
            name = getattr(key, "idx", None)
        parts.append(str(key if name is None else name))
    return ".".join(parts)


def matrix_leaf(path, x) -> bool:
    """True for the leaves Muon should own.

    Under `nn.scan` a real matrix is 3-D -- (layers, in, out) -- while a bias,
    a Spin diagonal or a norm gain is 2-D and the attention gate is 1-D.
    Orthogonalising a vector is meaningless, so the rank test does the work.
    Lookup tables are excluded by name because they are 3-D too and their rows
    are memory, not a map.
    """
    name = path_str(path).lower()
    return getattr(x, "ndim", 0) >= 3 and not any(h in name for h in LOOKUP_HINTS)


def label_tree(params, is_matrix=None):
    """A pytree of "muon"/"adamw" labels, matching `params` leaf for leaf."""
    import jax

    test = is_matrix or matrix_leaf
    return jax.tree_util.tree_map_with_path(
        lambda path, x: "muon" if test(path, x) else "adamw", params)


def split_summary(params, is_matrix=None) -> dict[str, dict[str, int]]:
    """Tensor and parameter counts per arm, for the line the run prints.

    Counted after `nn.scan` has stacked the layers, so one entry here is
    twenty seven tensors on the device.
    """
    import jax

    test = is_matrix or matrix_leaf
    out = {"muon": {"tensors": 0, "params": 0}, "adamw": {"tensors": 0, "params": 0}}
    leaves, _ = jax.tree_util.tree_flatten_with_path(params)
    for path, leaf in leaves:
        arm = "muon" if test(path, leaf) else "adamw"
        out[arm]["tensors"] += 1
        out[arm]["params"] += int(getattr(leaf, "size", 0))
    return out


def build_optimiser(args: Any, params, *, is_matrix=None):
    """The two-armed chain every stage trains under.

    The order of the chain is the whole point:

    * Muon returns a raw direction with no learning rate and no sign, so it
      needs its own `scale_by_schedule` with a NEGATIVE schedule. Leave the
      minus off and every matrix in the model ascends the loss.
    * `optax.adamw` has already folded -lr into its update. Wrapping it in the
      same negative scaling multiplies two negatives, flips the sign back to
      positive, and ascends on the vectors and the embedding instead. AdamW
      must NOT be scaled a second time.
    * Clipping is first, on the raw gradient. Clipping after orthogonalisation
      would rescale something whose norm Newton-Schulz has already fixed by
      construction, which is both pointless and a silent learning-rate change.

    `optimiser="adamw"` is the ablation the post reports against: the chain is
    unchanged and the Muon arm simply gets nothing, so the two runs differ in
    the update rule for the matrices and in nothing else.
    """
    import optax

    a = as_args(OptimArgs, args)
    if a.optimiser not in ("muon", "adamw"):
        raise ValueError(
            f"optimiser must be 'muon' or 'adamw', got {a.optimiser!r}")
    schedule = learning_rate_schedule(a)
    labels = (label_tree(params, is_matrix) if a.optimiser == "muon"
              else label_tree(params, lambda path, x: False))
    arms = {
        "muon": optax.chain(
            muon(momentum=a.momentum, ns_steps=a.ns_steps),
            optax.scale_by_schedule(lambda count: -schedule(count)),
        ),
        "adamw": optax.adamw(schedule, b1=a.b1, b2=a.b2, weight_decay=a.weight_decay),
    }
    return optax.chain(optax.clip_by_global_norm(a.clip),
                       optax.partition(arms, labels))


# --- checkpoints -----------------------------------------------------------
def unpack_checkpoint(obj) -> tuple[Any, QuartzConfig | None]:
    """Normalise whatever `quartz.model.run` hands back into (params, config).

    The record on disk carries a format version and its geometry beside the
    weights; readers differ on whether they return the record or the pair.
    Both are accepted here so every stage has one code path.
    """
    params, cfg = obj, None
    if isinstance(obj, tuple):
        params, cfg = obj[0], (obj[1] if len(obj) > 1 else None)
    elif isinstance(obj, dict) and "params" in obj:
        params, cfg = obj["params"], obj.get("config")
    if isinstance(cfg, dict):
        cfg = QuartzConfig(**cfg)
    if params is None:
        raise ValueError("checkpoint contained no parameters")
    return params, cfg


def read_checkpoint(path: str) -> tuple[Any, QuartzConfig | None]:
    """Load a stage checkpoint. One call site, so the run.py contract is
    honoured in a single place."""
    from quartz.model.run import load_checkpoint

    return unpack_checkpoint(load_checkpoint(path))


def write_checkpoint(path: str, params, cfg: QuartzConfig):
    """Save a stage checkpoint, creating the directory if it is missing."""
    from quartz.model.run import save_checkpoint

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return save_checkpoint(path, params, cfg)


def average_stage_checkpoints(paths):
    """The mean of the last few checkpoints, in parameter space.

    Not an ensemble: the last few thousand steps of a cosine schedule bounce
    around one basin, and the mean of those samples is a better point than any
    single one of them.
    """
    from quartz.model.run import average_checkpoints

    return unpack_checkpoint(average_checkpoints(list(paths)))[0]
