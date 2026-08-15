"""Stage three: quantisation-aware post-training.

Grist rounds a rotated weight group to a two-bit index and one half-precision
length, and at two bits that rounding costs one and eight tenths of a point on
Errand. Most of that is recoverable, because the weights are free to move
towards the lattice they are about to be snapped onto -- if training can see
the snapping.

So the forward pass runs on quantised weights while the optimiser keeps a full
precision master copy. Two mechanisms, in that order:

* **Noise.** For the first part of the run the weights get additive Gaussian
  noise whose scale is the distortion the codebook actually makes on that
  tensor, measured once with Grist itself. It is the same magnitude of
  perturbation without the discontinuity, so the model anneals into it.
* **Straight-through.** After that the real codebook is in the forward pass and
  the gradient is passed through the rounding unchanged, because the true
  derivative is zero almost everywhere and would train nothing.

At the end the bits map is stamped onto `config.weight_bits`, so the exporter
quantises exactly the scheme this run adapted to and nobody has to remember a
flag.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from quartz.model.config import CQ_GROUP, DEFAULT_BITS_MAP
from quartz.train.optim import (
    LOOKUP_HINTS,
    OptimArgs,
    as_args,
    config_from_args,
    path_str,
    write_checkpoint,
)
from quartz.train.pretrain import Z_LOSS_WEIGHT, load_or_init, make_state
from quartz.train.sft import batches, dataset, fit_max_len, masked_losses, steps_per_epoch

#: Fraction of the run spent on noise before the hard codebook takes over.
#: Long enough for the weights to stop moving in response to the perturbation,
#: short enough that most of the run trains against the real rounding.
NOISE_FRACTION = 0.3


@dataclass
class QaptArgs(OptimArgs):
    """Stage three's knobs. 1.8B tokens is three more passes over the stage two
    data, at a tenth of the stage two learning rate: this is an adaptation, not
    a refit."""

    config: str = ""
    preset_name: str = "base"
    data: str = ""
    init: str = "ckpt/quartz_sft.pkl"
    out: str = "ckpt/quartz_qapt.pkl"
    tokenizer: str = ""
    bits_map: str = DEFAULT_BITS_MAP
    group: int = CQ_GROUP
    tokens: float = 0.0            # 0 means use `epochs` instead
    epochs: int = 3
    batch_size: int = 16
    cap: int = 1024
    seed: int = 0
    log_every: int = 500
    noise_fraction: float = NOISE_FRACTION
    noise_scale: float = 1.0       # multiplies the measured distortion; 1.0 is it
    lr: float = 3e-5
    z_weight: float = Z_LOSS_WEIGHT


class LeafPlan:
    """What the quantiser will do to one leaf, decided once."""

    __slots__ = ("axis", "bits", "index", "sigma")

    def __init__(self, bits: float, axis: int, index: int, sigma: float = 0.0):
        self.bits, self.axis, self.index, self.sigma = bits, axis, index, sigma

    def __repr__(self) -> str:  # pragma: no cover - debugging only
        return f"LeafPlan(bits={self.bits}, axis={self.axis}, sigma={self.sigma:.3g})"


# --- the codebook, in jnp --------------------------------------------------
@lru_cache(maxsize=8)
def _levels(bits: float, group: int) -> np.ndarray:
    """Grist's codebook, sorted. Cached because Lloyd-Max runs 200 sweeps and
    the answer depends on nothing that changes during a run."""
    from quartz.model.grist import codebook

    return np.sort(np.asarray(codebook(bits, group), dtype=np.float32))


@lru_cache(maxsize=8)
def _walsh(group: int) -> np.ndarray:
    from quartz.model.grist import walsh_np

    return np.asarray(walsh_np(group), dtype=np.float32)


def default_axis(ndim: int) -> int:
    """The kernel convention: group along the reduction axis of a matrix."""
    return -2 if ndim >= 2 else -1


def quant_axis(name: str, ndim: int) -> int:
    """Which axis the quantiser groups along.

    For a kernel that is the reduction axis -- (in, out), or (layers, in, out)
    once nn.scan has stacked it -- because Ingot writes tensors as [out, in] so
    each output row is contiguous along the axis that was grouped. A lookup
    table is different: its row IS the vector, so it groups along the last one.
    """
    if ndim < 2 or any(hint in name.lower() for hint in LOOKUP_HINTS):
        return -1
    return default_axis(ndim)


def cq_forward(w, bits: float, group: int = CQ_GROUP, axis: int | None = None):
    """Grist's rotate-normalise-index-dequantise path, differentiably.

    A jnp mirror of `quartz.model.grist.cq_quantize`, sharing its codebook and
    its Walsh matrix, so what training sees is what the exporter will write.
    The length is rounded through fp16 here for the same reason it is there:
    quantise and dequantise have to agree on it.

    With no axis given the reduction axis is assumed, which is right for every
    kernel and wrong for a lookup table; `quant_plan` decides it per leaf.
    """
    import jax.numpy as jnp

    x = jnp.asarray(w)
    if x.ndim < 1:
        return x
    axis = default_axis(x.ndim) if axis is None else axis
    moved = jnp.moveaxis(x, axis, -1)
    width = moved.shape[-1]
    pad = (-width) % group
    if pad:
        moved = jnp.pad(moved, [(0, 0)] * (moved.ndim - 1) + [(0, pad)])
    padded_shape = moved.shape

    groups = moved.reshape(-1, group).astype(jnp.float32)
    walsh = jnp.asarray(_walsh(group))
    rot = groups @ walsh                       # incoherence processing
    norm = jnp.sqrt(jnp.sum(rot ** 2, axis=-1, keepdims=True))
    unit = rot / jnp.maximum(norm, 1e-12)
    norm = norm.astype(jnp.float16).astype(jnp.float32)

    levels = jnp.asarray(_levels(bits, group))
    edges = (levels[:-1] + levels[1:]) / 2.0
    # searchsorted against the midpoints is the nearest-centroid rule without
    # ever materialising a (values, levels) distance matrix.
    nearest = jnp.take(levels, jnp.searchsorted(edges, unit))
    out = ((nearest * norm) @ walsh).reshape(padded_shape)
    if pad:
        out = out[..., :width]
    return jnp.moveaxis(out, -1, axis).astype(x.dtype)


def cq_ste(w, bits: float, group: int = CQ_GROUP, axis: int | None = None):
    """Quantise forwards, pass the gradient straight through.

    The derivative of a rounding is zero almost everywhere and undefined at the
    boundaries, so the honest gradient trains nothing at all. The estimator
    pretends the rounding is the identity, which is exactly true in expectation
    for a weight sitting mid-bin and wrong in a way that averages out.
    """
    import jax

    return w + jax.lax.stop_gradient(cq_forward(w, bits, group, axis) - w)


def cq_noise_scale(w, bits: float, group: int = CQ_GROUP,
                   axis: int | None = None) -> float:
    """The RMS error the codebook actually makes on THIS tensor.

    Measured with Grist rather than assumed from the bit width, because the
    distortion of a group depends on how Gaussian it is after rotation, and
    that differs between an attention projection and a router.
    """
    from quartz.model.grist import cq_quantize

    x = np.asarray(w, dtype=np.float32)
    if x.ndim >= 1:
        x = np.moveaxis(x, default_axis(x.ndim) if axis is None else axis, -1)
    q = np.asarray(cq_quantize(x, bits, group), dtype=np.float32)
    return float(np.sqrt(np.mean((q - x) ** 2)))


def add_cq_noise(w, sigma: float, key):
    """Additive Gaussian noise at the measured distortion scale.

    Isotropic in weight space is the right shape: the Walsh transform is
    orthogonal, so noise that is isotropic before the rotation is isotropic
    after it, which is where the quantiser's error lives.
    """
    import jax
    import jax.numpy as jnp

    noise = jax.random.normal(key, w.shape, dtype=jnp.float32) * sigma
    return w + noise.astype(w.dtype)


# --- which leaf gets which width ------------------------------------------
def _parts(name: Any) -> tuple[str, ...]:
    """Split any leaf name into lowercase components, however it was written.

    `loom.phi`, `loom_phi` and `('loom', 'phi_res')` all have to land on the
    same components, because the bits map is written by a human and the tree is
    written by flax.
    """
    text = ".".join(str(p) for p in name) if isinstance(name, tuple) else str(name)
    return tuple(p for p in re.split(r"[^a-z0-9]+", text.lower()) if p)


def _contains(parts: tuple[str, ...], wanted: tuple[str, ...]) -> bool:
    """True when `wanted` appears as a contiguous run inside `parts`."""
    n = len(wanted)
    if not n or n > len(parts):
        return False
    return any(parts[i:i + n] == wanted for i in range(len(parts) - n + 1))


def quant_plan(params, bits_map: str = DEFAULT_BITS_MAP) -> dict[str, LeafPlan]:
    """Decide, once, what happens to every leaf. Keyed by dotted path.

    Grist owns both halves of this decision: `quant_leaf_names` says which
    leaves are quantised at all, and `parse_bits_map` says how wide each family
    is, longest match winning. We only join them to the live parameter tree.
    """
    import jax

    from quartz.model.grist import parse_bits_map, quant_leaf_names

    table, default = parse_bits_map(bits_map)
    wanted = [_parts(n) for n in (quant_leaf_names(params) or ())]
    families = sorted(((_parts(k), v) for k, v in table.items()),
                      key=lambda kv: len(kv[0]), reverse=True)

    plan: dict[str, LeafPlan] = {}
    leaves, _ = jax.tree_util.tree_flatten_with_path(params)
    for index, (path, leaf) in enumerate(leaves):
        name = path_str(path)
        parts = _parts(name)
        if not any(_contains(parts, w) for w in wanted):
            continue
        bits = next((v for key, v in families if _contains(parts, key)), default)
        plan[name] = LeafPlan(bits, quant_axis(name, leaf.ndim), index)
    if not plan:
        raise ValueError(
            "no leaf matched grist.quant_leaf_names for this parameter tree; "
            "the exporter and this stage disagree about what gets quantised")
    return plan


def measure_distortion(params, plan: dict[str, LeafPlan],
                       group: int = CQ_GROUP) -> dict[str, LeafPlan]:
    """Fill in each leaf's noise scale, on the weights we are starting from.

    Measured once. It drifts by a few percent over a run, and the point of the
    number is its scale, not its digits.
    """
    import jax

    leaves, _ = jax.tree_util.tree_flatten_with_path(params)
    for path, leaf in leaves:
        name = path_str(path)
        entry = plan.get(name)
        if entry is not None:
            entry.sigma = cq_noise_scale(np.asarray(leaf), entry.bits, group,
                                         entry.axis)
    return plan


def fake_quant(params, plan: dict[str, LeafPlan], *, mode: str = "ste",
               key=None, group: int = CQ_GROUP, noise_scale: float = 1.0):
    """Return a parameter tree with the quantiser in the forward path.

    `mode` is a Python string, not a traced value, so the two phases compile to
    two graphs instead of one graph that computes both and throws half away.

    `noise_scale` multiplies each leaf's measured distortion. At 1.0, which is
    what ships, the perturbation is exactly the error the codebook makes; it is
    a knob because the anneal is the part of this stage worth ablating, and the
    measured sigma is the only defensible unit to ablate it in.
    """
    import jax

    if mode not in ("ste", "noise"):
        raise ValueError(f"mode must be 'ste' or 'noise', got {mode!r}")
    if mode == "noise" and key is None:
        raise ValueError("the noise phase needs a PRNG key")

    def one(path, leaf):
        entry = plan.get(path_str(path))
        if entry is None:
            return leaf
        if mode == "ste":
            return cq_ste(leaf, entry.bits, group, entry.axis)
        return add_cq_noise(leaf, entry.sigma * noise_scale,
                            jax.random.fold_in(key, entry.index))

    return jax.tree_util.tree_map_with_path(one, params)


def mean_bits(params, plan: dict[str, LeafPlan]) -> float:
    """The scheme's average width over the weights it quantises, for the log."""
    import jax

    leaves, _ = jax.tree_util.tree_flatten_with_path(params)
    total = weighted = 0.0
    for path, leaf in leaves:
        entry = plan.get(path_str(path))
        if entry is not None:
            total += int(leaf.size)
            weighted += int(leaf.size) * entry.bits
    return weighted / max(total, 1.0)


# --- the loop --------------------------------------------------------------
def _make_step(z_weight: float, plan: dict[str, LeafPlan], mode: str, group: int,
               noise_scale: float = 1.0):
    """A jitted step whose forward pass runs on quantised weights."""
    import jax

    def step(state, ids, weights, key):
        def loss_fn(params):
            quantised = fake_quant(params, plan, mode=mode, key=key, group=group,
                                   noise_scale=noise_scale)
            return masked_losses(state.apply_fn, quantised, ids, weights,
                                 z_weight=z_weight)

        (total, ce), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        return state.apply_gradients(grads=grads), {"loss": total, "ce": ce}

    return jax.jit(step)


def train(args: Any = None, *, examples: Any = None, **overrides) -> dict:
    """Run stage three and return the run's statistics.

    The checkpoint holds the MASTER weights, not the quantised ones: Grist
    quantises at export and it has to see the same weights this stage was
    optimising. What changes on disk is `config.weight_bits`.
    """
    import jax

    from quartz.model.scribe import get_tokenizer

    a = as_args(QaptArgs, args, **overrides)
    sp = get_tokenizer(a.tokenizer) if a.tokenizer else get_tokenizer()
    source = examples if examples is not None else a.data
    if not isinstance(source, str | Path):
        source = list(source)
    max_len = fit_max_len(source, sp, a.cap)
    ids, weights = dataset(source, sp, max_len)

    per_epoch = steps_per_epoch(ids.shape[0], a.batch_size)
    if a.tokens:
        total = max(1, int(a.tokens // (a.batch_size * max_len)))
    else:
        total = per_epoch * a.epochs
    a.total_steps = a.total_steps or total
    switch = int(a.total_steps * a.noise_fraction)

    model, params, cfg = load_or_init(a, config_from_args(a))
    plan = measure_distortion(params, quant_plan(params, a.bits_map), a.group)
    state = make_state(model, params, a)

    noise_step = _make_step(a.z_weight, plan, "noise", a.group, a.noise_scale)
    ste_step = _make_step(a.z_weight, plan, "ste", a.group)
    key = jax.random.PRNGKey(a.seed)

    worst = sorted(plan.items(), key=lambda kv: -kv[1].sigma)[:3]
    print(f"  scheme    {a.bits_map}")
    print(f"  leaves    {len(plan)} quantised  mean {mean_bits(params, plan):.3f} bits")
    for name, entry in worst:
        print(f"  distortion {entry.sigma:.5f} rms at {entry.bits:g} bits  {name}")
    print(f"  schedule  {a.total_steps:,} steps  noise for {switch:,}, "
          f"straight-through for {a.total_steps - switch:,}")

    stream = batches(ids, weights, a.batch_size, seed=a.seed,
                     epochs=max(1, a.total_steps // max(per_epoch, 1) + 1))
    stats: dict[str, Any] = {"steps": a.total_steps, "bits_map": a.bits_map}
    started = time.time()
    for step in range(1, a.total_steps + 1):
        batch_ids, batch_w = next(stream)
        key, sub = jax.random.split(key)
        fn = noise_step if step <= switch else ste_step
        state, metrics = fn(state, jax.numpy.asarray(batch_ids),
                            jax.numpy.asarray(batch_w), sub)
        if step % a.log_every == 0 or step == a.total_steps:
            phase = "noise" if step <= switch else "ste"
            loss = float(metrics["loss"])
            print(f"  step      {step:,}/{a.total_steps:,}  loss {loss:.4f}  {phase}")
            stats["loss"] = loss

    # The stamp. Everything downstream -- Grist, Ingot, the runtime -- reads the
    # scheme off the config rather than off a command line that has scrolled away.
    cfg.weight_bits = a.bits_map
    write_checkpoint(a.out, jax.device_get(state.params), cfg)
    stats["out"] = a.out
    stats["weight_bits"] = cfg.weight_bits
    stats["elapsed"] = time.time() - started
    print(f"  stamped   config.weight_bits = {cfg.weight_bits!r} -> {a.out}")
    return stats
