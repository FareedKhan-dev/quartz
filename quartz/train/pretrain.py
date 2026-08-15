"""Stage one: 120 billion packed tokens, from random initialisation.

Two things here are not the usual next-token loop.

The first is the packing mask. Documents are concatenated into fixed rows and
every token carries a segment id, so a token that starts document two must not
predict itself from the end of document one. That is a mask on the attention
*and* a mask on the loss, and forgetting the second is the subtler bug: the
model learns a transition that never happened.

The second is the z-loss. In bf16 the logits are one overflow away from a NaN
that arrives ten thousand steps later with no explanation. A tax of 1e-4 on the
squared log-partition keeps them small early, then stops mattering, which is
what a good regulariser looks like.
"""
from __future__ import annotations

import math
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from quartz.model.config import EOS_ID, PAD_ID, QuartzConfig
from quartz.train.optim import (
    OptimArgs,
    as_args,
    average_stage_checkpoints,
    build_optimiser,
    config_from_args,
    split_summary,
    write_checkpoint,
)

#: Small enough to be invisible in the loss, large enough to hold the logits
#: down while they find their scale.
Z_LOSS_WEIGHT = 1e-4

#: Foresight predicts token t+2 from the trunk state at t and the embedding of
#: t+1. Half weight: it is there to push the trunk towards a representation
#: that survives past the next symbol, not to steer the run.
FORESIGHT_WEIGHT = 0.5


@dataclass
class PretrainArgs(OptimArgs):
    """Stage one's knobs. The defaults are the run in the post: 120B tokens,
    1024 rows of 2048, 57,220 steps at a peak of 3e-3."""

    config: str = ""              # a yaml geometry; empty means use the preset
    preset_name: str = "base"
    data: str = ""                # packed rows from quartz.data.winnow
    init: str = ""                # a checkpoint to continue from; empty starts cold
    out: str = "ckpt/quartz_base.pkl"
    ckpt_dir: str = "ckpt"
    tokens: float = 120e9
    batch_size: int = 1024
    seq_len: int = 0              # 0 means cfg.max_seq_len
    seed: int = 0
    log_every: int = 100
    ckpt_every: int = 2000
    avg_last: int = 6             # checkpoints averaged into the final weights
    z_weight: float = Z_LOSS_WEIGHT
    foresight_weight: float = FORESIGHT_WEIGHT


# --- data ------------------------------------------------------------------
def segments_from_eos(tokens: np.ndarray) -> np.ndarray:
    """Recover segment ids from a plain token stream.

    A packer that only wrote token ids still marked its document boundaries,
    because every document ends with EOS. Padding gets segment 0, so a padded
    position can never satisfy `seg > 0` and never contributes a loss term or
    an attention row.
    """
    tokens = np.asarray(tokens)
    starts = np.zeros(tokens.shape, dtype=np.int32)
    starts[:, 1:] = (tokens[:, :-1] == EOS_ID).astype(np.int32)
    seg = np.cumsum(starts, axis=1, dtype=np.int32) + 1
    seg[tokens == PAD_ID] = 0
    return seg


def _open_rows(source: Any, seq_len: int):
    """Memory-map the packed corpus without reading it.

    Accepts a .npy of shape (rows, seq_len) or (rows, seq_len, 2) -- tokens and
    segment ids -- a flat uint16 dump, or an array already in memory.
    """
    if isinstance(source, str | Path):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(
                f"no packed corpus at {path}. Build one with quartz.data.winnow "
                f"(`python -m quartz.cli winnow`), which writes rows of "
                f"{seq_len} tokens.")
        arr = (np.load(path, mmap_mode="r") if path.suffix == ".npy"
               else np.memmap(path, dtype=np.uint16, mode="r"))
    else:
        arr = np.asarray(source)
    if arr.ndim == 1:
        usable = (arr.shape[0] // seq_len) * seq_len
        if usable == 0:
            raise ValueError(f"corpus holds fewer than {seq_len} tokens")
        arr = arr[:usable].reshape(-1, seq_len)
    if arr.ndim not in (2, 3):
        raise ValueError(f"packed corpus must be 1-D, 2-D or 3-D, got {arr.shape}")
    if arr.shape[1] < seq_len:
        raise ValueError(
            f"packed rows are {arr.shape[1]} tokens, shorter than seq_len {seq_len}")
    return arr


def packed_batches(source: Any, batch_size: int, seq_len: int, *, seed: int = 0,
                   loop: bool = True) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (tokens, seg_ids) batches, reshuffled every pass.

    Rows are drawn by index out of a memory map, so a 120 billion token corpus
    costs one batch of resident memory rather than all of it.
    """
    arr = _open_rows(source, seq_len)
    n = arr.shape[0]
    if n < batch_size:
        raise ValueError(f"corpus holds {n} rows, fewer than one batch of {batch_size}")
    rng = np.random.default_rng(seed)
    while True:
        order = rng.permutation(n)
        for i in range(0, n - batch_size + 1, batch_size):
            rows = np.asarray(arr[np.sort(order[i:i + batch_size])])[:, :seq_len]
            if rows.ndim == 3:
                tokens = rows[..., 0].astype(np.int32)
                segs = rows[..., 1].astype(np.int32)
            else:
                tokens = rows.astype(np.int32)
                segs = segments_from_eos(tokens)
            yield tokens, segs
        if not loop:
            return


def as_stream(data: Any, batch_size: int, seq_len: int, *, seed: int = 0):
    """A batch iterator from a path, an array, or an iterator of batches."""
    if isinstance(data, Iterator):
        return data
    if isinstance(data, Iterable) and not isinstance(data, str | Path | np.ndarray):
        return iter(data)
    return packed_batches(data, batch_size, seq_len, seed=seed)


def data_sharding():
    """Split the batch axis across every visible device, or None on one card.

    Parameters stay replicated: at 45M parameters there is nothing to gain from
    splitting them, and the gradient all-reduce is what four cards are for.
    """
    import jax

    devices = jax.devices()
    if len(devices) < 2:
        return None
    mesh = jax.sharding.Mesh(np.asarray(devices), ("data",))
    return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))


def put(x, sharding):
    """Place a host batch on the devices, sharded if there is more than one."""
    import jax

    return jax.device_put(x, sharding) if sharding is not None else jax.numpy.asarray(x)


# --- model -----------------------------------------------------------------
def init_model(cfg: QuartzConfig, seed: int = 0):
    """Build QuartzNetwork and its parameters.

    Initialised on a short row: no parameter shape depends on the sequence
    length, and a 2048-wide trace here would cost a minute for nothing.
    """
    import jax
    import jax.numpy as jnp

    from quartz.model.architecture import QuartzNetwork, make_packing_mask

    model = QuartzNetwork(cfg)
    t = min(cfg.max_seq_len, 16)
    tokens = jnp.ones((1, t), jnp.int32)
    mask = make_packing_mask(jnp.ones((1, t), jnp.int32))
    params = model.init(jax.random.PRNGKey(seed), tokens, mask=mask)["params"]
    return model, params


def load_or_init(args: Any, cfg: QuartzConfig | None = None):
    """Continue from a previous stage if `args.init` names one, else start cold.

    The checkpoint carries its own geometry, and that one wins: a stage two run
    pointed at a stage one checkpoint must not quietly re-shape the model.
    """
    from quartz.model.architecture import QuartzNetwork
    from quartz.train.optim import read_checkpoint

    init = getattr(args, "init", "")
    if not init:
        cfg = cfg or config_from_args(args)
        model, params = init_model(cfg, getattr(args, "seed", 0))
        return model, params, cfg
    params, saved = read_checkpoint(init)
    cfg = saved or cfg or config_from_args(args)
    return QuartzNetwork(cfg), params, cfg


def make_state(model, params, args: Any):
    """A flax TrainState carrying the two-armed optimiser."""
    from flax.training import train_state

    tx = build_optimiser(args, params)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


# --- loss ------------------------------------------------------------------
def losses(apply_fn, params, tokens, seg_ids, *, z_weight: float = Z_LOSS_WEIGHT,
           foresight_weight: float = FORESIGHT_WEIGHT):
    """Next-token cross entropy, the z-loss, and Foresight. Returns (total, ce).

    `loss_mask` requires the input position and its target to sit in the same
    document, which drops exactly one token per packed document: the boundary,
    where the model would otherwise be taught a transition that packing was
    meant to hide.
    """
    import jax
    import jax.numpy as jnp
    import optax

    from quartz.model.architecture import make_packing_mask

    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    seg_in, seg_tgt = seg_ids[:, :-1], seg_ids[:, 1:]
    # `> 0` also drops padding, which shares segment id 0 with nothing.
    loss_mask = ((seg_in == seg_tgt) & (seg_in > 0)).astype(jnp.float32)
    denom = jnp.maximum(jnp.sum(loss_mask), 1.0)

    mask = make_packing_mask(seg_in)
    xe = optax.softmax_cross_entropy_with_integer_labels
    if foresight_weight:
        logits, foresight = apply_fn({"params": params}, inputs, mask=mask,
                                     return_foresight=True)
    else:
        logits, foresight = apply_fn({"params": params}, inputs, mask=mask), None

    ce = jnp.sum(xe(logits, targets) * loss_mask) / denom
    # In bf16 this term is the difference between training and silent overflow.
    z = jnp.sum(jax.nn.logsumexp(logits, axis=-1) ** 2 * loss_mask) / denom
    total = ce + z_weight * z

    if foresight is not None:
        pad = jnp.zeros((tokens.shape[0], 1), tokens.dtype)
        far = jnp.concatenate([tokens[:, 2:], pad], axis=1)
        seg_far = jnp.concatenate([seg_ids[:, 2:],
                                   jnp.zeros((tokens.shape[0], 1), seg_ids.dtype)],
                                  axis=1)
        # Same rule two tokens out, which also drops the last position, whose
        # t+2 target does not exist.
        fs_mask = ((seg_in == seg_far) & (seg_in > 0)).astype(jnp.float32)
        fs_ce = (jnp.sum(xe(foresight, far) * fs_mask)
                 / jnp.maximum(jnp.sum(fs_mask), 1.0))
        total = total + foresight_weight * fs_ce
    return total, ce


def train_step(state, tokens, seg_ids, *, z_weight: float = Z_LOSS_WEIGHT,
               foresight_weight: float = FORESIGHT_WEIGHT):
    """One optimiser step. Returns (state, metrics)."""
    import jax

    def loss_fn(params):
        return losses(state.apply_fn, params, tokens, seg_ids,
                      z_weight=z_weight, foresight_weight=foresight_weight)

    (total, ce), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    return state.apply_gradients(grads=grads), {"loss": total, "ce": ce}


# --- the loop --------------------------------------------------------------
def train(args: Any = None, *, data: Any = None, **overrides) -> dict:
    """Run stage one and return the run's statistics.

    Checkpoints land in `ckpt_dir` every `ckpt_every` steps; the last
    `avg_last` of them are averaged into `out` at the end.
    """
    import functools

    import jax

    a = as_args(PretrainArgs, args, **overrides)
    # load_or_init, not init_model: with `init` empty the two are the same cold
    # start, and with it set this is what makes `--resume` mean anything. Stages
    # two, three and four all come through here, and stage one silently not
    # doing so is how a resumed run quietly restarts from noise.
    #
    # It comes first because a resumed checkpoint's geometry wins over the one
    # the flags name, and seq_len is read off that geometry. Deriving the
    # schedule from the config we asked for and then training the config we got
    # is how a resumed run ends up with a row length the data does not have.
    model, params, cfg = load_or_init(a, config_from_args(a))
    seq_len = a.seq_len or cfg.max_seq_len
    per_step = a.batch_size * seq_len
    total_steps = a.total_steps or max(1, int(math.ceil(a.tokens / per_step)))
    a.total_steps = total_steps

    state = make_state(model, params, a)
    split = split_summary(params)
    sharding = data_sharding()
    stream = as_stream(data if data is not None else a.data, a.batch_size, seq_len,
                       seed=a.seed)

    step_fn = jax.jit(functools.partial(
        train_step, z_weight=a.z_weight, foresight_weight=a.foresight_weight))

    print(f"  data      {int(total_steps * per_step):,} tokens  seq_len {seq_len}  packed")
    print(f"  batch     {a.batch_size} seqs x {seq_len} tok = {per_step:,} tokens a step")
    print(f"  schedule  {total_steps:,} steps  warmup "
          f"{a.warmup or int(total_steps * a.warmup_frac):,}  clip {a.clip}")
    print(f"  split     muon {split['muon']['tensors']} tensors  "
          f"adamw {split['adamw']['tensors']} tensors  (each carries the layer axis)")

    kept: list[str] = []
    stats = {"steps": total_steps, "tokens": total_steps * per_step, "split": split}
    started = time.time()
    for step in range(1, total_steps + 1):
        tokens, seg_ids = next(stream)
        state, metrics = step_fn(state, put(tokens, sharding), put(seg_ids, sharding))
        if step % a.log_every == 0 or step == total_steps:
            loss = float(metrics["loss"])
            rate = step * per_step / max(time.time() - started, 1e-9) / 1e3
            print(f"  step      {step:,}/{total_steps:,}  loss {loss:.4f}  "
                  f"ce {float(metrics['ce']):.4f}  {rate:,.0f} tok/ms")
            stats["loss"] = loss
        if step % a.ckpt_every == 0 or step == total_steps:
            path = str(Path(a.ckpt_dir) / f"step_{step:07d}.pkl")
            write_checkpoint(path, jax.device_get(state.params), cfg)
            kept.append(path)

    tail = kept[-a.avg_last:]
    final = average_stage_checkpoints(tail) if len(tail) > 1 \
        else jax.device_get(state.params)
    write_checkpoint(a.out, final, cfg)
    stats["averaged"] = len(tail)
    stats["out"] = a.out
    stats["elapsed"] = time.time() - started
    print(f"  averaged  {len(tail)} checkpoints -> {a.out}")
    return stats
