"""Stage two: 1.2 million teacher-written calls, and the mask that is half the
method.

Roughly three quarters of every training row is context -- the tool schemas and
the request -- which the model reads and never writes. Scoring it spends three
quarters of the run teaching the model to reproduce JSON Schema documents.

So the loss sees the reasoning line, the call, and the EOS that ends the turn,
and nothing else. The prompt is encoded separately from the target for the same
reason: run them through the tokenizer as one string and BPE merges the last
character of the prompt into the first token of the target, which puts a scored
token half inside the part we masked.
"""
from __future__ import annotations

import json
import math
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from quartz.model.config import BOS_ID, EOS_ID, PAD_ID
from quartz.train.optim import (
    OptimArgs,
    as_args,
    config_from_args,
    write_checkpoint,
)
from quartz.train.pretrain import Z_LOSS_WEIGHT, load_or_init, make_state

#: The shortest row worth padding to. Bucketing is powers of two from here, so
#: a corpus whose longest turn is 490 tokens trains at 512 and not at the 1024
#: cap, where three quarters of every batch would be padding.
MIN_BUCKET = 128


@dataclass
class SftArgs(OptimArgs):
    """Stage two's knobs. Three epochs is the run in the post; the fourth
    reaches 0.441 training and 0.713 validation, which is the set, not the
    task."""

    config: str = ""
    preset_name: str = "base"
    data: str = ""                     # jsonl of {query, tools, reasoning, answers}
    init: str = "ckpt/quartz_base.pkl"  # the stage one weights
    out: str = "ckpt/quartz_sft.pkl"
    tokenizer: str = ""
    epochs: int = 3
    batch_size: int = 16
    max_len: int = 0                   # 0 fits the bucket to the data
    cap: int = 1024                    # the longest row we will ever pad to
    holdout: float = 0.1
    seed: int = 0
    log_every: int = 500
    lr: float = 3e-4                   # an order below stage one: this is a refit
    z_weight: float = Z_LOSS_WEIGHT


# --- data ------------------------------------------------------------------
def load_jsonl(path: str) -> Iterator[dict]:
    """Stream a jsonl file of examples, skipping blank lines."""
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(
            f"no training data at {file}. Generate it with quartz.data.quarry "
            f"(`python -m quartz.cli quarry`), which writes one example a line.")
    with file.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def pad_to(values, length: int, fill, dtype) -> np.ndarray:
    """Right-pad to a fixed row. Padding is PAD_ID, which is segment 0 and is
    therefore invisible to both the attention mask and the loss."""
    if len(values) > length:
        raise ValueError(f"row of {len(values)} does not fit {length}")
    out = np.full((length,), fill, dtype=dtype)
    out[:len(values)] = values
    return out


def encode(sp, example: dict, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Encode prompt and target SEPARATELY, and score only the target.

    The mask lines up with `ids`, so position i is scored when it is a target
    of position i-1. EOS scores too, because stopping is a thing we teach: a
    model that never learns to stop writes a second call nobody asked for.
    """
    from quartz.model.scribe import pre_tokenize, render

    prompt, target = render(example)
    p_ids = list(sp.encode(pre_tokenize(prompt)))
    t_ids = list(sp.encode(pre_tokenize(target)))
    room = max_len - len(t_ids) - 2                 # BOS and EOS
    if room < 0:
        raise ValueError(
            f"target is {len(t_ids)} tokens, longer than the {max_len} row; "
            f"raise --cap or drop the example")
    if len(p_ids) > room:
        # Keep the tail of the prompt: the request sits next to the target, and
        # the schemas at the front are what a short window drops anyway.
        p_ids = p_ids[-room:]
    ids = [BOS_ID] + p_ids + t_ids + [EOS_ID]
    mask = [0.0] * (1 + len(p_ids)) + [1.0] * (len(t_ids) + 1)
    return (pad_to(ids, max_len, PAD_ID, np.int32),
            pad_to(mask, max_len, 0.0, np.float32))


def fit_max_len(path: Any, sp, cap: int, floor: int = MIN_BUCKET) -> int:
    """The smallest power-of-two row that holds the corpus, capped.

    Padding to a fixed 1024 when the data is 245 tokens long makes three
    quarters of every batch padding, which is a four to six times slowdown for
    nothing at all. Takes a jsonl path or a list of examples.
    """
    from quartz.model.scribe import pre_tokenize, render

    rows = load_jsonl(path) if isinstance(path, str | Path) else path
    longest = 0
    for example in rows:
        prompt, target = render(example)
        longest = max(longest, len(sp.encode(pre_tokenize(prompt)))
                      + len(sp.encode(pre_tokenize(target))) + 2)
    bucket = floor
    while bucket < min(longest, cap):
        bucket *= 2
    return min(bucket, cap)


def dataset(source: Any, sp, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Encode a whole jsonl (or a list of examples) into two fixed arrays.

    Rows whose target alone overflows the cap are dropped rather than truncated:
    a half-written tool call is a wrong label, not a short one.
    """
    rows = load_jsonl(source) if isinstance(source, str | Path) else source
    ids: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    dropped = 0
    for example in rows:
        try:
            row, mask = encode(sp, example, max_len)
        except ValueError:
            dropped += 1
            continue
        ids.append(row)
        weights.append(mask)
    if not ids:
        raise ValueError("no example fitted the row length; raise --cap")
    if dropped:
        print(f"  dropped   {dropped:,} examples longer than {max_len} tokens")
    return np.stack(ids), np.stack(weights)


def batches(ids: np.ndarray, weights: np.ndarray, batch_size: int, *, seed: int = 0,
            epochs: int = 1) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Shuffled fixed-size batches. The tail of an epoch is dropped so every
    step compiles once and runs at one shape."""
    rng = np.random.default_rng(seed)
    n = ids.shape[0]
    if n < batch_size:
        raise ValueError(f"{n} examples is fewer than one batch of {batch_size}")
    for _ in range(epochs):
        order = rng.permutation(n)
        for i in range(0, n - batch_size + 1, batch_size):
            take = order[i:i + batch_size]
            yield ids[take], weights[take]


def steps_per_epoch(n: int, batch_size: int) -> int:
    """Full batches in one pass. The tail is dropped, so this matches what
    `batches()` actually yields and the schedule cannot outrun the data."""
    return max(1, n // batch_size)


# --- loss ------------------------------------------------------------------
def attention_mask(ids):
    """Causal, and blind to padding.

    This is the packing mask with one document a row: segment 1 everywhere real
    and 0 on padding. One code path with stage one, and padding that cannot
    attend to itself and produce a NaN out of the softmax.
    """
    import jax.numpy as jnp

    from quartz.model.architecture import make_packing_mask

    return make_packing_mask((ids != PAD_ID).astype(jnp.int32))


def masked_losses(apply_fn, params, ids, weights, *, z_weight: float = Z_LOSS_WEIGHT):
    """Cross entropy over the scored span only. Returns (total, ce).

    `weights[:, 1:]` is the shift: a weight sits on the token being predicted,
    so it lines up with the targets and not with the inputs.
    """
    import jax
    import jax.numpy as jnp
    import optax

    inputs, targets = ids[:, :-1], ids[:, 1:]
    w = weights[:, 1:].astype(jnp.float32)
    denom = jnp.maximum(jnp.sum(w), 1.0)
    logits = apply_fn({"params": params}, inputs, mask=attention_mask(inputs))
    ce = jnp.sum(optax.softmax_cross_entropy_with_integer_labels(logits, targets)
                 * w) / denom
    # The same tax as stage one, because these are the same logits.
    z = jnp.sum(jax.nn.logsumexp(logits, axis=-1) ** 2 * w) / denom
    return ce + z_weight * z, ce


def _make_step(z_weight: float):
    """A jitted (state, ids, weights) -> (state, metrics) step."""
    import jax

    def step(state, ids, weights):
        def loss_fn(params):
            return masked_losses(state.apply_fn, params, ids, weights,
                                 z_weight=z_weight)

        (total, ce), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        return state.apply_gradients(grads=grads), {"loss": total, "ce": ce}

    return jax.jit(step)


def make_eval(apply_fn):
    """A jitted forward-only cross entropy. Built once per run, because a fresh
    closure here would be a fresh compilation on every epoch."""
    import jax

    return jax.jit(lambda p, i, w: masked_losses(apply_fn, p, i, w, z_weight=0.0)[1])


def evaluate(apply_fn, params, ids, weights, batch_size: int, eval_fn=None) -> float:
    """Mean scored cross entropy over a held-out split, forward only."""
    fn = eval_fn or make_eval(apply_fn)
    total, n = 0.0, 0
    for i in range(0, ids.shape[0] - batch_size + 1, batch_size):
        total += float(fn(params, ids[i:i + batch_size], weights[i:i + batch_size]))
        n += 1
    return total / max(n, 1)


# --- the loop --------------------------------------------------------------
def train(args: Any = None, *, examples: Any = None, **overrides) -> dict:
    """Run stage two and return the run's statistics.

    Watch the gap rather than the level: the loss starts near 1.1 instead of
    near 9 because most of a tool call is boilerplate the base model already
    predicts. Train minus validation is 0.027 after one epoch and 0.158 after
    three, which is the signal to stop.
    """
    import jax

    from quartz.model.scribe import get_tokenizer

    a = as_args(SftArgs, args, **overrides)
    sp = get_tokenizer(a.tokenizer) if a.tokenizer else get_tokenizer()
    source = examples if examples is not None else a.data
    if not isinstance(source, str | Path):
        source = list(source)
    # An explicit row length wins; 0 fits the bucket to the data. Deriving it
    # unconditionally means a caller who knows their corpus cannot pin the shape
    # every step compiles at.
    max_len = a.max_len or fit_max_len(source, sp, a.cap)
    ids, weights = dataset(source, sp, max_len)

    rng = np.random.default_rng(a.seed)
    order = rng.permutation(ids.shape[0])
    n_val = int(ids.shape[0] * a.holdout)
    val, train_rows = order[:n_val], order[n_val:]
    tr_ids, tr_w = ids[train_rows], weights[train_rows]
    va_ids, va_w = ids[val], weights[val]

    per_epoch = steps_per_epoch(tr_ids.shape[0], a.batch_size)
    a.total_steps = a.total_steps or per_epoch * a.epochs

    model, params, cfg = load_or_init(a, config_from_args(a))
    state = make_state(model, params, a)
    step_fn = _make_step(a.z_weight)
    eval_fn = make_eval(state.apply_fn)

    print(f"  data      {tr_ids.shape[0]:,} examples  seq_len {max_len}  cap {a.cap}")
    print(f"  holdout   {va_ids.shape[0]:,} examples for validation")
    print(f"  batch     {a.batch_size} seqs x {max_len} tok")
    print(f"  schedule  {a.total_steps:,} steps  warmup "
          f"{a.warmup or int(a.total_steps * a.warmup_frac):,}  clip {a.clip}")

    stats: dict[str, Any] = {"max_len": max_len, "steps": a.total_steps, "epochs": []}
    started = time.time()
    stream = batches(tr_ids, tr_w, a.batch_size, seed=a.seed, epochs=a.epochs)
    step = 0
    for epoch in range(1, a.epochs + 1):
        running = 0.0
        for _ in range(per_epoch):
            batch_ids, batch_w = next(stream)
            state, metrics = step_fn(state, jax.numpy.asarray(batch_ids),
                                     jax.numpy.asarray(batch_w))
            running += float(metrics["ce"])
            step += 1
            if step % a.log_every == 0:
                print(f"  step      {step:,}/{a.total_steps:,}  "
                      f"loss {float(metrics['loss']):.4f}")
        train_ce = running / per_epoch
        val_ce = (evaluate(state.apply_fn, state.params, va_ids, va_w, a.batch_size,
                           eval_fn=eval_fn)
                  if va_ids.shape[0] >= a.batch_size else math.nan)
        print(f"  epoch     {epoch}/{a.epochs}  loss {train_ce:.4f}  val {val_ce:.4f}")
        stats["epochs"].append({"epoch": epoch, "loss": train_ce, "val": val_ce})

    write_checkpoint(a.out, jax.device_get(state.params), cfg)
    stats["out"] = a.out
    stats["elapsed"] = time.time() - started
    return stats
