"""Writing a checkpoint down, reading it back, and averaging the last few.

Three rules hold this file together.

**A checkpoint carries its format version and a mismatch is refused.** Not
migrated, not best-effort loaded: refused, by name, with the version it was
written at. A parameter tree that loads into the wrong reader gives a model
that runs and is quietly wrong, which is the most expensive failure in the
build because nothing surfaces until an evaluation number is a point low.

**A checkpoint holds numpy, never a device array or a Flax container.** Loading
one must not need JAX, Flax, or the ml_dtypes registration that bfloat16 rides
on, because the exporter, the quantiser and the container all run on numpy
alone. Anything exotic is cast to float32 on the way out, and the cast is
announced in the docstring rather than discovered in a stack trace.

**The last checkpoint is not the best point.** The final few thousand steps of
a cosine schedule still bounce around a basin, so
:func:`average_checkpoints` takes the mean of the last few. That mean is
computed in float32 and stored in float16: accumulating six half-precision
trees in half precision would lose more than the averaging buys.

The container is pickle, which executes arbitrary code on load. Only load
checkpoints you produced or trust; the ingot (:mod:`quartz.model.ingot`) is
the format for shipping to a device and it parses nothing.
"""
from __future__ import annotations

import os
import pickle
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .config import QuartzConfig

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "CheckpointFormatError",
    "average_checkpoints",
    "load_checkpoint",
    "save_checkpoint",
]

#: Bumped whenever the payload's meaning changes: a renamed tensor, a different
#: parameter layout, a new required key. Readers refuse anything else.
CHECKPOINT_FORMAT_VERSION = 1

#: Written into every payload so a stray pickle is recognisable as not ours
#: before we start trusting its keys.
MAGIC = "quartz-checkpoint"

#: Dtypes that survive a pickle round trip on a machine with numpy alone. jax
#: bfloat16 arrays do not: they need ml_dtypes registered to unpickle, and the
#: device the ingot is built for has neither.
_PORTABLE = frozenset({
    np.dtype(t) for t in
    (np.float16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64,
     np.uint8, np.uint16, np.uint32, np.uint64, np.bool_)
})


class CheckpointFormatError(RuntimeError):
    """Raised when a file is not a Quartz checkpoint of the version we read."""


# --- tree helpers -----------------------------------------------------------
def _to_numpy(tree: Any) -> Any:
    """Copy a parameter tree onto the host as plain dicts of plain numpy.

    Flax hands back a FrozenDict of device arrays. Neither should end up in the
    file: the dict would make loading depend on Flax, and the arrays would make
    it depend on a live JAX install pointing at the same backend.
    """
    if isinstance(tree, Mapping):
        return {str(k): _to_numpy(v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(_to_numpy(v) for v in tree)
    arr = np.asarray(tree)
    return arr if arr.dtype in _PORTABLE else arr.astype(np.float32)


def _cast(tree: Any, dtype: Any) -> Any:
    """Cast every floating leaf, leaving integer leaves (counters, ids) alone."""
    if isinstance(tree, Mapping):
        return {k: _cast(v, dtype) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(_cast(v, dtype) for v in tree)
    arr = np.asarray(tree)
    return arr.astype(dtype) if arr.dtype.kind == "f" else arr


def _flat(tree: Any, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], np.ndarray]:
    """Flatten to {path: array}, so two trees can be compared leaf by leaf."""
    out: dict[tuple[str, ...], np.ndarray] = {}
    if isinstance(tree, Mapping):
        for key, value in tree.items():
            out.update(_flat(value, (*prefix, str(key))))
    elif isinstance(tree, (list, tuple)):
        for i, value in enumerate(tree):
            out.update(_flat(value, (*prefix, str(i))))
    else:
        out[prefix] = np.asarray(tree)
    return out


def _unflatten(flat: Mapping[tuple[str, ...], np.ndarray]) -> dict[str, Any]:
    """Rebuild the nested tree a flat map came from."""
    tree: dict[str, Any] = {}
    for path, value in flat.items():
        node = tree
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value
    return tree


def _param_count(tree: Any) -> int:
    return int(sum(int(a.size) for a in _flat(tree).values()))


# --- save and load ----------------------------------------------------------
def save_checkpoint(path: str | os.PathLike[str], params: Any,
                    config: QuartzConfig | Mapping[str, Any] | None = None, *,
                    step: int = 0, meta: Mapping[str, Any] | None = None,
                    dtype: Any = None) -> Path:
    """Write a parameter tree, its geometry and its step to one file.

    The config travels with the parameters. A tree of arrays is not
    self-describing: 27 stacked layers of (512, 512) look identical whatever
    the imprint sites or the window were, and rebuilding a model against a
    guessed geometry is how a checkpoint silently becomes a different model.

    The write is atomic. A run that dies mid-save leaves the previous
    checkpoint intact rather than a truncated file that loads as far as layer
    nineteen.
    """
    path = Path(path)
    if isinstance(config, QuartzConfig):
        config_dict: dict[str, Any] | None = config.to_dict()
    elif config is None:
        config_dict = None
    else:
        config_dict = dict(config)

    tree = _to_numpy(params)
    if dtype is not None:
        tree = _cast(tree, np.dtype(dtype))

    payload = {
        "magic": MAGIC,
        "format": CHECKPOINT_FORMAT_VERSION,
        "config": config_dict,
        "step": int(step),
        "params": tree,
        "meta": {
            "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "params": _param_count(tree),
            **dict(meta or {}),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with open(tmp, "wb") as fh:
        pickle.dump(payload, fh, protocol=4)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str | os.PathLike[str], *,
                    expect: QuartzConfig | None = None) -> dict[str, Any]:
    """Read a checkpoint, or refuse it.

    Returns ``{"params", "config", "step", "meta", "format"}``, where
    ``config`` is a :class:`QuartzConfig` when the file carried one and None
    when it did not. Pass ``expect`` to also require that the geometry matches
    the model you are about to build, which turns a shape error deep inside the
    first forward pass into one sentence here.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no checkpoint at {path}; train one with `quartz pretrain` or point "
            "--checkpoint at an existing file")
    with open(path, "rb") as fh:
        payload = pickle.load(fh)

    if not isinstance(payload, Mapping) or payload.get("magic") != MAGIC:
        raise CheckpointFormatError(
            f"{path} is not a Quartz checkpoint: no {MAGIC!r} marker")
    found = payload.get("format")
    if found != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointFormatError(
            f"{path} is checkpoint format {found!r} and this build reads format "
            f"{CHECKPOINT_FORMAT_VERSION}. Refusing to load it: the parameter "
            "layout changed between the two, and a tree that loads into the "
            "wrong reader gives a model that runs and is wrong. Re-export it "
            "with the version that wrote it.")

    config = QuartzConfig(**payload["config"]) if payload.get("config") else None
    if expect is not None and config is not None:
        differ = [k for k, v in expect.to_dict().items()
                  if config.to_dict().get(k) != v]
        if differ:
            raise CheckpointFormatError(
                f"{path} was written for a different geometry: "
                f"{', '.join(sorted(differ))} disagree with the config in hand")
    return {
        "params": payload["params"],
        "config": config,
        "step": int(payload.get("step", 0)),
        "meta": dict(payload.get("meta", {})),
        "format": found,
    }


# --- averaging --------------------------------------------------------------
def average_checkpoints(paths: Sequence[str | os.PathLike[str]],
                        out: str | os.PathLike[str] | None = None, *,
                        dtype: Any = np.float16) -> dict[str, Any]:
    """A uniform mean of several checkpoints, accumulated in float32.

    Not an ensemble. The last few thousand steps of a cosine schedule still
    bounce around one basin, and the mean of those samples is a better point
    than any single one of them, for no inference cost at all. The post
    averages the final six.

    Uniform, deliberately: weighting the later checkpoints more is a knob with
    no measurement behind it, and it would reintroduce the bounce the average
    exists to cancel.

    The accumulator is float32 whatever the inputs were, then the result is
    stored at ``dtype`` (float16 by default). Summing six half-precision trees
    in half precision loses more than the averaging buys back.
    """
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("average_checkpoints needs at least one checkpoint")

    loaded = [load_checkpoint(p) for p in paths]
    first = _flat(loaded[0]["params"])
    reference = loaded[0]["config"]

    accum = {k: v.astype(np.float32) for k, v in first.items() if v.dtype.kind == "f"}
    verbatim = {k: v for k, v in first.items() if v.dtype.kind != "f"}

    for path, ckpt in zip(paths[1:], loaded[1:], strict=True):
        other = ckpt["config"]
        if reference is not None and other is not None \
                and other.to_dict() != reference.to_dict():
            raise ValueError(
                f"{path} has a different geometry from {paths[0]}; averaging "
                "parameters across two models is meaningless")
        flat = _flat(ckpt["params"])
        if set(flat) != set(first):
            missing = sorted("/".join(k) for k in set(first) ^ set(flat))[:4]
            raise ValueError(
                f"{path} has a different parameter tree from {paths[0]}: "
                f"{len(set(first) ^ set(flat))} leaves differ ({', '.join(missing)})")
        for key, value in flat.items():
            if value.shape != first[key].shape:
                raise ValueError(
                    f"{path}: {'/'.join(key)} is {value.shape}, expected "
                    f"{first[key].shape}")
            if key in accum:
                accum[key] += value.astype(np.float32)
            elif not np.array_equal(value, verbatim[key]):
                # integer leaves are ids and counters; a mean of two of them is
                # not a thing, so they have to agree
                raise ValueError(
                    f"{path}: non-float leaf {'/'.join(key)} differs between "
                    "checkpoints and cannot be averaged")

    n = len(loaded)
    target = np.dtype(dtype)
    merged = {k: (v / n).astype(target) for k, v in accum.items()}
    merged.update(verbatim)

    steps = [c["step"] for c in loaded]
    result = {
        "params": _unflatten(merged),
        "config": reference,
        "step": max(steps),
        "format": CHECKPOINT_FORMAT_VERSION,
        "meta": {
            "averaged": n,
            "sources": [str(p) for p in paths],
            "steps": steps,
            "dtype": target.name,
        },
    }
    if out is not None:
        save_checkpoint(out, result["params"], reference, step=result["step"],
                        meta=result["meta"])
    return result
