"""Grist, two bits without a calibration set.

Absmax quantisation dies on real weights: six outliers out of four thousand set
the scale and everything else collapses onto zero. Grist does not tolerate
outliers, it removes them. A Walsh rotation makes every coordinate of a 128 wide
group a signed sum of 128 weights, which is Gaussian by construction, so one
codebook fitted offline to a standard normal is the right codebook for every
group in every tensor of the model. That is why nothing in this file needs a
calibration set, a Hessian, or a single forward pass.

What is stored per group is one two bit index per direction and one half
precision length. What stays in floating point is every vector, every scale and
both heads: they are three tenths of one percent of the model and quantising
them buys nothing.

Numpy only. Nothing here imports JAX.
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from functools import cache

import numpy as np

from .config import (
    CQ_BITS,
    CQ_GROUP,
    DEFAULT_BITS_MAP,
    IMPRINT_TAPS,
    TERNARY_BITS,
    TERNARY_CENTROID,
    QuartzConfig,
)

__all__ = [
    "bits_for",
    "canonical",
    "codebook",
    "cq_decode",
    "cq_encode",
    "cq_quantize",
    "family",
    "flatten_params",
    "group_axis",
    "is_head",
    "is_quantised",
    "is_training_only",
    "leaf_bytes",
    "lloyd_max_gaussian",
    "map_leaves",
    "mixed_quantise",
    "model_bytes",
    "natural_key",
    "nearest",
    "parse_bits_map",
    "quant_leaf_names",
    "quantise_params",
    "shipped_tensors",
    "size_report",
    "walsh_np",
]

#: Widths that mean "leave this leaf in floating point", in bits.
STORE_WIDTHS = (16, 32)
#: Widths the codebook path understands. 1.58 is ternary, three levels.
QUANT_WIDTHS = (TERNARY_BITS, *CQ_BITS)


# --------------------------------------------------------------------------
# the rotation
# --------------------------------------------------------------------------
@cache
def _walsh_cached(n: int) -> np.ndarray:
    if n < 1 or n & (n - 1):
        raise ValueError(f"a Walsh matrix needs a power of two, got {n}")
    h = np.array([[1.0]], dtype=np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    h = (h / np.sqrt(n)).astype(np.float32)
    h.flags.writeable = False          # it is a constant, shared by every caller
    return h


def walsh_np(n: int) -> np.ndarray:
    """The normalised Sylvester-Hadamard matrix: symmetric and its own inverse.

    Every entry is plus or minus one over sqrt(n), so the rotation is a butterfly
    of additions with one scale at the end and never a multiply. Being its own
    inverse is what lets dequantisation undo it with the same matrix. The result
    is cached and read-only, because the quantiser asks for the same 128 by 128
    constant a few hundred thousand times.
    """
    return _walsh_cached(int(n))


# --------------------------------------------------------------------------
# the codebook, fitted to a Gaussian rather than to our weights
# --------------------------------------------------------------------------
def _levels_for(bits: float) -> int:
    """How many reconstruction levels a width buys. 1.58 bits is three."""
    levels = int(round(2.0 ** float(bits)))
    if levels < 2:
        raise ValueError(f"{bits} bits leaves fewer than two levels")
    return levels


@cache
def _lloyd_cached(bits: float, iters: int, samples: int, seed: int) -> np.ndarray:
    levels = _levels_for(bits)
    if levels == 3:
        # Pinned to a constant rather than refitted. Ternary is the one width
        # whose codebook a device reconstructs from a header field, so it must
        # not depend on a random draw.
        c = np.array([-TERNARY_CENTROID, 0.0, TERNARY_CENTROID], dtype=np.float64)
        c.flags.writeable = False
        return c

    x = np.sort(np.random.RandomState(seed).randn(samples))
    # start on equal-mass quantiles, so no level is born in an empty cell
    q = (np.arange(levels) + 0.5) / levels
    c = x[(q * (samples - 1)).astype(np.int64)].astype(np.float64)
    for _ in range(iters):
        idx = np.searchsorted((c[:-1] + c[1:]) / 2.0, x)
        counts = np.bincount(idx, minlength=levels)
        sums = np.bincount(idx, weights=x, minlength=levels)
        nxt = np.where(counts > 0, sums / np.maximum(counts, 1), c)
        moved = float(np.max(np.abs(nxt - c)))
        c = nxt
        if moved < 1e-9:
            break
    c = np.sort(c)
    c.flags.writeable = False
    return c


#: The fit every shipped codebook comes from: iterations, samples, seed. Frozen,
#: because a file written last year decodes against these numbers and a nicer
#: fit found later would change what a two bit index means.
SHIPPED_FIT = (200, 400_000, 0)


def lloyd_max_gaussian(bits: float, iters: int = 200, samples: int = 400_000,
                       seed: int = 0) -> np.ndarray:
    """Reconstruction levels fitted to the NORMAL DISTRIBUTION.

    Not to our weights. After the Walsh rotation a group is Gaussian whatever it
    was before, so fitting the distribution instead of the data is not an
    approximation, it is the same answer without the calibration set. Lloyd's
    algorithm alternates the two conditions of an optimal quantiser: each level
    is the mean of its cell, each boundary is the midpoint of two levels.
    """
    return np.array(_lloyd_cached(float(bits), int(iters), int(samples), int(seed)))


def codebook(bits: float, group: int = CQ_GROUP) -> np.ndarray:
    """The levels a rotated unit vector is indexed against, in float32.

    A unit vector in R^group has coordinates of size about 1/sqrt(group), so the
    Gaussian levels are divided by sqrt(group) once here rather than a scale
    being carried at every use. The fit is always the shipped one: what a stored
    index means is part of the file format, not a knob.
    """
    return (_lloyd_cached(float(bits), *SHIPPED_FIT) / math.sqrt(group)).astype(np.float32)


def nearest(x: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Nearest centroid, with a tie broken toward the lower index, always.

    A search over the cell boundaries is both cheaper than a distance matrix and
    exactly reproducible: a value sitting on a boundary falls to the lower level
    on every machine, so two runs of the exporter produce identical bytes.
    """
    levels = np.asarray(levels, dtype=np.float32)
    mid = ((levels[:-1] + levels[1:]) * 0.5).astype(np.float32)
    return np.searchsorted(mid, np.asarray(x, dtype=np.float32), side="left").astype(np.uint8)


# --------------------------------------------------------------------------
# encode, decode, and the round trip between them
# --------------------------------------------------------------------------
def cq_encode(w: np.ndarray, bits: float, group: int = CQ_GROUP,
              axis: int = -1) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Rotate, keep the length, index the direction.

    Returns the codes in the layout the container stores them in: `axis` moved
    last and padded up to a whole number of groups, so one row of codes is
    contiguous along the axis the quantiser grouped on. The lengths come back
    already rounded through fp16, because that is the value the file will hold
    and the reconstruction has to agree with it.
    """
    w = np.asarray(w)
    if w.ndim < 1:
        raise ValueError("a scalar has no axis to group along")
    ax = axis if axis >= 0 else w.ndim + axis
    if not 0 <= ax < w.ndim:
        raise ValueError(f"axis {axis} is outside a rank {w.ndim} tensor")
    if group < 1 or group & (group - 1):
        raise ValueError(f"group must be a power of two, got {group}")

    x = np.moveaxis(w, ax, -1)
    lead, n = x.shape[:-1], x.shape[-1]
    pad = (-n) % group
    if pad:
        x = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)])
    rows = np.ascontiguousarray(x).reshape(-1, group).astype(np.float32)

    rot = rows @ walsh_np(group)              # incoherence processing
    norm = np.sqrt(np.sum(rot * rot, axis=-1, keepdims=True))
    unit = rot / np.maximum(norm, 1e-12)
    # round the length through fp16 HERE, so quantise and dequantise agree
    norm = norm.astype(np.float16)
    codes = nearest(unit, codebook(bits, group))
    n_pad = n + pad
    return (codes.reshape(*lead, n_pad),
            norm.reshape(*lead, n_pad // group),
            tuple(w.shape))


def cq_decode(codes: np.ndarray, norm: np.ndarray, bits: float,
              shape: Sequence[int], group: int = CQ_GROUP, axis: int = -1,
              levels: np.ndarray | None = None) -> np.ndarray:
    """Levels times length, rotated back, unpadded, and put back on `axis`.

    Pass `levels` to decode against a codebook that arrived with a file rather
    than one refitted here: an ingot carries its own, so a reader is never at
    the mercy of a later change to the fit.
    """
    shape = tuple(int(s) for s in shape)
    ax = axis if axis >= 0 else len(shape) + axis
    moved = shape[:ax] + shape[ax + 1:] + (shape[ax],)
    codes = np.asarray(codes)
    n_pad = codes.shape[-1]

    cb = codebook(bits, group) if levels is None else np.asarray(levels, dtype=np.float32)
    if int(codes.max(initial=0)) >= cb.size:
        raise ValueError(f"a code is outside the {cb.size} level codebook for {bits} bits")
    flat = cb[codes.reshape(-1, group)] * np.asarray(norm, dtype=np.float32).reshape(-1, 1)
    back = (flat @ walsh_np(group)).reshape(*moved[:-1], n_pad)
    return np.moveaxis(back[..., :moved[-1]], -1, ax)


def cq_quantize(w: np.ndarray, bits: float = 2, group: int = CQ_GROUP,
                axis: int = -1) -> np.ndarray:
    """One tensor through the whole path, out the other side as floats.

    This is the function the training loops and the evaluations call: it answers
    "what will this weight be on the device", without producing a file.
    """
    codes, norm, shape = cq_encode(w, bits, group, axis)
    out = cq_decode(codes, norm, bits, shape, group, axis)
    dtype = np.asarray(w).dtype
    return out.astype(dtype) if dtype.kind == "f" else out


# --------------------------------------------------------------------------
# names, and the map from a name to a width
# --------------------------------------------------------------------------
#: Module and parameter names that carry no information about which tensor this
#: is. Dropping them is what lets a bits map be written the way a person thinks
#: about the model rather than the way Flax nests it.
_DROP = frozenset({
    "params", "layers", "layer", "trunk", "body", "loombody", "quartznetwork",
    "kernel", "weight", "scale", "value",
})
_ALIAS = {
    "self_attn": "attn", "self_attention": "attn", "attention": "attn",
    "porthole": "attn", "embed": "embedding", "embedder": "embedding",
    "norm_f": "final_norm", "ln_f": "final_norm",
}
#: Prefixes that name a family rather than a module, so they become a level.
_SPLIT_PREFIXES = ("loom_", "imprint_", "fs_")

#: The heads. Both stay in floating point: together they are 272,386 parameters,
#: six tenths of one percent of the model, and what they emit is a calibrated
#: confidence and a retrieval embedding, two numbers where two bit noise costs
#: more than the quarter megabyte it saves.
HEAD_FAMILIES = ("dowser", "gauge")
#: Foresight predicts token t+2 during training and is dropped at export, so it
#: is never counted and never written.
TRAINING_ONLY_FAMILIES = ("fs", "foresight")
#: Leaves read a row at a time rather than reduced over. They group along their
#: last axis; everything else groups along the axis a matmul contracts.
TABLE_HINTS = ("embedding", "table", "slot", "memory", "codebook")
#: A last name component ending this way is a per-channel scale, never a matrix.
#: The size test below catches these anyway at any sane depth; this is the guard
#: for a model deep enough that a stacked vector looks square.
_VECTOR_SUFFIXES = ("norm", "gamma", "gate", "bias", "taps", "temp", "scale")


def canonical(name: str | Sequence[str]) -> str:
    """One name for a leaf, whatever the module tree happened to call it.

    A Flax path like ('params', 'layers', 'self_attn', 'out_proj', 'kernel')
    becomes 'attn.out_proj', which is what a bits map is written against.
    """
    if not isinstance(name, str):
        name = ".".join(str(p) for p in name)
    text = name.replace("/", ".").strip().lower()
    for pre in _SPLIT_PREFIXES:
        text = text.replace(pre, f"{pre[:-1]}.")
    parts: list[str] = []
    for raw in text.split("."):
        part = _ALIAS.get(raw, raw)
        if not part or part in _DROP or (parts and parts[-1] == part):
            continue
        parts.append(part)
    return ".".join(parts) or text.split(".")[-1]


def family(name: str) -> str:
    """The first level of a canonical name: which part of the model this is.

    Matching on the family rather than on an exact prefix is what keeps these
    rules working when Flax names a module 'Dowser_0' instead of 'dowser'.
    """
    return name.split(".", 1)[0]


def is_head(name: str) -> bool:
    """The Dowser and the Gauge, however the module tree spelled them."""
    return family(name).startswith(HEAD_FAMILIES)


def is_training_only(name: str) -> bool:
    """Foresight, which is trained and then thrown away at export."""
    head = family(name)
    return head == "fs" or head.startswith("foresight")


def _width(text: str) -> float:
    value = float(text)
    if value == TERNARY_BITS:
        return TERNARY_BITS
    if value != int(value):
        raise ValueError(f"{text!r} is not a width, expected one of "
                         f"{[*QUANT_WIDTHS, *STORE_WIDTHS]}")
    value = int(value)
    if value not in (*CQ_BITS, *STORE_WIDTHS):
        raise ValueError(f"{value} bit weights are not supported, expected one of "
                         f"{[*QUANT_WIDTHS, *STORE_WIDTHS]}")
    return value


def parse_bits_map(spec: str) -> tuple[dict[str, float], float]:
    """Parse 'default=2,loom.phi=4,attn.out_proj=3' into a table and a default.

    The default is mandatory. A built-in fallback width would one day ship a
    model whose numerics nobody chose, and the failure would be a quiet two
    points of accuracy rather than an error.
    """
    table: dict[str, float] = {}
    for part in str(spec).split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"bits map entry {part!r} is not <name>=<bits>")
        key, value = part.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"bits map entry {part!r} has no name to match on")
        table["default" if key == "default" else canonical(key)] = _width(value.strip())
    default = table.pop("default", None)
    if default is None:
        raise ValueError(f"bits map {spec!r} needs a default=<b> entry")
    return table, default


def bits_for(name: str, table: Mapping[str, float], default: float) -> float:
    """Longest prefix wins, so 'attn.out_proj=3' beats 'attn.=2'.

    Prefixes are on the string, not on the dotted components, which is what lets
    'loom.phi' name phi_pre, phi_post and phi_res in one entry.
    """
    best, bits = -1, default
    for key, width in table.items():
        if len(key) > best and name.startswith(key):
            best, bits = len(key), width
    return bits


def group_axis(name: str, ndim: int) -> int | None:
    """Which axis the 128 wide groups run along, or None for a leaf we leave.

    Tables are read a row at a time, so their row is the group. Everything else
    is reduced over its input axis, which is the second to last once the layer
    axis has been stacked on the front, and grouping along the reduction axis is
    what makes one output row contiguous in the file.
    """
    if ndim < 2:
        return None
    return ndim - 1 if any(hint in name for hint in TABLE_HINTS) else ndim - 2


def is_quantised(name: str, shape: Sequence[int], group: int = CQ_GROUP) -> bool:
    """Whether this leaf goes through the codebook at all.

    Four refusals. Vectors and scalars have no group to rotate. The two heads
    are too small to be worth the noise. A per-channel scale is a vector however
    many layers were stacked in front of it. And an axis shorter than the group
    would be more padding than data, which is paying bytes to lose accuracy.
    """
    shape = tuple(int(s) for s in shape)
    if len(shape) < 2 or is_head(name):
        return False
    if name.split(".")[-1].endswith(_VECTOR_SUFFIXES):
        return False
    axis = group_axis(name, len(shape))
    return axis is not None and shape[axis] >= group


# --------------------------------------------------------------------------
# walking a parameter tree
# --------------------------------------------------------------------------
def natural_key(text: str) -> tuple:
    """Sort 'layer2' before 'layer10' rather than after it.

    Shared with the exporter, which needs its directory order to be the one a
    person would write down.
    """
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", text))


def _walk(tree, prefix: tuple[str, ...] = ()):
    if isinstance(tree, Mapping):
        for key in sorted(tree, key=lambda k: natural_key(str(k))):
            yield from _walk(tree[key], (*prefix, str(key)))
    else:
        yield prefix, tree


def _map_leaves(tree, fn, prefix: tuple[str, ...]):
    if isinstance(tree, Mapping):
        return {k: _map_leaves(v, fn, (*prefix, str(k))) for k, v in tree.items()}
    return fn(prefix, tree)


def map_leaves(tree, fn: Callable[[tuple[str, ...], object], object]):
    """Rebuild a parameter tree leaf by leaf, structure and key order intact.

    `fn` is handed the path to each leaf, so it can decide per tensor. Returns
    plain dicts, which Flax accepts wherever it accepts a FrozenDict.
    """
    return _map_leaves(tree, fn, ())


def flatten_params(params) -> list[tuple[str, np.ndarray]]:
    """Every leaf as (canonical name, array), in a deterministic order.

    Two leaves that canonicalise to the same name is an error rather than a last
    writer wins, because the second one would silently take the first one's bit
    width.
    """
    out: list[tuple[str, np.ndarray]] = []
    seen: dict[str, tuple[str, ...]] = {}
    for path, leaf in _walk(params):
        name = canonical(path)
        if name in seen:
            raise ValueError(f"{'.'.join(path)} and {'.'.join(seen[name])} both "
                             f"canonicalise to {name!r}; the alias table needs a rule")
        seen[name] = path
        out.append((name, np.asarray(leaf)))
    return out


def quant_leaf_names(params, group: int = CQ_GROUP) -> list[str]:
    """The canonical names of the leaves the quantiser is allowed to touch.

    A bits map can still send one of them to 16 and leave it in floating point;
    this is the structural set, not the chosen scheme.
    """
    return [name for name, leaf in flatten_params(params)
            if is_quantised(name, leaf.shape, group)]


# --------------------------------------------------------------------------
# quantising a whole model
# --------------------------------------------------------------------------
def _fmt_bits(bits: float) -> str:
    return f"{bits:g}"


def mixed_quantise(params, bits_map: str = DEFAULT_BITS_MAP,
                   group: int = CQ_GROUP):
    """Quantise a tree with a different width per family.

    Every leaf that does not go through the codebook is still rounded through
    fp16, because that is what the container stores. So the tree this returns is
    exactly what an ingot written with the same map will read back, and an
    evaluation of it is an evaluation of the shipped file.
    """
    table, default = parse_bits_map(bits_map)

    def leaf(path: tuple[str, ...], value):
        arr = np.asarray(value)
        if arr.dtype.kind not in "fc":
            return arr                     # token ids and masks are not weights
        name = canonical(path)
        bits = bits_for(name, table, default)
        if bits >= 32:
            return arr.astype(np.float32)
        if bits >= 16 or not is_quantised(name, arr.shape, group):
            return arr.astype(np.float16).astype(np.float32)
        return cq_quantize(arr, bits, group, group_axis(name, arr.ndim))

    return map_leaves(params, leaf)


def quantise_params(params, bits: float = 2, group: int = CQ_GROUP):
    """One width everywhere, which is the ladder the post walks down."""
    return mixed_quantise(params, f"default={_fmt_bits(bits)}", group)


# --------------------------------------------------------------------------
# byte accounting
# --------------------------------------------------------------------------
def leaf_bytes(name: str, shape: Sequence[int], bits: float,
               group: int = CQ_GROUP) -> float:
    """What one tensor weighs in the file, indices plus its fp16 lengths.

    Ternary is counted at its 1.58 bits of entropy, which is the number the
    ladder in the post reports. The container rounds each value up to a two bit
    crumb, so an ingot written at 1.58 bits is the size of one written at 2.
    """
    shape = tuple(int(s) for s in shape)
    n = math.prod(shape) if shape else 1
    if not is_quantised(name, shape, group) or bits >= 32:
        return n * (4 if bits >= 32 else 2)
    if bits >= 16:
        return n * 2
    axis = group_axis(name, len(shape))
    rows = n // shape[axis]
    padded = -(-shape[axis] // group) * group
    return rows * (padded * float(bits) / 8.0 + (padded // group) * 2)


def shipped_tensors(cfg: QuartzConfig) -> list[tuple[str, tuple[int, ...]]]:
    """Every tensor that reaches the device, with the shape it has in the tree.

    This mirrors architecture.QuartzNetwork so a size can be quoted before a
    single parameter has been initialised. When a real tree is at hand, pass it
    to model_bytes instead: the tree is the authority and this is the map.
    Foresight is absent because it is a training head, dropped at export.
    """
    d, layers = cfg.d_model, cfg.num_layers
    head_dim, kv_dim, n_had = cfg.head_dim, cfg.kv_dim, cfg.hadamard_n
    lanes, wide = cfg.lanes, cfg.lane_dim
    _, _, sub_dim = cfg.imprint_geometry

    out: list[tuple[str, tuple[int, ...]]] = [("embedding", (cfg.vocab_size, d))]

    # the fourteen tensors of one block, each with the layer axis scan gave it
    per_layer: list[tuple[str, tuple[int, ...]]] = [
        ("attn.q_proj", (d, d)),
        ("attn.k_proj", (d, kv_dim)),
        ("attn.v_proj", (d, kv_dim)),
        ("attn.gate_proj", (d, d)),
        ("attn.out_proj", (d, d)),
        ("attn_norm", (d,)),
        ("attn.q_norm", (head_dim,)),
        ("attn.k_norm", (head_dim,)),
        ("post_attn_norm", (d,)),
        ("attn_gate", ()),
        ("pre_spin_norm", (d,)),
        ("spin.d1", (n_had,)),
        ("spin.d2", (n_had,)),
        ("spin.d3", (n_had,)),
    ]
    out += [(name, (layers, *shape)) for name, shape in per_layer]

    # nine router tensors, kept stacked: a sweep reads all the layers at once
    out += [
        ("loom.phi_pre", (layers, wide, lanes)),
        ("loom.phi_post", (layers, wide, lanes)),
        ("loom.phi_res", (layers, wide, lanes * lanes)),
        ("loom.b_pre", (layers, lanes)),
        ("loom.b_post", (layers, lanes)),
        ("loom.b_res", (layers, lanes, lanes)),
        ("loom.a_pre", (layers,)),
        ("loom.a_post", (layers,)),
        ("loom.a_res", (layers,)),
    ]

    for site in cfg.imprint_sites:
        out += [
            (f"imprint.{site}.table", (cfg.imprint_tables, cfg.imprint_slots, sub_dim)),
            (f"imprint.{site}.k_proj", (d, d)),
            (f"imprint.{site}.v_proj", (d, d)),
            (f"imprint.{site}.taps", (IMPRINT_TAPS, d)),
        ]

    out += [
        ("final_norm", (d,)),
        ("dowser.probe", (cfg.dowser_probes, d)),
        ("dowser.proj", (cfg.dowser_probes, d, cfg.dowser_dim)),
        ("dowser.temp", ()),
        ("gauge.probe", (cfg.gauge_probes, d)),
        ("gauge.gate", (cfg.gauge_probes, d)),
        ("gauge.bias", ()),
    ]
    return out


def _inventory(cfg: QuartzConfig, params=None) -> list[tuple[str, tuple[int, ...]]]:
    if params is None:
        return shipped_tensors(cfg)
    return [(name, tuple(leaf.shape)) for name, leaf in flatten_params(params)
            if not is_training_only(name)]


def _table_for(bits: float | str) -> tuple[dict[str, float], float]:
    spec = bits if isinstance(bits, str) else f"default={_fmt_bits(bits)}"
    return parse_bits_map(spec)


def model_bytes(cfg: QuartzConfig, bits: float | str = DEFAULT_BITS_MAP,
                group: int = CQ_GROUP, params=None) -> int:
    """What the weights weigh, in bytes, under a width or a whole bits map.

    Counts the shipped set only. Foresight is 1,577,089 parameters that never
    leave the training host, and quoting a size that includes them would be
    quoting a file nobody downloads.
    """
    table, default = _table_for(bits)
    total = 0.0
    for name, shape in _inventory(cfg, params):
        total += leaf_bytes(name, shape, bits_for(name, table, default), group)
    return int(round(total))


def size_report(cfg: QuartzConfig, bits: float | str = DEFAULT_BITS_MAP,
                group: int = CQ_GROUP, params=None) -> dict[str, float]:
    """Bytes, the split between coded and stored weights, and the mean width."""
    table, default = _table_for(bits)
    total_bytes, coded, stored, coded_bits = 0.0, 0, 0, 0.0
    for name, shape in _inventory(cfg, params):
        width = bits_for(name, table, default)
        n = math.prod(shape) if shape else 1
        total_bytes += leaf_bytes(name, shape, width, group)
        if is_quantised(name, shape, group) and width < 16:
            coded += n
            coded_bits += n * float(width)
        else:
            stored += n
    return {
        "bytes": int(round(total_bytes)),
        "params": coded + stored,
        "quantised": coded,
        "stored": stored,
        "mean_bits": coded_bits / coded if coded else 0.0,
    }


if __name__ == "__main__":     # pragma: no cover - a size table, not a test
    from .config import preset

    cfg = preset("base")
    for label, width in [("fp16", 16), ("W4", 4), ("W3", 3), ("W2", 2),
                         ("ternary", TERNARY_BITS)]:
        print(f"  {label:<9} {model_bytes(cfg, width) / 1e6:>6.2f} MB")
    report = size_report(cfg, DEFAULT_BITS_MAP)
    print(f"  {'mixed':<9} {report['bytes'] / 1e6:>6.2f} MB   "
          f"{report['mean_bits']:.3f} bits per quantised weight")
