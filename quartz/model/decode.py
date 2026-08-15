"""The KV-cached decoder: one token in, one token out, at bounded memory.

:mod:`quartz.model.architecture` is a training-time module. It reads a whole
sequence, builds a mask over it and recomputes every key and value on every
call, which is right for a 2,097,152 token batch and wrong for a watch. This
module is the other half: the same arithmetic, driven one step at a time
against a cache that never grows.

Four things here are decisions rather than mechanics.

**The cache is a ring with a frozen head.** It holds exactly ``cache_len``
positions, laid out as ``pinned`` slots that are written once and never touched
again, then a rotating region for the conversation. That is what makes pinning
the tool schemas free: those positions were already in the cache, and the
window is enforced in the mask rather than in the storage.

**Keys are stored after QK-norm and after the rotation.** The cached tensor is
then exactly the one the dot product consumes, so a cached key is never
re-normalised or re-rotated on a later step, and the shipped runtime can hold
it as int8 with one fp32 scale per :data:`quartz.model.config.KV_GROUP` lanes
without the quantisation ever landing mid-computation.

**Imprint gets a token history, not a value cache.** The memory sites hash
n-grams of token ids, so their keys and values are a pure function of the last
few tokens. Keeping ``IMPRINT_TAPS * max(orders)`` token ids and recomputing
the four taps is smaller than caching the value stream, and the history is
initialised to ``PAD_ID`` so the warm-up rows at the start of a session hash
exactly the way the trainer's zero padding does.

**The trunk is re-implemented here, not re-used.** Porthole takes no cache, and
threading one through the training module would put a decode-time branch inside
the scanned body that every training step then pays for. The layer loop below
is a ``lax.scan`` over the same stacked parameters, reads the same tensors by
name, and reuses every free function in ``architecture`` that does not touch the
cache, so there is one source of truth for the maths and two for the plumbing.

The parameter tree it reads is the one ``QuartzNetwork.init`` produces::

    embedding/embedding                            (vocab, d)
    memory_{i}/table                               (tables, slots, sub_dim)
    memory_{i}/{k_proj,v_proj}/kernel              (d, d)
    memory_{i}/taps                                (taps, d)
    trunk/loom_phi_{pre,post}                      (L, lane_dim, lanes)
    trunk/loom_phi_res                             (L, lane_dim, lanes * lanes)
    trunk/loom_{a,b}_{pre,post,res}                (L, ...)
    trunk/layers/block/ZCRMSNorm_0/scale           (L, d)
    trunk/layers/block/self_attn/*_proj/kernel     (L, d, *)
    trunk/layers/block/self_attn/{q,k}_norm/scale  (L, head_dim)
    trunk/layers/block/{post_attn,pre_spin}_norm/scale  (L, d)
    trunk/layers/block/attn_gate                   (L,)
    trunk/layers/block/spin/d{1,2,3}               (L, hadamard_n)
    trunk/final_norm/scale                         (d,)

Leaves are found by path tail and shape, so a re-nesting of the module tree is
a clear error naming the tensor rather than a silently wrong model. Foresight
is skipped: it is training-only and dropped at export, and its block would
otherwise collide with the trunk's on every projection name.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .architecture import (
    _lane_offsets,
    _site_flags,
    apply_rope,
    imprint_indices,
    rms_unit,
    rope_tables,
    sinkhorn,
    walsh,
)
from .config import EOS_ID, IMPRINT_TAPS, PAD_ID, QuartzConfig

__all__ = [
    "ALLOCATOR_ARENA_BYTES",
    "DecodeCfg",
    "as_decode_cfg",
    "decode_cfg",
    "forward_cached",
    "generate",
    "init_kv_cache",
    "kv_cache_bytes",
    "sample_token",
    "session_budget",
]

#: Measured on the reference runtime, not derived from the config: the pool the
#: allocator reserves, the loader plus its constant tables, the compiled Trellis
#: tries and the tokenizer image. They are here so the budget below adds up to
#: the number the post quotes rather than to the interesting half of it.
ALLOCATOR_ARENA_BYTES = 2_998_272
LOADER_BYTES = 1_343_488
TRELLIS_TRIE_BYTES = 918_016
TOKENIZER_BYTES = 412_800

#: Segments whose subtree the decoder ignores. Foresight is a training-time
#: head, dropped by the exporter, and it owns a second copy of every block name.
_TRAINING_ONLY = ("fs_", "foresight")


# --- the static half --------------------------------------------------------
@dataclass(frozen=True)
class DecodeCfg:
    """Everything the compiled step needs to know before it sees a token.

    Frozen and built out of tuples, so it hashes. That is the point: it is
    passed to ``jax.jit`` as a static argument, so the window, the cache length
    and the layer count are constants inside the graph and the cache indexing
    compiles to fixed slicing rather than to traced arithmetic.
    """

    vocab_size: int
    d_model: int
    num_heads: int
    num_kv_heads: int
    num_layers: int
    head_dim: int
    lanes: int
    sinkhorn_iters: int
    hadamard_n: int
    max_seq_len: int
    rope_theta: float
    window: int
    cache_len: int
    imprint_sites: tuple[int, ...]
    imprint_orders: tuple[int, ...]
    imprint_heads: int
    imprint_sub_dim: int
    imprint_slots: int
    embed_scale: float
    eos_id: int = EOS_ID
    pad_id: int = PAD_ID

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def lane_dim(self) -> int:
        return self.lanes * self.d_model

    @property
    def group_size(self) -> int:
        """Query heads per key/value head."""
        return self.num_heads // self.num_kv_heads

    @property
    def imprint_tables(self) -> int:
        return len(self.imprint_orders) * self.imprint_heads

    @property
    def imprint_dilation(self) -> int:
        """The tap spacing: the longest n-gram order, so the four taps land on
        n-grams that do not overlap each other."""
        return max(self.imprint_orders) if self.imprint_orders else 1

    @property
    def history_len(self) -> int:
        """Token ids to carry between steps.

        Tap three reaches back ``3 * dilation`` positions and the n-gram ending
        there reads a further ``order - 1``, so ``taps * dilation`` ids always
        covers the deepest read with one position to spare.
        """
        return IMPRINT_TAPS * self.imprint_dilation


def decode_cfg(config: QuartzConfig, *, window: int | None = None,
               cache_len: int | None = None) -> DecodeCfg:
    """Freeze a :class:`QuartzConfig` into the static half of a decode.

    The window defaults to ``config.effective_window()``, which is the smaller
    of what the KV budget allows and what the config asked for. The cache is
    sized to the window, because a slot outside the window can never be
    attended to and holding it would be paying for memory to ignore it.

    The two numbers do different jobs, and pinning is where they part. The
    cache is ``cache_len`` positions and that is the memory bill; the window is
    how far back a query may reach inside it. At the default
    ``cache_len == window`` the pinned schemas take their slots out of the
    window, so 256 positions with 168 pinned leaves 88 of conversation. Ask for
    ``cache_len = window + pinned`` instead and the full window sits on top of
    the pinned block, which is the mask
    :func:`quartz.model.architecture.window_mask` builds at training time, for
    the memory those extra positions cost. The budget table in the post is the
    first arrangement.
    """
    win = config.effective_window() if window is None else int(window)
    if win < 1:
        raise ValueError(f"window must be positive, got {win}")
    length = win if cache_len is None else int(cache_len)
    if length < win:
        raise ValueError(f"cache_len {length} cannot be shorter than the window {win}")
    orders, heads, sub_dim = config.imprint_geometry
    return DecodeCfg(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_kv_heads=config.num_kv_heads,
        num_layers=config.num_layers,
        head_dim=config.head_dim,
        lanes=config.lanes,
        sinkhorn_iters=config.sinkhorn_iters,
        hadamard_n=config.hadamard_n,
        max_seq_len=config.max_seq_len,
        rope_theta=float(config.rope_theta),
        window=win,
        cache_len=length,
        imprint_sites=tuple(config.imprint_sites),
        imprint_orders=tuple(orders),
        imprint_heads=int(heads),
        imprint_sub_dim=int(sub_dim),
        imprint_slots=config.imprint_slots,
        # the embedding is initialised at 1/sqrt(d) and scaled back up on the
        # way in, exactly as QuartzNetwork.embed_scale does
        embed_scale=math.sqrt(config.d_model),
        pad_id=config.pad_token_id,
    )


# --- reading a checkpoint ---------------------------------------------------
class _ParamView:
    """A flat, read-only view of a parameter tree, addressed by path tail.

    The decoder does not own the module tree it reads, so it never indexes into
    it by full path. A leaf is found by the tail of its path plus its expected
    shape, which is enough to be unambiguous and tolerant of the tree being
    re-nested, and a miss raises with the tensor's name in it.
    """

    def __init__(self, params: Mapping[str, Any]):
        if len(params) == 1 and "params" in params:
            params = params["params"]        # a full variable dict was handed in
        leaves: dict[tuple[str, ...], Any] = {}

        def walk(node: Any, path: tuple[str, ...]) -> None:
            if isinstance(node, Mapping):
                for key, value in node.items():
                    walk(value, (*path, str(key)))
            else:
                leaves[path] = node

        walk(params, ())
        self._leaves = {
            path: leaf for path, leaf in leaves.items()
            if not any(seg.startswith(_TRAINING_ONLY) for seg in path)
        }

    def __len__(self) -> int:
        return len(self._leaves)

    def find(self, tail: Sequence[str], shape: tuple[int, ...] | None = None
             ) -> list[tuple[tuple[str, ...], Any]]:
        tail = tuple(tail)
        n = len(tail)
        hits = [(p, a) for p, a in self._leaves.items() if p[-n:] == tail]
        if shape is not None:
            hits = [(p, a) for p, a in hits if tuple(a.shape) == shape]
        return sorted(hits)

    def one(self, candidates: Sequence[Sequence[str]], shape: tuple[int, ...] | None,
            what: str) -> Any:
        """The unique leaf matching the first candidate tail that matches once."""
        for tail in candidates:
            hits = self.find(tail, shape)
            if len(hits) == 1:
                return hits[0][1]
            if len(hits) > 1:
                found = ", ".join("/".join(p) for p, _ in hits[:4])
                raise KeyError(
                    f"{what}: {'/'.join(tail)} matches {len(hits)} tensors ({found}); "
                    "the checkpoint layout is ambiguous for this decoder")
        wanted = " or ".join("/".join(t) for t in candidates)
        raise KeyError(
            f"{what}: no tensor at {wanted} with shape {shape} in a tree of "
            f"{len(self._leaves)} leaves; this checkpoint was not written by "
            "quartz.model.architecture.QuartzNetwork")


def _resolve(params: Mapping[str, Any], cfg: DecodeCfg) -> dict[str, Any]:
    """Pull every tensor the step needs out of the tree, once, at trace time."""
    import jax.numpy as jnp

    view = _ParamView(params)
    d, layers, lanes = cfg.d_model, cfg.num_layers, cfg.lanes
    wide, head_dim, kv_dim = cfg.lane_dim, cfg.head_dim, cfg.kv_dim

    def get(name: str, candidates: Sequence[Sequence[str]],
            shape: tuple[int, ...] | None) -> Any:
        # float32 throughout: the training graph runs bf16 for throughput, the
        # decoder runs one token at a time and the precision is free
        return jnp.asarray(view.one(candidates, shape, name), dtype=jnp.float32)

    out: dict[str, Any] = {
        "embed": get("embedding", [("embedding", "embedding")], (cfg.vocab_size, d)),
        "final_norm": get("final norm", [("final_norm", "scale")], (d,)),
    }

    layer_spec = (
        ("n_in", [("block", "ZCRMSNorm_0", "scale")], (layers, d)),
        ("q_proj", [("self_attn", "q_proj", "kernel")], (layers, d, d)),
        ("k_proj", [("self_attn", "k_proj", "kernel")], (layers, d, kv_dim)),
        ("v_proj", [("self_attn", "v_proj", "kernel")], (layers, d, kv_dim)),
        ("gate_proj", [("self_attn", "gate_proj", "kernel")], (layers, d, d)),
        ("out_proj", [("self_attn", "out_proj", "kernel")], (layers, d, d)),
        ("q_norm", [("q_norm", "scale")], (layers, head_dim)),
        ("k_norm", [("k_norm", "scale")], (layers, head_dim)),
        ("n_post", [("post_attn_norm", "scale")], (layers, d)),
        ("attn_gate", [("block", "attn_gate")], (layers,)),
        ("n_spin", [("pre_spin_norm", "scale")], (layers, d)),
        ("d1", [("spin", "d1")], (layers, cfg.hadamard_n)),
        ("d2", [("spin", "d2")], (layers, cfg.hadamard_n)),
        ("d3", [("spin", "d3")], (layers, cfg.hadamard_n)),
        ("phi_pre", [("loom_phi_pre",)], (layers, wide, lanes)),
        ("phi_post", [("loom_phi_post",)], (layers, wide, lanes)),
        ("phi_res", [("loom_phi_res",)], (layers, wide, lanes * lanes)),
        ("b_pre", [("loom_b_pre",)], (layers, lanes)),
        ("b_post", [("loom_b_post",)], (layers, lanes)),
        ("b_res", [("loom_b_res",)], (layers, lanes, lanes)),
        ("a_pre", [("loom_a_pre",)], (layers,)),
        ("a_post", [("loom_a_post",)], (layers,)),
        ("a_res", [("loom_a_res",)], (layers,)),
    )
    out["layers"] = {name: get(name, cands, shape) for name, cands, shape in layer_spec}

    sites = []
    tables, sub = cfg.imprint_tables, cfg.imprint_sub_dim
    for i in range(len(cfg.imprint_sites)):
        site = (f"memory_{i}",)
        sites.append({
            "table": get(f"imprint {i} table", [(*site, "table")],
                         (tables, cfg.imprint_slots, sub)),
            "k_proj": get(f"imprint {i} k_proj", [(*site, "k_proj", "kernel")], (d, d)),
            "v_proj": get(f"imprint {i} v_proj", [(*site, "v_proj", "kernel")], (d, d)),
            "taps": get(f"imprint {i} taps", [(*site, "taps")], (IMPRINT_TAPS, d)),
        })
    out["sites"] = sites
    return out


# --- the cache --------------------------------------------------------------
def as_decode_cfg(cfg: DecodeCfg | QuartzConfig) -> DecodeCfg:
    """Take either config, hand back the frozen one.

    The caller that holds a model holds a :class:`QuartzConfig`, and making
    every one of them remember to freeze it first is a rule that will be
    forgotten once. A QuartzConfig cannot be a jit static argument anyway: it
    is mutable and therefore unhashable, and the failure it gives is a
    tracebackful of jax internals rather than a sentence.
    """
    if isinstance(cfg, DecodeCfg):
        return cfg
    if isinstance(cfg, QuartzConfig):
        return decode_cfg(cfg)
    raise TypeError(f"expected a DecodeCfg or a QuartzConfig, got {type(cfg).__name__}")


def init_kv_cache(cfg: DecodeCfg | QuartzConfig, batch: int = 1, *,
                  pinned: int = 0) -> dict[str, Any]:
    """An empty cache: keys, values, what each slot holds, and the history.

    ``pinned`` is how many leading positions are never evicted. In a turn that
    is the length of the rendered ``<tools>`` block, which is the longest and
    the oldest part of the prompt and therefore the first thing a bare window
    throws away, exactly when the model needs to spell a tool name. Those slots
    come out of ``cache_len``, so the conversation keeps ``cache_len - pinned``
    positions of recency; see :func:`decode_cfg` for the other arrangement.

    Every row of a batch decodes in lockstep, so the slot bookkeeping is shared
    across the batch and only the keys, values and token history carry a batch
    axis.
    """
    import jax.numpy as jnp

    cfg = as_decode_cfg(cfg)
    if batch < 1:
        raise ValueError(f"batch must be positive, got {batch}")
    if not 0 <= pinned < cfg.cache_len:
        raise ValueError(
            f"pinned must be in [0, {cfg.cache_len}), got {pinned}; the rotating "
            "region of the cache cannot be empty")
    shape = (cfg.num_layers, batch, cfg.num_kv_heads, cfg.cache_len, cfg.head_dim)
    return {
        "k": jnp.zeros(shape, jnp.float32),
        "v": jnp.zeros(shape, jnp.float32),
        # -1 marks a slot that has never been written, which is what keeps an
        # unfilled cache out of the softmax
        "slot_pos": jnp.full((cfg.cache_len,), -1, jnp.int32),
        "length": jnp.asarray(0, jnp.int32),
        "pinned": jnp.asarray(pinned, jnp.int32),
        # PAD_ID, so the first n-grams of a session hash the same way the
        # trainer's zero-padded warm-up rows do
        "history": jnp.full((batch, cfg.history_len), cfg.pad_id, jnp.int32),
    }


def kv_cache_bytes(config: QuartzConfig, window: int | None = None) -> int:
    """What the shipped cache costs at this window, int8 with its group scales.

    The decoder in this file holds float32, because on a host that is free and
    exact. The number a device budget has to live inside is this one.
    """
    win = config.effective_window() if window is None else int(window)
    return win * config.kv_bytes_per_position()


def _slot_of(positions: Any, pinned: Any, cache_len: int) -> Any:
    """Where an absolute position lives: pinned slots hold still, the rest ring."""
    import jax.numpy as jnp

    rotating = cache_len - pinned
    return jnp.where(positions < pinned, positions,
                     pinned + (positions - pinned) % rotating)


# --- the step ---------------------------------------------------------------
def _zc_norm(x: Any, gamma: Any) -> Any:
    """ZCRMSNorm applied out of line: a (1 + gamma) gain over the last axis."""
    import jax.numpy as jnp

    rms = jnp.sqrt(jnp.mean(x.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + 1e-6)
    return (1.0 + gamma) * x / rms


def _spin(x: Any, p: Mapping[str, Any], hadamard: Any, d_model: int) -> Any:
    """Two fixed Walsh transforms with a learned diagonal on either side."""
    import jax.numpy as jnp
    from jax import nn as jnn

    pad = hadamard.shape[0] - d_model
    z = jnp.pad(x, ((0, 0), (0, 0), (0, pad))) if pad else x
    z = (p["d1"] * z) @ hadamard
    z = jnn.silu(p["d2"] * z) @ hadamard      # the only nonlinearity
    return (p["d3"] * z)[..., :d_model]


def _attend(x: Any, p: Mapping[str, Any], k_cache: Any, v_cache: Any, mask: Any,
            rope: tuple[Any, Any], slots: Any, cfg: DecodeCfg) -> tuple[Any, Any, Any]:
    """Porthole against the cache: write this step's keys, read the whole window."""
    import jax.numpy as jnp
    from jax import nn as jnn

    b, t, d = x.shape
    heads, kv_heads, head_dim = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
    cos, sin = rope

    q = (x @ p["q_proj"]).reshape(b, t, heads, head_dim).transpose(0, 2, 1, 3)
    k = (x @ p["k_proj"]).reshape(b, t, kv_heads, head_dim).transpose(0, 2, 1, 3)
    v = (x @ p["v_proj"]).reshape(b, t, kv_heads, head_dim).transpose(0, 2, 1, 3)

    q = _zc_norm(q, p["q_norm"])
    k = _zc_norm(k, p["k_norm"])
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

    # what lands in the cache is post-norm and post-rotation, so a key is never
    # touched again after the step that produced it
    k_cache = k_cache.at[:, :, slots, :].set(k)
    v_cache = v_cache.at[:, :, slots, :].set(v)

    kk = jnp.repeat(k_cache, heads // kv_heads, axis=1)
    vv = jnp.repeat(v_cache, heads // kv_heads, axis=1)
    scores = jnp.matmul(q, kk.transpose(0, 1, 3, 2)) / jnp.sqrt(jnp.float32(head_dim))
    att = jnn.softmax(jnp.where(mask, scores, jnp.finfo(scores.dtype).min), axis=-1)
    out = jnp.matmul(att, vv).transpose(0, 2, 1, 3).reshape(b, t, d)
    # the gate reads the block input, so a position can refuse attention this
    # layer before seeing what attention would have returned
    out = out * jnn.sigmoid(x @ p["gate_proj"])
    return out @ p["out_proj"], k_cache, v_cache


def _imprint(w: Mapping[str, Any], history: Any, tokens: Any, positions: Any,
             cfg: DecodeCfg) -> tuple[Any, Any] | None:
    """Both memory sites for this chunk, from the token history.

    One hash pass over ``history + tokens`` gives the rows for every position
    the four taps reach, and each tap is a static slice of that. A tap whose
    reach falls before the start of the session is switched off, which is the
    same warm-up rule the training module applies with ``pos >= j * dilation``.
    """
    import jax.numpy as jnp

    if not cfg.imprint_sites:
        return None

    b, t = tokens.shape
    tables, sub = cfg.imprint_tables, cfg.imprint_sub_dim
    base, dilation = cfg.history_len, cfg.imprint_dilation
    ext = jnp.concatenate([history, tokens], axis=1)
    rows = imprint_indices(ext, cfg.imprint_orders, cfg.imprint_heads, cfg.imprint_slots)
    table_axis = jnp.arange(tables)[None, None, :]
    # tap j reads the n-grams ending j * dilation positions back, and is dead
    # until the session is that long
    lags = [j * dilation for j in range(IMPRINT_TAPS)]
    live = [(positions >= lag).astype(jnp.float32)[None, :, None] for lag in lags]

    keys, values = [], []
    for site in w["sites"]:
        rows_at = [rows[:, base - lag:base - lag + t, :] for lag in lags]
        gathered = [site["table"][table_axis, idx].reshape(b, t, tables * sub)
                    for idx in rows_at]
        keys.append(gathered[0] @ site["k_proj"])
        values.append(sum(site["taps"][j] * (gathered[j] @ site["v_proj"]) * live[j]
                          for j in range(IMPRINT_TAPS)))
    return jnp.stack(keys), jnp.stack(values)


def _forward(params: Mapping[str, Any], cache: Mapping[str, Any], tokens: Any,
             cfg: DecodeCfg, all_logits: bool, return_cells: bool) -> tuple[Any, ...]:
    """One chunk of tokens through the trunk, with the cache in place."""
    import jax.numpy as jnp
    from jax import lax
    from jax import nn as jnn

    w = _resolve(params, cfg)
    b, t = tokens.shape
    start = cache["length"]
    positions = start + jnp.arange(t, dtype=jnp.int32)
    slots = _slot_of(positions, cache["pinned"], cfg.cache_len)
    slot_pos = cache["slot_pos"].at[slots].set(positions)

    # --- the mask is where the window and the pinned schemas both live ---
    filled = slot_pos[None, :] >= 0
    causal = slot_pos[None, :] <= positions[:, None]
    recent = (positions[:, None] - slot_pos[None, :]) < cfg.window
    pinned_slot = jnp.arange(cfg.cache_len) < cache["pinned"]
    mask = (filled & causal & (pinned_slot[None, :] | recent))[None, None]

    # rotary tables are absolute: the slice starts at this chunk's first
    # position, so a pinned key keeps the angle it was written with
    cos_all, sin_all = rope_tables(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
    half = cfg.head_dim // 2
    rope = (lax.dynamic_slice(cos_all, (start, 0), (t, half)),
            lax.dynamic_slice(sin_all, (start, 0), (t, half)))

    x = jnp.take(w["embed"], tokens, axis=0) * cfg.embed_scale
    imprint = _imprint(w, cache["history"], tokens, positions, cfg)
    hadamard = walsh(cfg.hadamard_n).astype(jnp.float32)

    lanes, layers = cfg.lanes, cfg.num_layers
    pre_off, post_off = _lane_offsets(layers, lanes)
    flags = jnp.asarray(_site_flags(layers, cfg.imprint_sites))
    x = jnp.repeat(x[:, :, None, :], lanes, axis=2)

    def body(state: Any, per_layer: Any) -> tuple[Any, Any]:
        p, k_cache, v_cache, pre, post, site_flags = per_layer
        # all lanes normalised jointly, so the router can see which lane is loud
        nx = rms_unit(state.reshape(b, t, lanes * cfg.d_model))
        h_pre = jnn.sigmoid(p["a_pre"] * (nx @ p["phi_pre"]) + p["b_pre"] + pre)
        u = jnp.einsum("btn,btnc->btc", h_pre, state)     # a sum, not an average

        h = u
        if imprint is not None:
            k_mem, v_mem = imprint
            alpha = jnn.sigmoid(
                jnp.einsum("btd,sbtd->sbt", rms_unit(h), rms_unit(k_mem))
                / math.sqrt(cfg.d_model))
            h = h + jnp.einsum("s,sbt,sbtd->btd", site_flags, alpha, v_mem)

        skip = h
        a, k_cache, v_cache = _attend(_zc_norm(h, p["n_in"]), p, k_cache, v_cache,
                                      mask, rope, slots, cfg)
        h = skip + jnn.sigmoid(p["attn_gate"]) * _zc_norm(a, p["n_post"])
        h = h + _spin(_zc_norm(h, p["n_spin"]), p, hadamard, cfg.d_model)

        y = h - u                                          # the delta, not the output
        h_post = 2.0 * jnn.sigmoid(p["a_post"] * (nx @ p["phi_post"])
                                   + p["b_post"] + post)
        routing = p["a_res"] * (nx @ p["phi_res"]).reshape(b, t, lanes, lanes) + p["b_res"]
        transport = sinkhorn(routing, cfg.sinkhorn_iters)
        # transport conserves mass, so growth can only come from the gated delta
        state = (jnp.einsum("btij,btjc->btic", transport, state)
                 + h_post[..., None] * y[:, :, None, :])
        return state, (k_cache, v_cache, h)

    x, (k_new, v_new, hidden) = lax.scan(
        body, x, (w["layers"], cache["k"], cache["v"], jnp.asarray(pre_off),
                  jnp.asarray(post_off), flags))

    x = jnp.mean(x, axis=2)                                # the lanes become one
    x = _zc_norm(x, w["final_norm"])
    tail = x if all_logits else x[:, -1:, :]
    logits = tail @ w["embed"].T                           # tied embedding

    new_cache = {
        "k": k_new,
        "v": v_new,
        "slot_pos": slot_pos,
        "length": start + t,
        "pinned": cache["pinned"],
        "history": jnp.concatenate([cache["history"], tokens],
                                   axis=1)[:, -cfg.history_len:],
    }
    if not return_cells:
        return logits, new_cache
    cells = jnp.concatenate([jnp.moveaxis(hidden, 0, 2), x[:, :, None, :]], axis=2)
    return logits, new_cache, cells


_STEP: Any = None


def _step_fn() -> Any:
    """The jitted step, compiled once per (DecodeCfg, flags) combination."""
    global _STEP
    if _STEP is None:
        import jax

        _STEP = jax.jit(_forward, static_argnums=(3, 4, 5))
    return _STEP


def forward_cached(params: Mapping[str, Any], cache: Mapping[str, Any], tokens: Any,
                   cfg: DecodeCfg | QuartzConfig, *, all_logits: bool = False,
                   return_cells: bool = False, jit: bool = True) -> tuple[Any, ...]:
    """Run ``tokens`` (batch, time) against the cache and return (logits, cache).

    The same call does prefill and decode: a chunk of the prompt and a single
    sampled token differ only in ``time``. Chunks must be no longer than the
    rotating region of the cache, or a key one query in the chunk still needs
    would be overwritten by a later token of the same chunk.

    With ``return_cells`` it also returns the (batch, time, layers + 1, d) grid
    the Dowser and Gauge heads read.
    """
    import jax.numpy as jnp

    cfg = as_decode_cfg(cfg)
    tokens = jnp.asarray(tokens, dtype=jnp.int32)
    if tokens.ndim != 2:
        raise ValueError(f"tokens must be (batch, time), got shape {tokens.shape}")
    if tokens.shape[1] > cfg.cache_len:
        raise ValueError(
            f"{tokens.shape[1]} tokens will not fit in a {cfg.cache_len} slot cache; "
            "prefill in chunks of at most cache_len minus the pinned prefix, which "
            "is what generate() does")
    fn = _step_fn() if jit else _forward
    return fn(params, cache, tokens, cfg, all_logits, return_cells)


# --- sampling ---------------------------------------------------------------
def sample_token(logits: np.ndarray, *, temperature: float = 0.0, top_k: int = 0,
                 top_p: float = 1.0, rng: np.random.Generator | None = None) -> int:
    """One token id out of one row of logits.

    Sampling happens on the host, in numpy, and deliberately so: Trellis masks
    logits with a trie walk that is Python, and a device-side sampler would
    mean copying the distribution back and forth to consult it. At 38 decoded
    tokens a turn the copy is nothing and the constraint is everything.

    ``temperature`` at or below zero is greedy, which is what the tool-calling
    path ships with: there is one right call, and creativity is a defect.
    """
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 1:
        raise ValueError(f"sample_token wants one row of logits, got {logits.shape}")
    if temperature <= 0.0:
        return int(np.argmax(logits))

    scaled = logits / float(temperature)
    if top_k and top_k < scaled.shape[0]:
        cut = np.partition(scaled, -top_k)[-top_k]
        scaled = np.where(scaled < cut, -np.inf, scaled)
    finite = np.isfinite(scaled)
    if not finite.any():
        # every candidate was masked out; falling back to the raw argmax beats
        # emitting nothing, which is the worse failure
        return int(np.argmax(logits))
    probs = np.exp(scaled - np.max(scaled[finite]), where=finite, out=np.zeros_like(scaled))
    probs /= probs.sum()
    if 0.0 < top_p < 1.0:
        order = np.argsort(-probs)
        keep = np.cumsum(probs[order]) - probs[order] < top_p
        mask = np.zeros_like(probs, dtype=bool)
        mask[order[keep]] = True
        probs = np.where(mask, probs, 0.0)
        probs /= probs.sum()
    rng = np.random.default_rng() if rng is None else rng
    return int(rng.choice(probs.shape[0], p=probs))


def generate(params: Mapping[str, Any], cfg: DecodeCfg | QuartzConfig,
             prompt: Sequence[int], *, max_new_tokens: int = 64,
             temperature: float = 0.0, top_k: int = 0, top_p: float = 1.0,
             seed: int = 0, pinned: int = 0, stop_ids: Sequence[int] = (EOS_ID,),
             decoder: Any = None,
             processor: Callable[[np.ndarray, list[int]], np.ndarray] | None = None,
             ) -> list[int]:
    """Prefill the prompt, then decode until a stop id or the budget runs out.

    ``pinned`` is the number of leading prompt tokens that are never evicted,
    which for a rendered turn is where the ``<tools>`` block ends.

    ``decoder`` is a :class:`quartz.model.trellis.ConstrainedDecoder`, driven
    through the two calls it asks for: ``mask`` before sampling and ``accept``
    after. Accepting the id that was actually sampled rather than the one that
    looked likely is what keeps the trie walk in step with the text, so a
    turn where the mask had to give up cannot desynchronise it. ``processor``
    is the same hook one level down, for anything that is not a Trellis.

    Returns the generated ids only. The prompt is not echoed, because the
    caller already has it and the tokenizer round trip is not free.
    """
    cfg = as_decode_cfg(cfg)
    ids = np.asarray(list(prompt), dtype=np.int32)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError("prompt must be a non-empty sequence of token ids")
    total = int(ids.size) + int(max_new_tokens)
    if total > cfg.max_seq_len:
        raise ValueError(
            f"{ids.size} prompt + {max_new_tokens} new tokens is {total} positions, "
            f"past the {cfg.max_seq_len} rotary positions this model was trained "
            "with; start a new turn rather than growing this one")

    cache = init_kv_cache(cfg, 1, pinned=pinned)
    chunk = max(1, cfg.cache_len - pinned)
    logits = None
    for begin in range(0, ids.size, chunk):
        piece = ids[None, begin:begin + chunk]
        logits, cache = forward_cached(params, cache, piece, cfg)

    rng = np.random.default_rng(seed)
    stop = {int(s) for s in stop_ids}
    out: list[int] = []
    for _ in range(int(max_new_tokens)):
        row = np.asarray(logits[0, -1], dtype=np.float32)
        if decoder is not None:
            row = np.asarray(decoder.mask(row), dtype=np.float32)
        if processor is not None:
            row = np.asarray(processor(row, out), dtype=np.float32)
        token = sample_token(row, temperature=temperature, top_k=top_k, top_p=top_p,
                             rng=rng)
        if decoder is not None:
            decoder.accept(token)
        out.append(token)
        if token in stop:
            break
        logits, cache = forward_cached(params, cache,
                                       np.asarray([[token]], dtype=np.int32), cfg)
    return out


# --- what a session costs ---------------------------------------------------
def session_budget(config: QuartzConfig, weight_bytes: int, *, window: int | None = None,
                   tools_rendered: int = 5) -> dict[str, int]:
    """Every byte a live session holds, weights included.

    ``weight_bytes`` comes from the quantiser (:func:`quartz.model.grist.model_bytes`)
    rather than from a constant here, because the whole point of the bits map is
    that this number is a choice.

    Only one line of this table grows with the conversation, and it stops at the
    window. A model whose memory grows with the chat is killed by the operating
    system on the fourth turn.
    """
    win = config.effective_window() if window is None else int(window)
    return {
        "weights": int(weight_bytes),
        "prefill scratch": win * config.d_model * config.lanes * 4
                           + config.num_heads * win * win * 4,
        "KV cache": kv_cache_bytes(config, win),
        "allocator arena": ALLOCATOR_ARENA_BYTES,
        "loader and constants": LOADER_BYTES,
        "Trellis tries": TRELLIS_TRIE_BYTES,
        "tokenizer": TOKENIZER_BYTES,
        # the full catalogue index lives on flash and is never resident
        "Dowser index": tools_rendered * config.dowser_dim * 2,
        "logits": config.vocab_size * 4,
    }


if __name__ == "__main__":     # pragma: no cover
    from .config import KV_BUDGET_BYTES, preset

    base = preset("base")
    per_pos = base.kv_bytes_per_position()
    print(f"  {'per pos':<9} {per_pos:,} bytes")
    print(f"  {'budget':<9} {base.budget_window()} positions fit in "
          f"{KV_BUDGET_BYTES / 1024 / 1024:.1f} MiB")
    for win in (128, 256, 512, base.budget_window()):
        print(f"  {'window':<9} {win:>4} -> {win * per_pos / 1e6:>6.2f} MB")
    print(f"  {'taken':<9} {base.effective_window()}, which is a choice, not a "
          "derivation")
