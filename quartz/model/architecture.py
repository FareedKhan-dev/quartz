"""The whole model: Spin, Porthole, Imprint, Loom, and the two heads.

Every shape here comes from a :class:`~quartz.model.config.QuartzConfig`. Nothing
is hard-coded, so the two-layer laptop preset and the 45,211,383 parameter model
run the same code path.

JAX and Flax are an optional dependency of this package. The Flax modules are
therefore built on first attribute access rather than at import time, and the
free functions import JAX inside their bodies. ``import quartz`` works with only
numpy, sentencepiece and pyyaml; touching ``architecture.Spin`` is what pulls
Flax in.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .config import IMPRINT_PRIME, IMPRINT_SEED, IMPRINT_TAPS, QuartzConfig

#: The Flax modules, named here and defined in `_build_namespace` below. They
#: are resolved through the module-level `__getattr__`, so nothing in this file
#: touches Flax until one of them is looked up.
_LAZY = ("ZCRMSNorm", "Spin", "Porthole", "Imprint", "Block", "LoomBody", "Trunk",
         "Dowser", "Gauge", "QuartzNetwork")

__all__ = sorted([
    "BASE_PARAM_COUNT",
    "EXPORT_DROPPED",
    "SHIPPED_PARAM_COUNT",
    "apply_rope",
    "imprint_indices",
    "make_causal_mask",
    "make_packing_mask",
    "param_count",
    "probe_pool",
    "rms_unit",
    "rope_tables",
    "shift_right",
    "sinkhorn",
    "walsh",
    "window_mask",
    *_LAZY,
])

#: What ``sum(param_count(preset("base")).values())`` must equal. Quoted in the
#: post, asserted by the tests, and the reason every count below is derived.
BASE_PARAM_COUNT = 45_211_383

#: The components the exporter drops, because Foresight is a training-time
#: signal that the device never evaluates.
EXPORT_DROPPED = ("foresight",)

#: What survives into the ingot for the base preset.
SHIPPED_PARAM_COUNT = 43_634_294

_EPS = 1e-6


# --- pure numpy -------------------------------------------------------------
def _walsh_np(n: int) -> np.ndarray:
    """Sylvester-Hadamard at width `n`, normalised so it is its own inverse."""
    if n < 1 or n & (n - 1):
        raise ValueError(f"walsh needs a power of two width, got {n}")
    h = np.array([[1.0]], dtype=np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return (h / np.sqrt(n)).astype(np.float32)


def _lane_offsets(num_layers: int, lanes: int) -> tuple[np.ndarray, np.ndarray]:
    """The fixed per-layer lane preferences Loom starts from.

    Every layer owns one lane at initialisation, cycling. Through the sigmoid
    that is 0.982 on your own lane and 0.018 on the others, so the stack begins
    as `lanes` nearly independent networks. They are only offsets, so the router
    can learn its way out of them. Recomputed rather than checkpointed.
    """
    lane_of = np.eye(lanes, dtype=np.float32)[np.arange(num_layers) % lanes]
    return 8.0 * lane_of - 4.0, -4.0 * (1.0 - lane_of)


def _site_flags(num_layers: int, sites: tuple[int, ...]) -> np.ndarray:
    """A (layers, sites) one-hot, because the layer loop has a traced index.

    `if layer in (2, 15)` is not available inside a scan, so every layer runs
    both memory sites against a row that is zero for twenty five of twenty
    seven layers. Two wasted gathers buys one compiled scan.
    """
    if not sites:
        return np.zeros((num_layers, 0), dtype=np.float32)
    rows = [(np.arange(num_layers) == s).astype(np.float32) for s in sites]
    return np.stack(rows, axis=1)


def _dtype_of(name: str) -> Any:
    """Resolve the config's dtype string against jax.numpy."""
    import jax.numpy as jnp

    dtype = getattr(jnp, name, None)
    if dtype is None:
        raise ValueError(f"config.dtype {name!r} is not a jax.numpy dtype")
    return dtype


# --- free functions ---------------------------------------------------------
def walsh(n: int) -> Any:
    """The normalised Sylvester-Hadamard matrix: symmetric and its own inverse.

    Every entry is plus or minus one over sqrt(n), so the transform is a
    butterfly of additions with one final scale and never a multiply.
    """
    import jax.numpy as jnp

    return jnp.asarray(_walsh_np(n))


def rope_tables(head_dim: int, seq_len: int, theta: float = 1e5) -> tuple[Any, Any]:
    """Cosine and sine tables for rotary positions, shaped (seq_len, head_dim/2).

    theta is 1e5 rather than the usual 1e4, because the larger base decays the
    low frequencies more slowly when a model trained at 2048 runs at 256.
    """
    import jax.numpy as jnp

    if head_dim % 2:
        raise ValueError(f"rope needs an even head_dim, got {head_dim}")
    inv = 1.0 / (theta ** (jnp.arange(0, head_dim, 2).astype(jnp.float32) / head_dim))
    angles = jnp.outer(jnp.arange(seq_len).astype(jnp.float32), inv)
    return jnp.cos(angles), jnp.sin(angles)


def apply_rope(x: Any, cos: Any, sin: Any) -> Any:
    """Rotate `x`, laid out (batch, heads, time, head_dim), in the half-split
    convention, so dimension i pairs with i + head_dim/2 and not its neighbour.

    Pass tables already offset by the cache position to decode past the prefix:
    the function only ever reads their first `time` rows.
    """
    import jax.numpy as jnp

    t, half = x.shape[2], x.shape[-1] // 2
    if cos.shape[0] < t:
        raise ValueError(f"rope tables hold {cos.shape[0]} positions, need {t}")
    cos, sin = cos[:t][None, None], sin[:t][None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * cos - x2 * sin,
                            x2 * cos + x1 * sin], axis=-1).astype(x.dtype)


def rms_unit(x: Any, eps: float = _EPS) -> Any:
    """Scale a vector to unit root mean square, so its length is sqrt(d).

    Two such vectors dotted together give d times their cosine, which is what
    makes the sqrt(d) divisor in the Imprint gate come out as a sharpened
    cosine rather than a damped one.
    """
    import jax.numpy as jnp

    z = x.astype(jnp.float32)
    return z / jnp.sqrt(jnp.mean(z ** 2, axis=-1, keepdims=True) + eps)


def sinkhorn(logits: Any, iters: int = 20) -> Any:
    """Project a matrix onto the doubly stochastic ones, in the log domain.

    The loop ends on a column normalisation, so the result is exactly column
    stochastic and only approximately row stochastic. Column sums control the
    mass arriving in each lane, and a lane quietly receiving too much is what
    this construction exists to prevent.
    """
    import jax
    import jax.numpy as jnp

    log_k = logits
    for _ in range(iters):
        log_k = log_k - jax.nn.logsumexp(log_k, axis=-1, keepdims=True)
        log_k = log_k - jax.nn.logsumexp(log_k, axis=-2, keepdims=True)
    return jnp.exp(log_k)


def shift_right(x: Any, n: int) -> Any:
    """x[..., t - n, ...] along the time axis, zero filled at the front.

    Causality for both the n-gram hash and the Imprint convolution rests on
    this: nothing at position t may read a position after t.
    """
    import jax.numpy as jnp

    if n == 0:
        return x
    pad = [(0, 0)] * x.ndim
    pad[1] = (n, 0)
    return jnp.pad(x, pad)[:, :x.shape[1]]


def probe_pool(cells: Any, probes: Any) -> Any:
    """Attention-pool a (batch, time, layers + 1, d) cell grid down to one vector.

    The grid is flattened across time and depth together, so a probe can read a
    middle layer as easily as the final one. Small models keep a lot of their
    signal in the middle.
    """
    import jax
    import jax.numpy as jnp

    b, t, layers, d = cells.shape
    flat = cells.reshape(b, t * layers, d).astype(jnp.float32)
    w = jax.nn.softmax(jnp.einsum("bcd,kd->bkc", flat, probes) / math.sqrt(d), axis=-1)
    return jnp.einsum("bkc,bcd->bkd", w, flat).reshape(b, -1)


def imprint_indices(tokens: Any, orders: tuple[int, ...] | None = None,
                    heads: int | None = None, slots: int | None = None) -> Any:
    """Hash each n-gram ending at t into one row per table.

    FNV-1a over the token ids themselves, not over embeddings, so the row a
    rare tool name lands in is fixed before training starts and never drifts.
    Anything left as None falls back to the default geometry, which is what an
    interactive check wants; the model always passes its own config through.
    Returns (batch, time, len(orders) * heads) int32 rows.
    """
    import jax.numpy as jnp

    if orders is None or heads is None or slots is None:
        default = QuartzConfig()
        d_orders, d_heads, _ = default.imprint_geometry
        orders = d_orders if orders is None else orders
        heads = d_heads if heads is None else heads
        slots = default.imprint_slots if slots is None else slots

    u = tokens.astype(jnp.uint32)
    out = []
    for oi, order in enumerate(orders):
        for h in range(heads):
            # golden-ratio stride, so the per-table seeds decorrelate
            seed = (IMPRINT_SEED * (oi * heads + h + 1)) & 0xFFFFFFFF
            acc = jnp.full_like(u, jnp.uint32(seed))
            for j in range(order):
                acc = (acc ^ shift_right(u, j)) * jnp.uint32(IMPRINT_PRIME)
            acc = acc ^ (acc >> jnp.uint32(15))     # avalanche the low bits
            out.append((acc % jnp.uint32(slots)).astype(jnp.int32))
    return jnp.stack(out, axis=-1)


# --- masks ------------------------------------------------------------------
def make_causal_mask(seq_len: int) -> Any:
    """The plain lower triangle, broadcast to (1, 1, time, time)."""
    import jax.numpy as jnp

    return jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))[None, None]


def make_packing_mask(seg_ids: Any) -> Any:
    """Causal, and confined to one packed document.

    Documents are concatenated into fixed rows to save the third of every batch
    that padding wastes, and this closes the seam: without it a token starting
    document two attends to the end of document one and learns a transition that
    never happened.

    The `> 0` on the query side blanks padding rows entirely rather than letting
    padding attend to padding. Porthole fills a blank row with the float minimum
    before the softmax, so it comes back uniform rather than NaN, and the loss
    mask drops those positions anyway.
    """
    import jax.numpy as jnp

    seg_ids = jnp.asarray(seg_ids)
    t = seg_ids.shape[1]
    causal = jnp.tril(jnp.ones((t, t), dtype=bool))
    same = (seg_ids[:, :, None] == seg_ids[:, None, :]) & (seg_ids[:, :, None] > 0)
    return (same & causal[None, :, :])[:, None, :, :]


def window_mask(seq_len: int, window: int, sink: Any = None) -> Any:
    """Causal, plus a sliding window, plus keys that are never evicted.

    Where `sink` is true, that key stays attendable however far the query moves.
    The tool schemas are declared at the front of the turn and are the longest
    part of the prompt, so a bare window drops them first and the model is asked
    to call tools it can no longer see. Pinning costs nothing: those positions
    are already in the cache.
    """
    import jax.numpy as jnp

    pos = jnp.arange(seq_len)
    recent = (pos[:, None] - pos[None, :]) < window
    keep = recent if sink is None else (recent | jnp.asarray(sink)[:, None, :])
    return ((pos[None, :] <= pos[:, None])[None] & keep)[:, None, :, :]


# --- the Flax modules, built on first use -----------------------------------
_NAMESPACE: dict[str, Any] | None = None


def _build_namespace() -> dict[str, Any]:
    """Define every Flax module. Called once, the first time one is looked up."""
    import jax.numpy as jnp
    from flax import linen as nn

    jinit = nn.initializers

    class ZCRMSNorm(nn.Module):
        """A gain written as (1 + gamma) with gamma at zero.

        The layer starts as an exact identity in scale, where a gain
        initialised at 1.0 would be a large parameter with no gradient signal
        to move it. The epsilon sits inside the square root, added to the mean
        square rather than to the root afterwards: that is the stable placement
        and easy to get backwards.
        """

        dtype: Any = jnp.bfloat16
        epsilon: float = _EPS

        @nn.compact
        def __call__(self, x: Any) -> Any:
            gamma = self.param("scale", jinit.zeros, (x.shape[-1],))
            # the mean square is in fp32 even in a bf16 model, because a small
            # denominator here becomes a large error
            rms = jnp.sqrt(jnp.mean(x.astype(jnp.float32) ** 2, axis=-1,
                                    keepdims=True) + self.epsilon)
            return ((1 + gamma) * x / rms).astype(self.dtype)

    class Spin(nn.Module):
        """Two fixed transforms and three learned diagonals, at next_pow2(d_model).

        A gated feed-forward network is three d by 4d matrices, 3.1 million
        parameters a layer at d_model 512. This keeps the shape without the
        matrices: the mixing is a Walsh transform that costs nothing to store,
        and only a diagonal scale on either side is learned. Eighty five million
        parameters across twenty seven layers become forty one thousand.
        """

        d_model: int
        dtype: Any = jnp.bfloat16

        @nn.compact
        def __call__(self, x: Any) -> Any:
            n = 1 << (self.d_model - 1).bit_length()
            hm = walsh(n).astype(self.dtype)
            d1 = self.param("d1", jinit.ones, (n,)).astype(self.dtype)
            d2 = self.param("d2", jinit.ones, (n,)).astype(self.dtype)
            # d3 starts at 0.02, not 1.0, so this branch has to earn its way in
            d3 = self.param("d3", jinit.constant(0.02), (n,)).astype(self.dtype)

            pad = n - self.d_model
            z = jnp.pad(x, ((0, 0), (0, 0), (0, pad))) if pad else x
            z = (d1 * z) @ hm
            z = nn.silu(d2 * z) @ hm         # the only nonlinearity
            return (d3 * z)[..., :self.d_model]

    class Porthole(nn.Module):
        """Grouped-query attention with QK-norm, an input gate and a bounded window.

        The front half is ordinary: query heads share key/value heads to shrink
        the cache, and the window lives in the mask rather than in the storage,
        so nothing here grows with the conversation.
        """

        config: QuartzConfig

        @nn.compact
        def __call__(self, x: Any, mask: Any = None, rope: Any = None) -> Any:
            cfg = self.config
            dtype = _dtype_of(cfg.dtype)
            b, t, d = x.shape
            heads, kv_heads, head_dim = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim

            def dense(features: int, name: str, stddev: float = 0.02):
                return nn.Dense(features, use_bias=False, dtype=dtype,
                                param_dtype=jnp.float32, name=name,
                                kernel_init=jinit.normal(stddev=stddev))

            q = dense(d, "q_proj")(x).reshape(b, t, heads, head_dim)
            k = dense(cfg.kv_dim, "k_proj")(x).reshape(b, t, kv_heads, head_dim)
            v = dense(cfg.kv_dim, "v_proj")(x).reshape(b, t, kv_heads, head_dim)
            q, k, v = (z.transpose(0, 2, 1, 3) for z in (q, k, v))

            # QK-norm: 3,456 parameters for 0.050 nats, the best ratio in the
            # ablation, and what keeps bf16 logits from blowing up
            q = ZCRMSNorm(dtype=dtype, name="q_norm")(q)
            k = ZCRMSNorm(dtype=dtype, name="k_norm")(k)
            if rope is not None:
                cos, sin = rope
                q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

            k = jnp.repeat(k, heads // kv_heads, axis=1)
            v = jnp.repeat(v, heads // kv_heads, axis=1)
            scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(
                jnp.float32(head_dim))
            if mask is None:
                mask = make_causal_mask(t)
            out = jnp.matmul(nn.softmax(
                jnp.where(mask, scores, jnp.finfo(scores.dtype).min), axis=-1), v)
            out = out.transpose(0, 2, 1, 3).reshape(b, t, d)
            # the gate is read from x, the block input, not from `out`, so it
            # decides about the CONTEXT rather than the RESULT
            out = out * nn.sigmoid(dense(d, "gate_proj")(x))
            # sqrt(2L) scaling, so L stacked residual branches do not compound
            return dense(d, "out_proj",
                         stddev=0.02 / math.sqrt(2 * cfg.num_layers))(out)

    def conv_identity_init(key: Any, shape: Any, dtype: Any = jnp.float32) -> Any:
        """A pass-through at first: looking back has to be learned."""
        del key
        return jnp.zeros(shape, dtype).at[0].set(1.0)

    class Imprint(nn.Module):
        """Hash tables instead of arithmetic.

        A tool called `set_thermostat_schedule_weekday` appears a handful of
        times in training and this many parameters at this depth has no room for
        it, so it gets a lookup instead. One hash per table indexes one row each,
        and the tables are concatenated back to d_model. A shared row in a single
        table is not a collision the model feels: at the shipped geometry two
        n-grams must share a row in all four tables at once, one pair in fifty
        billion.
        """

        config: QuartzConfig

        @nn.compact
        def __call__(self, idx: Any) -> tuple[Any, Any]:
            cfg = self.config
            dtype = _dtype_of(cfg.dtype)
            orders, _, sub_dim = cfg.imprint_geometry
            tables = cfg.imprint_tables
            b, t, _ = idx.shape

            table = self.param("table", jinit.normal(stddev=0.02),
                               (tables, cfg.imprint_slots, sub_dim))
            rows = table[jnp.arange(tables)[None, None, :], idx]
            e = rows.reshape(b, t, tables * sub_dim).astype(dtype)

            def dense(name: str):
                return nn.Dense(cfg.d_model, use_bias=False, dtype=dtype,
                                param_dtype=jnp.float32, name=name,
                                kernel_init=jinit.normal(stddev=0.02))

            k, v = dense("k_proj")(e), dense("v_proj")(e)

            # dilated by the longest n-gram order, so the taps land on
            # non-overlapping n-grams rather than re-reading the same one
            dilation = max(orders)
            taps = self.param("taps", conv_identity_init, (IMPRINT_TAPS, cfg.d_model))
            pos = jnp.arange(t)
            v = sum(taps[j].astype(dtype) * shift_right(v, j * dilation)
                    * (pos >= j * dilation)[:, None].astype(dtype)
                    for j in range(IMPRINT_TAPS))
            return k, v

    class Block(nn.Module):
        """One layer: optional memory, then gated attention, then Spin.

        The ordering is deliberately asymmetric. Attention gets a norm on both
        sides and a learned scalar gate starting at exactly one half, because
        its output magnitude swings with sequence content. Spin gets a norm in
        and nothing out: its d3 diagonal at 0.02 already sets its scale.
        """

        config: QuartzConfig

        @nn.compact
        def __call__(self, x: Any, mask: Any = None, rope: Any = None,
                     imprint: Any = None, site_flags: Any = None) -> Any:
            cfg = self.config
            dtype = _dtype_of(cfg.dtype)

            if imprint is not None and site_flags is not None:
                # fires BEFORE the skip is captured, so the memory is part of
                # what attention and Spin both read
                k, v = imprint
                alpha = nn.sigmoid(jnp.einsum("btd,sbtd->sbt", rms_unit(x),
                                              rms_unit(k)) / math.sqrt(cfg.d_model))
                x = x + jnp.einsum("s,sbt,sbtd->btd", site_flags, alpha,
                                   v.astype(jnp.float32)).astype(x.dtype)

            skip = x
            x = ZCRMSNorm(dtype=dtype)(x)
            x = Porthole(cfg, name="self_attn")(x, mask=mask, rope=rope)
            x = ZCRMSNorm(dtype=dtype, name="post_attn_norm")(x)
            gate = nn.sigmoid(self.param("attn_gate", jinit.zeros, ()))
            x = skip + gate.astype(dtype) * x

            skip = x
            return skip + Spin(cfg.d_model, dtype=dtype, name="spin")(
                ZCRMSNorm(dtype=dtype, name="pre_spin_norm")(x))

    class LoomBody(nn.Module):
        """One scanned layer of Loom: read the lanes, run the block, write back.

        A normal transformer has one residual stream and at this depth and width
        it is a bottleneck, with every layer competing for the same 512 numbers.
        Loom gives several streams and lets each layer choose which to read and
        where to put the result.
        """

        config: QuartzConfig

        @nn.compact
        def __call__(self, x: Any, layer_in: Any, mask: Any, rope: Any,
                     imprint: Any) -> tuple[Any, Any]:
            cfg = self.config
            site_flags, hc, offsets = layer_in
            pre_off, post_off = offsets
            b, t, n, c = x.shape

            # all lanes are normalised JOINTLY, as one lane_dim-wide vector, so
            # the router can see which lane is loud
            nx = rms_unit(x.reshape(b, t, n * c))

            h_pre = nn.sigmoid(hc["a_pre"] * (nx @ hc["phi_pre"]) + hc["b_pre"] + pre_off)
            # a SUM, not a convex combination: nothing forces these to add to
            # one, so a layer can open two lanes and read more than either
            u = jnp.einsum("btn,btnc->btc", h_pre, x)

            blk = Block(cfg, name="block")(u, mask=mask, rope=rope,
                                           imprint=imprint, site_flags=site_flags)
            y = blk - u          # the delta, not the output

            h_post = 2.0 * nn.sigmoid(
                hc["a_post"] * (nx @ hc["phi_post"]) + hc["b_post"] + post_off)
            routing = hc["a_res"] * (nx @ hc["phi_res"]).reshape(b, t, n, n) + hc["b_res"]
            p = sinkhorn(routing, cfg.sinkhorn_iters)

            # a stream that is only ever added to grows without bound as you
            # stack layers; one transported by a doubly stochastic matrix cannot
            # grow from the transport at all
            x = (jnp.einsum("btij,btjc->btic", p, x)
                 + h_post[..., None] * y[:, :, None, :].astype(jnp.float32))
            return x, blk

    class Trunk(nn.Module):
        """The layer stack, written once and scanned.

        Twenty seven copies of a block means twenty seven copies of the compiled
        graph and twenty seven times the live activations. `nn.scan` compiles one
        and gives every parameter a leading layer axis, which the quantiser and
        the exporter both rely on afterwards.
        """

        config: QuartzConfig

        @nn.compact
        def __call__(self, x: Any, mask: Any = None, rope: Any = None,
                     imprint: Any = None) -> tuple[Any, Any]:
            cfg = self.config
            n, layers, wide = cfg.lanes, cfg.num_layers, cfg.lane_dim

            def res_bias(key: Any, shape: Any, dtype: Any = jnp.float32) -> Any:
                # four times the identity, so P starts at 0.948 on the diagonal
                del key
                return jnp.broadcast_to(4.0 * jnp.eye(shape[-1], dtype=dtype), shape)

            phi = jinit.normal(stddev=0.02)
            specs = (
                ("phi_pre", phi, (layers, wide, n)),
                ("phi_post", phi, (layers, wide, n)),
                ("phi_res", phi, (layers, wide, n * n)),
                ("b_pre", jinit.zeros, (layers, n)),
                ("b_post", jinit.zeros, (layers, n)),
                ("b_res", res_bias, (layers, n, n)),
                # The amplitudes start small but NOT at zero. At zero the whole
                # phi block would take an exactly zero gradient, since d/dphi is
                # a times the input, and 1.3M parameters would sit dead for the
                # first step. At 0.02 the router contributes about +/-0.02 to a
                # logit of +/-4, so the offsets still decide the routing and the
                # lanes still read 0.982 and 0.018, but phi learns from step one.
                ("a_pre", jinit.constant(0.02), (layers,)),
                ("a_post", jinit.constant(0.02), (layers,)),
                ("a_res", jinit.constant(0.02), (layers,)),
            )
            # every router parameter carries a leading layer axis for scan to slice
            hc = {nm: self.param(f"loom_{nm}", ini, sh) for nm, ini, sh in specs}

            pre_off, post_off = _lane_offsets(layers, n)
            layer_in = (jnp.asarray(_site_flags(layers, cfg.imprint_sites)), hc,
                        (jnp.asarray(pre_off), jnp.asarray(post_off)))

            x = jnp.repeat(x.astype(jnp.float32)[:, :, None, :], n, axis=2)
            # remat discards each block's activations and recomputes them on the
            # backward pass: about thirty percent more compute, a large drop in
            # peak memory, and what makes a two million token batch fit
            body = nn.remat(LoomBody) if cfg.remat else LoomBody
            scanned = nn.scan(
                body,
                variable_axes={"params": 0},      # params gain a leading L axis
                split_rngs={"params": True},      # each layer initialised separately
                length=layers, unroll=cfg.scan_unroll,
                in_axes=(0, nn.broadcast, nn.broadcast, nn.broadcast))
            x, hidden = scanned(cfg, name="layers")(x, layer_in, mask, rope, imprint)
            x = jnp.mean(x, axis=2)               # the lanes become one
            return ZCRMSNorm(dtype=_dtype_of(cfg.dtype), name="final_norm")(x), hidden

    class Dowser(nn.Module):
        """Retrieval over a catalogue that does not fit in the window.

        A phone assistant has hundreds of tools and 41,000 is the design target,
        none of which fits in 256 tokens. This head embeds the request into the
        same space as the tool descriptions so the top few can be rendered.
        """

        config: QuartzConfig

        @nn.compact
        def __call__(self, cells: Any) -> tuple[Any, Any]:
            cfg = self.config
            probes = self.param("probes", jinit.normal(stddev=0.02),
                                (cfg.dowser_probes, cells.shape[-1]))
            pooled = probe_pool(cells, probes)
            p = nn.Dense(cfg.dowser_dim, use_bias=False, param_dtype=jnp.float32,
                         kernel_init=jinit.normal(stddev=0.02))(pooled)
            # the temperature is learned but clipped by the loss, since a
            # sharper distribution always lowers the training objective
            log_temp = self.param("log_temp", jinit.constant(math.log(0.07)), ())
            return p / jnp.sqrt(jnp.sum(p ** 2, -1, keepdims=True) + 1e-12), log_temp

    class Gauge(nn.Module):
        """Should we act at all, or hand this to something bigger?

        The output layer is initialised to ZERO so an untrained head returns 0.5
        for every input. A head that starts confident is wrong in the most
        dangerous direction.
        """

        config: QuartzConfig

        @nn.compact
        def __call__(self, cells: Any) -> Any:
            cfg = self.config
            probes = self.param("probes", jinit.normal(stddev=0.02),
                                (cfg.gauge_probes, cells.shape[-1]))
            pooled = probe_pool(cells, probes)
            return nn.Dense(1, param_dtype=jnp.float32,
                            kernel_init=jinit.zeros)(pooled)[..., 0]

    class QuartzNetwork(nn.Module):
        """Embedding, memory sites, trunk, heads, and the tied output head.

        The embedding is tied, so its parameters are both the input table and
        the output projection. Untied, this model would be 49.4 million
        parameters rather than 45.2 million.
        """

        config: QuartzConfig

        def setup(self) -> None:
            cfg = self.config
            dtype = _dtype_of(cfg.dtype)
            self.embedding = nn.Embed(
                cfg.vocab_size, cfg.d_model, param_dtype=jnp.float32,
                embedding_init=jinit.normal(stddev=1.0 / math.sqrt(cfg.d_model)))
            self.memory = [Imprint(cfg) for _ in cfg.imprint_sites]
            self.trunk = Trunk(cfg)
            self.dowser = Dowser(cfg)
            self.gauge = Gauge(cfg)
            # Foresight: token t+2 from the trunk state at t and the embedding
            # of token t+1. Dropped entirely at export, so the device pays nothing.
            self.fs_combine = nn.Dense(cfg.d_model, use_bias=False, dtype=dtype,
                                       param_dtype=jnp.float32,
                                       kernel_init=jinit.normal(stddev=0.02))
            self.fs_emb_norm = ZCRMSNorm(dtype=dtype)
            self.fs_block = Block(cfg)
            self.fs_final_norm = ZCRMSNorm(dtype=dtype)

        @property
        def embed_scale(self) -> float:
            """Embeddings are initialised at 1/sqrt(d) and scaled back up here,
            so the stream enters the trunk at unit RMS whatever d_model is."""
            return math.sqrt(self.config.d_model)

        def _body(self, tokens: Any, mask: Any, rope: Any) -> tuple[Any, Any, Any]:
            """Shared forward: the final state, the cell grid, and the mask/rope
            actually used, which the Foresight branch needs to reuse."""
            cfg = self.config
            t = tokens.shape[1]
            if mask is None:
                mask = make_causal_mask(t)
            if rope is None:
                rope = rope_tables(cfg.head_dim, t, cfg.rope_theta)

            x = self.embedding(tokens) * self.embed_scale
            imprint = None
            if self.memory:
                orders, heads, _ = cfg.imprint_geometry
                idx = imprint_indices(tokens, orders, heads, cfg.imprint_slots)
                pairs = [site(idx) for site in self.memory]
                imprint = (jnp.stack([k for k, _ in pairs]),
                           jnp.stack([v for _, v in pairs]))

            x, hidden = self.trunk(x, mask=mask, rope=rope, imprint=imprint)
            # (layers, batch, time, d) -> (batch, time, layers + 1, d), so the
            # probes see every layer and the final state on the same footing
            cells = jnp.concatenate(
                [jnp.moveaxis(hidden, 0, 2).astype(jnp.float32),
                 x[:, :, None, :].astype(jnp.float32)], axis=2)
            return x, cells, (mask, rope)

        def _foresight(self, tokens: Any, x: Any, mask: Any, rope: Any) -> Any:
            """Predict token t+2 from the state at t and the embedding of t+1.

            A denser signal for the same forward pass: it pushes the trunk
            towards a representation useful beyond the next symbol.
            """
            nxt = jnp.pad(tokens[:, 1:], ((0, 0), (0, 1)))
            e2 = self.fs_emb_norm(self.embedding(nxt) * self.embed_scale)
            m = self.fs_block(self.fs_combine(jnp.concatenate([x, e2], axis=-1)),
                              mask=mask, rope=rope)      # no Imprint here
            return self.fs_final_norm(m).astype(jnp.float32) @ self.embedding.embedding.T

        def __call__(self, tokens: Any, mask: Any = None, rope: Any = None,
                     return_foresight: bool = False) -> Any:
            """Logits, or (logits, foresight logits) when asked.

            Every head runs during `init` whether it was asked for or not,
            because Flax only materialises the parameters of submodules that
            are actually called and the count has to be complete.
            """
            x, cells, (mask, rope) = self._body(tokens, mask, rope)
            logits = x.astype(jnp.float32) @ self.embedding.embedding.T
            if not (return_foresight or self.is_initializing()):
                return logits
            if self.is_initializing():
                self.dowser(cells)
                self.gauge(cells)
            fs = self._foresight(tokens, x, mask, rope)
            return (logits, fs) if return_foresight else logits

        def features(self, tokens: Any, mask: Any = None, rope: Any = None) -> Any:
            """(logits, cells). The cell grid is what the two heads read."""
            x, cells, _ = self._body(tokens, mask, rope)
            return x.astype(jnp.float32) @ self.embedding.embedding.T, cells

        def heads(self, tokens: Any, mask: Any = None, rope: Any = None) -> dict[str, Any]:
            """Both auxiliary heads in one pass, for retrieval and for confidence."""
            _, cells, _ = self._body(tokens, mask, rope)
            embedding, log_temp = self.dowser(cells)
            return {"dowser": embedding, "log_temp": log_temp,
                    "gauge": self.gauge(cells)}

    return {
        "ZCRMSNorm": ZCRMSNorm, "Spin": Spin, "Porthole": Porthole, "Imprint": Imprint,
        "Block": Block, "LoomBody": LoomBody, "Trunk": Trunk, "Dowser": Dowser,
        "Gauge": Gauge, "QuartzNetwork": QuartzNetwork,
    }


def __getattr__(name: str) -> Any:
    """Materialise the Flax modules on first touch, so importing this module
    does not require JAX to be installed."""
    if name in _LAZY:
        global _NAMESPACE
        if _NAMESPACE is None:
            _NAMESPACE = _build_namespace()
        return _NAMESPACE[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


# --- the count --------------------------------------------------------------
def param_count(config: QuartzConfig) -> dict[str, int]:
    """Every parameter in the model, by component, derived from the config.

    The shape of the answer is the argument for the design, so nothing here is
    a quoted figure: change d_model or num_layers and the table follows. For
    ``preset("base")`` the components sum to 45,211,383, of which the
    ``foresight`` entry is dropped at export.
    """
    v, d, layers = config.vocab_size, config.d_model, config.num_layers
    head_dim, kv_dim, n_had = config.head_dim, config.kv_dim, config.hadamard_n
    lanes, wide = config.lanes, config.lane_dim
    _, _, sub_dim = config.imprint_geometry
    sites = len(config.imprint_sites)

    block = (
        d * d                       # q_proj
        + 2 * d * kv_dim            # k_proj and v_proj, shrunk by the KV grouping
        + d * d                     # gate_proj, read from the block input
        + d * d                     # out_proj, depth scaled
        + 2 * head_dim              # QK-norm gains
        + 3 * d                     # pre-attn, post-attn and pre-Spin gains
        + 1                         # the scalar attention gate
        + 3 * n_had                 # Spin's three diagonals
    )
    routers = (
        layers * wide * lanes * 2               # phi_pre and phi_post
        + layers * wide * lanes * lanes         # phi_res, the transport logits
        + 2 * layers * lanes                    # b_pre and b_post
        + layers * lanes * lanes                # b_res, four times the identity
        + 3 * layers                            # a_pre, a_post, a_res
    )
    site = (config.imprint_tables * config.imprint_slots * sub_dim   # the tables
            + 2 * d * d                                              # k_proj, v_proj
            + IMPRINT_TAPS * d)                                      # the conv taps
    dowser = (config.dowser_probes * d                    # the probes
              + config.dowser_probes * d * config.dowser_dim   # the projection
              + 1)                                        # the learned temperature
    gauge = (config.gauge_probes * d                      # the probes
             + config.gauge_probes * d + 1)               # one logit, kernel and bias

    return {
        "embedding": v * d,                     # tied: input table and output head
        "blocks": layers * block,
        "loom_routers": routers,
        "imprint": sites * site,
        "dowser": dowser,
        "gauge": gauge,
        "final_norm": d,
        "foresight": 2 * d * d + block + 2 * d,  # training only, dropped at export
    }


if __name__ == "__main__":     # pragma: no cover
    from .config import preset

    counts = param_count(preset("base"))
    for name, n in counts.items():
        print(f"  {name:<16} {n:>12,}")
    total = sum(counts.values())
    shipped = total - sum(counts[k] for k in EXPORT_DROPPED)
    print(f"  {'TOTAL':<16} {total:>12,}")
    print(f"  {'SHIPPED':<16} {shipped:>12,}")
    assert total == BASE_PARAM_COUNT, f"{total} != {BASE_PARAM_COUNT}"
    assert shipped == SHIPPED_PARAM_COUNT, f"{shipped} != {SHIPPED_PARAM_COUNT}"
