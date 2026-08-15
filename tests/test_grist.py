"""Grist: two bits, no calibration set.

The claim being tested is a chain of three links. A Walsh rotation makes every
coordinate of a group a signed sum of 128 weights, so the group is Gaussian
whatever it was before. A Gaussian is what the codebook was fitted to, offline,
so it is the right codebook for every group in the model. And that is why
nothing here needs a calibration set, a Hessian, or a forward pass.

The first test is the one that carries the argument: rotation removes the
outliers that make absmax quantisation fail, and the round trip that follows it
loses a fraction of what absmax loses at the same width.
"""
from __future__ import annotations

import numpy as np
import pytest

from quartz.model.config import (
    CQ_BITS,
    CQ_GROUP,
    DEFAULT_BITS_MAP,
    TERNARY_BITS,
    TERNARY_CENTROID,
)
from quartz.model.grist import (
    bits_for,
    canonical,
    codebook,
    cq_decode,
    cq_encode,
    cq_quantize,
    flatten_params,
    group_axis,
    is_quantised,
    leaf_bytes,
    lloyd_max_gaussian,
    mixed_quantise,
    model_bytes,
    nearest,
    parse_bits_map,
    quant_leaf_names,
    quantise_params,
    size_report,
    walsh_np,
)


def kurtosis(x) -> float:
    """Excess kurtosis. Zero is Gaussian, and 41 is six outliers in four thousand."""
    x = np.asarray(x, dtype=np.float64).ravel()
    centred = x - x.mean()
    return float(np.mean(centred ** 4) / np.mean(centred ** 2) ** 2 - 3.0)


def relative_mse(q, w) -> float:
    q, w = np.asarray(q, np.float64), np.asarray(w, np.float64)
    return float(np.mean((q - w) ** 2) / np.mean(w ** 2))


def absmax_quantise(w, bits: int):
    """The scheme Grist replaces: one scale for the whole tensor.

    Kept in the tests rather than in the package, because it exists only to be
    beaten and shipping it would invite someone to call it.
    """
    lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    scale = np.max(np.abs(w)) / hi
    return np.clip(np.round(w / scale), lo, hi) * scale


def outlier_weights(n: int = 4096, spikes: int = 6, size: float = 8.0):
    """A weight vector shaped like a real one: Gaussian, plus a few spikes."""
    rng = np.random.default_rng(0)
    w = rng.standard_normal(n)
    w[-spikes:] = size * rng.standard_normal(spikes)
    return w


# --- the rotation -----------------------------------------------------------
def test_rotation_reduces_kurtosis():
    """Six outliers in four thousand, gone after one butterfly of additions.

    Every rotated coordinate is a fixed signed sum of all 128 inputs, so no
    weight survives averaging with 127 others.
    """
    w = outlier_weights()
    rotated = w.reshape(-1, CQ_GROUP) @ walsh_np(CQ_GROUP)

    # the group holding the spikes is the one a shared codebook has to survive
    before, after = kurtosis(w[-CQ_GROUP:]), kurtosis(rotated[-1])
    assert before > 10.0
    assert abs(after) < 1.0                       # Gaussian, to one decimal place
    assert abs(kurtosis(rotated)) < kurtosis(w) / 10.0

    def peak(a) -> float:
        return float(np.abs(a).max() / np.asarray(a).std())

    assert peak(rotated) < peak(w) / 2.0


def test_walsh_is_symmetric_and_its_own_inverse():
    for n in (2, 8, CQ_GROUP):
        h = walsh_np(n)
        assert np.allclose(h, h.T)
        assert np.allclose(h @ h, np.eye(n), atol=1e-6)
        assert np.allclose(np.abs(h), 1.0 / np.sqrt(n))


def test_the_walsh_matrix_is_a_shared_constant():
    """It is asked for a few hundred thousand times, so it is cached and frozen."""
    assert walsh_np(CQ_GROUP) is walsh_np(CQ_GROUP)
    assert not walsh_np(CQ_GROUP).flags.writeable


def test_walsh_needs_a_power_of_two():
    with pytest.raises(ValueError, match="power of two"):
        walsh_np(100)


# --- the codebook -----------------------------------------------------------
@pytest.mark.parametrize("bits", CQ_BITS)
def test_the_codebook_is_symmetric(bits):
    """A Gaussian is symmetric, so its optimal quantiser is too.

    An asymmetric codebook would put a bias on every group in the model, which
    at 45 million weights is not a rounding detail. What is left is the sampling
    error of a 400,000 point fit, which is under two percent of the outermost
    level and an order below the spacing between levels.
    """
    levels = codebook(bits, CQ_GROUP)
    assert levels.size == 2 ** bits
    assert np.all(np.diff(levels) > 0)
    assert np.allclose(levels, -levels[::-1], atol=0.02 * np.abs(levels).max())
    assert abs(float(levels.mean())) < 0.1 * float(np.diff(levels).min())


def test_the_codebook_is_scaled_to_a_unit_vector():
    """A unit vector in R^group has coordinates of about 1/sqrt(group)."""
    for bits in CQ_BITS:
        assert np.allclose(codebook(bits, CQ_GROUP) * np.sqrt(CQ_GROUP),
                           lloyd_max_gaussian(bits), atol=1e-5)


def test_ternary_is_a_constant_not_a_fit():
    """A device rebuilds these three levels from a header field, so a refit
    would change what a stored index means."""
    levels = codebook(TERNARY_BITS, CQ_GROUP) * np.sqrt(CQ_GROUP)
    assert np.allclose(levels, [-TERNARY_CENTROID, 0.0, TERNARY_CENTROID])
    assert np.array_equal(lloyd_max_gaussian(TERNARY_BITS, iters=1, samples=10),
                          lloyd_max_gaussian(TERNARY_BITS))


def test_more_bits_is_a_finer_codebook():
    errors = [relative_mse(cq_quantize(outlier_weights().reshape(-1, CQ_GROUP), bits),
                           outlier_weights().reshape(-1, CQ_GROUP))
              for bits in CQ_BITS]
    assert errors == sorted(errors, reverse=True)


def test_nearest_breaks_a_tie_towards_the_lower_level():
    """Two runs of the exporter have to produce identical bytes."""
    levels = codebook(2, CQ_GROUP)
    boundary = (levels[0] + levels[1]) / 2.0
    assert nearest(np.array([boundary]), levels)[0] == 0
    assert nearest(np.array([levels[-1] * 10]), levels)[0] == levels.size - 1


# --- the round trip ---------------------------------------------------------
def test_quantise_dequantise_error_is_far_below_absmax():
    """The headline: absmax destroys most of the signal, Grist loses a tenth.

    Same weights, same two bits. The difference is entirely the rotation, which
    is why this is the test that carries the design.
    """
    w = outlier_weights()
    absmax = relative_mse(absmax_quantise(w, 2), w)
    grist = relative_mse(cq_quantize(w.reshape(-1, CQ_GROUP), 2),
                         w.reshape(-1, CQ_GROUP))

    assert absmax > 0.5                 # six weights set the scale, the rest die
    assert grist < 0.2
    assert grist < absmax / 4.0


def test_the_round_trip_is_deterministic():
    w = outlier_weights().reshape(-1, CQ_GROUP)
    assert np.array_equal(cq_quantize(w, 2), cq_quantize(w, 2))


def test_encode_keeps_the_shape_and_the_lengths_it_will_store():
    """The lengths come back already rounded through fp16, because that is what
    the file holds and the reconstruction has to agree with it."""
    w = np.asarray(outlier_weights(512).reshape(4, CQ_GROUP), dtype=np.float32)
    codes, norm, shape = cq_encode(w, 2, CQ_GROUP, axis=-1)

    assert shape == w.shape
    assert codes.shape == w.shape
    assert norm.shape == (4, 1)
    assert norm.dtype == np.float16
    assert codes.max() < 4
    assert np.allclose(cq_decode(codes, norm, 2, shape), cq_quantize(w, 2))


def test_a_short_axis_is_padded_not_refused():
    w = np.asarray(outlier_weights(CQ_GROUP + 32).reshape(1, -1), dtype=np.float32)
    out = cq_quantize(w, 2, CQ_GROUP, axis=-1)
    assert out.shape == w.shape


def test_encode_refuses_what_it_cannot_group():
    with pytest.raises(ValueError, match="scalar"):
        cq_encode(np.float32(1.0), 2)
    with pytest.raises(ValueError, match="axis 3 is outside"):
        cq_encode(np.zeros((4, 4), np.float32), 2, CQ_GROUP, axis=3)
    with pytest.raises(ValueError, match="power of two"):
        cq_encode(np.zeros((4, 100), np.float32), 2, group=100)


# --- the bits map -----------------------------------------------------------
def test_a_bits_map_with_no_default_is_rejected():
    """A built-in fallback would one day ship a model whose numerics nobody
    chose, and the failure would be two quiet points of accuracy."""
    with pytest.raises(ValueError, match="needs a default"):
        parse_bits_map("loom.phi=4,attn.out_proj=3")


def test_the_shipped_scheme_parses_to_two_promotions():
    table, default = parse_bits_map(DEFAULT_BITS_MAP)
    assert default == 2
    assert table == {"loom.phi": 4, "attn.out_proj": 3}


@pytest.mark.parametrize("spec", ["default=2,loom.phi", "default=5", "default",
                                  "default=2,=4", "default=2.5"])
def test_a_bits_map_that_cannot_be_read_is_an_error(spec):
    with pytest.raises(ValueError):
        parse_bits_map(spec)


def test_widths_that_mean_leave_it_alone_are_accepted():
    table, default = parse_bits_map("default=2,dowser=16,gauge=32")
    assert (table["dowser"], table["gauge"], default) == (16, 32, 2)
    assert parse_bits_map(f"default={TERNARY_BITS}")[1] == TERNARY_BITS


def test_the_longest_prefix_wins():
    table, default = parse_bits_map("default=2,attn=4,attn.out_proj=3")
    assert bits_for("attn.out_proj", table, default) == 3
    assert bits_for("attn.q_proj", table, default) == 4
    assert bits_for("embedding", table, default) == 2

    # a prefix on the string, not on the components, so one entry names phi_pre,
    # phi_post and phi_res at once
    shipped, shipped_default = parse_bits_map(DEFAULT_BITS_MAP)
    for name in ("loom.phi_pre", "loom.phi_post", "loom.phi_res"):
        assert bits_for(name, shipped, shipped_default) == 4


# --- names ------------------------------------------------------------------
def test_a_flax_path_canonicalises_to_what_a_person_writes():
    assert canonical(("params", "layers", "self_attn", "out_proj", "kernel")) \
        == "attn.out_proj"
    assert canonical("params/trunk/layers/embed/embedding") == "embedding"
    assert canonical(("params", "loom_phi_pre")) == "loom.phi_pre"
    assert canonical(("params", "Dowser_0", "probes")) == "dowser_0.probes"


def test_group_axis_reads_a_table_differently_from_a_matrix():
    """A table's row is the vector; everything else groups along the axis a
    matmul contracts."""
    assert group_axis("embedding", 2) == 1
    assert group_axis("imprint.2.table", 3) == 2
    assert group_axis("attn.q_proj", 2) == 0
    assert group_axis("attn.q_proj", 3) == 1
    assert group_axis("attn_gate", 1) is None


def test_is_quantised_has_four_refusals():
    assert is_quantised("attn.q_proj", (512, 512))
    assert not is_quantised("attn_norm", (512,))            # a vector has no group
    assert not is_quantised("gauge.probe", (8, 512))        # the heads stay float
    assert not is_quantised("attn_norm", (27, 512))         # a scale, 27 layers deep
    assert not is_quantised("attn.k_proj", (64, 64))        # more padding than data


def test_quant_leaf_names_is_the_structural_set(tiny_params):
    names = quant_leaf_names(tiny_params)
    assert "embedding" in names
    assert "attn.q_proj" in names
    assert not [name for name in names if name.startswith(("dowser", "gauge"))]
    assert all(is_quantised(name, dict(flatten_params(tiny_params))[name].shape)
               for name in names)


def test_two_leaves_that_canonicalise_the_same_are_an_error():
    tree = {"self_attn": {"out_proj": {"kernel": np.zeros((4, 4), np.float32)}},
            "attention": {"out_proj": {"weight": np.zeros((4, 4), np.float32)}}}
    with pytest.raises(ValueError, match="canonicalise"):
        flatten_params(tree)


# --- whole trees ------------------------------------------------------------
def test_mixed_quantise_returns_what_the_file_will_hold(tiny_params):
    """Every leaf that skips the codebook is still rounded through fp16.

    So an evaluation of this tree is an evaluation of the shipped file, which
    is the only reason the number in the post means anything.
    """
    out = mixed_quantise(tiny_params, DEFAULT_BITS_MAP)
    flat_in = dict(flatten_params(tiny_params))
    flat_out = dict(flatten_params(out))

    assert set(flat_out) == set(flat_in)
    for name, leaf in flat_out.items():
        assert leaf.shape == flat_in[name].shape
        if not is_quantised(name, leaf.shape):
            assert np.array_equal(
                leaf, flat_in[name].astype(np.float16).astype(np.float32))

    # the two promoted families really are finer than the default
    coarse = quantise_params(tiny_params, 2)
    phi = "loom.phi_pre"
    assert (relative_mse(flat_out[phi], flat_in[phi])
            < relative_mse(dict(flatten_params(coarse))[phi], flat_in[phi]))


def test_integer_leaves_are_left_alone():
    tree = {"tokens": np.arange(8, dtype=np.int32)}
    assert np.array_equal(mixed_quantise(tree, "default=2")["tokens"], tree["tokens"])


# --- byte accounting --------------------------------------------------------
def test_the_model_shrinks_with_the_width(base_cfg):
    sizes = [model_bytes(base_cfg, bits) for bits in (32, 16, 4, 3, 2, TERNARY_BITS)]
    assert sizes == sorted(sizes, reverse=True)
    assert model_bytes(base_cfg, 2) < model_bytes(base_cfg, 16) / 4


def test_the_shipped_scheme_sits_between_two_and_four_bits(base_cfg):
    mixed = model_bytes(base_cfg, DEFAULT_BITS_MAP)
    assert model_bytes(base_cfg, 2) < mixed < model_bytes(base_cfg, 4)

    report = size_report(base_cfg, DEFAULT_BITS_MAP)
    assert report["bytes"] == mixed
    assert 2.0 < report["mean_bits"] < 4.0
    assert report["quantised"] > report["stored"] * 10
    assert report["params"] == report["quantised"] + report["stored"]


def test_a_parameter_tree_beats_the_map_when_one_is_given(tiny_cfg, tiny_params):
    """The tree is the authority; `shipped_tensors` is the map of it.

    They have to agree, and Foresight has to be missing from both, or the size
    quoted for the download is a size nobody downloads.
    """
    assert model_bytes(tiny_cfg, 2, params=tiny_params) == model_bytes(tiny_cfg, 2)


def test_leaf_bytes_counts_the_indices_and_the_lengths():
    rows, width = 4, CQ_GROUP
    got = leaf_bytes("attn.q_proj", (width, rows), 2, CQ_GROUP)
    # one two-bit index per weight, plus one fp16 length per group
    assert got == rows * (width * 2 / 8 + 2)
    assert leaf_bytes("attn_norm", (512,), 2) == 512 * 2       # left in fp16
    assert leaf_bytes("attn_norm", (512,), 32) == 512 * 4
