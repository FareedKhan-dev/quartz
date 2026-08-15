"""The model: the parameter table, the masks, the hash, and the two heads.

The parameter table is the argument for the design, so the first test is that
it is *derived*: change the width or the depth and every component follows. The
second is that it is *true*: a real Flax initialisation of the same geometry has
exactly the number of parameters the table claims, component for component.

Everything below that needs JAX and says so. `param_count` does not, which is
the point of keeping it out of the Flax namespace: a size can be quoted before
a single parameter has been initialised.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from quartz.model import architecture
from quartz.model.architecture import (
    BASE_PARAM_COUNT,
    EXPORT_DROPPED,
    SHIPPED_PARAM_COUNT,
    param_count,
)
from quartz.model.config import IMPRINT_TAPS, preset

from .conftest import requires_jax


@pytest.fixture(scope="module")
def tiny_model():
    """One initialised two-layer model, shared by every test that needs one.

    Initialising is the expensive part and nothing here mutates it, so it is
    built once for the module rather than once a test.
    """
    jax = requires_jax()
    jnp = jax.numpy

    cfg = preset("tiny", vocab_size=512, imprint_slots=64)
    model = architecture.QuartzNetwork(cfg)
    tokens = jnp.ones((1, 8), jnp.int32)
    params = model.init(jax.random.PRNGKey(0), tokens)["params"]
    return cfg, model, params


def leaf_total(params) -> int:
    import jax

    return sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(params))


# --- the count, on numpy alone ----------------------------------------------
def test_the_components_sum_to_the_shipped_count(base_cfg):
    counts = param_count(base_cfg)
    assert sum(counts.values()) == BASE_PARAM_COUNT == 45_211_383
    shipped = sum(counts.values()) - sum(counts[name] for name in EXPORT_DROPPED)
    assert shipped == SHIPPED_PARAM_COUNT


def test_every_component_is_derived_from_the_config(base_cfg):
    """Nothing in the table is a quoted figure."""
    counts = param_count(base_cfg)
    _, _, sub_dim = base_cfg.imprint_geometry

    assert counts["embedding"] == base_cfg.vocab_size * base_cfg.d_model
    assert counts["final_norm"] == base_cfg.d_model
    assert counts["imprint"] == len(base_cfg.imprint_sites) * (
        base_cfg.imprint_tables * base_cfg.imprint_slots * sub_dim
        + 2 * base_cfg.d_model ** 2 + IMPRINT_TAPS * base_cfg.d_model)
    assert counts["dowser"] == (base_cfg.dowser_probes * base_cfg.d_model
                                + base_cfg.dowser_probes * base_cfg.d_model
                                * base_cfg.dowser_dim + 1)
    assert counts["gauge"] == 2 * base_cfg.gauge_probes * base_cfg.d_model + 1


def test_spin_is_what_makes_the_depth_affordable(base_cfg):
    """A gated feed-forward network at this width would be 3.1M a layer.

    Spin keeps the shape and drops the matrices: three diagonals at the
    Hadamard width, so 85 million parameters become forty one thousand.
    """
    counts = param_count(base_cfg)
    spin = 3 * base_cfg.hadamard_n * base_cfg.num_layers
    gated_ffn = 3 * base_cfg.d_model * 4 * base_cfg.d_model * base_cfg.num_layers

    assert spin == 41_472
    assert gated_ffn > 80_000_000
    assert counts["blocks"] > spin


def test_the_table_scales_with_the_geometry():
    wide = param_count(preset("wide"))
    base = param_count(preset("base"))
    assert sum(wide.values()) > sum(base.values())
    assert all(wide[name] >= base[name] for name in base)


# --- the free functions -----------------------------------------------------
@pytest.mark.needs_jax
def test_walsh_matches_the_numpy_one_and_inverts_itself():
    import jax.numpy as jnp

    from quartz.model.grist import walsh_np

    h = architecture.walsh(128)
    assert np.allclose(np.asarray(h), walsh_np(128))
    assert np.allclose(np.asarray(h @ h), np.eye(128), atol=1e-5)
    assert jnp.asarray(h).shape == (128, 128)


@pytest.mark.needs_jax
def test_rope_rotates_without_changing_a_length():
    import jax.numpy as jnp

    cfg = preset("tiny")
    cos, sin = architecture.rope_tables(cfg.head_dim, 16, cfg.rope_theta)
    assert cos.shape == sin.shape == (16, cfg.head_dim // 2)

    x = jnp.asarray(np.random.default_rng(0).standard_normal(
        (2, cfg.num_heads, 16, cfg.head_dim)), jnp.float32)
    y = architecture.apply_rope(x, cos, sin)

    assert y.shape == x.shape
    assert np.allclose(np.linalg.norm(np.asarray(y), axis=-1),
                       np.linalg.norm(np.asarray(x), axis=-1), atol=1e-4)
    # position zero is an identity, so a prefix of one token is untouched
    assert np.allclose(np.asarray(y[:, :, :1]), np.asarray(x[:, :, :1]), atol=1e-5)


@pytest.mark.needs_jax
def test_rope_refuses_a_table_that_is_too_short():
    import jax.numpy as jnp

    cos, sin = architecture.rope_tables(8, 4)
    with pytest.raises(ValueError, match="rope tables hold"):
        architecture.apply_rope(jnp.zeros((1, 1, 8, 8)), cos, sin)
    with pytest.raises(ValueError, match="even head_dim"):
        architecture.rope_tables(7, 4)


@pytest.mark.needs_jax
def test_the_causal_mask_is_the_lower_triangle():
    mask = np.asarray(architecture.make_causal_mask(4))
    assert mask.shape == (1, 1, 4, 4)
    assert np.array_equal(mask[0, 0], np.tril(np.ones((4, 4), bool)))


@pytest.mark.needs_jax
def test_the_packing_mask_closes_the_seam_between_documents():
    """Without it a token starting document two attends to the end of document
    one and learns a transition that never happened."""
    import jax.numpy as jnp

    seg = jnp.asarray([[1, 1, 2, 2, 0]])          # two documents, then padding
    mask = np.asarray(architecture.make_packing_mask(seg))[0, 0]

    assert not mask[2, 1]                         # across the seam
    assert mask[3, 2]                             # inside document two
    assert not mask[1, 2]                         # and still causal
    assert not mask[4].any()                      # a padding row attends to nothing


@pytest.mark.needs_jax
def test_the_window_mask_pins_the_keys_that_must_not_be_evicted():
    """The schemas sit at the front of the turn, so a bare window drops them
    first and the model is asked to call tools it can no longer see."""
    import jax.numpy as jnp

    sink = jnp.asarray([[True, False, False, False, False, False]])
    plain = np.asarray(architecture.window_mask(6, 2))[0, 0]
    pinned = np.asarray(architecture.window_mask(6, 2, sink))[0, 0]

    assert not plain[5, 3]                        # outside the window
    assert plain[5, 4]                            # inside it
    assert not plain[5, 0]                        # the schemas have been evicted
    assert pinned[5, 0]                           # unless they are pinned
    assert not pinned[0, 1:].any()                # pinning never breaks causality


@pytest.mark.needs_jax
def test_shift_right_is_where_causality_comes_from():
    import jax.numpy as jnp

    x = jnp.arange(12, dtype=jnp.float32).reshape(1, 4, 3)
    out = np.asarray(architecture.shift_right(x, 1))
    assert out.shape == (1, 4, 3)
    assert not out[0, 0].any()
    assert np.array_equal(out[0, 1:], np.asarray(x)[0, :-1])
    assert np.array_equal(np.asarray(architecture.shift_right(x, 0)), np.asarray(x))


@pytest.mark.needs_jax
def test_sinkhorn_leaves_the_columns_stochastic():
    """The loop ends on a column normalisation, and column sums are what control
    the mass arriving in each lane."""
    import jax.numpy as jnp

    logits = jnp.asarray(np.random.default_rng(0).standard_normal((2, 3, 4, 4)),
                         jnp.float32)
    p = np.asarray(architecture.sinkhorn(logits, 20))
    assert np.allclose(p.sum(axis=-2), 1.0, atol=1e-5)
    assert (p >= 0).all()


# --- the hash ---------------------------------------------------------------
@pytest.mark.needs_jax
def test_imprint_indices_are_causal_and_in_range():
    """The row a rare tool name lands in is fixed before training starts, and
    a token at t may never read a token after it."""
    import jax.numpy as jnp

    cfg = preset("tiny", imprint_slots=64)
    orders, heads, _ = cfg.imprint_geometry
    tokens = jnp.asarray([[5, 6, 7, 8]], jnp.int32)
    idx = np.asarray(architecture.imprint_indices(tokens, orders, heads,
                                                  cfg.imprint_slots))

    assert idx.shape == (1, 4, cfg.imprint_tables)
    assert idx.min() >= 0 and idx.max() < cfg.imprint_slots

    changed = jnp.asarray([[5, 6, 7, 99]], jnp.int32)
    after = np.asarray(architecture.imprint_indices(changed, orders, heads,
                                                    cfg.imprint_slots))
    assert np.array_equal(idx[:, :3], after[:, :3])
    assert not np.array_equal(idx[:, 3], after[:, 3])


@pytest.mark.needs_jax
def test_the_tables_disagree_about_where_an_n_gram_goes():
    """Per-table seeds decorrelate, which is what makes a collision in all four
    tables at once a one in fifty billion event rather than a one in slots."""
    import jax.numpy as jnp

    cfg = preset("tiny", imprint_slots=64)
    orders, heads, _ = cfg.imprint_geometry
    tokens = jnp.asarray([list(range(4, 36))], jnp.int32)
    idx = np.asarray(architecture.imprint_indices(tokens, orders, heads,
                                                  cfg.imprint_slots))
    columns = [tuple(idx[0, :, i]) for i in range(idx.shape[-1])]
    assert len(set(columns)) == len(columns)


# --- the modules ------------------------------------------------------------
@pytest.mark.needs_jax
def test_the_norm_starts_as_an_exact_identity_in_scale():
    """The gain is written as (1 + gamma) with gamma at zero, where a gain
    initialised at 1.0 would be a large parameter with no signal to move it."""
    import jax
    import jax.numpy as jnp

    norm = architecture.ZCRMSNorm(dtype=jnp.float32)
    x = jnp.asarray(np.random.default_rng(0).standard_normal((2, 3, 16)), jnp.float32)
    variables = norm.init(jax.random.PRNGKey(0), x)

    assert not np.asarray(variables["params"]["scale"]).any()
    out = np.asarray(norm.apply(variables, x))
    assert np.allclose(np.sqrt(np.mean(out ** 2, axis=-1)), 1.0, atol=1e-3)


@pytest.mark.needs_jax
def test_spin_keeps_the_width_and_enters_quietly():
    """d3 starts at 0.02, not 1.0, so the branch has to earn its way in."""
    import jax
    import jax.numpy as jnp

    cfg = preset("tiny")
    spin = architecture.Spin(cfg.d_model, dtype=jnp.float32)
    x = jnp.asarray(np.random.default_rng(0).standard_normal((1, 4, cfg.d_model)),
                    jnp.float32)
    variables = spin.init(jax.random.PRNGKey(0), x)
    params = variables["params"]

    assert set(params) == {"d1", "d2", "d3"}
    assert all(int(params[name].shape[0]) == cfg.hadamard_n for name in params)
    assert np.allclose(np.asarray(params["d3"]), 0.02)
    assert np.asarray(spin.apply(variables, x)).shape == (1, 4, cfg.d_model)


# --- the whole network ------------------------------------------------------
@pytest.mark.needs_jax
def test_a_real_initialisation_has_exactly_the_counted_parameters(tiny_model):
    """The table and the model, component for component.

    This is the test that makes `param_count` quotable: a figure derived from
    the config and never checked against Flax is a figure about a model nobody
    built.
    """
    cfg, _, params = tiny_model
    assert leaf_total(params) == sum(param_count(cfg).values())


@pytest.mark.needs_jax
def test_the_forward_pass_is_logits_over_the_vocabulary(tiny_model):
    import jax.numpy as jnp

    cfg, model, params = tiny_model
    tokens = jnp.asarray([[4, 5, 6, 7, 8, 9, 10, 11]], jnp.int32)
    logits = model.apply({"params": params}, tokens)

    assert logits.shape == (1, tokens.shape[1], cfg.vocab_size)
    assert np.isfinite(np.asarray(logits)).all()


@pytest.mark.needs_jax
def test_foresight_comes_back_only_when_it_is_asked_for(tiny_model):
    import jax.numpy as jnp

    cfg, model, params = tiny_model
    tokens = jnp.asarray([[4, 5, 6, 7, 8, 9, 10, 11]], jnp.int32)
    logits, foresight = model.apply({"params": params}, tokens, return_foresight=True)

    assert logits.shape == foresight.shape == (1, 8, cfg.vocab_size)
    assert not np.array_equal(np.asarray(logits), np.asarray(foresight))


@pytest.mark.needs_jax
def test_the_gauge_starts_at_exactly_one_half(tiny_model):
    """A head that starts confident is wrong in the most dangerous direction,
    so its output layer is initialised to zero."""
    import jax.numpy as jnp

    _, model, params = tiny_model
    tokens = jnp.asarray([[4, 5, 6, 7, 8, 9, 10, 11]], jnp.int32)
    heads = model.apply({"params": params}, tokens, method="heads")

    assert np.allclose(np.asarray(heads["gauge"]), 0.0)


@pytest.mark.needs_jax
def test_the_dowser_returns_a_unit_vector(tiny_model):
    """It has to live on the same sphere as the tool embeddings it is scored
    against, or a dot product is not a similarity."""
    import jax.numpy as jnp

    cfg, model, params = tiny_model
    tokens = jnp.asarray([[4, 5, 6, 7, 8, 9, 10, 11]], jnp.int32)
    heads = model.apply({"params": params}, tokens, method="heads")

    embedding = np.asarray(heads["dowser"])
    assert embedding.shape == (1, cfg.dowser_dim)
    assert np.allclose(np.linalg.norm(embedding, axis=-1), 1.0, atol=1e-5)
    assert math.isclose(float(np.exp(np.asarray(heads["log_temp"]))), 0.07,
                        rel_tol=1e-3)


@pytest.mark.needs_jax
def test_the_cell_grid_holds_every_layer_and_the_final_state(tiny_model):
    """Small models keep a lot of their signal in the middle, so a probe has to
    be able to read a middle layer as easily as the last one."""
    import jax.numpy as jnp

    cfg, model, params = tiny_model
    tokens = jnp.asarray([[4, 5, 6, 7, 8, 9, 10, 11]], jnp.int32)
    _, cells = model.apply({"params": params}, tokens, method="features")

    assert cells.shape == (1, 8, cfg.num_layers + 1, cfg.d_model)


@pytest.mark.needs_jax
def test_the_layer_axis_is_where_scan_put_it(tiny_model):
    """The quantiser and the exporter both rely on it being the leading one."""
    import jax

    cfg, _, params = tiny_model
    flat = jax.tree_util.tree_flatten_with_path(params)[0]
    stacked = {path[-2].key if len(path) > 1 else "": leaf for path, leaf in flat
               if "layers" in "/".join(str(part) for part in path)}

    assert stacked, "nothing carries a layer axis"
    assert all(leaf.shape[0] == cfg.num_layers for leaf in stacked.values())


# --- the lazy namespace -----------------------------------------------------
def test_the_flax_modules_are_named_but_not_built():
    """They resolve through the module __getattr__, which is what keeps
    importing this file free on a machine with no accelerator."""
    for name in ("ZCRMSNorm", "Spin", "Porthole", "Imprint", "Block", "Trunk",
                 "Dowser", "Gauge", "QuartzNetwork"):
        assert name in architecture.__all__
        assert name in dir(architecture)

    missing = "NotAModule"
    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(architecture, missing)
