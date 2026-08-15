"""The four training stages, and the optimiser all of them share.

Split three ways by cost. The data plumbing -- packing, segment ids, encoding,
the confidence curve, the synthetic curriculum's arithmetic -- is numpy and runs
in the fast suite, because that is where the bugs that quietly ruin a run live:
a loss mask off by one position teaches a transition that never happened, and
nothing about the loss curve will tell you.

Everything that needs JAX is marked `needs_jax`, and everything that takes an
optimiser step is marked `slow` as well. Those initialise a two-layer model on
CPU and run a handful of steps, which is enough to catch a shape error, a
frozen tensor that is not frozen, or a sign that points the wrong way.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pytest

from quartz.model.config import (
    BOS_ID,
    CALL_END,
    CALL_START,
    CQ_GROUP,
    DEFAULT_BITS_MAP,
    EOS_ID,
    IM_END,
    PAD_ID,
    TOOLS_END,
    TOOLS_START,
    preset,
)
from quartz.train import heads, optim, pretrain, qapt, sft

from .conftest import requires_jax


@pytest.fixture(scope="module")
def tiny_trainable():
    """One initialised model for the stages to step on.

    Smaller than the `tiny` preset in two places that cost compile time and
    prove nothing: the vocabulary, and the number of Sinkhorn iterations. Both
    are config fields, which is the point -- the stages read them like any other
    geometry.
    """
    requires_jax()
    cfg = preset("tiny", vocab_size=128, imprint_slots=32, sinkhorn_iters=4)
    model, params = pretrain.init_model(cfg, seed=0)
    return cfg, model, params


@pytest.fixture
def batch(tiny_trainable):
    """Two rows of two packed documents each."""
    cfg, _, _ = tiny_trainable
    rng = np.random.default_rng(0)
    tokens = rng.integers(4, cfg.vocab_size, (2, 16)).astype(np.int32)
    tokens[:, 7] = EOS_ID                       # a document boundary in each row
    tokens[:, -2:] = PAD_ID
    return tokens, pretrain.segments_from_eos(tokens)


def example(query: str = "dim the kitchen") -> dict:
    return {
        "query": query,
        "tools": [{"name": "set_lights",
                   "parameters": {"type": "object",
                                  "properties": {"room": {"type": "string"}},
                                  "required": ["room"]}}],
        "reasoning": "the room is named",
        "answers": [{"name": "set_lights", "arguments": {"room": "kitchen"}}],
    }


# --- stage one: packing -----------------------------------------------------
def test_segment_ids_come_back_out_of_the_token_stream():
    """A packer that wrote only ids still marked its boundaries, because every
    document ends with EOS. Padding gets segment 0 and is invisible to both the
    attention mask and the loss."""
    tokens = np.array([[9, 8, EOS_ID, 7, 6, EOS_ID, PAD_ID, PAD_ID]], np.int32)
    seg = pretrain.segments_from_eos(tokens)

    assert list(seg[0]) == [1, 1, 1, 2, 2, 2, 0, 0]
    assert (seg[tokens == PAD_ID] == 0).all()


def test_packed_batches_draw_rows_out_of_the_corpus():
    rows = np.arange(8 * 16, dtype=np.int32).reshape(8, 16)
    stream = pretrain.packed_batches(rows, 4, 16, seed=0, loop=False)
    tokens, seg = next(stream)

    assert tokens.shape == seg.shape == (4, 16)
    assert len(list(stream)) == 1                 # eight rows, two batches
    assert set(np.unique(tokens)) <= set(np.unique(rows))


def test_a_corpus_that_carries_its_own_segments_is_used_as_written():
    rows = np.stack([np.full((4, 16), 5, np.int32),
                     np.full((4, 16), 3, np.int32)], axis=-1)
    tokens, seg = next(pretrain.packed_batches(rows, 2, 16, loop=False))
    assert (tokens == 5).all()
    assert (seg == 3).all()


def test_a_corpus_smaller_than_a_batch_is_an_error():
    with pytest.raises(ValueError, match="fewer than one batch"):
        next(pretrain.packed_batches(np.zeros((2, 16), np.int32), 4, 16))


def test_a_missing_corpus_says_how_to_build_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="winnow"):
        next(pretrain.packed_batches(tmp_path / "absent.npy", 1, 16))


# --- stage two: what the loss is allowed to see -----------------------------
def test_encode_scores_the_target_and_nothing_else(scribe):
    """The prompt is context. Scoring it would train the model to write the
    schemas back out, which is most of the tokens and none of the task."""
    from quartz.model.scribe import pre_tokenize, render

    row = example()
    prompt, target = render(row)
    p_ids = scribe.encode(pre_tokenize(prompt))
    t_ids = scribe.encode(pre_tokenize(target))

    ids, weights = sft.encode(scribe, row, 256)
    assert ids.shape == weights.shape == (256,)
    assert weights[:1 + len(p_ids)].sum() == 0.0
    # the target, plus the EOS: stopping is a thing we teach
    assert weights.sum() == len(t_ids) + 1
    assert ids[1 + len(p_ids) + len(t_ids)] == EOS_ID
    assert (ids[2 + len(p_ids) + len(t_ids):] == PAD_ID).all()


def test_a_prompt_that_does_not_fit_keeps_its_tail(scribe):
    """The request sits next to the target; the schemas at the front are what a
    short window drops anyway."""
    row = example("dim the kitchen " * 40)
    ids, weights = sft.encode(scribe, row, 128)
    assert ids[0] == 2                               # BOS
    assert weights.sum() > 0
    assert ids.shape == (128,)


def test_a_target_longer_than_the_row_is_refused(scribe):
    with pytest.raises(ValueError, match="raise --cap"):
        sft.encode(scribe, example(), 8)


def test_fit_max_len_buckets_by_powers_of_two(scribe):
    rows = [example(), example("turn the hall lights up")]
    length = sft.fit_max_len(rows, scribe, cap=1024)

    assert length >= sft.MIN_BUCKET
    assert length & (length - 1) == 0
    assert length <= 1024
    assert sft.fit_max_len(rows, scribe, cap=sft.MIN_BUCKET) == sft.MIN_BUCKET


def test_dataset_drops_what_it_cannot_encode(scribe, capsys):
    """A half-written tool call is a wrong label, not a short one."""
    overlong = example()
    overlong["answers"][0]["arguments"]["room"] = "kitchen " * 40
    ids, weights = sft.dataset([example(), overlong], scribe, 64)

    assert ids.shape[0] == 1
    assert "dropped" in capsys.readouterr().out
    assert ids.shape == weights.shape


def test_batches_drop_the_tail_so_every_step_is_one_shape():
    ids = np.arange(10 * 4, dtype=np.int32).reshape(10, 4)
    weights = np.ones_like(ids, np.float32)
    got = list(sft.batches(ids, weights, 4, epochs=2))

    assert len(got) == 4                            # two whole batches an epoch
    assert all(chunk.shape == (4, 4) for chunk, _ in got)
    assert sft.steps_per_epoch(10, 4) == 2
    with pytest.raises(ValueError, match="fewer than one batch"):
        list(sft.batches(ids[:2], weights[:2], 4))


def test_load_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(example()) + "\n\n" + json.dumps(example()) + "\n",
                    encoding="utf-8")
    assert len(list(sft.load_jsonl(str(path)))) == 2


def test_missing_training_data_says_how_to_make_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="quarry"):
        list(sft.load_jsonl(str(tmp_path / "absent.jsonl")))


def test_pad_to_refuses_to_truncate():
    assert list(sft.pad_to([1, 2], 4, PAD_ID, np.int32)) == [1, 2, 0, 0]
    with pytest.raises(ValueError, match="does not fit"):
        sft.pad_to([1, 2, 3], 2, PAD_ID, np.int32)


# --- the optimiser split ----------------------------------------------------
def test_muon_takes_the_matrices_and_nothing_else():
    """Under nn.scan a real matrix is 3-D. A bias, a Spin diagonal and a norm
    gain are not linear maps, and orthogonalising a lookup table would force
    every row to the same length."""
    stacked = np.zeros((27, 512, 512), np.float32)
    assert optim.matrix_leaf(("params", "layers", "q_proj", "kernel"), stacked)
    assert not optim.matrix_leaf(("params", "embedding", "embedding"), stacked)
    assert not optim.matrix_leaf(("params", "imprint", "table"), stacked)
    assert not optim.matrix_leaf(("params", "layers", "spin", "d1"),
                                 np.zeros((27, 512), np.float32))
    assert not optim.matrix_leaf(("params", "attn_gate"), np.zeros((27,), np.float32))


def test_path_str_reads_a_tree_path_without_importing_jax():
    assert optim.path_str(("params", "layers", "q_proj")) == "params.layers.q_proj"


def test_arguments_arrive_from_a_namespace_a_dict_or_nothing():
    """The CLI passes a Namespace, a test passes the dataclass, a notebook
    passes a few keywords."""
    from_cli = optim.as_args(optim.OptimArgs,
                             argparse.Namespace(lr=1e-4, nonsense=True))
    assert from_cli.lr == 1e-4
    assert not hasattr(from_cli, "nonsense")

    # an unset CLI flag is None, and must not clobber a default
    assert optim.as_args(optim.OptimArgs, None, lr=None).lr == optim.OptimArgs().lr
    assert optim.as_args(optim.OptimArgs, {"clip": 0.5}).clip == 0.5


def test_the_geometry_of_a_run_comes_from_a_yaml_or_a_preset():
    cfg = optim.config_from_args(argparse.Namespace(config="", preset_name="tiny"))
    assert cfg.num_layers == preset("tiny").num_layers


def test_a_checkpoint_is_read_in_whatever_shape_it_arrives():
    params = {"w": np.zeros(4)}
    assert optim.unpack_checkpoint(params)[0] is params
    assert optim.unpack_checkpoint((params, None))[0] is params
    got, cfg = optim.unpack_checkpoint({"params": params,
                                        "config": {"d_model": 256, "num_heads": 8,
                                                   "num_kv_heads": 4}})
    assert got is params
    assert cfg.d_model == 256
    with pytest.raises(ValueError, match="no parameters"):
        optim.unpack_checkpoint({"params": None})


# --- stage three: which leaf gets which width -------------------------------
def test_quant_axis_follows_what_the_tensor_is_read_as():
    assert qapt.quant_axis("layers.q_proj.kernel", 3) == -2
    assert qapt.quant_axis("embedding.embedding", 2) == -1
    assert qapt.quant_axis("imprint.table", 3) == -1
    assert qapt.quant_axis("attn_gate", 1) == -1


def test_the_noise_scale_is_measured_not_assumed():
    """The distortion of a group depends on how Gaussian it is after rotation,
    and that differs between an attention projection and a router."""
    rng = np.random.default_rng(0)
    w = rng.standard_normal((CQ_GROUP, 64)).astype(np.float32)

    sigma_two = qapt.cq_noise_scale(w, 2, CQ_GROUP, -2)
    sigma_four = qapt.cq_noise_scale(w, 4, CQ_GROUP, -2)

    assert 0.0 < sigma_four < sigma_two < 1.0
    from quartz.model.grist import cq_quantize

    want = float(np.sqrt(np.mean((cq_quantize(w, 2, CQ_GROUP, 0) - w) ** 2)))
    assert abs(sigma_two - want) < 1e-6


# --- stage four: the confidence curve ---------------------------------------
def test_perplexity_becomes_a_confidence_in_zero_to_one():
    """mu is the 80th percentile, so about a fifth of examples land below 0.5."""
    ppl = np.array([1.5, 2.0, 4.0, 8.0, 40.0])
    conf, mu = heads.perplexity_to_confidence(ppl, k=5.0)

    assert mu == float(np.percentile(ppl, 80))
    assert ((conf > 0.0) & (conf < 1.0)).all()
    assert list(conf) == sorted(conf, reverse=True)      # lower perplexity, higher
    assert abs(float(heads.perplexity_to_confidence(np.array([mu]), mu=mu)[0][0])
               - 0.5) < 1e-6


def test_k_is_how_sharp_the_switch_is():
    ppl = np.array([2.0, 20.0])
    ramp, _ = heads.perplexity_to_confidence(ppl, mu=6.0, k=2.0)
    step, _ = heads.perplexity_to_confidence(ppl, mu=6.0, k=12.0)
    assert step[0] > ramp[0]
    assert step[1] < ramp[1]


def test_a_temperature_pulls_everything_towards_one_half():
    conf = np.array([0.9, 0.1])
    tempered = heads.apply_temperature(conf, 2.0)
    assert tempered[0] < conf[0]
    assert tempered[1] > conf[1]


# --- stage zero: the synthetic curriculum -----------------------------------
def test_a_synthetic_turn_is_exactly_as_long_as_the_arithmetic_says():
    rng = np.random.default_rng(0)
    n_tools, n_args, name_len = 3, 2, heads.NAME_LEN
    row_len = heads.stage0_row_length(n_tools, n_args, name_len)

    ids, weights = heads.synthetic_batch(rng, 4, n_tools, (n_args, n_args),
                                         vocab=128, max_len=row_len + 8)
    assert ids.shape == weights.shape == (4, row_len + 8)
    assert (ids[:, row_len:] == PAD_ID).all()
    assert ids[0, 1] == heads.marker_id(TOOLS_START)
    # the call, minus the marker that opens it
    assert weights.sum(axis=1).tolist() == [name_len + 2 * n_args + 2] * 4


def test_the_synthetic_names_are_never_control_ids():
    """Names and values are resampled every call, so nothing about a particular
    identifier can generalise. What generalises is the copying."""
    rng = np.random.default_rng(0)
    ids, _ = heads.synthetic_batch(rng, 8, 4, (1, 3), vocab=64, max_len=96)
    body = ids[ids != PAD_ID]
    structure = {BOS_ID, EOS_ID, *(heads.marker_id(marker) for marker in
                                   (TOOLS_START, TOOLS_END, IM_END, CALL_START,
                                    CALL_END))}

    reserved = body[body < heads.first_content_id()]
    assert set(reserved.tolist()) <= structure
    assert (body >= heads.first_content_id()).sum() > 0


def test_the_catalogue_shrinks_until_the_turn_fits():
    long_row = heads.stage0_row_length(41, 5)
    rng = np.random.default_rng(0)
    ids, _ = heads.synthetic_batch(rng, 2, 41, (5, 5), vocab=64, max_len=long_row // 4)
    assert (ids != PAD_ID).sum(axis=1).max() <= long_row // 4


def test_pairs_and_recall_read_the_stage_two_data():
    rows = [example(), {"query": "no call", "tools": [], "answers": []}]
    pairs = heads.pairs_from_examples(rows)
    assert len(pairs) == 1                          # a refusal has no positive
    assert pairs[0][0] == "dim the kitchen"
    assert '"name":"set_lights"' in pairs[0][1]

    emb = np.eye(3, dtype=np.float32)
    assert heads.recall_at_k(emb, emb, np.arange(3), k=1) == 1.0
    assert heads.recall_at_k(emb, np.roll(emb, 1, axis=0), np.arange(3), k=1) == 0.0
    assert heads.recall_at_k(emb, np.roll(emb, 1, axis=0), np.arange(3), k=3) == 1.0


# --- the pieces that need JAX ----------------------------------------------
@pytest.mark.needs_jax
def test_newton_schulz_squeezes_the_singular_values():
    """Not exactly orthogonal, and it does not need to be. What matters is that
    no direction of the update dominates the rest."""
    import jax.numpy as jnp

    rng = np.random.default_rng(0)
    u, _, vt = np.linalg.svd(rng.standard_normal((8, 16)), full_matrices=False)
    spread = np.diag([10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.01])
    g = (u @ spread @ vt).astype(np.float32)

    out = np.asarray(optim.newton_schulz(jnp.asarray(g), optim.NS_STEPS))
    before = np.linalg.svd(g, compute_uv=False)
    after = np.linalg.svd(out, compute_uv=False)

    assert out.shape == g.shape
    assert before.max() / before.min() > 100.0
    assert after.max() / after.min() < 5.0
    assert after.max() / after.min() < before.max() / before.min() / 100.0
    assert after.min() > 0.3
    assert after.max() < 1.5


@pytest.mark.needs_jax
def test_newton_schulz_wants_a_matrix():
    import jax.numpy as jnp

    with pytest.raises(ValueError, match="wants a matrix"):
        optim.newton_schulz(jnp.zeros((8,)))


@pytest.mark.needs_jax
def test_orthogonalise_treats_every_layer_on_its_own():
    """Under nn.scan the layer axis leads, and the twenty seven layers share
    nothing, so orthogonalising them as one tall block would mix them."""
    import jax.numpy as jnp

    rng = np.random.default_rng(0)
    g = jnp.asarray(rng.standard_normal((3, 8, 8)), jnp.float32)
    out = np.asarray(optim.orthogonalise(g, optim.NS_STEPS))

    assert out.shape == (3, 8, 8)
    for layer in range(3):
        one = np.asarray(optim.newton_schulz(g[layer], optim.NS_STEPS))
        assert np.allclose(out[layer], one, atol=1e-4)


@pytest.mark.needs_jax
@pytest.mark.slow
def test_both_arms_descend():
    """The trap the chain is written around.

    Muon emits a direction with no learning rate and no sign, so it needs its
    own negative schedule; AdamW has already folded -lr in, so scaling it again
    would flip it. Get either wrong and half the model ascends the loss.
    """
    import jax.numpy as jnp
    import optax

    params = {"kernel": jnp.full((2, 8, 8), 0.1), "gain": jnp.full((8,), 0.1)}
    args = optim.OptimArgs(lr=0.05, total_steps=8, warmup=1, weight_decay=0.0)
    tx = optim.build_optimiser(args, params)
    state = tx.init(params)

    current = params
    for _ in range(3):
        grads = current                    # the gradient of 0.5 * sum(p ** 2)
        updates, state = tx.update(grads, state, current)
        current = optax.apply_updates(current, updates)

    for name in params:
        assert float(jnp.linalg.norm(current[name])) < float(
            jnp.linalg.norm(params[name])), name


@pytest.mark.needs_jax
def test_the_straight_through_estimator_quantises_forwards_and_not_backwards():
    """The derivative of a rounding is zero almost everywhere, so the honest
    gradient trains nothing at all."""
    import jax
    import jax.numpy as jnp

    from quartz.model.grist import cq_quantize

    rng = np.random.default_rng(0)
    w = jnp.asarray(rng.standard_normal((4, CQ_GROUP)), jnp.float32)

    forward = np.asarray(qapt.cq_ste(w, 2, CQ_GROUP, -1))
    assert np.allclose(forward, cq_quantize(np.asarray(w), 2, CQ_GROUP, -1), atol=1e-4)
    assert not np.allclose(forward, np.asarray(w))

    grad = jax.grad(lambda x: jnp.sum(qapt.cq_ste(x, 2, CQ_GROUP, -1)))(w)
    assert np.allclose(np.asarray(grad), 1.0)


@pytest.mark.needs_jax
def test_the_noise_phase_is_isotropic_and_reproducible():
    import jax

    rng = np.random.default_rng(0)
    w = jax.numpy.asarray(rng.standard_normal((64, 64)), jax.numpy.float32)
    key = jax.random.PRNGKey(0)

    noisy = np.asarray(qapt.add_cq_noise(w, 0.05, key))
    again = np.asarray(qapt.add_cq_noise(w, 0.05, key))

    assert np.array_equal(noisy, again)
    assert abs(float(np.std(noisy - np.asarray(w))) - 0.05) < 0.005


@pytest.mark.needs_jax
def test_the_plan_says_what_happens_to_every_leaf(tiny_params):
    plan = qapt.quant_plan(tiny_params, DEFAULT_BITS_MAP)
    widths = {entry.bits for entry in plan.values()}

    assert widths <= {2, 3, 4}
    assert 2.0 < qapt.mean_bits(tiny_params, plan) < 4.0
    assert not [name for name in plan if "dowser" in name or "gauge" in name]
    assert any("phi" in name for name in plan)
    with pytest.raises(ValueError, match="no leaf matched"):
        qapt.quant_plan({"scalar": np.zeros((), np.float32)}, DEFAULT_BITS_MAP)


@pytest.mark.needs_jax
def test_fake_quant_puts_the_codebook_in_the_forward_path(tiny_params):
    import jax

    plan = qapt.measure_distortion(tiny_params, qapt.quant_plan(tiny_params))
    assert all(entry.sigma > 0 for entry in plan.values())

    quantised = qapt.fake_quant(tiny_params, plan, mode="ste")
    noised = qapt.fake_quant(tiny_params, plan, mode="noise",
                             key=jax.random.PRNGKey(0))
    name = next(iter(plan))

    for tree in (quantised, noised):
        leaf = jax.tree_util.tree_flatten_with_path(tree)[0]
        assert len(leaf) == len(jax.tree_util.tree_flatten_with_path(tiny_params)[0])
    assert not np.allclose(np.asarray(jax.tree_util.tree_leaves(quantised)[0]),
                           np.asarray(jax.tree_util.tree_leaves(tiny_params)[0]))
    assert name in plan
    with pytest.raises(ValueError, match="'ste' or 'noise'"):
        qapt.fake_quant(tiny_params, plan, mode="magic")
    with pytest.raises(ValueError, match="needs a PRNG key"):
        qapt.fake_quant(tiny_params, plan, mode="noise")


@pytest.mark.needs_jax
def test_the_contrastive_loss_scores_both_directions():
    """A query has to find its tool, and a tool has to find its query."""
    import jax.numpy as jnp

    q = jnp.eye(4, dtype=jnp.float32)
    log_temp = jnp.asarray(np.log(0.07), jnp.float32)

    matched = float(heads.contrastive_loss(q, q, log_temp))
    shuffled = float(heads.contrastive_loss(q, q[::-1], log_temp))

    assert matched < shuffled
    assert abs(matched - float(heads.contrastive_loss(q, q, log_temp))) < 1e-6
    assert abs(float(heads.contrastive_loss(q, q[::-1], log_temp))
               - float(heads.contrastive_loss(q[::-1], q, log_temp))) < 1e-5


@pytest.mark.needs_jax
def test_freezing_is_a_gradient_mask(tiny_params):
    """One apply, one tree, and provably the same thing as splitting it -- as
    long as the optimiser has no decoupled weight decay."""
    import jax

    grads = jax.tree_util.tree_map(np.ones_like, tiny_params)
    frozen = heads.freeze_except(grads, "gauge")
    flat = {optim.path_str(path): leaf
            for path, leaf in jax.tree_util.tree_flatten_with_path(frozen)[0]}

    assert any(np.asarray(leaf).any() for name, leaf in flat.items() if "gauge" in name)
    assert not any(np.asarray(leaf).any() for name, leaf in flat.items()
                   if "gauge" not in name)
    assert 0 < heads.trainable_count(tiny_params, "gauge") < sum(
        int(leaf.size) for leaf in jax.tree_util.tree_leaves(tiny_params))


# --- the stages that actually step -----------------------------------------
@pytest.mark.needs_jax
@pytest.mark.slow
def test_the_pretraining_loss_drops_the_boundary_token(tiny_trainable, batch):
    """One token per packed document: the boundary, where the model would
    otherwise be taught a transition that packing was meant to hide."""
    import jax.numpy as jnp

    _, model, params = tiny_trainable
    tokens, seg = batch

    total, ce = pretrain.losses(model.apply, params, jnp.asarray(tokens),
                                jnp.asarray(seg))
    assert float(ce) > 0.0
    assert float(total) > float(ce)             # the z-loss and Foresight ride on top

    # every position masked out: nothing is scored, and the loss is exactly zero
    blank = jnp.zeros_like(jnp.asarray(seg))
    _, empty_ce = pretrain.losses(model.apply, params, jnp.asarray(tokens), blank)
    assert float(empty_ce) == 0.0


@pytest.mark.needs_jax
@pytest.mark.slow
def test_a_pretraining_step_warms_up_and_then_moves_the_weights(tiny_trainable, batch):
    """The schedule starts at exactly zero, so the first step is a no-op.

    Worth pinning: a run that looked frozen for its first few hundred steps
    would otherwise send someone hunting for a bug in the gradient.
    """
    import jax
    import jax.numpy as jnp

    _, model, params = tiny_trainable
    tokens, seg = jnp.asarray(batch[0]), jnp.asarray(batch[1])
    args = pretrain.PretrainArgs(lr=1e-3, total_steps=4, warmup=1, preset_name="tiny")
    state = pretrain.make_state(model, params, args)

    def moved(one, other) -> list[bool]:
        return [not np.array_equal(np.asarray(a), np.asarray(b))
                for a, b in zip(jax.tree_util.tree_leaves(one),
                                jax.tree_util.tree_leaves(other), strict=True)]

    state, metrics = pretrain.train_step(state, tokens, seg)
    assert np.isfinite(float(metrics["loss"]))
    assert float(metrics["ce"]) > 0.0
    assert not any(moved(params, state.params))          # warmup step, lr is zero

    state, _ = pretrain.train_step(state, tokens, seg)
    assert any(moved(params, state.params))


@pytest.mark.needs_jax
@pytest.mark.slow
def test_the_supervised_loss_reads_the_weight_beside_the_target(tiny_trainable):
    """`weights[:, 1:]` is the shift: a weight sits on the token being predicted,
    so it lines up with the targets and not with the inputs."""
    import jax.numpy as jnp

    cfg, model, params = tiny_trainable
    rng = np.random.default_rng(0)
    ids = jnp.asarray(rng.integers(4, cfg.vocab_size, (2, 16)), jnp.int32)

    scored = jnp.asarray(np.tile([0.0] * 8 + [1.0] * 8, (2, 1)), jnp.float32)
    total, ce = sft.masked_losses(model.apply, params, ids, scored)
    assert float(ce) > 0.0
    assert float(total) >= float(ce)

    _, empty = sft.masked_losses(model.apply, params, ids, jnp.zeros_like(scored))
    assert float(empty) == 0.0


@pytest.mark.needs_jax
@pytest.mark.slow
def test_stage_zero_trains_adapters_and_leaves_the_table_where_it_was(
        tiny_trainable):
    """The embedding is scrambled under the adapters and restored at the end,
    so nothing about token identity can be memorised and only the pattern
    survives into the real geometry."""
    from flax.traverse_util import flatten_dict

    cfg, _, params = tiny_trainable
    out, stats = heads.stage0_curriculum(
        params, cfg, phases=[(2, 2, (1, 1), None)], rank=2, batch=2, lr=1e-3)

    before, after = flatten_dict(params), flatten_dict(out)
    assert set(before) == set(after)
    assert stats["rank"] == 2
    assert stats["phases"][0]["steps"] == 2

    epath = heads.embedding_path(params)
    assert np.array_equal(np.asarray(before[epath]), np.asarray(after[epath]))

    adapted = [path for path in before
               if not np.array_equal(np.asarray(before[path]), np.asarray(after[path]))]
    assert adapted, "the adapters merged into nothing"
    assert all(path[-1] == "kernel" for path in adapted)
