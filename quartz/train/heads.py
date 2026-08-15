"""Stage four: the two heads, and the curriculum that comes before them.

Gauge answers "should this device act at all, or hand the request to something
bigger". Its label is not correctness, which is sparse and says nothing about
*how* wrong an answer was, but the model's own perplexity on the call it wrote,
squashed through a sigmoid. That is a proxy, and the sigmoid was chosen rather
than fitted, so the head inherits whatever miscalibration mu and k introduce --
which is why the temperature fit at the end of this file exists and why it is
measured against outcomes rather than against perplexity.

Dowser is retrieval: query and tool description into one 128-dimensional space,
trained with symmetric InfoNCE so a query finds its tool and a tool finds its
query.

Stage zero comes first. Both heads need the trunk to already know that copying
is a thing, and that pattern -- an induction head -- forms far faster on random
symbols than on English, so 2,500 steps of pure nonsense buy it before any real
data is spent on it.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from quartz.model.config import (
    BOS_ID,
    CALL_END,
    CALL_START,
    CHAT_MARKERS,
    EOS_ID,
    FIRST_MARKER_ID,
    IM_END,
    PAD_ID,
    TOOLS_END,
    TOOLS_START,
    QuartzConfig,
)
from quartz.train.optim import as_args, config_from_args, path_str, write_checkpoint
from quartz.train.sft import attention_mask, fit_max_len, load_jsonl, pad_to

#: The heads are reached as named methods on QuartzNetwork. Named here so a
#: rename in the architecture is one edit rather than six.
GAUGE_METHOD = "gauge"
DOWSER_METHOD = "dowser"

#: (steps, tools declared, arguments per tool, embedding noise sigma).
#: sigma None means "resample the embedding table every step", so nothing about
#: token identity can be memorised and only the pattern survives.
STAGE0 = [
    (500, 2, (1, 1), None),
    (500, 5, (2, 3), None),
    (1000, 10, (2, 5), (1.0, 0.3)),
    (500, 41, (2, 5), (0.1, 0.1)),
]

#: Tokens per synthetic tool name. Three is enough that a name cannot be
#: recognised from one piece, which is the whole point of the exercise.
NAME_LEN = 3


@dataclass
class HeadsArgs:
    """Stage four's knobs."""

    config: str = ""
    preset_name: str = "base"
    data: str = ""                      # the stage two jsonl, reused
    init: str = "ckpt/quartz_qapt.pkl"
    out: str = "ckpt/quartz_heads.pkl"
    tokenizer: str = ""
    stage0: int = 2500                  # 0 skips the curriculum
    gauge: bool = True
    dowser: bool = True
    epochs: int = 1
    batch_size: int = 128
    dowser_batch: int = 64
    cap: int = 1024
    lr: float = 1e-3
    seed: int = 0
    rank: int = 16
    alpha: float = 32.0
    k: float = 5.0                      # the sigmoid's sharpness
    holdout: float = 0.1


# --- small numpy pieces ----------------------------------------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def perplexity_to_confidence(ppl, mu: float | None = None,
                             k: float = 5.0) -> tuple[np.ndarray, float]:
    """Squash perplexity into a confidence in (0, 1).

    mu is the 80th percentile of observed perplexity, so about a fifth of
    examples land below 0.5, and k sets how sharp the switch is: a ramp at 2, a
    near step at 12. Five ships. Working in log perplexity is what makes the
    curve symmetric about mu, since perplexity is a ratio scale.
    """
    ppl = np.asarray(ppl, dtype=np.float64)
    mu = float(np.percentile(ppl, 80)) if mu is None else float(mu)
    log_ppl = np.log(np.maximum(ppl, 1.0))
    conf = 1.0 / (1.0 + np.exp(k * (log_ppl - np.log(max(mu, 1.0 + 1e-9)))))
    return conf.astype(np.float32), mu


def fit_temperature(conf, correct, lo: float = 0.05, hi: float = 20.0,
                    points: int = 400) -> float:
    """One scalar that rescales the head's logit, fitted by minimum NLL.

    A grid rather than a solver: the objective is one-dimensional and smooth,
    the grid is deterministic, and it costs no dependency.
    """
    z = _logit(np.asarray(conf, dtype=np.float64))
    y = np.asarray(correct, dtype=np.float64)
    grid = np.exp(np.linspace(math.log(lo), math.log(hi), points))
    p = np.clip(_sigmoid(z[None, :] / grid[:, None]), 1e-9, 1.0 - 1e-9)
    nll = -np.mean(y[None, :] * np.log(p) + (1 - y[None, :]) * np.log(1 - p), axis=1)
    return float(grid[int(np.argmin(nll))])


def apply_temperature(conf, temperature: float) -> np.ndarray:
    """The calibrated confidence. T > 1 pulls everything towards 0.5."""
    return _sigmoid(_logit(np.asarray(conf, dtype=np.float64)) / temperature)


def calibration_report(conf, correct, temperature: float | None = None) -> dict:
    """Expected and maximum calibration error, before and after the fit.

    Binned by *predicted confidence*, not by perplexity: binning by the
    training signal would only show that the head tracks its own label.

    Both errors come from :mod:`quartz.errand.score`, which owns the binning.
    A second copy of the rule here would be free to disagree about the edges --
    and it did: a head that returns exactly 0.0 falls outside every half-open
    bin, so a local `(lo, hi]` walk silently dropped it from the average while
    the suite counted it. One implementation, one set of numbers.
    """
    from quartz.errand.score import (
        expected_calibration_error,
        maximum_calibration_error,
    )

    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    temperature = fit_temperature(conf, correct) if temperature is None else temperature
    tempered = apply_temperature(conf, temperature)
    return {
        "temperature": float(temperature),
        "ece": float(expected_calibration_error(conf, correct)),
        "mce": float(maximum_calibration_error(conf, correct)),
        "ece_tempered": float(expected_calibration_error(tempered, correct)),
        "mce_tempered": float(maximum_calibration_error(tempered, correct)),
    }


# --- shared training plumbing ---------------------------------------------
def head_apply(cfg: QuartzConfig, apply_fn=None):
    """The model's apply function, built from the config if none was given."""
    if apply_fn is not None:
        return apply_fn
    from quartz.model.architecture import QuartzNetwork

    return QuartzNetwork(cfg).apply


def freeze_except(grads, keep: str):
    """Zero every gradient outside `keep`, by name.

    The heads read pooled cells out of the trunk, so splitting the parameter
    tree in two would mean threading two trees through one apply. Masking is
    one line and provably the same thing -- as long as the optimiser has no
    decoupled weight decay, which would shrink the frozen trunk even on a zero
    gradient. That is why the head fits below use adam and not adamw.
    """
    import jax
    import jax.numpy as jnp

    return jax.tree_util.tree_map_with_path(
        lambda path, g: g if keep in path_str(path).lower() else jnp.zeros_like(g),
        grads)


def trainable_count(params, keep: str) -> int:
    """How many parameters `freeze_except(keep)` will actually let move.

    Printed beside the total at the start of a head fit, because "8,193 of
    45,211,383" is the claim that the trunk is frozen, and it is worth stating
    from the tree rather than from the design.
    """
    import jax

    leaves, _ = jax.tree_util.tree_flatten_with_path(params)
    return sum(int(leaf.size) for path, leaf in leaves
               if keep in path_str(path).lower())


def _head_state(apply_fn, params, lr: float, clip: float = 1.0):
    import optax
    from flax.training import train_state

    tx = optax.chain(optax.clip_by_global_norm(clip), optax.adam(lr))
    return train_state.TrainState.create(apply_fn=apply_fn, params=params, tx=tx)


def encode_text(sp, text: str, max_len: int) -> np.ndarray:
    """One string into one padded row. Used for both sides of Dowser."""
    from quartz.model.scribe import pre_tokenize

    ids = list(sp.encode(pre_tokenize(text)))[:max_len - 2]
    return pad_to([BOS_ID, *ids, EOS_ID], max_len, PAD_ID, np.int32)


# --- Gauge -----------------------------------------------------------------
def measure_perplexity(apply_fn, params, ids: np.ndarray, weights: np.ndarray,
                       batch_size: int = 32) -> np.ndarray:
    """Per-example perplexity over the scored span. Forward only.

    This is pass one of two: the label for pass two is what the model was
    already going to be surprised by, so it costs one extra forward over the
    training set and no gradients at all.
    """
    import jax
    import jax.numpy as jnp
    import optax

    def row_ce(p, row, weight):
        inputs, targets = row[:, :-1], row[:, 1:]
        w = weight[:, 1:].astype(jnp.float32)
        logits = apply_fn({"params": p}, inputs, mask=attention_mask(inputs))
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return jnp.sum(ce * w, axis=-1) / jnp.maximum(jnp.sum(w, axis=-1), 1.0)

    fn = jax.jit(row_ce)
    out = []
    for i in range(0, ids.shape[0], batch_size):
        chunk, weight = ids[i:i + batch_size], weights[i:i + batch_size]
        out.append(np.asarray(fn(params, jnp.asarray(chunk), jnp.asarray(weight))))
    return np.exp(np.concatenate(out)).astype(np.float64)


def train_gauge(params, data, epochs: int = 1, lr: float = 1e-3, batch: int = 128, *,
                cfg: QuartzConfig | None = None, sp=None, apply_fn=None,
                max_len: int | None = None, mu: float | None = None, k: float = 5.0,
                seed: int = 0):
    """Fit the confidence head. Returns (params, stats).

    Pass one measures perplexity, pass two regresses the head onto it. The
    trunk is frozen by MASKING GRADIENTS rather than by splitting the parameter
    tree, so 8,193 of 45 million parameters move and the rest are held exactly.
    """
    import jax
    import jax.numpy as jnp

    from quartz.model.scribe import get_tokenizer
    from quartz.train.sft import dataset

    cfg = cfg or QuartzConfig()
    apply_fn = head_apply(cfg, apply_fn)
    if isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], np.ndarray):
        ids, weights = data
    else:
        sp = sp or get_tokenizer()
        rows = list(load_jsonl(data)) if isinstance(data, str) else list(data)
        max_len = max_len or fit_max_len(rows, sp, cfg.max_seq_len)
        ids, weights = dataset(rows, sp, max_len)

    ppl = measure_perplexity(apply_fn, params, ids, weights)
    targets, mu = perplexity_to_confidence(ppl, mu, k)
    print(f"  ppl       p50 {np.percentile(ppl, 50):.2f}  "
          f"p80 {np.percentile(ppl, 80):.2f}  p99 {np.percentile(ppl, 99):.2f}")
    print(f"  head      {trainable_count(params, GAUGE_METHOD):,} params trainable "
          f"of {sum(int(x.size) for x in jax.tree_util.tree_leaves(params)):,}")

    def loss_fn(p, row, target):
        logit = apply_fn({"params": p}, row, mask=attention_mask(row),
                         method=GAUGE_METHOD)
        pred = jax.nn.sigmoid(jnp.asarray(logit, jnp.float32).reshape(-1))
        return jnp.mean((pred - target) ** 2), pred

    @jax.jit
    def step(state, row, target):
        (mse, pred), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, row, target)
        grads = freeze_except(grads, GAUGE_METHOD)
        return state.apply_gradients(grads=grads), mse, pred

    state = _head_state(apply_fn, params, lr)
    rng = np.random.default_rng(seed)
    batch = max(1, min(batch, ids.shape[0]))
    stats: dict[str, Any] = {"mu": float(mu), "k": float(k), "epochs": []}
    for epoch in range(1, epochs + 1):
        order = rng.permutation(ids.shape[0])
        preds, se, ae, n = [], 0.0, 0.0, 0
        for i in range(0, ids.shape[0] - batch + 1, batch):
            take = order[i:i + batch]
            state, mse, pred = step(state, jnp.asarray(ids[take]),
                                    jnp.asarray(targets[take]))
            pred = np.asarray(pred)
            preds.append((take, pred))
            se += float(mse)
            ae += float(np.mean(np.abs(pred - targets[take])))
            n += 1
        flat_idx = np.concatenate([t for t, _ in preds])
        flat_pred = np.concatenate([p for _, p in preds])
        seen = targets[flat_idx]
        # A constant column has no correlation to report, and corrcoef would
        # divide by its zero standard deviation to say so.
        pearson = (float(np.corrcoef(flat_pred, seen)[0, 1])
                   if flat_pred.std() > 0 and seen.std() > 0 else math.nan)
        print(f"  epoch     {epoch}/{epochs}  mse {se / max(n, 1):.4f}  "
              f"mae {ae / max(n, 1):.4f}  pearson {pearson:.3f}")
        stats["epochs"].append({"epoch": epoch, "mse": se / max(n, 1),
                                "mae": ae / max(n, 1), "pearson": pearson})
    return state.params, stats


# --- Dowser ----------------------------------------------------------------
def contrastive_loss(q_emb, t_emb, log_temp):
    """Symmetric InfoNCE.

    The temperature is clipped to [0.01, 100] because otherwise it runs away
    towards zero: a sharper distribution always lowers the training loss, and
    nothing in the objective pushes back. Both directions are scored, so a
    query finds its tool and a tool finds its query.
    """
    import jax.numpy as jnp
    import optax

    temp = jnp.exp(jnp.clip(log_temp, -jnp.log(100.0), jnp.log(100.0)))
    logits = q_emb @ t_emb.T / temp
    labels = jnp.arange(logits.shape[0])
    xe = optax.softmax_cross_entropy_with_integer_labels
    return (jnp.mean(xe(logits, labels)) + jnp.mean(xe(logits.T, labels))) / 2.0


def pairs_from_examples(rows) -> list[tuple[str, str]]:
    """(query, tool schema) pairs out of the stage two data.

    The positive is the schema of the tool the teacher actually called, dumped
    the same way the prompt dumps it, so the tool side of the index sees the
    text it will see at serving time. Refusals have no positive and are skipped.
    """
    pairs: list[tuple[str, str]] = []
    for row in rows:
        answers = row.get("answers") or []
        if not answers or not isinstance(answers[0], dict):
            continue
        tools = {t.get("name"): t for t in row.get("tools", []) if isinstance(t, dict)}
        tool = tools.get(answers[0].get("name"))
        if tool is None:
            continue
        pairs.append((row.get("query", ""), json.dumps(tool, separators=(",", ":"))))
    return pairs


def recall_at_k(q_emb: np.ndarray, t_emb: np.ndarray, labels: np.ndarray,
                k: int = 5) -> float:
    """Share of queries whose own tool is in the top k of the catalogue."""
    scores = np.asarray(q_emb) @ np.asarray(t_emb).T
    k = min(k, scores.shape[1])
    top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    return float(np.mean([labels[i] in top[i] for i in range(len(labels))]))


def _unique_batches(rng, tool_ids: np.ndarray, batch: int):
    """Batches whose tools are all different.

    A repeated tool inside a batch is a false negative: InfoNCE would push a
    query away from a tool that is also a correct answer for it.
    """
    order = rng.permutation(len(tool_ids))
    rows: list[int] = []
    seen: set[int] = set()
    for i in order:
        if int(tool_ids[i]) in seen:
            continue
        seen.add(int(tool_ids[i]))
        rows.append(int(i))
        if len(rows) == batch:
            yield np.asarray(rows)
            rows, seen = [], set()


def train_dowser(params, pairs, epochs: int = 1, lr: float = 1e-3, batch: int = 64, *,
                 cfg: QuartzConfig | None = None, sp=None, apply_fn=None,
                 max_len: int | None = None, holdout: float = 0.1, seed: int = 0):
    """Fit the retrieval head on (query, tool) pairs. Returns (params, stats).

    The trunk is frozen here too. The contrastive objective has no language
    modelling term beside it, so an unfrozen trunk would happily trade the
    decoder away for a slightly better recall number.
    """
    import jax
    import jax.numpy as jnp

    from quartz.model.scribe import get_tokenizer

    cfg = cfg or QuartzConfig()
    apply_fn = head_apply(cfg, apply_fn)
    sp = sp or get_tokenizer()
    pairs = list(pairs)
    if not pairs:
        raise ValueError("no (query, tool) pairs to train Dowser on")
    max_len = max_len or min(cfg.max_seq_len, 256)

    texts: list[str] = []
    index: dict[str, int] = {}
    tool_ids = []
    for _, tool in pairs:
        if tool not in index:
            index[tool] = len(texts)
            texts.append(tool)
        tool_ids.append(index[tool])
    tool_ids = np.asarray(tool_ids)
    q_rows = np.stack([encode_text(sp, q, max_len) for q, _ in pairs])
    t_rows = np.stack([encode_text(sp, t, max_len) for t in texts])
    # In-batch negatives are the only negatives, and every one of them has to be
    # a different tool, so the batch can never exceed the catalogue.
    batch = max(2, min(batch, len(texts)))

    n_val = int(len(pairs) * holdout)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(pairs))
    val, train_rows = shuffled[:n_val], shuffled[n_val:]

    def embed(p, rows):
        emb, log_temp = apply_fn({"params": p}, rows, mask=attention_mask(rows),
                                 method=DOWSER_METHOD)
        return emb, log_temp

    def loss_fn(p, queries, tools):
        q_emb, log_temp = embed(p, queries)
        t_emb, _ = embed(p, tools)
        return contrastive_loss(q_emb, t_emb, log_temp)

    @jax.jit
    def step(state, queries, tools):
        loss, grads = jax.value_and_grad(loss_fn)(state.params, queries, tools)
        return state.apply_gradients(grads=freeze_except(grads, DOWSER_METHOD)), loss

    embed_fn = jax.jit(lambda p, rows: embed(p, rows)[0])

    def embed_all(p, rows):
        out = [np.asarray(embed_fn(p, jnp.asarray(rows[i:i + batch])))
               for i in range(0, rows.shape[0], batch)]
        return np.concatenate(out)

    state = _head_state(apply_fn, params, lr)
    stats: dict[str, Any] = {"pairs": len(pairs), "tools": len(texts), "epochs": []}
    for epoch in range(1, epochs + 1):
        total, n = 0.0, 0
        for rows in _unique_batches(rng, tool_ids[train_rows], batch):
            take = train_rows[rows]
            state, loss = step(state, jnp.asarray(q_rows[take]),
                               jnp.asarray(t_rows[tool_ids[take]]))
            total += float(loss)
            n += 1
        record = {"epoch": epoch, "loss": total / max(n, 1)}
        if n_val:
            record["recall@5"] = recall_at_k(embed_all(state.params, q_rows[val]),
                                             embed_all(state.params, t_rows),
                                             tool_ids[val], 5)
        print(f"  epoch     {epoch}/{epochs}  loss {record['loss']:.4f}  "
              f"recall@5 {record.get('recall@5', math.nan):.3f}")
        stats["epochs"].append(record)
    return state.params, stats


# --- stage zero ------------------------------------------------------------
def marker_id(marker: str) -> int:
    """The token id of a chat marker. Positions 4 to 13, asserted at fit time."""
    return FIRST_MARKER_ID + CHAT_MARKERS.index(marker)


def first_content_id() -> int:
    """The lowest id that is neither a control token nor a marker."""
    return FIRST_MARKER_ID + len(CHAT_MARKERS)


def stage0_row_length(n_tools: int, n_args: int, name_len: int = NAME_LEN) -> int:
    """Exactly how long one synthetic turn is."""
    tools_block = 2 + n_tools * (name_len + n_args) + 1
    query = name_len + n_args + 1
    call = 1 + name_len + 2 * n_args + 2
    return tools_block + query + call


def _tools_that_fit(n_tools: int, n_args: int, max_len: int,
                    name_len: int = NAME_LEN) -> int:
    """Shrink the catalogue until the turn fits the row, never below one."""
    while n_tools > 1 and stage0_row_length(n_tools, n_args, name_len) > max_len:
        n_tools -= 1
    return n_tools


def synthetic_batch(rng, batch: int, n_tools: int, n_args: tuple[int, int],
                    vocab: int, max_len: int, name_len: int = NAME_LEN):
    """A batch of pure symbol-manipulation turns. Returns (ids, weights).

    Names and values are random token ids, resampled every call, so nothing
    about a particular identifier generalises. What generalises is the pattern:
    find the marker, copy what follows, and interleave the keys declared in the
    tools block with the values given in the query.
    """
    ids = np.full((batch, max_len), PAD_ID, dtype=np.int32)
    weights = np.zeros((batch, max_len), dtype=np.float32)
    lo, hi = n_args
    first = first_content_id()
    for b in range(batch):
        k = int(rng.integers(lo, hi + 1))
        tools = _tools_that_fit(n_tools, k, max_len, name_len)
        names = rng.integers(first, vocab, size=(tools, name_len))
        keys = rng.integers(first, vocab, size=(tools, k))
        values = rng.integers(first, vocab, size=k)
        pick = int(rng.integers(0, tools))

        row = [BOS_ID, marker_id(TOOLS_START)]
        for i in range(tools):
            row += names[i].tolist() + keys[i].tolist()
        row += [marker_id(TOOLS_END)]
        row += names[pick].tolist() + values.tolist() + [marker_id(IM_END)]
        scored_from = len(row) + 1          # the call, not the marker that opens it
        row += [marker_id(CALL_START)] + names[pick].tolist()
        for j in range(k):
            row += [int(keys[pick][j]), int(values[j])]
        row += [marker_id(CALL_END), EOS_ID]

        ids[b, :len(row)] = row
        weights[b, scored_from:len(row)] = 1.0
    return ids, weights


def _copy_loss(apply_fn, params, ids, weights):
    """Cross entropy over the call span, and exact-row accuracy beside it."""
    import jax.numpy as jnp
    import optax

    inputs, targets = ids[:, :-1], ids[:, 1:]
    w = weights[:, 1:].astype(jnp.float32)
    logits = apply_fn({"params": params}, inputs, mask=attention_mask(inputs))
    ce = jnp.sum(optax.softmax_cross_entropy_with_integer_labels(logits, targets) * w)
    ce = ce / jnp.maximum(jnp.sum(w), 1.0)
    hit = (jnp.argmax(logits, axis=-1) == targets) | (w == 0)
    return ce, jnp.mean(jnp.all(hit, axis=-1))


def embedding_path(params):
    """Find the token embedding table: the tallest two-dimensional leaf whose
    name mentions an embedding."""
    from flax.traverse_util import flatten_dict

    flat = flatten_dict(params)
    hits = [p for p, v in flat.items()
            if getattr(v, "ndim", 0) == 2 and "embed" in "/".join(map(str, p)).lower()]
    if not hits:
        raise KeyError("no embedding table found in the parameter tree")
    return max(hits, key=lambda p: flat[p].shape[0])


def stage0_curriculum(params, cfg: QuartzConfig | None = None, *, apply_fn=None,
                      phases=STAGE0, rank: int = 16, alpha: float = 32.0,
                      lr: float = 1e-3, batch: int = 16, seed: int = 0,
                      max_len: int | None = None, steps: int | None = None):
    """Four phases of synthetic copying, on adapters. Returns (params, stats).

    Rank 16 adapters train alone while the embedding table is being scrambled
    underneath them: the base weights never move, so the scrambling leaves
    nothing behind, and at the end the adapters merge into the weights and
    everything is unfrozen again. Noise falls from "resample every step" to a
    tenth, so the head carries the pattern onto the real geometry rather than
    learning both at once.
    """
    import jax
    import jax.numpy as jnp
    import optax
    from flax.traverse_util import flatten_dict, unflatten_dict

    from quartz.model.graft import init_lora, lora_target_paths

    cfg = cfg or QuartzConfig()
    apply_fn = head_apply(cfg, apply_fn)
    flat = dict(flatten_dict(params))
    epath = embedding_path(params)
    base_embed = jnp.asarray(flat[epath])
    embed_rms = float(np.sqrt(np.mean(np.asarray(base_embed, np.float32) ** 2)))

    paths = list(lora_target_paths(params))
    if not paths:
        raise ValueError("graft.lora_target_paths found nothing to adapt")
    keys = jax.random.split(jax.random.PRNGKey(seed), len(paths))
    adapters = {p: init_lora(flat[p], rank, k) for p, k in zip(paths, keys, strict=True)}
    scale = alpha / rank

    need = max(stage0_row_length(n, hi, NAME_LEN) for _, n, (_, hi), _ in phases)
    row_len = max_len or min(cfg.max_seq_len, 1 << (need - 1).bit_length())

    def merged(base, ad, embed):
        cur = dict(base)
        cur[epath] = embed
        for path, ab in ad.items():
            delta = jnp.einsum("...ir,...ro->...io", ab["A"], ab["B"])
            cur[path] = cur[path] + scale * delta.astype(cur[path].dtype)
        return unflatten_dict(cur)

    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))

    @jax.jit
    def step(ad, opt_state, base, ids, weights, key, keep, sigma):
        # keep=0 resamples the table outright, keep=1 keeps it and adds sigma
        # times its own RMS as noise. One graph covers every phase.
        noise = jax.random.normal(key, base_embed.shape, jnp.float32) * embed_rms
        embed = (keep * base_embed.astype(jnp.float32) + sigma * noise)
        embed = embed.astype(base_embed.dtype)

        def loss_fn(a):
            return _copy_loss(apply_fn, merged(base, a, embed), ids, weights)

        (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(ad)
        updates, opt_state = tx.update(grads, opt_state, ad)
        return optax.apply_updates(ad, updates), opt_state, loss, acc

    opt_state = tx.init(adapters)
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed + 1)
    scale_total = sum(p[0] for p in phases)
    stats: dict[str, Any] = {"row_len": row_len, "rank": rank, "phases": []}
    base_tree = {k: jnp.asarray(v) for k, v in flat.items()}

    for phase_i, (n_steps, n_tools, n_args, sigma_spec) in enumerate(phases):
        if steps:                       # honour a shortened budget proportionally
            n_steps = max(1, round(n_steps * steps / scale_total))
        acc, last = 0.0, math.nan
        for i in range(n_steps):
            ids, weights = synthetic_batch(rng, batch, n_tools, n_args,
                                           cfg.vocab_size, row_len)
            if sigma_spec is None:
                keep, sigma = 0.0, 1.0
            else:
                s0, s1 = sigma_spec
                keep = 1.0
                sigma = s0 + (s1 - s0) * (i / max(n_steps - 1, 1))
            key, sub = jax.random.split(key)
            adapters, opt_state, loss, batch_acc = step(
                adapters, opt_state, base_tree, jnp.asarray(ids), jnp.asarray(weights),
                sub, jnp.float32(keep), jnp.float32(sigma))
            acc, last = float(batch_acc), float(loss)
        label = "0" + "abcdefgh"[min(phase_i, 7)]
        print(f"  {label:<4} {n_steps:>5} steps  {n_tools:>3} tools  "
              f"args {n_args[0]}-{n_args[1]}  loss {last:.4f}  acc {acc:.3f}")
        stats["phases"].append({"phase": label, "steps": n_steps, "tools": n_tools,
                                "loss": last, "accuracy": acc})

    final = merged(base_tree, adapters, base_embed)
    print(f"  merged    LoRA rank {rank}, unfroze everything")
    return jax.device_get(final), stats


# --- the stage -------------------------------------------------------------
def train(args: Any = None, *, examples: Any = None, **overrides) -> dict:
    """Run stage four: curriculum, then Gauge, then Dowser. Returns statistics.

    The temperature is deliberately not fitted here. It has to be fitted
    against outcomes -- whether the call was right -- and those come from
    Errand, so `calibration_report` is called after scoring, not during
    training, where the only label available is the head's own target.
    """
    import jax

    from quartz.model.scribe import get_tokenizer
    from quartz.train.pretrain import load_or_init
    from quartz.train.sft import dataset

    a = as_args(HeadsArgs, args, **overrides)
    model, params, cfg = load_or_init(a, config_from_args(a))
    apply_fn = model.apply
    sp = get_tokenizer(a.tokenizer) if a.tokenizer else get_tokenizer()
    rows = list(load_jsonl(a.data)) if examples is None else list(examples)
    max_len = fit_max_len(rows, sp, a.cap)

    stats: dict[str, Any] = {}
    started = time.time()
    if a.stage0:
        params, stats["stage0"] = stage0_curriculum(
            params, cfg, apply_fn=apply_fn, rank=a.rank, alpha=a.alpha,
            lr=a.lr, seed=a.seed, steps=a.stage0)
    if a.gauge:
        encoded = dataset(rows, sp, max_len)
        params, stats["gauge"] = train_gauge(
            params, encoded, epochs=a.epochs, lr=a.lr, batch=a.batch_size,
            cfg=cfg, apply_fn=apply_fn, k=a.k, seed=a.seed)
    if a.dowser:
        params, stats["dowser"] = train_dowser(
            params, pairs_from_examples(rows), epochs=a.epochs, lr=a.lr,
            batch=a.dowser_batch, cfg=cfg, sp=sp, apply_fn=apply_fn,
            holdout=a.holdout, seed=a.seed)

    write_checkpoint(a.out, jax.device_get(params), cfg)
    stats["out"] = a.out
    stats["elapsed"] = time.time() - started
    return stats
