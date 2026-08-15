"""The build, one subcommand per stage, in the order they run.

    quartz train-tokenizer --corpus data/mixed.txt   # the isolation rule
    quartz winnow --tokens 120e9                     # the corpus
    quartz pretrain --config configs/base.yaml       # fifteen hours, four cards
    quartz quarry --schemas data/schemas.json        # the teacher's homework
    quartz sft --data data/tools.jsonl               # the loss mask
    quartz qapt --tokens 1.8e9                       # train where we will ship
    quartz heads --stage0 2500 --gauge --dowser      # retrieval and confidence
    quartz build checkpoints/quartz_heads.pkl        # cast the Ingot
    quartz run "dim the kitchen to 10 percent"       # ask it something
    quartz errand                                    # score it

Every stage module is imported inside its own handler, never at the top of this
file. `quartz run` on a phone-class machine should not import optax, and
`quartz --help` should not import JAX at all: a help screen that takes four
seconds is a help screen nobody reads twice.

Every flag has a default, so each line above is runnable as written and the
defaults spell out the recipe the post used. Geometry defaults are read from
QuartzConfig rather than typed here, because the number of layers has exactly
one home.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

from .model.config import DEFAULT_BITS_MAP, PRESETS, QuartzConfig

#: The default geometry, used only to source flag defaults. The real geometry
#: for a run comes from --config.
_CFG = QuartzConfig()

_DEFAULT_CONFIG = "configs/base.yaml"
_DEFAULT_TOKENIZER = "artifacts/scribe.model"
_CKPT = "checkpoints"


def main(argv: list[str] | None = None) -> int:
    """Parse and dispatch. Returns a process exit code."""
    parser = _parser()
    args = parser.parse_args(argv)
    if not getattr(args, "handler", None):
        parser.print_help()
        return 1
    try:
        args.handler(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"quartz {args.command}: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    return 0


def _parser() -> argparse.ArgumentParser:
    from . import __version__

    parser = argparse.ArgumentParser(
        prog="quartz", description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--version", action="version", version=f"quartz {__version__}")
    subs = parser.add_subparsers(dest="command", metavar="stage")

    _add_train_tokenizer(subs)
    _add_winnow(subs)
    _add_pretrain(subs)
    _add_quarry(subs)
    _add_sft(subs)
    _add_qapt(subs)
    _add_heads(subs)
    _add_build(subs)
    _add_run(subs)
    _add_errand(subs)
    return parser


def _sub(subs: Any, name: str, help_text: str) -> argparse.ArgumentParser:
    return subs.add_parser(
        name, help=help_text, description=help_text,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)


# --- 1. the tokenizer ------------------------------------------------------
def _add_train_tokenizer(subs: Any) -> None:
    p = _sub(subs, "train-tokenizer",
             "Fit Scribe: isolate the eight structural characters, then run BPE.")
    p.add_argument("--corpus", default="data/corpus/mixed.txt",
                   help="one document per line, already sampled from the mix")
    p.add_argument("--prefix", default="artifacts/scribe",
                   help="output prefix; writes PREFIX.model and PREFIX.vocab")
    p.add_argument("--vocab", type=int, default=_CFG.vocab_size,
                   help="pieces, including 4 control ids, 10 markers, 256 bytes")
    p.set_defaults(handler=_train_tokenizer, command="train-tokenizer")


def _train_tokenizer(args: argparse.Namespace) -> None:
    from .model import scribe

    _need(args.corpus, "corpus")
    Path(args.prefix).parent.mkdir(parents=True, exist_ok=True)
    sp = _invoke(scribe.train_tokenizer, args, corpus_path=args.corpus,
                 prefix=args.prefix, vocab_size=args.vocab)
    size = sp.get_piece_size() if hasattr(sp, "get_piece_size") else args.vocab
    _say("scribe", f"{size:,} pieces -> {args.prefix}.model")


# --- 2. the corpus ---------------------------------------------------------
def _add_winnow(subs: Any) -> None:
    p = _sub(subs, "winnow",
             "Filter, mix and pack the corpus: 62 percent web, 18 code, 12 "
             "schemas, 8 synthetic tool traces.")
    p.add_argument("--sources", default="data/sources",
                   help="raw shards, one subdirectory per key of winnow.MIX")
    p.add_argument("--out", default="data/corpus",
                   help="where the packed token rows are written")
    p.add_argument("--tokens", type=float, default=120e9,
                   help="total tokens to emit across the mix")
    p.add_argument("--seq-len", type=int, default=_CFG.max_seq_len,
                   help="row length; every row carries seg_ids, 0 for padding")
    p.add_argument("--tokenizer", default=_DEFAULT_TOKENIZER,
                   help="the fitted Scribe model")
    p.add_argument("--min-chars", type=int, default=200,
                   help="drop documents shorter than this before packing")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 8,
                   help="parallel shard readers")
    p.add_argument("--seed", type=int, default=0, help="shuffle seed")
    p.set_defaults(handler=_winnow, command="winnow")


def _winnow(args: argparse.Namespace) -> None:
    from .data import winnow

    _need(args.tokenizer, "tokenizer")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    out = _invoke(winnow.build, args)
    _say("winnow", f"{int(args.tokens):,} tokens packed to {args.seq_len} "
                   f"-> {out or args.out}")


# --- 3. pretraining --------------------------------------------------------
def _add_pretrain(subs: Any) -> None:
    p = _sub(subs, "pretrain",
             "Stage one: next-token cross entropy plus a z-loss, Muon on the "
             "matrices and AdamW on everything else.")
    p.add_argument("--config", default=_DEFAULT_CONFIG, help="geometry YAML")
    # Every dest below is the field name on the stage's argument dataclass, not
    # a second spelling of it. as_args() keeps the keys it recognises and drops
    # the rest in silence, so a flag named for anything else parses, prints in
    # --help, and changes nothing about the run.
    p.add_argument("--data", default="data/corpus", help="packed rows from winnow")
    p.add_argument("--tokens", type=float, default=120e9, help="tokens to train on")
    p.add_argument("--batch", dest="batch_size", type=int, default=1024,
                   help="sequences per step; times seq_len gives tokens per step")
    p.add_argument("--lr", type=float, default=3e-3, help="peak learning rate")
    p.add_argument("--warmup", type=float, default=0.05,
                   help="warmup steps, or a fraction of total steps when below 1")
    p.add_argument("--total-steps", type=int, default=0,
                   help="0 derives it from --tokens, --batch and the config")
    p.add_argument("--clip", type=float, default=1.0,
                   help="global norm clip, applied before the update rules")
    p.add_argument("--z-loss", dest="z_weight", type=float, default=1e-4,
                   help="weight on the logsumexp penalty that keeps bf16 honest")
    p.add_argument("--optimiser", default="muon", choices=("muon", "adamw"),
                   help="muon orthogonalises matrix gradients; adamw is the ablation")
    p.add_argument("--average", dest="avg_last", type=int, default=6,
                   help="final checkpoints averaged in fp32 into the release")
    p.add_argument("--save-every", dest="ckpt_every", type=int, default=2000,
                   help="steps per checkpoint")
    p.add_argument("--log-every", type=int, default=100, help="steps per log line")
    p.add_argument("--out", default=f"{_CKPT}/quartz_base.pkl", help="output weights")
    p.add_argument("--resume", dest="init", default="",
                   help="checkpoint to continue from")
    p.add_argument("--seed", type=int, default=0, help="init and shuffle seed")
    p.set_defaults(handler=_pretrain, command="pretrain")


def _pretrain(args: argparse.Namespace) -> None:
    from .train import pretrain

    cfg = _load_config(args.config)
    if not args.total_steps:
        args.total_steps = max(1, int(args.tokens // (args.batch_size * cfg.max_seq_len)))
    args.warmup = _steps(args.warmup, args.total_steps)
    args.cfg = cfg
    _say("pretrain", f"{args.total_steps:,} steps, warmup {args.warmup:,}, "
                     f"{args.batch_size * cfg.max_seq_len:,} tokens a step")
    _invoke(pretrain.train, args, config=cfg, cfg=cfg)
    _say("pretrain", f"wrote {args.out}")


# --- 4. the teacher --------------------------------------------------------
def _add_quarry(subs: Any) -> None:
    p = _sub(subs, "quarry",
             "Ask a very large open model for tool-calling homework: a query, "
             "one line of reasoning, and the answers.")
    p.add_argument("--schemas", default="data/schemas.json",
                   help="the tool catalogue the teacher draws from")
    p.add_argument("--out", default="data/tools.jsonl", help="generated examples")
    p.add_argument("--n", type=int, default=1_200_000, help="examples to keep")
    p.add_argument("--batch", type=int, default=25, help="examples per request")
    p.add_argument("--refusals", type=int, default=3,
                   help="off-topic examples per batch; 3 of 25 is the one in eight "
                        "that costs half a point of accuracy and removes 92 "
                        "percent of false calls on nonsense")
    p.add_argument("--workers", type=int, default=16, help="concurrent requests")
    p.add_argument("--temperature", type=float, default=0.9,
                   help="high, because variety is the point and dedup catches the rest")
    p.add_argument("--endpoint", default="http://127.0.0.1:8000/v1",
                   help="an OpenAI-compatible server; see data/serve_teacher.sh")
    p.add_argument("--model", default="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
                   help="the teacher checkpoint the server is holding")
    p.add_argument("--seed", type=int, default=0, help="schema sampling seed")
    p.set_defaults(handler=_quarry, command="quarry")


def _quarry(args: argparse.Namespace) -> None:
    from .data import quarry

    _need(args.schemas, "schemas")
    tools = _read_json(args.schemas)
    rows = _invoke(quarry.generate, args, tools=tools, n_target=args.n)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    _say("quarry", f"{len(rows):,} examples -> {args.out}")


# --- 5. supervised fine-tuning --------------------------------------------
def _add_sft(subs: Any) -> None:
    p = _sub(subs, "sft",
             "Stage two: score the reasoning, the call and the EOS, and nothing "
             "before the assistant speaks.")
    p.add_argument("--data", default="data/tools.jsonl", help="examples from quarry")
    p.add_argument("--config", default=_DEFAULT_CONFIG, help="geometry YAML")
    p.add_argument("--init", default=f"{_CKPT}/quartz_base.pkl",
                   help="the pretrained checkpoint to start from")
    p.add_argument("--tokenizer", default=_DEFAULT_TOKENIZER, help="the Scribe model")
    p.add_argument("--epochs", type=int, default=3,
                   help="three; the fourth learns the set rather than the task")
    p.add_argument("--batch", dest="batch_size", type=int, default=16,
                   help="sequences per step")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="a thirtieth of the pretrain peak")
    p.add_argument("--warmup", type=float, default=0.05,
                   help="warmup steps, or a fraction of total steps when below 1")
    p.add_argument("--max-len", type=int, default=0,
                   help="0 fits the bucket to the data, capped by --cap")
    p.add_argument("--cap", type=int, default=1024,
                   help="the longest sequence allowed; padding beyond the data "
                        "costs four to six times the CPU time")
    p.add_argument("--holdout", type=float, default=0.1,
                   help="validation share; watch the gap, not the level")
    p.add_argument("--out", default=f"{_CKPT}/quartz_sft.pkl", help="output weights")
    p.add_argument("--seed", type=int, default=0, help="shuffle seed")
    p.set_defaults(handler=_sft, command="sft")


def _sft(args: argparse.Namespace) -> None:
    from .train import sft

    _need(args.data, "data")
    cfg = _load_config(args.config)
    args.cfg = cfg
    _invoke(sft.train, args, config=cfg, cfg=cfg)
    _say("sft", f"{args.epochs} epochs -> {args.out}")


# --- 6. quantisation-aware post-training ----------------------------------
def _add_qapt(subs: Any) -> None:
    p = _sub(subs, "qapt",
             "Train through the quantiser: rotate, index the direction, and let "
             "the straight-through estimator carry the gradient back.")
    p.add_argument("--data", default="data/tools.jsonl", help="the stage two data")
    p.add_argument("--config", default=_DEFAULT_CONFIG, help="geometry YAML")
    p.add_argument("--init", default=f"{_CKPT}/quartz_sft.pkl",
                   help="the fine-tuned checkpoint to start from")
    p.add_argument("--tokens", type=float, default=1.8e9, help="tokens to train on")
    p.add_argument("--bits-map", default=DEFAULT_BITS_MAP,
                   help="longest prefix wins; needs a default= entry")
    p.add_argument("--batch", dest="batch_size", type=int, default=16,
                   help="sequences per step")
    p.add_argument("--lr", type=float, default=3e-5,
                   help="low: the weights are already where they should be")
    p.add_argument("--noise", dest="noise_scale", type=float, default=1.0,
                   help="scale on the injected quantisation noise, over the "
                        "distortion the codebook actually makes")
    p.add_argument("--out", default=f"{_CKPT}/quartz_qapt.pkl", help="output weights")
    p.add_argument("--seed", type=int, default=0, help="noise seed")
    p.set_defaults(handler=_qapt, command="qapt")


def _qapt(args: argparse.Namespace) -> None:
    from .train import qapt

    cfg = _load_config(args.config)
    cfg.weight_bits = args.bits_map           # stamped so the exporter inherits it
    args.cfg = cfg
    _invoke(qapt.train, args, config=cfg, cfg=cfg)
    _say("qapt", f"{args.bits_map} -> {args.out}")


# --- 7. the two heads ------------------------------------------------------
def _add_heads(subs: Any) -> None:
    p = _sub(subs, "heads",
             "Dowser, warmed up on nonsense before it sees language, and Gauge, "
             "fitted to perplexity and then calibrated against being right.")
    p.add_argument("--config", default=_DEFAULT_CONFIG, help="geometry YAML")
    p.add_argument("--init", default=f"{_CKPT}/quartz_qapt.pkl",
                   help="the quantisation-aware checkpoint to start from")
    p.add_argument("--data", default="data/tools.jsonl", help="the stage two data")
    p.add_argument("--stage0", type=int, default=2500,
                   help="curriculum steps on random token ids, before real text")
    p.add_argument("--gauge", action="store_true", default=False,
                   help="train the confidence head")
    p.add_argument("--dowser", action="store_true", default=False,
                   help="train the retrieval head; with neither flag both run")
    p.add_argument("--lora-rank", type=int, default=16,
                   help="stage zero trains adapters, then merges and unfreezes")
    p.add_argument("--batch", type=int, default=128, help="examples per step")
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    p.add_argument("--out", default=f"{_CKPT}/quartz_heads.pkl", help="output weights")
    p.add_argument("--seed", type=int, default=0, help="curriculum seed")
    # Gauge's temperature is not fitted here. It needs stated confidence beside
    # observed correctness, which only exists once the suite has been run, so
    # `quartz errand` fits it and this stage trains the head that it corrects.
    p.set_defaults(handler=_heads, command="heads")


def _heads(args: argparse.Namespace) -> None:
    from . import _bind
    from .model import run as checkpoints
    from .train import heads

    _need(args.init, "checkpoint")
    bundle = checkpoints.load_checkpoint(args.init)
    params = bundle["params"]
    cfg = bundle["config"] or _load_config(args.config)

    # Neither flag means the shipped recipe, which is both. Asking for one head
    # and silently getting two would be worse than asking for neither.
    both = not (args.gauge or args.dowser)
    if args.dowser or both:
        if args.stage0:
            params, _ = _bind(heads.stage0_curriculum, params, cfg,
                              rank=args.lora_rank, lr=args.lr, batch=args.batch,
                              seed=args.seed, steps=args.stage0)
        params, _ = _bind(heads.train_dowser, params,
                          _dowser_pairs(args.data, heads), cfg=cfg, lr=args.lr,
                          batch=args.batch, seed=args.seed)
        _say("heads", "dowser trained")
    if args.gauge or both:
        params, _ = _bind(heads.train_gauge, params, args.data, cfg=cfg,
                          lr=args.lr, batch=args.batch, seed=args.seed)
        _say("heads", "gauge trained; fit its temperature with `quartz errand`")

    checkpoints.save_checkpoint(args.out, params, cfg)
    _say("heads", f"wrote {args.out}")


def _dowser_pairs(path: str, heads: Any) -> list[tuple[str, str]]:
    """(request, tool) pairs out of the stage two file.

    The pairing itself belongs to the stage, not to this file:
    `heads.pairs_from_examples` decides what the tool side of a pair is, and
    the tool side is the whole declared schema dumped exactly the way the
    prompt dumps it. A second implementation here was writing the same schemas
    with `sort_keys=True`, so the CLI trained Dowser on one spelling of a tool
    and the stage indexed it under another. All this adds is the error for an
    empty file, which is a CLI concern: the stage is entitled to be handed no
    pairs, a person who typed a path is not.
    """
    pairs = heads.pairs_from_examples(_read_json(path))
    if not pairs:
        raise ValueError(
            f"{path} yielded no (request, tool) pairs: a row needs 'query', "
            f"'tools' and 'answers' naming one of them")
    return pairs


# --- 8. the container ------------------------------------------------------
def _add_build(subs: Any) -> None:
    p = _sub(subs, "build",
             "Cast the checkpoint into one Ingot: pre-transposed, layer major, "
             "64-byte aligned, tensors found by position and not by name.")
    p.add_argument("checkpoint", nargs="?", default=f"{_CKPT}/quartz_heads.pkl",
                   help="the trained checkpoint to export")
    p.add_argument("--out", default="quartz.ingot", help="the container to write")
    p.add_argument("--config", default=_DEFAULT_CONFIG, help="geometry YAML")
    p.add_argument("--tokenizer", default=_DEFAULT_TOKENIZER,
                   help="the Scribe model, stored inline in the container")
    p.add_argument("--bits-map", default="",
                   help="empty takes the map stamped on the checkpoint's config, "
                        f"falling back to {DEFAULT_BITS_MAP!r}")
    p.set_defaults(handler=_build, command="build")


def _build(args: argparse.Namespace) -> None:
    from . import _bind
    from .model import ingot, scribe
    from .model import run as checkpoints

    _need(args.checkpoint, "checkpoint")
    bundle = checkpoints.load_checkpoint(args.checkpoint)
    params, config, tokenizer = _bundle(bundle, args.config)
    bits_map = args.bits_map or getattr(config, "weight_bits", "") or DEFAULT_BITS_MAP
    sp = tokenizer or scribe.get_tokenizer(args.tokenizer)

    # Foresight is dropped by the exporter, not by a flag here: it is a
    # training-only head, and a device that carried it would never run it.
    _bind(ingot.write_ingot, params, config, args.out,
          bits_map=bits_map, sp=sp, tokenizer=sp)
    size = Path(args.out).stat().st_size
    _say("ingot", f"{args.out}  {size / 1e6:.2f} MB  {bits_map}")


# --- 9. asking it something -----------------------------------------------
def _add_run(subs: Any) -> None:
    p = _sub(subs, "run",
             "One request in, one call out. The CLI has no functions to perform, "
             "so it prints the call; use quartz.Quartz(...).run() to execute it.")
    p.add_argument("prompt", nargs="?", default="",
                   help="the request; empty reads stdin")
    p.add_argument("--weights", default="",
                   help="a checkpoint; empty fetches the published release")
    p.add_argument("--tools", default="",
                   help="JSON file holding a list of tool schemas")
    p.add_argument("--tool-index", default="",
                   help="a catalogue too large for the window; .npz or .json")
    p.add_argument("--system", default="", help="a system turn, prepended verbatim")
    p.add_argument("--config", default="",
                   help="geometry, when the weights carry none")
    p.add_argument("--max-new-tokens", type=int, default=0,
                   help="0 uses what is left of the window")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 is greedy, which is what a tool caller wants")
    p.add_argument("--retrieve", type=int, default=5,
                   help="schemas rendered per turn when a catalogue is loaded")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="below this confidence, escalate instead of acting")
    p.add_argument("--no-constrain", action="store_true", default=False,
                   help="turn Trellis off; costs about 90 identifier errors per "
                        "thousand calls and saves 11.4 microseconds a token")
    p.add_argument("--json", action="store_true", default=False,
                   help="print the whole response dict instead of a readout")
    p.set_defaults(handler=_run, command="run")


def _run(args: argparse.Namespace) -> None:
    from . import Quartz

    prompt = args.prompt or sys.stdin.read().strip()
    if not prompt:
        raise ValueError("no request: pass one as an argument or on stdin")
    tools = _read_json(args.tools) if args.tools else []

    agent = Quartz(tools=tools, system=args.system or None,
                   weights=args.weights or None,
                   tool_index_path=args.tool_index or None,
                   config=args.config or None,
                   max_new_tokens=args.max_new_tokens,
                   temperature=args.temperature, retrieve=args.retrieve,
                   threshold=args.threshold, constrain=not args.no_constrain)
    response = agent.complete(prompt)

    if args.json:
        print(json.dumps(response, indent=2, default=str))
        return
    calls = response["function_calls"]
    if not calls:
        print("  (no call: nothing declared here serves this request)")
    for call in calls:
        print(f"  {call['name']}")
        print(f"    {json.dumps(call['arguments'], ensure_ascii=False)}")
    if response["reasoning"]:
        print(f"  why       {response['reasoning']}")
    if response["confidence"] is not None:
        print(f"  confidence {response['confidence']:.3f}")
    if response["error"]:
        print(f"  error     {response['error']}", file=sys.stderr)


# --- 10. the benchmark -----------------------------------------------------
def _add_errand(subs: Any) -> None:
    p = _sub(subs, "errand",
             "Score the suite: tool selection, argument exactness, refusal on "
             "off-topic input, and the calibration of the confidence.")
    p.add_argument("--weights", default="",
                   help="a checkpoint; empty fetches the published release")
    p.add_argument("--suite", default="",
                   help="a saved suite JSON; empty builds it from the seed")
    p.add_argument("--tool-index", default="",
                   help="score against a full catalogue rather than the declared "
                        "tools, which is where Dowser's recall becomes the ceiling")
    p.add_argument("--out", default="artifacts/errand.json",
                   help="where the per-errand results are written")
    p.add_argument("--limit", type=int, default=0, help="0 runs the whole suite")
    p.add_argument("--tools-per-turn", type=int, default=5,
                   help="schemas rendered per turn when a catalogue is loaded")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="below this confidence, escalate instead of acting")
    p.add_argument("--no-constrain", action="store_true", default=False,
                   help="score without Trellis, which is the unconstrained column")
    p.add_argument("--seed", type=int, default=0, help="suite seed")
    p.set_defaults(handler=_errand, command="errand")


def _errand(args: argparse.Namespace) -> None:
    from . import Quartz, _bind
    from .errand.score import score
    from .errand.suite import build_suite

    suite = _read_json(args.suite) if args.suite else build_suite(seed=args.seed)
    suite = list(suite)[: args.limit] if args.limit else list(suite)

    agent = Quartz(weights=args.weights or None,
                   tool_index_path=args.tool_index or None,
                   retrieve=args.tools_per_turn, threshold=args.threshold,
                   constrain=not args.no_constrain)
    predictions = []
    for i, errand in enumerate(suite, 1):
        predictions.append(agent.complete(errand.get("query", ""),
                                          tools=errand.get("tools") or None))
        if i % 100 == 0:
            print(f"  {i:,}/{len(suite):,}", end="\r", file=sys.stderr, flush=True)

    report = _bind(score, suite, predictions)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"report": report, "predictions": predictions},
                   indent=2, default=str), encoding="utf-8")
    _say("errand", "  ".join(f"{k} {_pct(v)}" for k, v in _flat(report).items()))


# --- shared plumbing -------------------------------------------------------
def _invoke(fn: Any, args: argparse.Namespace, **extra: Any) -> Any:
    """Call a stage entry point with the flags it declares.

    Each stage owns its own signature: some take the parsed namespace whole,
    some take named arguments. Binding by name here means a stage can be read on
    its own terms without this file being a second, drifting copy of its
    argument list. Anything required and unparsed raises with the name of what
    is missing rather than a bare TypeError.
    """
    values = {**vars(args), **extra}
    params = inspect.signature(fn).parameters
    names = [n for n, p in params.items()
             if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    if names[:1] == ["args"]:
        return fn(args, **{k: v for k, v in extra.items() if k in names[1:]})

    kwargs = {n: values[n] for n in names if n in values}
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        # Only what this handler asked for explicitly goes into **kwargs. The
        # namespace also holds flags named for other things, and a stage that
        # forwards its keywords onward would pass those straight through.
        kwargs.update({k: v for k, v in extra.items() if k not in names})

    missing = [n for n in names
               if n not in kwargs and params[n].default is inspect.Parameter.empty]
    if missing:
        raise RuntimeError(
            f"{fn.__module__}.{fn.__name__} needs {missing}, which this "
            f"subcommand does not parse")
    return fn(**kwargs)


def _load_config(path: str) -> QuartzConfig:
    """Geometry from a YAML file, or from a preset name for a quick run."""
    if path in PRESETS:
        from .model.config import preset

        return preset(path)
    _need(path, "config")
    return QuartzConfig.from_yaml(path)


def _bundle(bundle: Any, config_path: str) -> tuple[Any, QuartzConfig, Any]:
    """Split a loaded checkpoint, falling back to the YAML for the geometry."""
    from . import _unpack_bundle

    params, config, tokenizer = _unpack_bundle(bundle)
    if isinstance(config, dict):
        config = QuartzConfig(**config)
    return params, config or _load_config(config_path), tokenizer


def _steps(value: float, total: int) -> int:
    """A step count, given either a count or a fraction of the schedule."""
    return int(value) if value >= 1 else max(1, int(round(value * total)))


def _need(path: str, what: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"no {what} at {path}")


def _read_json(path: str) -> Any:
    _need(path, "file")
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith(".jsonl"):
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def _flat(report: Any) -> dict[str, Any]:
    return report if isinstance(report, dict) else {"score": report}


def _pct(value: Any) -> str:
    return f"{value:.1%}" if isinstance(value, float) and 0 <= value <= 1 else str(value)


def _say(stage: str, message: str) -> None:
    print(f"  [{stage}]".ljust(12) + message)


if __name__ == "__main__":
    raise SystemExit(main())
