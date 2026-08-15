"""Scribe: the tokenizer, and the rule that stops byte pair encoding eating braces.

To a model whose entire output is JSON a brace is not decoration, it is a load
bearing structural token. Byte pair encoding will happily learn a merge like
`{"room`, because that sequence is everywhere in JSON and merging it saves
tokens, and a merge table that has swallowed a brace can never be made to give
it back. Declaring the character as a user defined symbol does not help: that
guarantees a piece *exists*, it says nothing about what else the trainer may
build around it.

So the text is changed before the trainer ever sees it. The eight characters in
ISOLATED_CHARS are spaced out at corpus build and at every encode, which makes
the merge unlearnable and leaves every brace, quote and comma as its own piece.
That one rule is what lets quartz.model.trellis mask an *identifier* without
touching its punctuation.

The rule is never applied on the way back out. SentencePiece already models the
whitespace it inserted, and pre_tokenize is not idempotent, so a second pass
would double the spacing.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from quartz.model.config import (
    BOS_ID,
    CALL_END,
    CALL_START,
    CHAT_MARKERS,
    EOS_ID,
    FIRST_MARKER_ID,
    IM_END,
    IM_START,
    ISOLATED_CHARS,
    PAD_ID,
    THINK_END,
    THINK_START,
    TOOLS_END,
    TOOLS_START,
    UNK_ID,
    QuartzConfig,
)

#: The file name every loader looks for when nobody says otherwise.
DEFAULT_MODEL_NAME = "scribe.model"

# The markers that arm and disarm a constrained decoder live in
# quartz.model.trellis, beside the StateMachine that reads them. A second copy
# here would be a second thing to keep in step with the wire format.

# One alternation over the eight isolated characters, built from the config so
# the tokenizer and the decoder can never disagree about which they are.
_PRE_TOK = re.compile("([" + "".join(re.escape(c) for c in ISOLATED_CHARS) + "])")


def pre_tokenize(text: str) -> str:
    """Space out the eight structural characters.

    Applied to the fitting corpus and to every encode, never to decode. It is
    deliberately not idempotent: a second pass doubles the spaces it inserted,
    which is why Scribe.encode never applies it behind your back.
    """
    return _PRE_TOK.sub(r" \1 ", text)


def _dumps(obj: Any) -> str:
    """Compact JSON. On a 256 token window a pretty printer's spaces are money.

    ensure_ascii is off for the same reason the trainer runs with an identity
    normaliser: nothing in this pipeline may rewrite the user's bytes.
    """
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def render(example: Mapping[str, Any]) -> tuple[str, str]:
    """Turn one record into (prompt, target).

    The prompt is context we never score and the target is what the loss sees.
    The tools sit in the **user** turn rather than a system turn so they stay
    adjacent to the request, and the reasoning line sits inside the target so
    the model learns to derive each argument out loud before it commits.
    """
    if "query" not in example:
        raise KeyError(
            f"render() needs a 'query' field, got keys {sorted(example)}")
    tools = _dumps(example.get("tools", []))
    answers = _dumps(example.get("answers", []))
    prompt = (f"{IM_START}user\n{TOOLS_START}{tools}{TOOLS_END}\n"
              + example["query"] + IM_END + "\n" + f"{IM_START}assistant\n")
    target = (f"{THINK_START}{example.get('reasoning', '')}{THINK_END}\n"
              + f"{CALL_START}{answers}{CALL_END}{IM_END}")
    return prompt, target


#: Every argument the trainer gets. Three of them matter more than the rest:
#: `normalization_rule_name="identity"` stops the default Unicode rewriting,
#: which would change the user's text and break any claim that an argument was
#: copied verbatim; `byte_fallback=True` turns an unseen character into byte
#: pieces rather than an unknown, and device names are full of characters no
#: corpus contains; `character_coverage=0.9999` buys the tail of the world's
#: scripts for a few hundred slots, where the 0.9995 default quietly drops
#: whole scripts. The 4192 byte default for max_sentence_length truncates long
#: schemas, which is exactly the input we care about.
TRAIN_KWARGS: dict[str, Any] = {
    "model_type": "bpe",
    "pad_id": PAD_ID,
    "eos_id": EOS_ID,
    "bos_id": BOS_ID,
    "unk_id": UNK_ID,
    "user_defined_symbols": list(CHAT_MARKERS),
    "byte_fallback": True,             # so no input is ever <unk>
    "character_coverage": 0.9999,      # the 0.9995 default quietly drops scripts
    "normalization_rule_name": "identity",   # never rewrite the user's bytes
    "input_sentence_size": 10_000_000,
    "max_sentence_length": 32768,      # the 4192 default truncates long schemas
}


def _spm():
    """Import SentencePiece late, with an error a reader can act on."""
    try:
        import sentencepiece as spm
    except ImportError as exc:                      # pragma: no cover - env issue
        raise ImportError(
            "Scribe needs sentencepiece: pip install sentencepiece") from exc
    return spm


def assert_markers(sp) -> None:
    """The ten chat markers must land on ids 4 to 13.

    Every other file in the package hardcodes those positions instead of looking
    them up, so a marker that moved has to surface here as an error rather than
    as gibberish forty thousand steps into training.
    """
    sp = getattr(sp, "sp", sp)
    for i, marker in enumerate(CHAT_MARKERS):
        want = FIRST_MARKER_ID + i
        got = sp.piece_to_id(marker)
        if got != want:
            raise AssertionError(
                f"{marker} is on id {got}, not {want}. CHAT_MARKERS order is "
                f"load bearing and the tokenizer was fitted with a different one")


def _write_isolated(sources: Sequence[Path], dst: Path) -> int:
    """Stream the corpus through pre_tokenize once, line by line.

    Costs one pass and a second copy on disk, and buys a merge table that cannot
    contain a brace. Undecodable bytes become U+FFFD rather than killing a run
    that is hours in.
    """
    lines = 0
    with dst.open("w", encoding="utf-8") as out:
        for src in sources:
            with src.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    out.write(pre_tokenize(line.rstrip("\n")))
                    out.write("\n")
                    lines += 1
    return lines


def train_tokenizer(
    corpus_path: str | os.PathLike | Sequence[str | os.PathLike],
    prefix: str = "scribe",
    vocab_size: int | None = None,
    *,
    pretokenise: bool = True,
    **overrides: Any,
) -> Scribe:
    """Fit SentencePiece on the isolated corpus and verify the marker ids.

    `pretokenise` is only turned off when the caller has already applied the
    rule to the file, because applying it twice changes the spacing the trainer
    learns from.
    """
    spm = _spm()
    sources = ([Path(corpus_path)] if isinstance(corpus_path, str | os.PathLike)
               else [Path(p) for p in corpus_path])
    missing = [str(p) for p in sources if not p.is_file()]
    if missing:
        raise FileNotFoundError("no corpus to fit on: " + ", ".join(missing))
    vocab_size = QuartzConfig().vocab_size if vocab_size is None else int(vocab_size)

    work: Path | None = None
    try:
        if pretokenise:
            work = Path(f"{prefix}.pretok.txt")
            work.parent.mkdir(parents=True, exist_ok=True)
            _write_isolated(sources, work)
            inputs = [str(work)]
        else:
            inputs = [str(p) for p in sources]
        spm.SentencePieceTrainer.Train(
            input=inputs, model_prefix=str(prefix), vocab_size=vocab_size,
            **{**TRAIN_KWARGS, **overrides})
    finally:
        if work is not None and work.exists():
            work.unlink()

    model = Path(f"{prefix}.model")
    if not model.is_file():
        raise RuntimeError(f"the trainer produced no model at {model}")
    sp = Scribe(model)
    last = FIRST_MARKER_ID + len(CHAT_MARKERS) - 1
    print(f"  {'vocab':<9} {sp.vocab_size():,} pieces")
    print(f"  {'markers':<9} ids {FIRST_MARKER_ID} to {last} verified")
    return sp


class Scribe:
    """The trained tokenizer, plus the isolation rule and the marker ids.

    It wraps a SentencePieceProcessor rather than subclassing it, and forwards
    anything it does not define, so code written against the SentencePiece
    surface (`IsByte`, `IdToPiece`, `piece_to_id`) keeps working on a Scribe.

    `encode` is a faithful delegate that does **not** pre-tokenize, because
    pre_tokenize is not idempotent and callers following the post already apply
    it themselves. Use `encode_wire` when you are holding raw text.
    """

    def __init__(self, model_file: str | os.PathLike, *, verify: bool = True) -> None:
        path = Path(model_file)
        if not path.is_file():
            raise FileNotFoundError(
                f"no tokenizer at {path}. Train one with "
                f"`python -m quartz.cli train-tokenizer --corpus <file>`")
        spm = _spm()
        self.path = path
        self.sp = spm.SentencePieceProcessor(model_file=str(path))
        if verify:
            assert_markers(self.sp)
        self.marker_ids: dict[str, int] = {
            m: FIRST_MARKER_ID + i for i, m in enumerate(CHAT_MARKERS)}

    # --- the processor's own interface, unchanged ---------------------------
    def encode(self, text, out_type=int):
        """Encode text that has already been through pre_tokenize."""
        return self.sp.encode(text, out_type=out_type)

    def decode(self, ids: Iterable[int] | int) -> str:
        """Ids back to text. The isolation rule is never applied here."""
        if isinstance(ids, int):
            ids = [ids]
        return self.sp.decode([int(i) for i in ids])

    def vocab_size(self) -> int:
        return int(self.sp.vocab_size())

    # --- the parts that know about the rule ---------------------------------
    def encode_wire(self, text: str, *, bos: bool = False,
                    eos: bool = False) -> list[int]:
        """Raw text to ids, applying the isolation rule exactly once."""
        ids = list(self.sp.encode(pre_tokenize(text)))
        if bos:
            ids.insert(0, BOS_ID)
        if eos:
            ids.append(EOS_ID)
        return ids

    def pieces(self, text: str) -> list[str]:
        """The pieces raw text turns into, for looking at what the rule did."""
        return list(self.sp.encode(pre_tokenize(text), out_type=str))

    def marker_id(self, marker: str) -> int:
        try:
            return self.marker_ids[marker]
        except KeyError:
            raise KeyError(
                f"{marker!r} is not a chat marker; have {list(self.marker_ids)}"
            ) from None

    def __len__(self) -> int:
        return self.vocab_size()

    def __repr__(self) -> str:
        return f"Scribe({self.path.name!r}, {self.vocab_size():,} pieces)"

    def __getattr__(self, name: str):
        # Everything not defined above reaches the processor untouched.
        try:
            sp = self.__dict__["sp"]
        except KeyError:                            # pragma: no cover - partial init
            raise AttributeError(name) from None
        return getattr(sp, name)


_LOADED: dict[str, Scribe] = {}


def _candidate_paths(explicit: str | os.PathLike | None) -> list[Path]:
    here = Path(__file__).resolve()
    found = []
    if explicit is not None:
        found.append(Path(explicit))
    env = os.environ.get("QUARTZ_TOKENIZER")
    if env:
        found.append(Path(env))
    found += [
        Path.cwd() / DEFAULT_MODEL_NAME,
        here.parent / DEFAULT_MODEL_NAME,           # quartz/model/
        here.parents[2] / DEFAULT_MODEL_NAME,       # the repository root
        Path.home() / ".cache" / "quartz" / DEFAULT_MODEL_NAME,
    ]
    return found


def get_tokenizer(model_file: str | os.PathLike | None = None) -> Scribe:
    """Load the tokenizer once and hand out the same object afterwards.

    Loading costs about 400 kB of tables, and the constrained decoder builds a
    vocabulary-wide index from it, so this is a process-wide singleton per path.
    """
    looked = _candidate_paths(model_file)
    for path in looked:
        if path.is_file():
            key = str(path.resolve())
            if key not in _LOADED:
                _LOADED[key] = Scribe(path)
            return _LOADED[key]
    raise FileNotFoundError(
        "no tokenizer found. Looked in:\n  "
        + "\n  ".join(str(p) for p in looked)
        + "\nTrain one with `python -m quartz.cli train-tokenizer --corpus <file>`"
        + f", or point QUARTZ_TOKENIZER at a {DEFAULT_MODEL_NAME}.")
