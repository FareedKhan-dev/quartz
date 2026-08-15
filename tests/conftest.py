"""Fixtures the whole suite shares, and the one rule about JAX.

Three things live here.

`fake_tokenizer` is a SentencePiece-shaped object with no model file behind it.
A real processor needs a fitted `.model`, which needs a corpus and a training
run, and the fast suite has neither. What the package actually asks a processor
for is a piece table, four type predicates, a greedy encode and a decode, so
that is what this provides -- and because it records every string it was handed,
a test can prove exactly where the isolation rule was applied and where it was
not.

`tiny_cfg` is a two-layer geometry small enough to initialise in a second. It
comes from `preset("tiny")` rather than from a literal, because nothing in this
package is allowed to hard-code a shape, tests included.

The JAX skip is a collection hook rather than a fixture, so `needs_jax` marks a
test once at the top and the skip happens whether the test asked for a fixture
or not.
"""
from __future__ import annotations

import importlib.util
import string
from pathlib import Path

import numpy as np
import pytest

from quartz.model.config import (
    BOS_ID,
    CHAT_MARKERS,
    EOS_ID,
    FIRST_MARKER_ID,
    ISOLATED_CHARS,
    PAD_ID,
    UNK_ID,
    QuartzConfig,
    preset,
)

#: The character SentencePiece writes where a space was.
SPACE = "▁"

#: True when the train extra is installed. Used by the collection hook below;
#: a test can also read it to skip a branch rather than a whole function.
HAS_JAX = importlib.util.find_spec("jax") is not None

_MISSING_JAX = "needs the train extra: pip install -e '.[train]'"

#: Identifiers the constrained-decoder tests spell out. `set_lamp` is the near
#: miss the mask has to make unreachable: it shares four characters with a real
#: tool and then diverges, which a bucket-and-walk mask must still catch.
_WORDS = (
    "set_lights", "set_lamp", "set_temperature", "get_weather", "start_timer",
    "delete_all", "room", "kitchen", "brightness", "city", "mode", "eco",
    "auto", "name", "arguments", "true", "false", "null",
)
_SYMBOLS = (*dict.fromkeys(ISOLATED_CHARS), ":", ".", "-", "_", "/", "?")
_CHARS = (*string.ascii_lowercase, *string.digits)

#: One byte piece, so `build_token_strings` has a `<0xNN>` to decode.
_BYTE_PIECE = "<0x41>"

_CONTROL = (PAD_ID, EOS_ID, BOS_ID)


def vocabulary() -> list[str]:
    """The piece table, in the order the ids run.

    Positions 0 to 3 are the four control ids and 4 to 13 are the chat markers,
    because that layout is not a detail of this fake: the rest of the package
    asserts it rather than looking it up.
    """
    pieces = ["<pad>", "</s>", "<s>", "<unk>", *CHAT_MARKERS, SPACE]
    ordinary = [*_WORDS, *_SYMBOLS, *_CHARS]
    pieces += ordinary
    pieces += [SPACE + piece for piece in ordinary]
    pieces.append(_BYTE_PIECE)
    return pieces


class FakeSentencePiece:
    """The slice of the SentencePieceProcessor surface Quartz uses.

    Encoding is greedy longest match over the table, which is not how byte pair
    encoding works and does not need to be: every test here is about which
    pieces exist and what text they carry, not about how a trainer chose them.

    The dummy prefix is deliberately left off, so `decode(encode(text))` is
    exactly `text` and a test can say so without allowing for a leading space.
    """

    def __init__(self, pieces: list[str] | None = None) -> None:
        self.pieces = list(vocabulary() if pieces is None else pieces)
        if len(set(self.pieces)) != len(self.pieces):
            raise ValueError("a piece table has to be unique")
        self.ids = {piece: i for i, piece in enumerate(self.pieces)}
        self.special = {PAD_ID, EOS_ID, BOS_ID, UNK_ID}
        self.longest = max(len(piece) for piece in self.pieces)
        #: Every string `encode` was handed, so a test can prove what was applied.
        self.seen: list[str] = []

    # --- the processor's own names, spelled the way SentencePiece spells them
    def vocab_size(self) -> int:
        return len(self.pieces)

    def GetPieceSize(self) -> int:  # noqa: N802 - SentencePiece's own name
        return len(self.pieces)

    def IdToPiece(self, i: int) -> str:  # noqa: N802 - SentencePiece's own name
        return self.pieces[int(i)]

    def piece_to_id(self, piece: str) -> int:
        return self.ids.get(piece, UNK_ID)

    def IsControl(self, i: int) -> bool:  # noqa: N802 - SentencePiece's own name
        return int(i) in _CONTROL

    def IsUnknown(self, i: int) -> bool:  # noqa: N802 - SentencePiece's own name
        return int(i) == UNK_ID

    def IsByte(self, i: int) -> bool:  # noqa: N802 - SentencePiece's own name
        return self.pieces[int(i)].startswith("<0x")

    def encode(self, text, out_type=int):
        if not isinstance(text, str):
            return [self.encode(one, out_type) for one in text]
        self.seen.append(text)
        pieces = self._match(text.replace(" ", SPACE))
        if out_type is str:
            return pieces
        return [self.ids[piece] for piece in pieces]

    def decode(self, ids) -> str:
        if isinstance(ids, int):
            ids = [ids]
        kept = [self.pieces[int(i)] for i in ids if int(i) not in self.special]
        return "".join(kept).replace(SPACE, " ")

    def serialized_model_proto(self) -> bytes:
        """What Ingot writes inline. The bytes are opaque to everything but
        SentencePiece, so a deterministic filler is a faithful stand-in."""
        return bytes(range(256)) * 4

    # --- the greedy walk ----------------------------------------------------
    def _ordinary(self, piece: str) -> bool:
        return piece in self.ids and self.ids[piece] not in self.special

    def _match(self, norm: str) -> list[str]:
        out: list[str] = []
        i = 0
        while i < len(norm):
            end = min(len(norm), i + self.longest)
            piece = next((norm[i:j] for j in range(end, i, -1)
                          if self._ordinary(norm[i:j])), None)
            if piece is None:
                out.append("<unk>")     # byte fallback, without the bytes
                i += 1
            else:
                out.append(piece)
                i += len(piece)
        return out


class FakeSpmModule:
    """What `scribe._spm()` returns once it is patched: a processor and a trainer.

    The trainer writes the file `train_tokenizer` then loads, and keeps the text
    it was given, which is the only way to check that the corpus really went
    through the isolation rule exactly once on the way in.
    """

    def __init__(self, processor: FakeSentencePiece) -> None:
        self.processor = processor
        self.model_file: str | None = None
        self.kwargs: dict = {}
        self.corpus: str = ""
        module = self

        class SentencePieceTrainer:
            @staticmethod
            def Train(**kwargs):  # noqa: N802 - SentencePiece's own name
                module.kwargs = dict(kwargs)
                module.corpus = "".join(
                    Path(path).read_text(encoding="utf-8") for path in kwargs["input"])
                Path(f"{kwargs['model_prefix']}.model").write_bytes(b"fake")

        self.SentencePieceTrainer = SentencePieceTrainer

    def SentencePieceProcessor(self, model_file=None):  # noqa: N802 - their name
        self.model_file = None if model_file is None else str(model_file)
        return self.processor


def pytest_collection_modifyitems(config, items):
    """Skip everything marked `needs_jax` when the train extra is absent.

    Done here rather than with a skipif on each test, so the marker is the whole
    declaration: one word at the top of a test says both "this needs JAX" and
    "leave it out of the fast suite".
    """
    del config
    if HAS_JAX:
        return
    skip = pytest.mark.skip(reason=_MISSING_JAX)
    for item in items:
        if "needs_jax" in item.keywords:
            item.add_marker(skip)


def requires_jax():
    """Return the jax module, or skip. For a fixture, where a marker cannot
    reach: the collection hook above marks the tests, this guards the set-up
    that runs before them."""
    return pytest.importorskip("jax", reason=_MISSING_JAX)


# --- geometry ---------------------------------------------------------------
@pytest.fixture
def base_cfg() -> QuartzConfig:
    """The shipped 45,211,383 parameter geometry."""
    return preset("base")


@pytest.fixture
def tiny_cfg() -> QuartzConfig:
    """Two layers and a small vocabulary: the same code paths in a second.

    The vocabulary and the memory tables are shrunk further than the preset,
    because an 8,192 by 512 embedding is most of the time a round trip takes and
    none of what it proves.
    """
    return preset("tiny", vocab_size=512, imprint_slots=64)


@pytest.fixture
def tiny_params(tiny_cfg: QuartzConfig) -> dict:
    """A parameter tree in the shape the trunk hands over, filled with noise.

    Built from `grist.shipped_tensors`, so it has every tensor the exporter
    expects at the shapes the config says, without initialising a model. One
    Foresight leaf is added on purpose: it must not reach the file.
    """
    from quartz.model.grist import shipped_tensors

    rng = np.random.default_rng(0)
    params: dict = {}
    for name, shape in shipped_tensors(tiny_cfg):
        node = params
        parts = name.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = rng.standard_normal(shape).astype(np.float32)
    params["fs_block"] = {"q_proj": {"kernel": rng.standard_normal(
        (tiny_cfg.d_model, tiny_cfg.d_model)).astype(np.float32)}}
    return params


# --- the tokenizer ----------------------------------------------------------
@pytest.fixture
def fake_tokenizer() -> FakeSentencePiece:
    return FakeSentencePiece()


@pytest.fixture
def fake_spm(fake_tokenizer, monkeypatch) -> FakeSpmModule:
    """Patch `scribe._spm()` so Scribe loads the fake instead of a real model."""
    from quartz.model import scribe as scribe_module

    module = FakeSpmModule(fake_tokenizer)
    monkeypatch.setattr(scribe_module, "_spm", lambda: module)
    return module


@pytest.fixture
def scribe(fake_spm, tmp_path):
    """A real Scribe over the fake processor, marker check and all.

    The file has to exist because Scribe refuses a missing one, but nothing ever
    reads it: the patched loader hands back the fake whatever the path says.
    """
    from quartz.model.scribe import DEFAULT_MODEL_NAME, Scribe

    path = tmp_path / DEFAULT_MODEL_NAME
    path.write_bytes(b"fake")
    return Scribe(path)


@pytest.fixture
def token_strings(fake_tokenizer) -> list[str]:
    from quartz.model.trellis import build_token_strings

    return build_token_strings(fake_tokenizer)


@pytest.fixture
def token_index(token_strings):
    from quartz.model.trellis import TokenIndex

    return TokenIndex(token_strings)


# --- tools ------------------------------------------------------------------
#: Two tools whose every identifier is spellable in the fake vocabulary, with
#: one enum, one bounded number and one required key, so the three doors
#: SchemaConstraints closes are all represented.
TOOL_SCHEMAS = [
    {
        "name": "set_lights",
        "description": "set one room's brightness",
        "parameters": {
            "type": "object",
            "properties": {
                "room": {"type": "string"},
                "brightness": {"type": "integer", "minimum": 0, "maximum": 100},
                "mode": {"type": "string", "enum": ["eco", "auto"]},
            },
            "required": ["room"],
        },
    },
    {
        "name": "get_weather",
        "description": "the forecast for one city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
]


@pytest.fixture
def tool_schemas() -> list[dict]:
    import copy

    return copy.deepcopy(TOOL_SCHEMAS)


@pytest.fixture
def marker_ids() -> dict[str, int]:
    return {marker: FIRST_MARKER_ID + i for i, marker in enumerate(CHAT_MARKERS)}
