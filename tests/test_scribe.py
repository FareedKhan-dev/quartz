"""Scribe: the isolation rule, the turn format, and the marker ids.

The rule under test is one line long and the whole constrained decoder rests on
it. Space the eight structural characters out before the trainer ever sees the
corpus and no merge can contain a brace, which is what lets Trellis mask an
identifier without touching its punctuation. Apply it a second time and every
space doubles, which is why nothing on the decode path may touch it.
"""
from __future__ import annotations

import json

import pytest

from quartz.model.config import (
    CALL_END,
    CALL_START,
    CHAT_MARKERS,
    IM_END,
    IM_START,
    ISOLATED_CHARS,
    THINK_END,
    THINK_START,
    TOOLS_END,
    TOOLS_START,
    QuartzConfig,
)
from quartz.model.scribe import (
    TRAIN_KWARGS,
    Scribe,
    assert_markers,
    get_tokenizer,
    pre_tokenize,
    render,
    train_tokenizer,
)

from .conftest import SPACE, FakeSentencePiece


# --- the rule ---------------------------------------------------------------
@pytest.mark.parametrize("char", list(ISOLATED_CHARS))
def test_pre_tokenize_isolates_every_structural_character(char):
    """All eight, one test each, so a failure names the character that moved."""
    assert pre_tokenize(f"a{char}b") == f"a {char} b"


def test_pre_tokenize_covers_the_eight_and_nothing_else():
    text = "".join(ISOLATED_CHARS)
    assert pre_tokenize(text) == "".join(f" {char} " for char in ISOLATED_CHARS)
    # a colon carries no structure a mask needs, so it is left where it was
    assert pre_tokenize("room:kitchen") == "room:kitchen"
    assert pre_tokenize("") == ""


def test_pre_tokenize_is_not_idempotent():
    """The reason it is never applied on the way back out.

    A second pass doubles the spaces the first one inserted, so a decode that
    "helpfully" normalised the text would corrupt every value it touched.
    """
    once = pre_tokenize('{"room"}')
    assert pre_tokenize(once) != once
    assert pre_tokenize(once).count(" ") > once.count(" ")


def test_a_merge_can_no_longer_swallow_a_brace(scribe):
    """After the rule, no piece holds a structural character beside anything else."""
    pieces = scribe.pieces('{"room":"kitchen"}')
    for piece in pieces:
        body = piece.lstrip(SPACE)
        assert len(body) == 1 or not set(body) & set(ISOLATED_CHARS), piece
    assert '{' in [piece.lstrip(SPACE) for piece in pieces]


# --- what Scribe does and does not do --------------------------------------
def test_encode_does_not_apply_the_rule(scribe, fake_tokenizer):
    """`encode` is a faithful delegate: the caller owns the rule.

    Applying it here as well would double the spacing for every caller who
    followed the post and applied it themselves.
    """
    text = '{"room":"kitchen"}'
    scribe.encode(text)
    assert fake_tokenizer.seen[-1] == text


def test_encode_wire_applies_the_rule_exactly_once(scribe, fake_tokenizer):
    text = '{"room":"kitchen"}'
    scribe.encode_wire(text)
    assert fake_tokenizer.seen[-1] == pre_tokenize(text)
    assert fake_tokenizer.seen[-1] != pre_tokenize(pre_tokenize(text))


def test_encode_wire_adds_the_control_ids_on_request(scribe):
    from quartz.model.config import BOS_ID, EOS_ID

    bare = scribe.encode_wire("room")
    wrapped = scribe.encode_wire("room", bos=True, eos=True)
    assert wrapped == [BOS_ID, *bare, EOS_ID]


def test_decode_never_applies_the_rule(scribe, fake_tokenizer):
    """Decode hands back the spaced text unchanged, and encodes nothing."""
    text = '{"room":"kitchen"}'
    ids = scribe.encode_wire(text)
    seen = len(fake_tokenizer.seen)
    out = scribe.decode(ids)
    assert out == pre_tokenize(text)
    assert out != text                      # the caller undoes it, not Scribe
    assert len(fake_tokenizer.seen) == seen  # decode did not re-encode anything


def test_decode_takes_a_single_id(scribe, fake_tokenizer):
    tid = fake_tokenizer.piece_to_id("room")
    assert scribe.decode(tid) == "room"


def test_pieces_shows_what_the_rule_did(scribe):
    assert scribe.pieces("room") == ["room"]
    assert "".join(scribe.pieces('{"room"}')).replace(SPACE, " ") == pre_tokenize('{"room"}')


def test_scribe_forwards_the_processor_surface(scribe, fake_tokenizer):
    """Code written against SentencePiece keeps working on a Scribe."""
    assert scribe.vocab_size() == fake_tokenizer.vocab_size()
    assert len(scribe) == fake_tokenizer.vocab_size()
    assert scribe.IsByte(fake_tokenizer.piece_to_id("<0x41>"))
    assert "Scribe(" in repr(scribe)


def test_a_missing_model_says_how_to_make_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="train-tokenizer"):
        Scribe(tmp_path / "absent.model")


# --- the marker ids ---------------------------------------------------------
def test_markers_land_on_four_through_thirteen(scribe, fake_tokenizer, marker_ids):
    assert_markers(fake_tokenizer)
    assert_markers(scribe)
    for marker, want in marker_ids.items():
        assert scribe.marker_id(marker) == want
        assert fake_tokenizer.piece_to_id(marker) == want


def test_a_moved_marker_is_an_error_not_gibberish():
    """The rest of the package hardcodes these positions, so this must shout."""
    pieces = ["<pad>", "</s>", "<s>", "<unk>", "wedge", *CHAT_MARKERS]
    with pytest.raises(AssertionError, match="load bearing"):
        assert_markers(FakeSentencePiece(pieces))


def test_marker_id_rejects_something_that_is_not_a_marker(scribe):
    with pytest.raises(KeyError, match="not a chat marker"):
        scribe.marker_id("<|nonsense|>")


# --- the turn format --------------------------------------------------------
def test_render_puts_the_tools_in_the_user_turn():
    example = {
        "query": "dim the kitchen",
        "tools": [{"name": "set_lights"}],
        "reasoning": "the room is named",
        "answers": [{"name": "set_lights", "arguments": {"room": "kitchen"}}],
    }
    prompt, target = render(example)

    assert prompt.startswith(f"{IM_START}user\n{TOOLS_START}")
    assert prompt.endswith(f"{IM_START}assistant\n")
    assert TOOLS_END in prompt and example["query"] in prompt
    assert target == (f"{THINK_START}the room is named{THINK_END}\n"
                      f"{CALL_START}"
                      f"{json.dumps(example['answers'], separators=(',', ':'))}"
                      f"{CALL_END}{IM_END}")
    # the reasoning is inside the target, so the loss sees it
    assert THINK_START not in prompt


def test_render_dumps_compact_json_and_keeps_the_bytes():
    """Every space costs window, and no accent may be rewritten as an escape."""
    prompt, _ = render({"query": "q", "tools": [{"name": "café", "x": 1}]})
    assert '"name":"café"' in prompt
    assert ", " not in prompt.split(TOOLS_END)[0]


def test_render_defaults_a_missing_tool_list_and_answer():
    prompt, target = render({"query": "hello"})
    assert f"{TOOLS_START}[]{TOOLS_END}" in prompt
    assert f"{CALL_START}[]{CALL_END}" in target


def test_render_needs_a_query():
    with pytest.raises(KeyError, match="query"):
        render({"tools": []})


# --- loading and fitting ----------------------------------------------------
def test_get_tokenizer_is_a_singleton_per_path(fake_spm, tmp_path, monkeypatch):
    from quartz.model import scribe as scribe_module

    path = tmp_path / "scribe.model"
    path.write_bytes(b"fake")
    monkeypatch.setattr(scribe_module, "_LOADED", {})
    monkeypatch.setenv("QUARTZ_TOKENIZER", str(path))

    first = get_tokenizer()
    assert first is get_tokenizer()
    assert first is get_tokenizer(path)


def test_get_tokenizer_says_where_it_looked(tmp_path, monkeypatch):
    from quartz.model import scribe as scribe_module

    looked = [tmp_path / "nowhere" / "scribe.model"]
    monkeypatch.setattr(scribe_module, "_candidate_paths", lambda explicit: looked)
    with pytest.raises(FileNotFoundError, match="QUARTZ_TOKENIZER"):
        get_tokenizer()


def test_train_tokenizer_needs_a_corpus(fake_spm, tmp_path):
    with pytest.raises(FileNotFoundError, match="no corpus"):
        train_tokenizer(tmp_path / "absent.txt", prefix=str(tmp_path / "scribe"))


def test_train_tokenizer_fits_on_the_isolated_corpus(fake_spm, tmp_path, capsys):
    """The corpus reaching the trainer has been through the rule exactly once.

    That is the whole reason the fitting step costs a second pass over the data:
    a merge table fitted on the raw text could contain a brace, and no later
    configuration can take it back out.
    """
    lines = ['{"room":"kitchen"}', '{"room":"hall"}']
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(lines) + "\n", encoding="utf-8")
    prefix = tmp_path / "scribe"

    sp = train_tokenizer(corpus, prefix=str(prefix), vocab_size=64)

    assert fake_spm.corpus == "".join(pre_tokenize(line) + "\n" for line in lines)
    assert '{"' not in fake_spm.corpus        # the merge is unlearnable now
    assert fake_spm.kwargs["vocab_size"] == 64
    assert not list(tmp_path.glob("*.pretok.txt"))     # the scratch copy is gone
    assert sp.vocab_size() > 0
    assert "markers" in capsys.readouterr().out


def test_train_tokenizer_defaults_the_vocabulary_to_the_config(fake_spm, tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("room\n", encoding="utf-8")
    train_tokenizer(corpus, prefix=str(tmp_path / "scribe"))
    assert fake_spm.kwargs["vocab_size"] == QuartzConfig().vocab_size


def test_the_three_trainer_arguments_that_matter():
    """Each of these is a whole class of failure, so each is asserted by name."""
    assert TRAIN_KWARGS["normalization_rule_name"] == "identity"   # never rewrite bytes
    assert TRAIN_KWARGS["byte_fallback"] is True                   # nothing is <unk>
    assert TRAIN_KWARGS["character_coverage"] > 0.9995             # keep the tail
    assert TRAIN_KWARGS["max_sentence_length"] > 4192              # schemas are long
    assert TRAIN_KWARGS["user_defined_symbols"] == list(CHAT_MARKERS)
