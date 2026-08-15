"""Trellis: a wrong tool name has to be impossible, not unlikely.

The guarantee is narrow on purpose. Identifiers are masked -- the tool name, the
argument keys, and under a schema the values that have declared choices -- and
nothing else is touched. So the tests are about four things: what survives the
mask, what cannot, what happens when the model and the grammar disagree, and
that the walker reads a pre-tokenised call the same way it reads a compact one.

Every mask here soft-fails the same way. If nothing at all survives, the logits
come back untouched, because on disagreement emitting nothing is the worse
outcome.
"""
from __future__ import annotations

import numpy as np
import pytest

from quartz.model.config import CALL_END, CALL_START
from quartz.model.trellis import (
    ConstrainedDecoder,
    SchemaConstraints,
    State,
    StateMachine,
    ToolConstraints,
    Trie,
    apply_constraints,
    build_token_strings,
    digit_allowed,
)

from .conftest import SPACE, FakeSentencePiece

#: The opening of a call, in the compact form. The walker drops whitespace, so
#: this and its spaced-out wire form drive it identically -- which is asserted.
OPEN_NAME = '[{"name":"'
OPEN_ARGS = '[{"name":"set_lights","arguments":{'


@pytest.fixture
def decoder(fake_tokenizer, tool_schemas, token_strings, token_index):
    """A schema-constrained decoder over the fake vocabulary."""
    return ConstrainedDecoder.for_tools(
        fake_tokenizer, tool_schemas, schema=True,
        token_strings=token_strings, index=token_index)


@pytest.fixture
def pid(fake_tokenizer):
    """The id of one piece. `pid(' room')` is the word-initial piece."""
    return lambda text: fake_tokenizer.piece_to_id(text.replace(" ", SPACE))


@pytest.fixture
def logits(token_strings):
    return np.zeros(len(token_strings), dtype=np.float32)


# --- the tool name ----------------------------------------------------------
def test_a_wrong_tool_name_is_unreachable(decoder, logits, pid):
    """`set_lamp` shares four characters with a declared tool and then diverges.

    That is the whole failure mode: a 45M parameter model spelling a name it
    half remembers. It is a spelling failure, not a reasoning failure, and it
    can be forbidden outright.
    """
    decoder.machine.prime(OPEN_NAME)
    masked = decoder.mask(logits)

    assert np.isfinite(masked[pid(" set_lights")])
    assert np.isfinite(masked[pid(" get_weather")])
    assert np.isneginf(masked[pid(" set_lamp")])
    assert np.isneginf(masked[pid(" set_temperature")])
    assert np.isneginf(masked[pid(" delete_all")])
    assert np.isneginf(masked[pid(" kitchen")])


def test_a_right_name_survives_and_can_be_closed(decoder, logits, pid):
    """Having spelled a whole name, the only legal next move is the quote."""
    decoder.machine.prime(OPEN_NAME)
    decoder.mask(logits)
    decoder.accept(pid(" set_lights"))

    assert decoder.machine.state is State.IN_NAME
    masked = decoder.mask(logits)
    assert np.isfinite(masked[pid(' "')])
    assert np.isneginf(masked[pid(" room")])

    decoder.accept(pid(' "'))
    assert decoder.machine.tool_name == "set_lights"
    assert decoder.machine.state is State.FREE


def test_the_spaced_wire_form_reads_the_same_as_the_compact_one(
        fake_tokenizer, tool_schemas, token_strings, token_index, logits, pid):
    """The isolation rule puts a space either side of every quote and brace.

    The walker has to be blind to that, or every guarantee would hold on the
    training format and not on the emitted one.
    """
    from quartz.model.scribe import pre_tokenize

    spaced = ConstrainedDecoder.for_tools(
        fake_tokenizer, tool_schemas, token_strings=token_strings, index=token_index)
    compact = ConstrainedDecoder.for_tools(
        fake_tokenizer, tool_schemas, token_strings=token_strings, index=token_index)
    spaced.machine.prime(pre_tokenize(OPEN_NAME))
    compact.machine.prime(OPEN_NAME)

    assert spaced.machine.state is compact.machine.state is State.IN_NAME
    assert np.array_equal(spaced.mask(logits), compact.mask(logits))
    assert np.isfinite(spaced.mask(logits)[pid(" set_lights")])


# --- the argument keys ------------------------------------------------------
def test_argument_keys_come_from_the_tool_that_was_named(decoder, logits, pid):
    """`city` belongs to the other tool, so here it cannot be spelled at all."""
    decoder.machine.prime(OPEN_ARGS + '"')
    assert decoder.machine.state is State.IN_ARG_KEY

    masked = decoder.mask(logits)
    for key in (" room", " brightness", " mode"):
        assert np.isfinite(masked[pid(key)]), key
    assert np.isneginf(masked[pid(" city")])


def test_an_undeclared_tool_leaves_its_keys_unconstrained(
        fake_tokenizer, token_strings, token_index, logits):
    """The walker fell off the trie, so it stops pretending to a guarantee."""
    decoder = ConstrainedDecoder.for_tools(
        fake_tokenizer, [{"name": "set_lights", "parameters": {}}],
        token_strings=token_strings, index=token_index)
    decoder.machine.prime('[{"name":"set_lamp"')       # never on the trie
    decoder.machine.prime(',"arguments":{"')
    assert decoder.mask(logits) is logits


# --- values, which only SchemaConstraints has an opinion about --------------
def test_an_enum_value_is_masked_to_its_members(decoder, logits, pid):
    decoder.machine.prime(OPEN_ARGS + '"mode":')
    assert decoder.machine.state is State.AWAIT_VALUE

    opening = decoder.mask(logits)
    assert np.isfinite(opening[pid(' "')])
    assert np.isneginf(opening[pid(" eco")])          # the quote comes first

    decoder.accept(pid(' "'))
    masked = decoder.mask(logits)
    assert np.isfinite(masked[pid(" eco")])
    assert np.isfinite(masked[pid(" auto")])
    assert np.isneginf(masked[pid(" kitchen")])


def test_a_bounded_number_walks_its_digits(decoder, logits, pid):
    """100 is legal, and a fourth digit after it never can be.

    `room` is filled in first so the closing brace is legal on its own merits;
    the interaction with the required set is the next test down.
    """
    decoder.machine.prime(OPEN_ARGS + '"room":"kitchen","brightness":')
    start = decoder.mask(logits)
    assert np.isfinite(start[pid(" 1")])
    assert np.isneginf(start[pid(" -")])              # the range starts at zero

    decoder.machine.prime("100")
    masked = decoder.mask(logits)
    assert np.isneginf(masked[pid("0")])              # 1000 overshoots for good
    assert np.isfinite(masked[pid(" }")])             # closing here is legal


def test_plain_tool_constraints_have_no_opinion_about_values(
        fake_tokenizer, tool_schemas, token_strings, token_index, logits, pid):
    """Which is exactly why SchemaConstraints exists."""
    decoder = ConstrainedDecoder.for_tools(
        fake_tokenizer, tool_schemas, schema=False,
        token_strings=token_strings, index=token_index)
    assert isinstance(decoder.constraints, ToolConstraints)
    assert not decoder.constraints.tracks_values

    decoder.machine.prime(OPEN_ARGS + '"mode":"')
    assert decoder.machine.state is State.FREE
    assert decoder.mask(logits) is logits
    assert np.isfinite(decoder.mask(logits)[pid(" kitchen")])


# --- the one denial ---------------------------------------------------------
def test_the_arguments_object_cannot_close_on_a_missing_required_key(
        decoder, logits, pid):
    decoder.machine.prime(OPEN_ARGS)
    assert decoder.machine.pending_required() == frozenset({"room"})
    assert decoder.machine.forbids_close()

    masked = decoder.mask(logits)
    assert np.isneginf(masked[pid(" }")])
    assert np.isfinite(masked[pid(" room")])

    decoder.machine.prime('"room":"kitchen"')
    assert decoder.machine.pending_required() == frozenset()
    assert not decoder.machine.forbids_close()
    assert decoder.mask(logits) is logits


# --- the digit walker -------------------------------------------------------
def test_digit_allowed_accepts_a_value_already_in_range():
    assert digit_allowed("", "5", 0, 100)
    assert digit_allowed("1", "0", 0, 100)
    assert digit_allowed("9", "9", None, None)
    assert digit_allowed("-", "5", -20, 0)


def test_digit_allowed_accepts_a_prefix_that_can_still_get_there():
    """1 is not in [50, 100], but 100 is, so 1 has to be allowed.

    The walker decides about futures, not about the present.
    """
    assert not 50 <= 1 <= 100
    assert digit_allowed("", "1", 50, 100)


def test_digit_allowed_rejects_a_prefix_whose_completions_all_overshoot():
    assert not digit_allowed("9", "9", 0, 50)      # 99, 990, 999 ... all too big
    assert not digit_allowed("", "7", 0, 5)
    assert not digit_allowed("100", "0", 0, 100)
    assert not digit_allowed("-9", "9", -20, 0)    # more digits only go further down
    assert not digit_allowed("", "x", 0, 100)


# --- soft failure -----------------------------------------------------------
def test_an_empty_mask_falls_back_to_unmodified_logits(logits, token_strings,
                                                       token_index):
    """Nothing survived, so nothing is forbidden.

    The alternative is a sampler with no legal move, and a model that emits
    nothing at all is worse than one that disagrees with its grammar.
    """
    unreachable = Trie(["ζappa"])              # no piece begins with a zeta
    out = apply_constraints(logits, unreachable, token_strings, token_index, True)
    assert out is logits

    assert apply_constraints(logits, None, token_strings, token_index) is logits


def test_a_mask_that_survives_is_a_copy(logits, token_strings, token_index):
    """The caller's array is never written through."""
    node = Trie(["room"])
    out = apply_constraints(logits, node, token_strings, token_index, True)
    assert out is not logits
    assert np.all(logits == 0.0)
    assert np.isneginf(out).any()


# --- arming -----------------------------------------------------------------
def test_the_walker_sleeps_outside_the_call_block(
        fake_tokenizer, tool_schemas, token_strings, token_index, logits, pid):
    """Reasoning is prose, and prose must not be masked against a tool trie."""
    decoder = ConstrainedDecoder.for_tools(
        fake_tokenizer, tool_schemas, token_strings=token_strings,
        index=token_index, arm_on=CALL_START, disarm_on=CALL_END)

    decoder.machine.prime('the room is named [{"name":"')
    assert decoder.mask(logits) is logits          # still asleep

    decoder.machine.prime(CALL_START)
    decoder.machine.prime(OPEN_NAME)
    assert np.isneginf(decoder.mask(logits)[pid(" delete_all")])

    decoder.machine.prime('set_lights","arguments":{"room":"kitchen"}}]' + CALL_END)
    assert not decoder.machine.armed
    assert decoder.mask(logits) is logits


# --- the vocabulary index ---------------------------------------------------
def test_build_token_strings_reads_what_a_piece_contributes(fake_tokenizer,
                                                            token_strings, pid):
    from quartz.model.config import BOS_ID, EOS_ID, PAD_ID, UNK_ID

    assert len(token_strings) == fake_tokenizer.vocab_size()
    for control in (PAD_ID, EOS_ID, BOS_ID, UNK_ID):
        # a control piece contributes no text, so it can never satisfy a
        # constraint and a masked position can never emit one
        assert token_strings[control] == ""
    assert token_strings[pid(" room")] == " room"
    assert token_strings[pid("room")] == "room"
    assert token_strings[fake_tokenizer.piece_to_id("<0x41>")] == "A"


def test_a_malformed_byte_piece_is_an_error():
    pieces = [*FakeSentencePiece().pieces, "<0xZZ>"]
    with pytest.raises(ValueError, match="byte piece"):
        build_token_strings(FakeSentencePiece(pieces))


def test_the_index_buckets_by_first_character(token_index, pid):
    assert pid(" room") in token_index.candidates_for("r", True)
    assert pid(" room") not in token_index.candidates_for("r", False)
    assert pid("room") in token_index.candidates_for("r", False)
    assert token_index.candidates_for("ζ", True) == []
    assert f"{token_index.size:,}" in repr(token_index)


def test_the_index_covers_every_piece_that_carries_text(token_strings, token_index):
    carrying = sum(1 for text in token_strings if text)
    bucketed = sum(len(ids) for ids in token_index.by_first.values())
    assert bucketed == carrying


# --- the trie ---------------------------------------------------------------
def test_trie_walks_and_terminates():
    trie = Trie(["set_lights", "set_lamp"])
    assert trie.walk("set_l") is not None
    assert not trie.walk("set_l").is_terminal
    assert trie.walk("set_lights").is_terminal
    assert trie.walk("set_lightz") is None
    assert trie.step("z") is None
    assert "branches" in repr(trie)


def test_an_empty_trie_matches_nothing():
    trie = Trie()
    assert trie.walk("") is trie
    assert not trie.is_terminal
    assert trie.step("a") is None


# --- constraints, and what they refuse to build -----------------------------
def test_every_tool_needs_a_name():
    with pytest.raises(ValueError, match="non-empty 'name'"):
        ToolConstraints([{"description": "no name"}])


def test_properties_have_to_be_a_mapping():
    with pytest.raises(TypeError, match="not a mapping"):
        ToolConstraints([{"name": "t", "parameters": {"properties": ["room"]}}])


def test_schema_constraints_read_the_three_doors(tool_schemas):
    cons = SchemaConstraints(tool_schemas)
    assert cons.tracks_values
    assert cons.required_for("set_lights") == frozenset({"room"})
    assert cons.value_rule("set_lights", "mode").trie.walk("eco").is_terminal
    assert cons.value_rule("set_lights", "brightness").is_number
    assert cons.value_rule("set_lights", "room") is None
    assert cons.value_rule(None, "room") is None
    assert cons.param_trie("nonexistent") is None


def test_a_numeric_enum_is_not_quoted():
    cons = SchemaConstraints([{
        "name": "set_level",
        "parameters": {"properties": {"level": {"enum": [1, 2, 3]}}},
    }])
    assert not cons.value_rule("set_level", "level").quoted


# --- the decoder's own bookkeeping ------------------------------------------
def test_a_decoder_needs_a_vocabulary():
    with pytest.raises(ValueError, match="build_token_strings"):
        ConstrainedDecoder(None, None)


def test_accept_rejects_an_id_outside_the_vocabulary(decoder):
    with pytest.raises(IndexError, match="vocabulary"):
        decoder.accept(len(decoder.token_strings))


def test_only_a_minority_of_a_call_is_ever_masked(decoder, fake_tokenizer, logits):
    """About five percent of a real turn, because only identifiers are masked.

    A full JSON grammar would mask every token of the call; this masks the name,
    the keys, and the one brace that would close an unfilled object.
    """
    from quartz.model.scribe import pre_tokenize

    call = '[{"name":"set_lights","arguments":{"room":"kitchen"}}]'
    for tid in fake_tokenizer.encode(pre_tokenize(call)):
        decoder.mask(logits)
        decoder.accept(tid)

    assert decoder.tokens > 10
    assert 0.0 < decoder.constrained_fraction() < 0.5
    assert decoder.machine.tool_name == "set_lights"
    assert "masked" in repr(decoder)

    decoder.reset()
    assert decoder.constrained_fraction() == 0.0
    assert decoder.text == ""


def test_a_state_machine_with_no_constraints_masks_nothing():
    """The walker still tracks where it is; there is just nothing to forbid."""
    machine = StateMachine(None)
    machine.prime(OPEN_NAME)
    assert machine.state is State.IN_NAME
    assert machine.constraint() is None
    assert machine.pending_required() == frozenset()
    assert not machine.forbids_close()
