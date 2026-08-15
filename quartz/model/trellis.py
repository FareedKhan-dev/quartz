"""Trellis: making a wrong tool name impossible rather than unlikely.

A 45M parameter model will sometimes emit `set_light` when the tool is called
`set_lights`. That is a spelling failure, not a reasoning failure, and it can be
forbidden outright.

The instinct is a full JSON grammar compiled from the schema. The measurement
says otherwise: unconstrained, malformed JSON runs at 6.1 errors per thousand
calls, while wrong tool names and wrong argument keys run at 38.4 and 51.2. Both
of those are *identifier* errors, so Trellis constrains identifiers and nothing
else, at about 5 percent of emitted tokens.

Three pieces:

* a `Trie` per turn, rebuilt because Dowser's five tools change every turn;
* a `StateMachine` that walks the emitted text and decides when we are inside an
  identifier, which is the only thing that gets masked;
* a mask that buckets the vocabulary by first character, so a decode step walks
  about 180 strings instead of 8,192.

`SchemaConstraints` then closes the three doors the trie leaves open, with an
enum trie, a digit walker for bounded numbers and a required-set rule about when
the closing brace is legal.

Every mask soft-fails the same way: if nothing at all survives, the logits come
back untouched. On disagreement between the model and the grammar, emitting
nothing is the worse outcome.

This module needs numpy and the config, nothing else. The token strings come
from a tokenizer object the caller already has.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np

from quartz.model.config import CALL_END, CALL_START

#: What ends an unquoted value, and what the arguments object closes with.
_VALUE_END = ",}]"
_DIGITS = "0123456789-"

#: The cues the walker watches for, in the whitespace-free skeleton.
_NAME_CUE = '"name":"'
_ARGS_CUE = '"arguments":{'
_KEY_CUES = ('{"', ',"')

#: Only the tail of the skeleton is ever tested, and every cue is far shorter.
_SKELETON_KEEP = 64

#: Pass these to StateMachine(arm_on=..., disarm_on=...) to keep the walker
#: asleep outside the call block.
ARM_MARKER, DISARM_MARKER = CALL_START, CALL_END

_EMPTY: list[int] = []


class Trie:
    """A prefix tree over identifiers.

    Rebuilt every turn from five tools, so it is a few hundred nodes and the
    build cost never shows up next to a decode step.
    """

    __slots__ = ("children", "is_terminal")

    def __init__(self, words: Iterable[str] = ()) -> None:
        self.children: dict[str, Trie] = {}
        self.is_terminal = False
        for word in words:
            node = self
            for ch in str(word):
                node = node.children.setdefault(ch, Trie())
            node.is_terminal = True

    def step(self, ch: str) -> Trie | None:
        """The node one character on, or None when that path does not exist."""
        return self.children.get(ch)

    def walk(self, text: str) -> Trie | None:
        node: Trie | None = self
        for ch in text:
            if node is None:
                return None
            node = node.children.get(ch)
        return node

    def __repr__(self) -> str:
        return f"Trie({len(self.children)} branches, terminal={self.is_terminal})"


class State(Enum):
    """Where the walker is.

    FREE, IN_NAME and IN_ARG_KEY are all a plain ToolConstraints ever reaches:
    the name and the keys are masked, everything else runs free. AWAIT_VALUE and
    IN_ARG_VALUE only become reachable under SchemaConstraints, which has an
    opinion about values as well.
    """

    FREE = auto()
    IN_NAME = auto()
    IN_ARG_KEY = auto()
    AWAIT_VALUE = auto()
    IN_ARG_VALUE = auto()


@dataclass(frozen=True)
class _ValueRule:
    """What a declared argument's value is allowed to be.

    Either an enum, which is a trie over its members, or a bounded number, which
    is a pair of bounds for the digit walker. Anything else has no rule and is
    never masked.
    """

    trie: Trie | None = None
    quoted: bool = True
    lo: float | None = None
    hi: float | None = None

    @property
    def is_number(self) -> bool:
        return self.trie is None


@dataclass(frozen=True)
class _Constraint:
    """What the next token has to satisfy. Produced by StateMachine.constraint."""

    kind: str                       # identifier | bare | number | await
    node: Trie | None = None
    prefix: str = ""
    lo: float | None = None
    hi: float | None = None
    quoted: bool = True
    lead_space: bool = True
    seen_colon: bool = False


def _parameters(tool: Mapping[str, Any]) -> Mapping[str, Any]:
    params = tool.get("parameters") or {}
    return params if isinstance(params, Mapping) else {}


def _properties(tool: Mapping[str, Any]) -> Mapping[str, Any]:
    """The declared arguments of one tool, tolerating a schema with none."""
    props = _parameters(tool).get("properties") or {}
    if not isinstance(props, Mapping):
        raise TypeError(
            f"tool {tool.get('name')!r} declares properties as "
            f"{type(props).__name__}, which is not a mapping")
    return props


class ToolConstraints:
    """Rebuilt every turn, because Dowser's five tools change every turn.

    Two guarantees: the tool name exists, and every argument key exists. Values
    are not its business, which is why an enum can still come out wrong here and
    why SchemaConstraints exists.
    """

    tracks_values = False

    def __init__(self, tools: Sequence[Mapping[str, Any]]) -> None:
        self.tools = [dict(t) for t in tools]
        for tool in self.tools:
            if not isinstance(tool.get("name"), str) or not tool["name"]:
                raise ValueError(f"every tool needs a non-empty 'name': {tool!r}")
        self.name_trie = Trie(t["name"] for t in self.tools)
        self.param_tries: dict[str, Trie] = {
            t["name"]: Trie(_properties(t)) for t in self.tools}

    def param_trie(self, tool: str | None) -> Trie | None:
        """The argument keys of one tool, or None when we never declared it."""
        return None if tool is None else self.param_tries.get(tool)

    def required_for(self, tool: str | None) -> frozenset[str]:
        return frozenset()

    def value_rule(self, tool: str | None, key: str | None) -> _ValueRule | None:
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}({len(self.tools)} tools)"


class SchemaConstraints(ToolConstraints):
    """An enum becomes a trie, a bounded number becomes a digit walker, and a
    required set becomes a rule about when the closing brace is legal.

    That closes the three doors the trie leaves open: a value outside its
    choices, a number outside its range, and an object that closes before it has
    been filled in.
    """

    tracks_values = True

    def __init__(self, tools: Sequence[Mapping[str, Any]]) -> None:
        super().__init__(tools)
        self.value_tries: dict[tuple[str, str], Trie] = {}
        self.enum_quoted: dict[tuple[str, str], bool] = {}
        self.ranges: dict[tuple[str, str], tuple[float | None, float | None]] = {}
        self.required: dict[str, frozenset[str]] = {}
        for tool in self.tools:
            name = tool["name"]
            required = _parameters(tool).get("required") or ()
            self.required[name] = frozenset(str(k) for k in required)
            for key, spec in _properties(tool).items():
                if not isinstance(spec, Mapping):
                    continue
                if "enum" in spec:
                    members = list(spec["enum"])
                    self.value_tries[(name, key)] = Trie(str(v) for v in members)
                    # a numeric enum arrives unquoted, so the span ends on a
                    # comma or a brace rather than on a quote
                    self.enum_quoted[(name, key)] = all(
                        isinstance(v, str) for v in members)
                elif spec.keys() & {"minimum", "maximum"}:
                    self.ranges[(name, key)] = (spec.get("minimum"), spec.get("maximum"))

    def required_for(self, tool: str | None) -> frozenset[str]:
        return self.required.get(tool, frozenset()) if tool is not None else frozenset()

    def value_rule(self, tool: str | None, key: str | None) -> _ValueRule | None:
        if tool is None or key is None:
            return None
        trie = self.value_tries.get((tool, key))
        if trie is not None:
            return _ValueRule(trie=trie, quoted=self.enum_quoted.get((tool, key), True))
        bounds = self.ranges.get((tool, key))
        if bounds is None:
            return None
        return _ValueRule(lo=bounds[0], hi=bounds[1])


class StateMachine:
    """Walks the generated text and decides when we are inside an identifier.

    Three states do the work of the shipped guarantee: FREE, IN_NAME and
    IN_ARG_KEY, and only the last two are masked. The two value states are only
    entered when the constraints have a rule for that argument.

    The wire format is pre-tokenised, so the emitted stream carries a space
    before every brace, quote and comma (see quartz.model.scribe.pre_tokenize).
    The skeleton buffer therefore drops whitespace and identifier spans ignore
    it, which makes the walker read a spaced call and a compact one identically.
    """

    def __init__(self, constraints: ToolConstraints | None = None, *,
                 arm_on: str | None = None, disarm_on: str | None = None) -> None:
        self.constraints = constraints
        self.arm_on = arm_on
        self.disarm_on = disarm_on
        self.reset()

    def reset(self) -> None:
        """Forget everything, including the text, and start a fresh turn."""
        self.text = ""
        self._reset_structure()
        self.armed = not self.arm_on

    def _reset_structure(self) -> None:
        self.state = State.FREE
        self.buffer = ""
        self.constrained_buf = ""
        self.node: Trie | None = None
        self.in_string = False
        self.prev_escape = False
        self.nesting = 0
        self.in_arguments = False
        self.args_depth = 0
        self.tool_name: str | None = None
        self.arg_key: str | None = None
        self.seen_keys: set[str] = set()
        self.value_rule: _ValueRule | None = None
        self.value_mode = ""
        self.seen_colon = False

    # --- feeding ------------------------------------------------------------
    def feed(self, ch: str) -> None:
        """Consume one emitted character."""
        self.text += ch
        if not self.armed:
            if self.arm_on and self.text.endswith(self.arm_on):
                self.armed = True
                self._reset_structure()
            return
        if self.disarm_on and self.text.endswith(self.disarm_on):
            self.armed = False
            self._reset_structure()
            return
        if not ch.isspace():
            self.buffer = (self.buffer + ch)[-_SKELETON_KEEP:]
        if self.state is not State.FREE:
            self._inside(ch)
        else:
            self._free(ch)

    def prime(self, text: str) -> None:
        """Feed a whole string, for a prefix the sampler did not produce."""
        for ch in text:
            self.feed(ch)

    def _free(self, ch: str) -> None:
        if self.in_string:
            # a brace inside a string VALUE must not open an argument object
            if self.prev_escape:
                self.prev_escape = False
            elif ch == "\\":
                self.prev_escape = True
            elif ch == '"':
                self.in_string = False
            return
        if ch.isspace():
            return
        if ch in "{[":
            self.nesting += 1
        elif ch in "}]":
            self.nesting = max(0, self.nesting - 1)
            self.in_arguments &= self.nesting >= self.args_depth
        cons = self.constraints
        if self.buffer.endswith(_NAME_CUE) and not self.in_arguments:
            self._open(State.IN_NAME, None if cons is None else cons.name_trie)
        elif self.buffer.endswith(_ARGS_CUE):
            self.in_arguments, self.args_depth = True, self.nesting
            self.seen_keys = set()
        elif (self.in_arguments and self.nesting == self.args_depth
              and self.buffer[-2:] in _KEY_CUES):
            self._open(State.IN_ARG_KEY,
                       None if cons is None else cons.param_trie(self.tool_name))
        elif ch == '"':
            self.in_string = True

    def _open(self, state: State, node: Trie | None) -> None:
        self.state = state
        self.constrained_buf = ""
        self.node = node

    def _inside(self, ch: str) -> None:
        if self.state is State.AWAIT_VALUE:
            self._await(ch)
        elif self.state is State.IN_ARG_VALUE and self.value_mode != "quoted":
            self._unquoted(ch)
        else:
            self._span(ch)

    def _span(self, ch: str) -> None:
        """A quoted identifier: a tool name, an argument key, a string enum."""
        if ch == '"':
            self._close_span()
            return
        if ch.isspace():
            return          # the isolation rule spaces the quotes, never the name
        self.constrained_buf += ch
        if self.node is not None:
            self.node = self.node.step(ch)

    def _unquoted(self, ch: str) -> None:
        """A bare value: a number, or an enum whose members are not strings."""
        if ch.isspace() or ch in _VALUE_END:
            self.state, self.value_mode, self.node = State.FREE, "", None
            self.value_rule = None
            self._free(ch)          # let the terminator do its structural work
            return
        self.constrained_buf += ch
        if self.value_mode != "number" and self.node is not None:
            self.node = self.node.step(ch)

    def _close_span(self) -> None:
        text = self.constrained_buf
        state = self.state
        self.state, self.node, self.constrained_buf = State.FREE, None, ""
        if state is State.IN_NAME:
            self.tool_name = text
        elif state is State.IN_ARG_KEY:
            self.arg_key = text
            self.seen_keys.add(text)
            self._maybe_await()
        else:
            self.value_mode, self.value_rule = "", None

    def _maybe_await(self) -> None:
        rule = (None if self.constraints is None
                else self.constraints.value_rule(self.tool_name, self.arg_key))
        if rule is None:
            return              # a value we have no opinion about
        self.value_rule, self.seen_colon = rule, False
        self.state = State.AWAIT_VALUE

    def _await(self, ch: str) -> None:
        """Between a key and its value: the colon, then whatever starts it."""
        rule = self.value_rule
        if ch.isspace():
            return
        if ch == ":" and not self.seen_colon:
            self.seen_colon = True
            return
        if rule is not None and self.seen_colon:
            if rule.trie is not None and rule.quoted and ch == '"':
                self.state, self.node = State.IN_ARG_VALUE, rule.trie
                self.constrained_buf, self.value_mode = "", "quoted"
                return
            if rule.trie is not None and not rule.quoted:
                self.state, self.node = State.IN_ARG_VALUE, rule.trie.step(ch)
                self.constrained_buf, self.value_mode = ch, "bare"
                return
            if rule.is_number and (ch.isdigit() or ch == "-"):
                self.state, self.node = State.IN_ARG_VALUE, None
                self.constrained_buf, self.value_mode = ch, "number"
                return
        self.state, self.value_rule = State.FREE, None
        self._free(ch)

    # --- what the mask needs ------------------------------------------------
    def constraint(self) -> _Constraint | None:
        """What the next token has to satisfy, or None when nothing is masked.

        A None node means the walker fell off its trie, which happens after a
        soft failure. From there the span runs unconstrained rather than
        pretending to a guarantee it no longer has.
        """
        if not self.armed or self.constraints is None:
            return None
        lead = not self.constrained_buf
        if self.state in (State.IN_NAME, State.IN_ARG_KEY):
            if self.node is None:
                return None
            return _Constraint("identifier", node=self.node, lead_space=lead)
        if self.state is State.AWAIT_VALUE and self.value_rule is not None:
            rule = self.value_rule
            return _Constraint("await", node=rule.trie, lo=rule.lo, hi=rule.hi,
                               quoted=rule.quoted, lead_space=True,
                               seen_colon=self.seen_colon)
        if self.state is State.IN_ARG_VALUE:
            if self.value_mode == "number" and self.value_rule is not None:
                return _Constraint("number", prefix=self.constrained_buf,
                                   lo=self.value_rule.lo, hi=self.value_rule.hi,
                                   lead_space=False)
            if self.node is None:
                return None
            kind = "identifier" if self.value_mode == "quoted" else "bare"
            return _Constraint(kind, node=self.node, lead_space=lead)
        return None

    def pending_required(self) -> frozenset[str]:
        """Declared arguments this call has not filled in yet."""
        if self.constraints is None or not self.in_arguments:
            return frozenset()
        return self.constraints.required_for(self.tool_name) - self.seen_keys

    def forbids_close(self) -> bool:
        """Whether the arguments object may not close at this position.

        Relies on the isolation rule: in a Scribe vocabulary a brace is always
        its own piece, so denying every token that starts with one denies the
        close and nothing else.
        """
        return bool(self.armed and self.in_arguments
                    and self.nesting == self.args_depth
                    and self.pending_required())


def build_token_strings(sp) -> list[str]:
    """The text each id actually contributes, indexed by id.

    U+2581 becomes a real space, so a word-initial piece cannot sit in the
    middle of an identifier and drops out of the span for free. Control and
    unknown pieces get the empty string, so they can never satisfy a constraint
    and a constrained position can never emit one.
    """
    sp = getattr(sp, "sp", sp)
    out: list[str] = []
    for i in range(sp.vocab_size()):
        if sp.IsControl(i) or sp.IsUnknown(i):
            out.append("")
        elif sp.IsByte(i):
            piece = sp.IdToPiece(i)
            try:
                out.append(chr(int(piece[3:5], 16)))
            except ValueError as exc:
                raise ValueError(
                    f"id {i} is flagged as a byte piece but reads {piece!r}") from exc
        else:
            out.append(sp.IdToPiece(i).replace("▁", " "))
    return out


class TokenIndex:
    """First-character buckets over the vocabulary.

    Built once and shared across turns: the tries change every turn, this does
    not. It is what turns a mask into about 180 string walks instead of a scan
    over all 8,192 pieces.

    Two tables, because the wire format is pre-tokenised: one keyed on the first
    character of a piece, one on its first non-space character, for the pieces
    that open an identifier just after a spaced-out quote.
    """

    def __init__(self, token_strings: Sequence[str]) -> None:
        self.size = len(token_strings)
        self.by_first: dict[str, list[int]] = {}
        self.by_visible: dict[str, list[int]] = {}
        for tid, text in enumerate(token_strings):
            if not text:
                continue
            self.by_first.setdefault(text[0], []).append(tid)
            visible = text.lstrip()
            if visible:
                self.by_visible.setdefault(visible[0], []).append(tid)

    def candidates_for(self, ch: str, allow_leading_space: bool = False) -> list[int]:
        table = self.by_visible if allow_leading_space else self.by_first
        return table.get(ch, _EMPTY)

    def __repr__(self) -> str:
        return f"TokenIndex({self.size:,} pieces, {len(self.by_visible)} buckets)"


# --- the walks a single token has to survive -------------------------------


def _skip_space(text: str, i: int) -> int:
    while i < len(text) and text[i].isspace():
        i += 1
    return i


def _identifier_fits(text: str, node: Trie, lead_space: bool) -> bool:
    """Walk a token through the trie of a quoted identifier.

    Leading whitespace is skipped at the start of a span, and before a closing
    quote, because the isolation rule puts a space on both sides of every quote.
    It is not skipped in the middle, which is what keeps a word-initial piece
    out of the middle of a name.
    """
    if not text:
        return False
    i = _skip_space(text, 0) if (lead_space or text.lstrip()[:1] == '"') else 0
    if i >= len(text):
        return False
    while i < len(text):
        ch = text[i]
        if ch == '"':
            return node.is_terminal      # the identifier may end here
        child = node.step(ch)
        if child is None:
            return False
        node = child
        i += 1
    return True


def _bare_fits(text: str, node: Trie, lead_space: bool) -> bool:
    """The same walk for an unquoted enum: a space, comma or brace ends it."""
    if not text:
        return False
    i = _skip_space(text, 0) if lead_space else 0
    if i >= len(text):
        return False
    while i < len(text):
        ch = text[i]
        if ch.isspace() or ch in _VALUE_END:
            return node.is_terminal
        child = node.step(ch)
        if child is None:
            return False
        node = child
        i += 1
    return True


def digit_allowed(prefix: str, digit: str, lo: float | None, hi: float | None) -> bool:
    """Can `prefix + digit` still BECOME a number in [lo, hi]?

    '1' is not in [50, 100], but '10' leads to '100', so '1' has to be allowed.
    The walker decides about futures, not about the present: it asks whether any
    padding of the candidate still lands inside the range.
    """
    cand = prefix + digit
    negative = cand.startswith("-")
    digits = cand[1:] if negative else cand
    if not digits.isdigit():
        return False
    value = int(cand)
    if (lo is None or value >= lo) and (hi is None or value <= hi):
        return True
    # more digits move a positive number up and a negative number down, so only
    # the bound in that direction can still be reached
    bound = lo if negative else hi
    if bound is None:
        return True
    pad = len(str(abs(int(bound)))) - len(digits)
    if pad < 0:
        return False
    lowest, highest = int(digits + "0" * pad), int(digits + "9" * pad)
    if negative:
        lowest, highest = -highest, -lowest
    return (hi is None or lowest <= hi) and (lo is None or highest >= lo)


def _number_complete(text: str, lo: float | None, hi: float | None) -> bool:
    """Whether a finished number is inside its declared range."""
    if not text or not text.lstrip("-").isdigit():
        return False
    value = int(text)
    return (lo is None or value >= lo) and (hi is None or value <= hi)


def _number_fits(text: str, prefix: str, lo: float | None, hi: float | None,
                 lead_space: bool) -> bool:
    """Walk a token digit by digit against the declared bounds."""
    if not text:
        return False
    i = _skip_space(text, 0) if lead_space else 0
    if i >= len(text):
        return False
    cur = prefix
    while i < len(text):
        ch = text[i]
        if ch.isdigit():
            if cur in ("0", "-0"):
                return False            # JSON has no leading zeros
            if not digit_allowed(cur, ch, lo, hi):
                return False
            cur += ch
        elif ch == "-" and not cur:
            if lo is not None and lo >= 0:
                return False
            cur = "-"
        elif ch.isspace() or ch in _VALUE_END:
            return _number_complete(cur, lo, hi)
        else:
            return False
        i += 1
    return True


def _await_fits(text: str, c: _Constraint) -> bool:
    """Skip the spaces and the colon, then the value has to start legally.

    The colon and the first digits often arrive in one piece, because a colon is
    not one of the isolated characters, so this walk has to cross it.
    """
    i = _skip_space(text, 0)
    colon = c.seen_colon
    if not colon:
        if i >= len(text) or text[i] != ":":
            return False
        i = _skip_space(text, i + 1)
        colon = True
    if i >= len(text):
        return True                     # the value starts in the next token
    rest = text[i:]
    if c.node is not None and c.quoted:
        if rest[0] != '"':
            return False
        return len(rest) == 1 or _identifier_fits(rest[1:], c.node, False)
    if c.node is not None:
        return _bare_fits(rest, c.node, False)
    return _number_fits(rest, "", c.lo, c.hi, False)


# --- turning a walk into a mask --------------------------------------------


def _width(logits) -> int:
    shape = np.shape(logits)
    if len(shape) != 1:
        raise ValueError(
            f"constraints take one logit vector at a time, got shape {shape}")
    return int(shape[0])


def _mask_logits(logits, mask: np.ndarray):
    """Apply a keep-mask, or hand the logits back untouched if it is empty."""
    if not mask.any():
        # on disagreement, emitting nothing is the worse outcome
        return logits
    out = np.array(logits, copy=True)
    out[~mask] = -np.inf
    return out


def _mask_over(logits, groups, fits, token_strings: Sequence[str], index: TokenIndex):
    """Build a keep-mask from (characters, allow_leading_space) buckets."""
    width = _width(logits)
    mask = np.zeros(width, bool)
    for chars, allow_space in groups:
        for ch in chars:
            for tid in index.candidates_for(ch, allow_space):
                if tid < width and not mask[tid]:
                    mask[tid] = fits(token_strings[tid])
    return _mask_logits(logits, mask)


def apply_constraints(logits, node: Trie | None, token_strings: Sequence[str],
                      index: TokenIndex, allow_leading_space: bool = False):
    """Mask every token that cannot walk the identifier trie.

    Bucketing by first character makes this about 180 string walks per token,
    not the 8,192 of a vocabulary-wide scan. If nothing survives, the logits are
    returned unmodified rather than leaving the sampler with no legal move.
    """
    if node is None:
        return logits
    groups = [(node.children, allow_leading_space)]
    if node.is_terminal:
        # the closing quote arrives spaced, whatever the span has eaten so far
        groups.append(('"', True))
    return _mask_over(logits, groups,
                      lambda t: _identifier_fits(t, node, allow_leading_space),
                      token_strings, index)


def _apply_bare(logits, c: _Constraint, token_strings: Sequence[str],
                index: TokenIndex):
    groups = [(c.node.children, c.lead_space)]
    if c.node.is_terminal:
        groups += [(_VALUE_END, True), (" ", False)]
    return _mask_over(logits, groups, lambda t: _bare_fits(t, c.node, c.lead_space),
                      token_strings, index)


def _apply_number(logits, c: _Constraint, token_strings: Sequence[str],
                  index: TokenIndex):
    groups = [(_DIGITS, c.lead_space), (_VALUE_END, True), (" ", False)]
    return _mask_over(logits, groups,
                      lambda t: _number_fits(t, c.prefix, c.lo, c.hi, c.lead_space),
                      token_strings, index)


def _apply_await(logits, c: _Constraint, token_strings: Sequence[str],
                 index: TokenIndex):
    if not c.seen_colon:
        starts = ":"
    elif c.node is not None and c.quoted:
        starts = '"'
    elif c.node is not None:
        starts = "".join(c.node.children)
    else:
        starts = _DIGITS
    return _mask_over(logits, [(starts, True)], lambda t: _await_fits(t, c),
                      token_strings, index)


def _deny_first(logits, chars: str, token_strings: Sequence[str], index: TokenIndex):
    """Forbid every token that starts with one of `chars`.

    The one denial in the system, used for the closing brace of an arguments
    object that is still missing a required key. It is a denial rather than an
    allow-list because at that point every other character is still legal.
    """
    width = _width(logits)
    arr = np.asarray(logits)
    deny = np.zeros(width, bool)
    for ch in chars:
        for tid in index.candidates_for(ch, True):
            if tid < width:
                deny[tid] = True
    if not deny.any():
        return logits
    if not (~deny & np.isfinite(arr)).any():
        return logits               # denying everything leaves nothing to emit
    out = np.array(arr, copy=True)
    out[deny] = -np.inf
    return out


class ConstrainedDecoder:
    """Two calls a token: mask the logits, then accept the id that was sampled.

        logits = decoder.mask(logits)      # before sampling
        token = sample(logits)
        decoder.accept(token)              # after

    Nothing else is needed, and nothing else is safe: the walker only ever sees
    text that was really emitted, so a soft failure cannot desynchronise it from
    the sequence.

    The vocabulary index is built once. The tries come from the constraints,
    which are rebuilt every turn because the declared tools change every turn.
    """

    def __init__(self, sp, constraints: ToolConstraints | None, *,
                 token_strings: Sequence[str] | None = None,
                 index: TokenIndex | None = None,
                 arm_on: str | None = None, disarm_on: str | None = None) -> None:
        if token_strings is None:
            if sp is None:
                raise ValueError(
                    "ConstrainedDecoder needs a tokenizer, or token_strings from "
                    "build_token_strings()")
            token_strings = build_token_strings(sp)
        self.token_strings = list(token_strings)
        self.index = TokenIndex(self.token_strings) if index is None else index
        self.constraints = constraints
        self.machine = StateMachine(constraints, arm_on=arm_on, disarm_on=disarm_on)
        self.tokens = 0
        self.constrained = 0

    @classmethod
    def for_tools(cls, sp, tools: Sequence[Mapping[str, Any]], *,
                  schema: bool = True, **kwargs) -> ConstrainedDecoder:
        """Build the constraints for this turn's tools and wrap them."""
        cons = SchemaConstraints(tools) if schema else ToolConstraints(tools)
        return cls(sp, cons, **kwargs)

    def mask(self, logits):
        """Return the logits with every illegal token set to -inf."""
        self.tokens += 1
        out = self._apply(logits)
        if self.machine.forbids_close():
            out = _deny_first(out, "}", self.token_strings, self.index)
        if out is not logits:
            self.constrained += 1
        return out

    def _apply(self, logits):
        c = self.machine.constraint()
        if c is None:
            return logits
        if c.kind == "identifier":
            return apply_constraints(logits, c.node, self.token_strings, self.index,
                                     c.lead_space)
        if c.kind == "bare":
            return _apply_bare(logits, c, self.token_strings, self.index)
        if c.kind == "number":
            return _apply_number(logits, c, self.token_strings, self.index)
        if c.kind == "await":
            return _apply_await(logits, c, self.token_strings, self.index)
        raise ValueError(f"unknown constraint kind {c.kind!r}")

    def accept(self, token_id: int) -> str:
        """Feed the sampled token to the walker and return the text it added."""
        tid = int(token_id)
        if not 0 <= tid < len(self.token_strings):
            raise IndexError(
                f"token id {tid} is outside the {len(self.token_strings):,} piece "
                f"vocabulary")
        text = self.token_strings[tid]
        for ch in text:
            self.machine.feed(ch)
        return text

    def reset(self) -> None:
        """Start a new turn against the same vocabulary index."""
        self.machine.reset()
        self.tokens = self.constrained = 0

    def constrained_fraction(self) -> float:
        """The share of emitted tokens that were actually masked."""
        return self.constrained / self.tokens if self.tokens else 0.0

    @property
    def text(self) -> str:
        return self.machine.text

    @property
    def state(self) -> State:
        return self.machine.state

    def __repr__(self) -> str:
        return (f"ConstrainedDecoder({self.constraints!r}, "
                f"{self.constrained}/{self.tokens} tokens masked)")
