"""Quartz: a 45,211,383 parameter model whose entire output is a function call.

    from quartz import Quartz, tool

    @tool
    def set_lights(room: str, brightness: int = 100) -> str:
        \"\"\"Set one room's brightness.

        Args:
            room: which room.
            brightness: 0 to 100.
        \"\"\"
        return f"{room} at {brightness}"

    agent = Quartz(tools=[set_lights])
    agent.complete("dim the kitchen to 10 percent")["function_calls"]
    # [{'name': 'set_lights', 'arguments': {'room': 'kitchen', 'brightness': 10}}]

Importing this module costs the standard library and nothing else. JAX, Flax and
the tokenizer arrive on the first turn, not at import, so a program that builds
a Quartz at start-up and never receives a request pays nothing. A test enforces
the JAX half of that rule, because it is one convenience import away from being
broken at any time.

The response contract
---------------------
`complete()` returns a dict with every key below always present. Nothing is
optional and nothing appears conditionally, so a caller can index it without
guarding and a missing head shows up as a None rather than a KeyError:

    text              the assistant turn as decoded, markers stripped
    raw               the same turn with its markers intact, which is what the
                      loop replays into the next prompt
    reasoning         the one line inside <think>...</think>, "" when absent
    function_calls    [{"name": str, "arguments": dict}]; [] is the refusal
    confidence        Gauge, temperature-corrected, or None with no head loaded
    tools             the names rendered into this turn's window
    retrieval         "declared", "dowser" or "lexical": how those were chosen
    usage             {"prompt_tokens": int, "completion_tokens": int}
    stop_reason       "stop" when the model closed the turn, "length" when the
                      window ran out mid-call
    error             None, or why the emitted text could not be read as calls

`run()` returns a second contract, described on the method. An empty
`function_calls` list is a real answer: it is how the model refuses, and 12.5
percent of its supervision was learning to produce it.
"""
from __future__ import annotations

import inspect
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .agent.tools import Field, build_schema, tool
from .model.config import (
    BOS_ID,
    CALL_END,
    CALL_START,
    CHAT_MARKERS,
    EOS_ID,
    FIRST_MARKER_ID,
    IM_END,
    IM_START,
    ISOLATED_CHARS,
    PRESETS,
    RESULT_END,
    RESULT_START,
    THINK_END,
    THINK_START,
    TOOLS_END,
    QuartzConfig,
    preset,
)

__all__ = ["Field", "Quartz", "__version__", "extract", "tool"]
__version__ = "0.1.0"

#: The keys `complete()` always returns, and what each one means. Kept as data
#: so a test can assert the contract instead of trusting the docstring.
RESPONSE_FIELDS: dict[str, str] = {
    "text": "the assistant turn as decoded, chat markers stripped",
    "raw": "the same turn, markers intact, as replayed into the next prompt",
    "reasoning": "the line inside <think>...</think>, empty when absent",
    "function_calls": "[{'name': str, 'arguments': dict}]; [] is a refusal",
    "confidence": "Gauge, temperature-corrected, or None with no head loaded",
    "tools": "the tool names rendered into this turn's window",
    "retrieval": "how those were chosen: declared, dowser or lexical",
    "usage": "{'prompt_tokens': int, 'completion_tokens': int}",
    "stop_reason": "'stop' if the model closed the turn, 'length' if it ran out",
    "error": "None, or why the emitted text could not be read as calls",
}

#: One scalar temperature, fitted on 2,000 held-out calls, which drops the
#: expected calibration error of the raw Gauge logit from 0.1174 to 0.0212. This
#: is the shipped model's value and only the fallback: a checkpoint that carries
#: `gauge_temperature` in its meta is believed instead, because the number
#: belongs to the weights it was fitted on. A fine-tuned model inherits a
#: temperature fitted on a perplexity distribution that no longer holds, which
#: is why graft leaves the head alone and says so rather than reporting a
#: confident wrong number.
GAUGE_TEMPERATURE = 1.61

_ISOLATED_RE = re.compile(f"[ ]?([{re.escape(ISOLATED_CHARS)}])[ ]?")
_WORD_RE = re.compile(r"[a-z0-9]+")
_ASSISTANT = f"{IM_START}assistant\n"

#: Stop on the end of turn as well as on EOS. The supervision writes both, in
#: that order, so a model that closes the turn and then keeps talking has
#: already told us it is finished.
_STOP_IDS = (EOS_ID, FIRST_MARKER_ID + CHAT_MARKERS.index(IM_END))


def extract(text: str, schema: Any, *, weights: str | None = None,
            name: str = "extract", description: str | None = None,
            **kwargs: Any) -> dict[str, Any] | None:
    """Pull declared fields out of a passage.

    Performing an action and pulling structured data out of a passage are the
    same operation, and the only difference is what you declared. So extraction
    is one tool with no side effect::

        class Invoice(BaseModel):
            supplier: str
            total: float

        extract(letter, Invoice)
        # {'supplier': 'Ardent Freight', 'total': 1840.0}

    Returns the arguments the model filled in, or None when it declines. None is
    not an error: a passage that carries none of the declared fields should
    produce nothing, and the refusal behaviour is what makes that possible.

    Args:
        text: the passage to read.
        schema: a pydantic model, a function, or a raw JSON schema dict.
        weights: which checkpoint to use. Agents are cached per weights path,
            since reloading the model for each of a thousand documents is the
            whole cost of the job.
        name: the tool name to declare, which the model must spell exactly.
        description: overrides the docstring the schema would otherwise use.
        **kwargs: forwarded to Quartz for the cached agent's first construction.
    """
    spec = build_schema(schema, name=name, description=description)
    key = str(weights or "")
    agent = _EXTRACTORS.get(key)
    if agent is None:
        agent = _EXTRACTORS[key] = Quartz(weights=weights, **kwargs)
    response = agent.complete(text, tools=[spec])
    for call in response["function_calls"]:
        if call["name"] == spec["name"]:
            return call["arguments"]
    return None


#: One agent per weights path, kept alive between extract() calls.
_EXTRACTORS: dict[str, Quartz] = {}


class Quartz:
    """A tool-calling model behind three methods: complete, run, reset.

    `complete()` reads one request and returns one turn. It has no memory: the
    prompt is the tools, the request, and nothing else, which is the shape the
    model was supervised on and the shape a 256 token window can hold.

    `run()` is the loop: it executes the calls against the Python functions you
    declared, feeds the results back, and asks again until the model stops
    calling things or the step budget runs out.

    Args:
        tools: the tools to declare. Functions, pydantic models and raw schema
            dicts may be mixed. Only functions can be executed by `run()`; the
            rest are declared to the model and reported back to you to perform.
        system: a system turn, prepended verbatim. Supported because deployments
            want a persona line, but be aware the supervision put the tools in
            the user turn and used no system turn at all, so anything here is
            out-of-distribution context rather than an instruction the model was
            taught to obey.
        weights: path to a checkpoint. None fetches the published release,
            honouring QUARTZ_WEIGHTS first. Not an Ingot: that container is read
            positionally by a runtime that already knows the module tree, and
            Python is not that runtime.
        tool_index_path: a catalogue too large for the window. A .npz with
            `schemas` (JSON strings) and optionally `emb` (the Dowser vectors),
            or a .json/.jsonl list of schemas. Each turn renders the best few
            rather than all of them.
        config: a QuartzConfig, a preset name, a path to a YAML config, or a
            dict. Only needed when the checkpoint carries no geometry.
        tokenizer: an already-loaded Scribe or SentencePiece processor.
        max_steps: the default step budget for `run()`.
        max_new_tokens: 0 derives it per turn from what is left of the window,
            which is the only honest cap: you cannot decode past the cache.
        temperature: 0.0 is greedy, which is what a tool caller wants.
        top_k: 1 with a temperature of zero. Both are handed to the decoder.
        seed: sampling seed, used only when temperature is above zero.
        constrain: run Trellis. Off costs 38.4 wrong tool names and 51.2 wrong
            argument keys per thousand calls, and saves 11.4 microseconds a
            token against a decode step of about 1,175.
        retrieve: how many schemas the window holds when a catalogue is loaded.
        threshold: refuse to act below this confidence. `run()` stops with
            stop_reason "escalate" instead of executing, which is the deployable
            failure: a caller that says "ask something bigger" is fine, one that
            sends money to the wrong person is not. Requires a Gauge head.
        window: override the KV window. 0 takes it from the config, which takes
            it from the budget.
    """

    def __init__(self, tools: Iterable[Any] | None = None, system: str | None = None,
                 weights: str | Path | None = None,
                 tool_index_path: str | Path | None = None, *,
                 config: Any = None, tokenizer: Any = None, max_steps: int = 8,
                 max_new_tokens: int = 0, temperature: float = 0.0, top_k: int = 1,
                 seed: int = 0, constrain: bool = True, retrieve: int = 5,
                 threshold: float = 0.0, window: int = 0) -> None:
        if max_steps < 1:
            raise ValueError(f"max_steps must be at least 1, got {max_steps}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")

        self.system = system
        self.weights = str(weights) if weights else ""
        self.tool_index_path = str(tool_index_path) if tool_index_path else ""
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.seed = seed
        self.constrain = constrain
        self.retrieve = retrieve
        self.threshold = threshold
        self.window = window

        self.tools: list[dict[str, Any]] = []
        self.messages: list[dict[str, Any]] = []
        self._callables: dict[str, Callable[..., Any]] = {}
        for declaration in tools or ():
            self.register(declaration)

        self._config = _coerce_config(config)
        self._sp = tokenizer
        self._params: Any = None
        self._dcfg: Any = None
        self._token_strings: list[str] | None = None
        self._token_index: Any = None
        self._gauge_temperature = GAUGE_TEMPERATURE
        self._catalogue: list[dict[str, Any]] | None = None
        self._catalogue_emb: Any = None

    def __repr__(self) -> str:
        names = ", ".join(t["name"] for t in self.tools[:4])
        more = "" if len(self.tools) <= 4 else f", +{len(self.tools) - 4}"
        state = "loaded" if self._params is not None else "not loaded"
        return f"Quartz(tools=[{names}{more}], {state})"

    # --- declaring -------------------------------------------------------
    def register(self, declaration: Any, *, name: str | None = None,
                 description: str | None = None) -> dict[str, Any]:
        """Declare one tool and return the schema the model will see.

        A callable is remembered as well as declared, so `run()` can perform it.
        Anything else is declared only, and a call to it comes back to you in
        the response for your own dispatcher to handle.
        """
        schema = build_schema(declaration, name=name, description=description)
        self.tools = [t for t in self.tools if t["name"] != schema["name"]]
        self.tools.append(schema)
        if callable(declaration) and not isinstance(declaration, type):
            self._callables[schema["name"]] = declaration
        return schema

    def reset(self) -> None:
        """Drop the transcript and start the next request from nothing.

        The weights, the tokenizer, the token index and the tool catalogue stay
        resident. They are the thirteen and a half megabytes and the second of
        set-up cost, and a device that reloaded them between turns would spend
        longer loading than answering. What goes is the conversation, which is
        the only part that should not survive a new request.
        """
        self.messages.clear()

    # --- one turn --------------------------------------------------------
    def complete(self, query: str, *, tools: Iterable[Any] | None = None,
                 max_new_tokens: int | None = None,
                 constrain: bool | None = None) -> dict[str, Any]:
        """Read one request and return one turn, with no memory of any other.

        `tools` overrides the declared set for this call only, which is how the
        benchmark hands each errand its own catalogue. See RESPONSE_FIELDS for
        every key of the returned dict.
        """
        declared = ([build_schema(t) for t in tools] if tools is not None
                    else None)
        return self._step([{"role": "user", "content": query}], declared,
                          max_new_tokens=max_new_tokens, constrain=constrain)

    # --- the loop --------------------------------------------------------
    def run(self, query: str, *, max_steps: int | None = None,
            execute: bool = True) -> dict[str, Any]:
        """Call tools, feed the results back, and ask again until it stops.

        Returns:
            A dict with:

            - `function_calls`: every call made, across every step, in order
            - `results`: one entry per call, each `{"name", "arguments"}` plus
              either `"result"` or `"error"`
            - `steps`: the per-step `complete()` dicts, unmodified
            - `text`, `confidence`: from the last step
            - `messages`: the transcript, which is also kept on the instance
            - `stop_reason`: one of the four below
            - `exhausted`: True only when the budget ran out

        The stop reasons are the whole contract of this loop:

        - "done": the model returned an empty call list. That is a finished
          answer, and on an off-topic request it is the refusal.
        - "escalate": confidence fell below `threshold`, so nothing was
          executed and the request should go to something bigger. The threshold
          is checked before the call list, so an unconfident refusal escalates
          rather than standing as an answer.
        - "reported": `execute` was False, so the calls came back unperformed.
        - "max_steps": the budget ran out *while the model was still calling
          tools*. The work is unfinished. This loop does not retry, does not
          summarise, and does not pretend the last step was an answer, because
          silently returning a partial trace as if it were complete is how an
          agent ends up half-performing an action nobody can see it stopped in
          the middle of. Check `exhausted`, or raise on it.

        Args:
            max_steps: overrides the instance budget for this request.
            execute: when False, calls are reported and not performed, and the
              loop stops after the first step because there is no result to
              feed back.
        """
        limit = self.max_steps if max_steps is None else max_steps
        if limit < 1:
            raise ValueError(f"max_steps must be at least 1, got {limit}")

        self.messages.append({"role": "user", "content": query})
        steps: list[dict[str, Any]] = []
        made: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        # Only one path leaves this alone: falling out of the loop having used
        # every step with calls still outstanding.
        stop_reason = "max_steps"

        for _ in range(limit):
            response = self._step(self.messages, None)
            steps.append(response)
            # The raw turn, not the tidied one: the next prompt has to contain
            # what the model actually wrote, markers and all, or the format it
            # is continuing is not the format it was supervised on.
            self.messages.append({"role": "assistant", "content": response["raw"]})

            if self.threshold > 0.0:
                confidence = response["confidence"]
                if confidence is None:
                    raise RuntimeError(
                        "threshold is set but this checkpoint carries no Gauge "
                        "head; load a checkpoint with heads, or pass threshold=0")
                if confidence < self.threshold:
                    stop_reason = "escalate"
                    break

            if not response["function_calls"]:
                stop_reason = "done"
                break

            made.extend(response["function_calls"])
            if not execute:
                # Nothing was performed, so there is no result to feed back and
                # a second step would ask the same question of the same prompt.
                stop_reason = "reported"
                break

            batch = [self._execute(call) for call in response["function_calls"]]
            results.extend(batch)
            self.messages.append({"role": "tool", "content": batch})

        last = steps[-1] if steps else {"text": "", "confidence": None}
        return {
            "function_calls": made,
            "results": results,
            "steps": steps,
            "text": last["text"],
            "confidence": last["confidence"],
            "messages": list(self.messages),
            "stop_reason": stop_reason,
            "exhausted": stop_reason == "max_steps",
        }

    def _execute(self, call: Mapping[str, Any]) -> dict[str, Any]:
        """Perform one call, turning any failure into a result the model reads.

        A tool that raises is information: the model can pick a different tool
        or a different argument next step. A tool that raises *through* the loop
        is a crash on a device, so the exception is recorded and the turn
        continues.
        """
        name = call.get("name", "")
        arguments = call.get("arguments") or {}
        out: dict[str, Any] = {"name": name, "arguments": arguments}
        fn = self._callables.get(name)
        if fn is None:
            out["error"] = (f"no function registered for {name!r}; it was "
                            f"declared but not given a callable")
            return out
        try:
            out["result"] = fn(**arguments)
        except Exception as exc:                          # noqa: BLE001
            out["error"] = f"{type(exc).__name__}: {exc}"
        return out

    # --- the turn itself -------------------------------------------------
    def _step(self, messages: Sequence[Mapping[str, Any]],
              declared: list[dict[str, Any]] | None, *,
              max_new_tokens: int | None = None,
              constrain: bool | None = None) -> dict[str, Any]:
        """Render, generate, parse. Every path into the model comes through here."""
        self._ensure_loaded()
        query = next((m["content"] for m in messages if m["role"] == "user"), "")
        tools, how = self._tools_for(str(query), declared)

        prompt = self._render(messages, tools)
        prompt_ids = self._encode(prompt, bos=True)

        window = self.window or self._config.effective_window()
        pinned = self._pinned(prompt, window)
        budget = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        if not budget:
            # The rotating part of the cache is what a reply can occupy: the
            # pinned schemas hold their slots, and the prompt beyond them has
            # already scrolled through. The hard stop is the rotary positions
            # the model was trained with, which the decoder refuses to exceed,
            # so it is respected here rather than discovered there.
            budget = max(1, min(window - pinned,
                                self._config.max_seq_len - len(prompt_ids)))

        use_trellis = self.constrain if constrain is None else constrain
        new_ids, extra = self._generate(prompt_ids, tools, budget, use_trellis,
                                        pinned)
        text = self._decode(new_ids)
        calls, error = _parse_calls(text)

        return {
            "text": _strip_markers(text),
            "raw": text,
            "reasoning": _between(text, THINK_START, THINK_END),
            "function_calls": calls,
            "confidence": _confidence(extra, self._gauge_temperature),
            "tools": [t["name"] for t in tools],
            "retrieval": how,
            "usage": {"prompt_tokens": len(prompt_ids),
                      "completion_tokens": len(new_ids)},
            "stop_reason": str(extra.get("stop_reason")
                               or ("length" if len(new_ids) >= budget else "stop")),
            "error": error,
        }

    def _render(self, messages: Sequence[Mapping[str, Any]],
                tools: Sequence[Mapping[str, Any]]) -> str:
        """Build the prompt, byte for byte the way the supervision was written.

        The first user turn goes through scribe.render, so the tools block, the
        separators and the assistant opener are identical to training rather
        than merely similar. Later turns are appended here, because render()
        describes one turn and a loop has several.
        """
        from .model.scribe import render

        first = next((i for i, m in enumerate(messages) if m["role"] == "user"), 0)
        head, _ = render({"tools": list(tools),
                          "query": str(messages[first]["content"]) if messages else ""})
        parts = [f"{IM_START}system\n{self.system}{IM_END}\n"] if self.system else []
        parts.append(head)

        for message in messages[first + 1:]:
            role, content = message["role"], message["content"]
            if role == "assistant":
                # The turn usually ends with its own marker already, since the
                # model was taught to close what it opened.
                body = str(content).rstrip()
                if body.endswith(IM_END):
                    body = body[: -len(IM_END)].rstrip()
                parts.append(f"{body}{IM_END}\n")
            elif role == "tool":
                body = json.dumps(content, separators=(",", ":"), default=str)
                parts.append(f"{IM_START}user\n{RESULT_START}{body}{RESULT_END}"
                             f"{IM_END}\n{_ASSISTANT}")
            else:
                parts.append(f"{IM_START}user\n{content}{IM_END}\n{_ASSISTANT}")

        prompt = "".join(parts)
        return prompt if prompt.endswith(_ASSISTANT) else prompt + _ASSISTANT

    def _pinned(self, prompt: str, window: int) -> int:
        """How many leading tokens are never evicted: the declared tools.

        The schemas are the longest part of the prompt and they arrive first, so
        a sliding window drops them first and the model is left calling tools it
        can no longer see. Pinning them costs nothing, because those positions
        were already in the cache. It costs one extra encode of the block per
        turn to find where it ends, which is cheaper than the alternative.
        """
        if TOOLS_END not in prompt:
            return 0
        head = prompt.split(TOOLS_END, 1)[0] + TOOLS_END
        # The rotating part of the cache has to keep at least one slot, or a
        # turn longer than its own schemas would have nowhere to go.
        return min(len(self._encode(head, bos=True)), max(0, window - 1))

    def _generate(self, prompt_ids: list[int], tools: Sequence[Mapping[str, Any]],
                  budget: int, constrain: bool,
                  pinned: int) -> tuple[list[int], dict[str, Any]]:
        """Run the decoder, with the tries rebuilt over this turn's tools."""
        from .model import decode

        decoder = self._decoder(tools) if constrain else None
        raw = _bind(decode.generate, self._params, self._dcfg, prompt_ids,
                    decoder=decoder, constraints=decoder, max_new_tokens=budget,
                    temperature=self.temperature, top_k=self.top_k, seed=self.seed,
                    pinned=pinned, stop_ids=_STOP_IDS, sp=self._sp,
                    tokenizer=self._sp, tools=list(tools))
        ids, extra = _unpack_generation(raw)
        if len(ids) >= len(prompt_ids) and ids[:len(prompt_ids)] == prompt_ids:
            ids = ids[len(prompt_ids):]        # some decoders return the whole row
        return ids, extra

    def _decoder(self, tools: Sequence[Mapping[str, Any]]) -> Any:
        """Build the constrained decoder for this turn.

        Rebuilt every turn on purpose: retrieval changes which five tools are
        declared, and a trie over last turn's five is a guarantee about the
        wrong names. It costs about 0.9 MB of tries and no measurable time next
        to a forward pass.

        It is armed on the call marker and disarmed on the closing one, so the
        walker sleeps through the reasoning line. That matters: the turn opens
        with `<think>...</think>`, which is prose, and prose contains the same
        `{"` and `,"` cues the walker treats as the start of an identifier. An
        always-armed decoder starts masking inside the reasoning against a trie
        of tool names, which is a guarantee applied to the one span it was never
        meant to cover.
        """
        from .model import trellis

        constraints = trellis.SchemaConstraints(list(tools))
        return _bind(trellis.ConstrainedDecoder, sp=self._sp, tokenizer=self._sp,
                     tools=list(tools), constraints=constraints,
                     token_strings=self._token_strings, index=self._token_index,
                     token_index=self._token_index,
                     arm_on=trellis.ARM_MARKER, disarm_on=trellis.DISARM_MARKER)

    # --- retrieval -------------------------------------------------------
    def _tools_for(self, query: str,
                   declared: list[dict[str, Any]] | None
                   ) -> tuple[list[dict[str, Any]], str]:
        """Choose the schemas this turn renders, and say how they were chosen.

        With no catalogue this is the declared list. With one, the declared
        tools are kept first (they are the ones wired to real functions) and the
        catalogue fills the remaining slots, because the window holds about five
        schemas and a sixth would evict the request.
        """
        tools = list(self.tools if declared is None else declared)
        if not self.tool_index_path:
            return tools, "declared"

        catalogue = self._load_catalogue()
        room = max(0, self.retrieve - len(tools))
        if room == 0:
            return tools[: self.retrieve], "declared"

        scores, how = self._score_catalogue(query, catalogue)
        order = sorted(range(len(catalogue)), key=lambda i: -scores[i])
        have = {t["name"] for t in tools}
        for i in order:
            if len(tools) >= self.retrieve:
                break
            if catalogue[i]["name"] not in have:
                tools.append(catalogue[i])
        return tools, how

    def _score_catalogue(self, query: str, catalogue: list[dict[str, Any]]
                         ) -> tuple[list[float], str]:
        """Score every tool in the catalogue against the request.

        Dowser when the index carries embeddings and the loaded checkpoint has
        the head; the lexical scorer otherwise. The fallback is reported rather
        than hidden, because it is a different mechanism with different recall:
        Dowser finds a tool from a paraphrase, the word overlap here finds one
        the user has half-named and misses one they only described.
        """
        embedding = (self._query_embedding(query)
                     if self._catalogue_emb is not None else None)
        if embedding is not None:
            import numpy as np

            mat = np.asarray(self._catalogue_emb, dtype=np.float32)
            vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
            vec = vec / max(float(np.linalg.norm(vec)), 1e-12)
            return (mat @ vec).tolist(), "dowser"

        words = set(_WORD_RE.findall(query.lower()))
        haystack = " ".join(_WORD_RE.findall(query.lower()))
        return [_lexical_score(words, haystack, t) for t in catalogue], "lexical"

    def _query_embedding(self, query: str) -> Any:
        """Ask the loaded engine for the Dowser embedding of one request.

        Returns None when the decoder exposes no retrieval entry point, which is
        the ordinary case for a checkpoint trained without stage zero. The
        caller then falls back to the lexical scorer and says so in the
        response, rather than retrieving nothing and refusing every request.
        """
        from .model import decode

        embed = getattr(decode, "dowser_embed", None)
        if embed is None:
            return None
        return _bind(embed, self._params, self._dcfg,
                     self._encode(query, bos=True), sp=self._sp, tokenizer=self._sp)

    def _load_catalogue(self) -> list[dict[str, Any]]:
        """Read the tool index once. .npz carries embeddings, .json does not."""
        if self._catalogue is not None:
            return self._catalogue

        path = Path(self.tool_index_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"tool index {path} does not exist. It is a file you build: a "
                f".npz holding `schemas` (JSON strings) and optionally `emb` "
                f"(the Dowser vectors), or a .json/.jsonl list of schemas. Pass "
                f"tool_index_path=None to declare tools inline instead.")

        if path.suffix == ".npz":
            import numpy as np

            with np.load(path, allow_pickle=False) as data:
                raw = [str(s) for s in data["schemas"]]
                emb = (np.asarray(data["emb"], dtype=np.float32)
                       if "emb" in data else None)
            if emb is not None:
                norms = np.maximum(np.linalg.norm(emb, axis=-1, keepdims=True), 1e-12)
                emb = emb / norms
            self._catalogue_emb = emb
            self._catalogue = [build_schema(json.loads(s)) for s in raw]
        else:
            text = path.read_text(encoding="utf-8")
            rows = ([json.loads(line) for line in text.splitlines() if line.strip()]
                    if path.suffix == ".jsonl" else json.loads(text))
            self._catalogue = [build_schema(r) for r in rows]
            self._catalogue_emb = None
        return self._catalogue

    # --- loading ---------------------------------------------------------
    def _ensure_loaded(self) -> None:
        """Load weights, geometry and tokenizer once, on the first turn.

        Construction stays free so a service can build a Quartz at start-up and
        pay the megabytes only if a request ever arrives.
        """
        if self._params is not None:
            return

        path = self._resolve_weights()
        if path.suffix == ".ingot":
            raise RuntimeError(
                f"{path} is an Ingot, which is the device container: its "
                f"directory carries no tensor names, and a runtime finds each "
                f"tensor by position in an order it recomputes from the config. "
                f"Python inference loads a checkpoint instead, so pass weights= "
                f"a .pkl from `quartz pretrain`, `sft` or `heads`. To walk the "
                f"container itself, use quartz.model.ingot.read_ingot with "
                f"canonical_order.")

        from .model import run as checkpoints

        bundle = checkpoints.load_checkpoint(str(path))
        params, config, tokenizer = _unpack_bundle(bundle)
        meta = bundle.get("meta") or {} if isinstance(bundle, Mapping) else {}
        self._gauge_temperature = float(meta.get("gauge_temperature",
                                                 GAUGE_TEMPERATURE))
        if params is None:
            raise RuntimeError(f"{path} carried no weights")
        self._params = params
        self._config = self._config or _coerce_config(config)
        if self._config is None:
            raise RuntimeError(
                f"{path} carries no geometry; pass config='configs/base.yaml', "
                f"config='base', or a QuartzConfig")
        self._sp = self._sp or tokenizer or self._load_tokenizer()

        from .model import decode, trellis

        self._token_strings = trellis.build_token_strings(self._sp)
        self._token_index = trellis.TokenIndex(self._token_strings)
        self._dcfg = _bind(decode.decode_cfg, self._config,
                           window=self.window or self._config.effective_window(),
                           temperature=self.temperature, top_k=self.top_k,
                           seed=self.seed)

    def _resolve_weights(self) -> Path:
        """The checkpoint to load, fetching the release when none was named."""
        if not self.weights:
            from .agent.fetch import fetch_weights

            return fetch_weights()
        path = Path(self.weights).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"no weights at {path}; pass weights= a checkpoint, set "
                f"QUARTZ_WEIGHTS, or leave it unset to fetch the release")
        return path

    def _load_tokenizer(self) -> Any:
        """The tokenizer, when the checkpoint did not carry one inline."""
        from .model import scribe

        try:
            return _bind(scribe.get_tokenizer)
        except (TypeError, FileNotFoundError) as exc:
            raise RuntimeError(
                "no tokenizer: this checkpoint carries none inline and no "
                "scribe.model could be found. Pass tokenizer= a loaded Scribe, "
                "or train one with `quartz train-tokenizer`.") from exc

    # --- text in, text out ------------------------------------------------
    def _encode(self, text: str, *, bos: bool = False) -> list[int]:
        """Encode raw text, applying the isolation rule exactly once.

        Scribe.encode takes text that has already been through the rule, and
        Scribe.encode_wire takes raw text and applies it. Applying it twice
        doubles every space the tokenizer already models, so which one is being
        held decides, and a bare SentencePiece processor gets the rule applied
        here.
        """
        wire = getattr(self._sp, "encode_wire", None)
        if wire is not None:
            return list(wire(text, bos=bos))

        from .model import scribe

        ids = [int(i) for i in self._sp.encode(scribe.pre_tokenize(text))]
        return [BOS_ID, *ids] if bos else ids

    def _decode(self, ids: Sequence[int]) -> str:
        """Ids back to text. pre_tokenize is never inverted here: the JSON is
        recovered at parse time, where the string boundaries are known."""
        return str(self._sp.decode(list(ids)))


# --- parsing ---------------------------------------------------------------
def _parse_calls(text: str) -> tuple[list[dict[str, Any]], str | None]:
    """Read the emitted turn as a list of calls.

    Returns `(calls, error)` and never raises. A device caller needs a decision,
    and "the model wrote something I could not read" is a decision: it is an
    empty call list with a reason attached, which is the same shape as a refusal
    and can be handled in one branch.
    """
    body = text.split(CALL_START, 1)[1] if CALL_START in text else text
    body = body.split(CALL_END, 1)[0]
    body = _restore_json(_strip_markers(body)).strip()
    if not body:
        return [], None

    parsed, error = _loads(body)
    if error is not None:
        span = _outermost(body)
        if span is None:
            return [], error
        parsed, error = _loads(span)
        if error is not None:
            return [], error

    if isinstance(parsed, Mapping):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return [], f"expected a list of calls, got {type(parsed).__name__}"

    calls: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
            return calls, "a call had no name"
        arguments = item.get("arguments", item.get("parameters", {}))
        if isinstance(arguments, str):
            arguments, _ = _loads(arguments)          # some turns nest the args
        if not isinstance(arguments, Mapping):
            arguments = {}
        calls.append({"name": item["name"], "arguments": dict(arguments)})
    return calls, None


def _restore_json(text: str) -> str:
    """Undo the tokenizer's isolation rule.

    pre_tokenize puts a space either side of each of the eight structural
    characters before the trainer ever sees the text, so the model emits
    `{ " room " : " kitchen " }` and this puts it back. One space is removed on
    each side of each character, which is exactly what was added.

    It is not perfectly invertible inside a string value: `"late, sorry"` was
    written `" late ,  sorry "`, the tokenizer collapsed the double space, and
    what comes back is `"late,sorry"`. The lost space is the price of a merge
    table that can never swallow a brace, and it is paid on punctuation inside
    values, not on structure or on identifiers.
    """
    return _ISOLATED_RE.sub(r"\1", text)


def _loads(text: str) -> tuple[Any, str | None]:
    try:
        return json.loads(text), None
    except (ValueError, TypeError) as exc:
        return None, f"unparsable JSON: {exc}"


def _outermost(text: str) -> str | None:
    """The widest bracketed span in the text, for a turn with prose around it."""
    for opener, closer in (("[", "]"), ("{", "}")):
        start, end = text.find(opener), text.rfind(closer)
        if 0 <= start < end:
            return text[start:end + 1]
    return None


def _between(text: str, start: str, end: str) -> str:
    if start not in text:
        return ""
    tail = text.split(start, 1)[1]
    return tail.split(end, 1)[0].strip() if end in tail else tail.strip()


def _strip_markers(text: str) -> str:
    """Remove the chat markers, leaving what the model actually wrote."""
    for marker in CHAT_MARKERS:
        text = text.replace(marker, "")
    return text.strip()


def _confidence(extra: Mapping[str, Any], temperature: float) -> float | None:
    """The Gauge reading, calibrated when what came back is a raw logit.

    A head that reports 0.7 and is right 63 percent of the time is worse than no
    head at all, because a threshold set on it means something other than what
    it says. The temperature is applied here, once, at the boundary, and only to
    a logit: a decoder that hands back a number it already calls a confidence
    has made that decision itself and is not second-guessed.
    """
    if extra.get("confidence") is not None:
        return float(extra["confidence"])
    logit = extra.get("gauge_logit", extra.get("gauge"))
    if logit is None:
        return None
    return 1.0 / (1.0 + math.exp(-float(logit) / temperature))


def _lexical_score(words: set[str], haystack: str, schema: Mapping[str, Any]) -> float:
    """Word overlap between a request and one tool, normalised by tool length.

    A stand-in for Dowser, not a replacement: it scores the name, the
    description and the argument keys, and the square root keeps a tool with a
    long description from winning on volume.
    """
    properties = (schema.get("parameters") or {}).get("properties") or {}
    text = " ".join([str(schema.get("name", "")),
                     str(schema.get("description", "")), *properties])
    tokens = set(_WORD_RE.findall(text.lower()))
    if not tokens:
        return 0.0
    score = len(words & tokens) / math.sqrt(len(tokens))
    spelled = str(schema.get("name", "")).lower().replace("_", " ")
    return score + (1.0 if spelled and spelled in haystack else 0.0)


# --- talking to the rest of the package ------------------------------------
def _bind(fn: Callable[..., Any], *positional: Any, **candidates: Any) -> Any:
    """Call `fn` with `positional` up front and the rest matched by name.

    decode, trellis, scribe and run are separate modules with their own natural
    signatures. The leading arguments of a decoder are order-stable (weights,
    geometry, prompt) but its options are not, so the options are offered by
    name and whatever `fn` does not declare is dropped. A required parameter
    that nothing can fill raises here, beside the call, instead of arriving as a
    bare TypeError three frames into someone else's module.
    """
    params = inspect.signature(fn).parameters
    order = [n for n, p in params.items()
             if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    taken, tail = order[:len(positional)], order[len(positional):]
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        kwargs = {k: v for k, v in candidates.items() if k not in taken}
    else:
        kwargs = {k: candidates[k] for k in tail if k in candidates}

    missing = [k for k in tail
               if k not in kwargs and params[k].default is inspect.Parameter.empty]
    if missing:
        where = f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', fn)}"
        raise TypeError(f"{where} requires {missing}, which quartz cannot supply; "
                        f"its signature has moved away from the package contract")
    return fn(*positional, **kwargs)


def _unpack_generation(raw: Any) -> tuple[list[int], dict[str, Any]]:
    """Pull the new token ids, and anything else, out of a decoder's return."""
    if isinstance(raw, Mapping):
        for key in ("new_tokens", "tokens", "ids", "sequence", "output"):
            if key in raw:
                rest = {k: v for k, v in raw.items() if k != key}
                return _as_ids(raw[key]), rest
        raise TypeError(f"decode.generate returned no tokens, keys {sorted(raw)}")
    if isinstance(raw, tuple):
        extra = next((x for x in raw[1:] if isinstance(x, Mapping)), {})
        return _as_ids(raw[0]), dict(extra)
    return _as_ids(raw), {}


def _as_ids(values: Any) -> list[int]:
    """A row of ids as plain ints, whatever array type it arrived as."""
    return [int(v) for v in values]


def _unpack_bundle(bundle: Any) -> tuple[Any, Any, Any]:
    """Pull (params, config, tokenizer) out of whatever a loader returned.

    load_checkpoint returns a mapping and is read by key; a tuple is sniffed by
    type, for a caller who built the bundle by hand. The step counter and any
    optimiser state beside the weights are dropped here, which is correct:
    inference has no use for them and they are usually the larger half.
    """
    params = config = tokenizer = None
    if isinstance(bundle, Mapping):
        params = next((bundle[k] for k in ("params", "weights", "tensors")
                       if k in bundle), bundle)
        config = next((bundle[k] for k in ("config", "cfg")
                       if bundle.get(k) is not None), None)
        tokenizer = next((bundle[k] for k in ("tokenizer", "sp", "scribe")
                          if bundle.get(k) is not None), None)
    elif isinstance(bundle, (tuple, list)):
        for item in bundle:
            if isinstance(item, QuartzConfig) or _looks_like_config(item):
                config = item
            elif _looks_like_tokenizer(item):
                tokenizer = item
            elif params is None:
                params = item
    else:
        params = bundle
    return params, config, tokenizer


def _looks_like_config(item: Any) -> bool:
    return isinstance(item, Mapping) and {"d_model", "num_layers"} <= set(item)


def _looks_like_tokenizer(item: Any) -> bool:
    return (callable(getattr(item, "encode", None))
            and callable(getattr(item, "decode", None)))


def _coerce_config(config: Any) -> QuartzConfig | None:
    """A QuartzConfig from a config, a preset name, a YAML path or a dict."""
    if config is None or isinstance(config, QuartzConfig):
        return config
    if isinstance(config, Mapping):
        return QuartzConfig(**config)
    text = str(config)
    if text in PRESETS:
        return preset(text)
    if text.endswith((".yaml", ".yml")):
        path = Path(text).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"no config at {path}")
        return QuartzConfig.from_yaml(str(path))
    raise ValueError(
        f"cannot read {config!r} as a config; pass a QuartzConfig, a preset name "
        f"({sorted(PRESETS)}), a path to a .yaml, or a dict")
