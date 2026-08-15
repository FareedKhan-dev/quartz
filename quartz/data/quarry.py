"""Quarry: asking a very large open model for 1.2 million pieces of homework.

The base model completes text and has seen tool traces at eight percent of
pretraining, but never a labelled call at the density stage two needs. Nobody
hands out 1.2 million labelled tool calls across 41,000 schemas, so an open
model writes ours, behind an OpenAI-compatible server on localhost::

    python -m vllm.entrypoints.openai.api_server \\
      --model Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \\
      --tensor-parallel-size 4 --max-model-len 8192 \\
      --gpu-memory-utilization 0.92 --port 8000

Every clause of TEMPLATE is load bearing, and one of them more than the rest:
**"containing only values evidenced in the query"**. A model trained on invented
values learns to invent them. A wrong tool is obvious to a user; a hallucinated
phone number is not. That clause is asked for in the prompt and then enforced
here in `validate`, because a rule that is only requested is a rule that is only
mostly followed.

The client is `urllib` on purpose. Talking to an OpenAI-compatible endpoint is
one POST with a JSON body, and this package ships with three dependencies; the
teacher is a build-time tool and should not add a fourth.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

#: The local vLLM server the post runs against. Anything speaking the
#: OpenAI chat completions API works; nothing here is provider specific.
DEFAULT_ENDPOINT = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"

#: Three off-topic inputs per batch of twenty five is where the one in eight
#: refusal share comes from. At zero the model calls a tool for 96.2 percent of
#: nonsense; at one in eight, 4.1 percent, for half a point of real accuracy.
BATCH_SIZE = 25
REFUSALS_PER_BATCH = 3

#: Thirty percent more batches than the arithmetic needs, because deduplication
#: eats about a quarter of what a generator at temperature 0.9 produces.
OVERSAMPLE = 1.3

#: How many schemas one prompt sees. Small enough that the teacher can hold all
#: of them in the request, large enough for multi-call examples to be possible.
TOOLS_PER_PROMPT = 8

SYSTEM = (
    "You write training data for a small model whose entire output is a "
    "function call. Answer with JSON and nothing else: no prose, no markdown "
    "fences, no commentary. Every argument you write must be spelled exactly as "
    "the schema declares it and must be evidenced in the query you wrote for it."
)

TEMPLATE = """Schemas available (JSON):
{tools}

Produce {n} varied examples as a JSON array, each with a "query", a one-line
"reasoning" deriving every argument from its source span, and "answers".

Rules:
- Use only the schemas above, with arguments matching them exactly and
  containing only values evidenced in the query.
- Cover single-call, multi-call, and about {refusals} off-topic inputs no
  schema can serve, whose "answers" is [].
- Vary phrasing. Return ONLY the array."""

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)
_WORD = re.compile(r"[a-z0-9]+")
_RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

#: Numbers a request spells out rather than digits. A timer set for "eight
#: minutes" carries the argument 8, and a grounding check that only looked for
#: the character '8' would throw the example away.
_NUMBER_WORDS: dict[str, int] = {
    "zero": 0, "one": 1, "a": 1, "an": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}


# --- the client ------------------------------------------------------------
class TeacherError(RuntimeError):
    """The endpoint refused, timed out, or answered with something unusable."""


class Client:
    """A minimal OpenAI-compatible chat client over urllib.

    Retries on the statuses a busy inference server actually returns (429 and
    the 5xx family) and on transport errors, with exponential backoff. It does
    not retry a 400: a malformed request will be malformed the second time too,
    and a generation job that quietly retries its own bugs for eleven hours is
    worse than one that stops.
    """

    def __init__(self, endpoint: str = DEFAULT_ENDPOINT, model: str = DEFAULT_MODEL,
                 *, api_key: str = "", timeout: float = 600.0, retries: int = 4,
                 backoff: float = 2.0) -> None:
        self.url = endpoint.rstrip("/") + "/chat/completions"
        self.model = model
        # vLLM ignores the key but the header has to be there for some proxies.
        self.api_key = (api_key or os.environ.get("QUARTZ_TEACHER_KEY")
                        or os.environ.get("OPENAI_API_KEY") or "EMPTY")
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff

    def chat(self, messages: Sequence[Mapping[str, str]], *, temperature: float = 0.9,
             max_tokens: int = 4096, seed: int | None = None) -> str:
        """One completion, as text. Raises TeacherError once retries run out."""
        payload: dict[str, Any] = {
            "model": self.model, "messages": list(messages),
            "temperature": temperature, "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = int(seed)
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.url, data=body, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})

        last = ""
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as fh:
                    answer = json.loads(fh.read().decode("utf-8"))
                return str(answer["choices"][0]["message"]["content"])
            except urllib.error.HTTPError as exc:
                last = f"HTTP {exc.code}: {exc.read()[:200].decode('utf-8', 'replace')}"
                if exc.code not in _RETRY_STATUS:
                    raise TeacherError(f"{self.url} rejected the request, {last}") from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = f"{type(exc).__name__}: {exc}"
            except (KeyError, IndexError, ValueError) as exc:
                last = f"unreadable reply, {type(exc).__name__}: {exc}"
            if attempt < self.retries:
                time.sleep(self.backoff * (2 ** attempt))
        raise TeacherError(
            f"{self.url} failed {self.retries + 1} times, last was {last}. Is the "
            f"teacher serving? See the vLLM command in this module's docstring.")


# --- reading what came back ------------------------------------------------
def parse_examples(text: str) -> list[dict[str, Any]]:
    """Pull the JSON array out of a reply, fences and preamble included.

    Asking for "ONLY the array" gets it most of the time. The rest of the time
    the array is wrapped in a code fence or introduced by a sentence, and
    throwing away a good batch of twenty five examples over a stray "Here you
    go:" is expensive at 1.2 million examples.
    """
    body = _FENCE.sub("", text).strip()
    start, end = body.find("["), body.rfind("]")
    if start < 0 or end < start:
        return []
    try:
        rows = json.loads(body[start:end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict) and "query" in r]


def _normalise(text: str) -> str:
    return " ".join(_WORD.findall(str(text).lower()))


def _number_in(value: float, query: str) -> bool:
    """Is a number evidenced, as digits or as a word?"""
    digits = f"{value:g}"
    if digits in query.replace(",", ""):
        return True
    words = set(_WORD.findall(query.lower()))
    return any(_NUMBER_WORDS.get(w) == value for w in words)


def ungrounded(example: Mapping[str, Any],
               schemas: Mapping[str, Mapping[str, Any]]) -> list[tuple[str, Any]]:
    """Every argument whose value is not evidenced in the query.

    An enum value is exempt. It is a declared choice rather than a copied span,
    so "set it to eco" may legitimately arrive as the enum member `economy`,
    and the constrained decoder guarantees the spelling anyway. Everything else
    -- names, places, messages, numbers -- has to be findable in the request,
    which is the difference between a model that copies and a model that
    invents.
    """
    query = _normalise(example.get("query", ""))
    raw = str(example.get("query", ""))
    missing: list[tuple[str, Any]] = []
    for call in example.get("answers") or []:
        if not isinstance(call, Mapping):
            continue
        spec = schemas.get(str(call.get("name", "")), {})
        props = (spec.get("parameters") or {}).get("properties", {})
        for key, value in (call.get("arguments") or {}).items():
            field = props.get(key, {}) if isinstance(props, Mapping) else {}
            if isinstance(field, Mapping) and value in (field.get("enum") or []):
                continue
            if isinstance(value, bool) or value is None:
                continue          # a flag is implied by phrasing, not quoted
            if isinstance(value, int | float):
                if not _number_in(float(value), raw):
                    missing.append((key, value))
            elif isinstance(value, str):
                if _normalise(value) not in query:
                    missing.append((key, value))
            else:
                missing.append((key, value))     # nested values are not copied
    return missing


def validate(example: Mapping[str, Any],
             schemas: Mapping[str, Mapping[str, Any]]) -> str:
    """Return "" when the example is usable, or why it is not.

    The teacher is asked for all of this in the prompt. It is checked here
    because the difference between asking and enforcing is a few percent of
    1.2 million examples, and every one of those percent is a lesson the small
    model would otherwise learn.
    """
    if not str(example.get("query", "")).strip():
        return "empty query"
    answers = example.get("answers")
    if not isinstance(answers, list):
        return "answers is not a list"
    for call in answers:
        if not isinstance(call, Mapping):
            return "a call is not an object"
        name = str(call.get("name", ""))
        if name not in schemas:
            return f"undeclared tool {name!r}"
        arguments = call.get("arguments")
        if not isinstance(arguments, Mapping):
            return f"{name} has no arguments object"
        params = schemas[name].get("parameters") or {}
        props = params.get("properties") or {}
        unknown = [k for k in arguments if k not in props]
        if unknown:
            return f"{name} has undeclared arguments {unknown}"
        absent = [k for k in params.get("required") or [] if k not in arguments]
        if absent:
            return f"{name} is missing required {absent}"
    if answers and (bad := ungrounded(example, schemas)):
        return f"values not evidenced in the query: {bad}"
    return ""


def dedup_key(example: Mapping[str, Any]) -> tuple[str, str]:
    """One phrasing can legitimately map to two calls, so the key is request
    and call together."""
    return (example.get("query", "").strip().lower(),
            json.dumps(example.get("answers", []), sort_keys=True))


# --- one batch -------------------------------------------------------------
def _schema_name(schema: Mapping[str, Any]) -> str:
    return str(schema.get("name", ""))


def one_batch(tools: Sequence[Mapping[str, Any]], n: int = BATCH_SIZE, *,
              client: Client | None = None, refusals: int = REFUSALS_PER_BATCH,
              temperature: float = 0.9, seed: int | None = None,
              tools_per_prompt: int = TOOLS_PER_PROMPT,
              max_tokens: int = 4096) -> list[dict[str, Any]]:
    """Sample a few schemas, ask for `n` examples, keep the ones that check out.

    The schema sample is drawn from `seed`, so batch 91,415 of a rerun sees the
    same eight tools it saw the first time and a failed run can be resumed
    rather than restarted.
    """
    client = client or Client()
    rng = random.Random(seed)
    picked = (list(tools) if len(tools) <= tools_per_prompt
              else rng.sample(list(tools), tools_per_prompt))
    schemas = {_schema_name(t): t for t in picked}
    prompt = TEMPLATE.format(
        tools=json.dumps(picked, separators=(",", ":"), ensure_ascii=False),
        n=n, refusals=refusals)
    text = client.chat(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        temperature=temperature, max_tokens=max_tokens, seed=seed)

    kept: list[dict[str, Any]] = []
    for example in parse_examples(text):
        if validate(example, schemas):
            continue
        kept.append({
            "query": str(example["query"]).strip(),
            "reasoning": str(example.get("reasoning", "")).strip(),
            "answers": example.get("answers") or [],
            "tools": picked,
        })
    return kept


# --- the run ---------------------------------------------------------------
def generate(tools: Sequence[Mapping[str, Any]], n_target: int, batch: int = BATCH_SIZE,
             refusals: int = REFUSALS_PER_BATCH, workers: int = 16,
             temperature: float = 0.9, endpoint: str = DEFAULT_ENDPOINT,
             model: str = DEFAULT_MODEL, seed: int = 0, *,
             oversample: float = OVERSAMPLE, timeout: float = 600.0,
             retries: int = 4, tools_per_prompt: int = TOOLS_PER_PROMPT,
             progress: bool = True) -> list[dict[str, Any]]:
    """Generate `n_target` deduplicated examples, concurrently.

    Thirty percent more batches are queued than the arithmetic needs, because
    a generator asked for variety at temperature 0.9 converges on the same few
    phrasings and about a quarter of what it writes is a duplicate. Requests are
    concurrent because a 235 billion parameter teacher with 22 billion active
    per token is throughput bound, not latency bound: sixteen in flight keeps
    the server's batcher full.

    The keys are `dedup_key`, so a phrasing that legitimately maps to two
    different calls survives as two examples, and the same phrasing mapping to
    the same call five times survives as one.

    Returns the examples, at most `n_target` of them, each `{query, reasoning,
    answers, tools}` ready for `quartz.model.scribe.render`.
    """
    tools = list(tools)
    if not tools:
        raise ValueError("quarry needs a tool catalogue; --schemas was empty")
    if n_target < 1:
        raise ValueError(f"n_target must be positive, got {n_target}")

    n_batches = max(1, int(n_target / max(1, batch) * oversample))
    client = Client(endpoint, model, timeout=timeout, retries=retries)
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    produced = failed = 0

    def run(index: int) -> list[dict[str, Any]] | None:
        """None is a dead request; an empty list is a batch nothing survived."""
        try:
            return one_batch(tools, batch, client=client, refusals=refusals,
                             temperature=temperature, seed=seed + index,
                             tools_per_prompt=tools_per_prompt)
        except TeacherError as exc:
            # One dead batch out of sixty thousand is weather. The count is
            # reported at the end so a run that is failing all of them is loud.
            print(f"  batch {index} failed: {exc}", file=sys.stderr)
            return None

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = pool.map(run, range(n_batches))
        try:
            for out in results:
                if out is None:
                    failed += 1
                    continue
                produced += len(out)
                for example in out:
                    rows.setdefault(dedup_key(example), example)
                if progress:
                    print(f"  generated {min(len(rows), n_target):,}/{n_target:,}  "
                          f"failed {failed}", end="\r", file=sys.stderr, flush=True)
                if len(rows) >= n_target:
                    break
        finally:
            # Stop early without waiting on the batches that never started:
            # closing the iterator cancels every future still queued.
            close = getattr(results, "close", None)
            if close is not None:
                close()

    kept = list(rows.values())[:n_target]
    if progress:
        duplicates = 1.0 - len(rows) / max(1, produced)
        empty = sum(1 for r in kept if not r["answers"])
        print(f"\n  {'generated':<9} {len(kept):,}/{n_target:,}  failed {failed}")
        print(f"  {'kept':<9} {len(kept):,} of {produced:,} produced  "
              f"({duplicates:.1%} duplicates)")
        print(f"  {'refusals':<9} {empty:,}  ({empty / max(1, len(kept)):.1%})")
        print(f"  {'schemas':<9} {len(tools):,}")
    return kept


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: str) -> int:
    """One example a line, which is what `quartz.train.sft.load_jsonl` reads."""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with file.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    return written


def read_schemas(path: str) -> list[dict[str, Any]]:
    """The tool catalogue, from a `.json` list or a `.jsonl` of schemas."""
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(
            f"no tool catalogue at {file}. It is a JSON list of "
            f"{{'name', 'description', 'parameters'}} schemas.")
    text = file.read_text(encoding="utf-8")
    rows: Iterator[Any]
    if file.suffix == ".jsonl":
        rows = (json.loads(line) for line in text.splitlines() if line.strip())
    else:
        payload = json.loads(text)
        rows = iter(payload if isinstance(payload, list) else [payload])
    return [r for r in rows if isinstance(r, dict) and "name" in r]
