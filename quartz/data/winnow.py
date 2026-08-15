"""Winnow: four sources, one filter, and rows that close their own seams.

The corpus is where the priors come from, and its shape is unusual: a fifth of
it is structured data on purpose. Twelve percent on JSON Schema and OpenAPI
looks eccentric until you remember what this model does. It has to know that
`"required"` is a list and that `"type": "integer"` forbids a decimal point, and
none of that is in web text at any density. The eight percent of synthetic tool
traces is the target distribution itself, so stage two refines rather than
introduces.

Three things happen here, in order:

- `clean` drops a document rather than repairing it. A corpus is large and a
  filter is cheap, so anything doubtful goes.
- `pack` concatenates documents into fixed rows and gives every token a segment
  id, with 0 for padding, so `make_packing_mask` can refuse the one attention
  edge packing would otherwise invent: the first token of document two looking
  back at the last token of document one.
- `build` mixes the four sources to their shares, encodes with Scribe, and
  streams the rows to one `.npy` a training run can memory-map.

Expected layout under `--sources`, one directory per key of `MIX`::

    data/sources/fineweb-edu/*.jsonl[.gz]     one record a line, "text" field
    data/sources/the-stack-v2/*.jsonl[.gz]    or "content" / "code"
    data/sources/schemas/*.json               a list of records, or one record
    data/sources/synth-tools/*.txt[.gz]       one document a LINE

Nothing here imports JAX. Packing is a numpy job and the tokenizer is
SentencePiece, so a corpus can be built on a machine with no accelerator in it.
"""
from __future__ import annotations

import gzip
import itertools
import json
import random
import struct
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from quartz.model.config import EOS_ID, PAD_ID, QuartzConfig

#: The mix, by share of the token budget. The comments are the reason each one
#: is here at all; the numbers are the argument of the post.
MIX: dict[str, float] = {
    "fineweb-edu":  0.62,   # the general language floor
    "the-stack-v2": 0.18,   # nesting and matched delimiters
    "schemas":      0.12,   # JSON Schema, OpenAPI, TOML, .proto
    "synth-tools":  0.08,   # the target's own shape
}

#: The budget of the shipped run. 120 billion tokens, one fifth structured.
TOTAL_TOKENS = 120_000_000_000

if abs(sum(MIX.values()) - 1.0) > 1e-9:          # a guard on the constant above
    raise ValueError(f"MIX shares sum to {sum(MIX.values())}, not 1.0")

#: Sources whose whole value is that their delimiters match. An unbalanced
#: sample from one of these teaches the opposite of what it was mixed in for.
STRUCTURED = frozenset({"the-stack-v2", "schemas", "synth-tools"})

#: Sources whose documents are prose, and are therefore allowed to be judged on
#: line length. Code has short lines and is not a link farm.
PROSE = frozenset({"fineweb-edu"})

#: Which field of a JSON record holds the document, in order of preference.
TEXT_FIELDS = ("text", "content", "code", "body", "document")

# --- what the filter drops -------------------------------------------------
MIN_CHARS = 200               # anything shorter is a title or a nav crumb
MAX_REPLACEMENT = 0.002       # U+FFFD above this is a decoding failure, not text
MAX_NONPRINT = 0.01           # control bytes mean a binary that reached the shard
MIN_MEAN_LINE = 12.0          # prose only: shorter means a list of links
MAX_DUPLICATE_LINES = 0.30    # boilerplate repeated down the page
MAX_DEPTH = 64                # a nesting depth no real schema reaches

_KEEP_CONTROL = {"\n", "\t"}
_OPEN, _CLOSE = "([{", ")]}"
_PAIR = {")": "(", "]": "[", "}": "{"}


def plan(total: int | float = TOTAL_TOKENS) -> dict[str, int]:
    """Token budget per source, summing to exactly `total`.

    Largest remainder rather than rounding each share on its own, so the parts
    add up to the whole and the report cannot print a total that disagrees with
    its own column.
    """
    total = int(total)
    exact = {k: total * v for k, v in MIX.items()}
    floors = {k: int(v) for k, v in exact.items()}
    short = total - sum(floors.values())
    order = sorted(MIX, key=lambda k: exact[k] - floors[k], reverse=True)
    for k in order[:short]:
        floors[k] += 1
    return floors


def _scaled(tokens: int, unit: float, suffix: str) -> str:
    if unit == 1.0:
        return f"{tokens:>9,} tokens"
    return f"{tokens / unit:>6.1f}{suffix} tokens"


def format_plan(total: int | float = TOTAL_TOKENS) -> str:
    """The mix table, as the post prints it.

    The unit follows the budget rather than being fixed at billions, so a ten
    thousand token smoke test does not report four sources of `0.0B`.
    """
    budget = plan(total)
    unit, suffix = ((1e9, "B") if total >= 1e9 else
                    (1e6, "M") if total >= 1e6 else (1.0, ""))
    lines = [f"  {k:<14} {MIX[k]:>6.1%}  {_scaled(budget[k], unit, suffix)}"
             for k in MIX]
    lines.append(f"  {'total':<14} {sum(MIX.values()):>6.1%}  "
                 f"{_scaled(sum(budget.values()), unit, suffix)}")
    return "\n".join(lines)


# --- the filter ------------------------------------------------------------
def balanced(text: str, max_depth: int = MAX_DEPTH) -> bool:
    """Do the brackets match, ignoring anything inside a double-quoted string?

    Only double quotes are honoured. A single quote is an apostrophe as often as
    it is a string delimiter, and guessing wrong would drop good documents; the
    formats in STRUCTURED all quote with `"` anyway.
    """
    stack: list[str] = []
    in_string = escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in _OPEN:
            stack.append(ch)
            if len(stack) > max_depth:
                return False
        elif ch in _CLOSE:
            opener = stack.pop() if stack else ""
            if opener != _PAIR[ch]:
                return False
    return not stack and not in_string


def clean(text: str, *, source: str = "", min_chars: int = MIN_CHARS) -> str | None:
    """Normalise the line endings and decide whether to keep the document.

    Returns the cleaned text, or None when the document is dropped. Rewriting is
    kept to line endings and runs of blank lines: everything else is a keep or a
    drop, because a filter that edits its input cannot later claim the model was
    trained on what the source actually said. That is also why the tokenizer
    runs with an identity normaliser -- the two rules have to agree.

    The tests differ by source. A prose document with a mean line of eight
    characters is a link farm; a source file with a mean line of eight
    characters is ordinary code. A structured document whose brackets do not
    close is the one thing the structured share was mixed in to prevent.
    """
    if not text:
        return None
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n")).strip("\n")
    while "\n\n\n" in text:                     # collapse gaps, keep paragraphs
        text = text.replace("\n\n\n", "\n\n")
    if len(text) < min_chars:
        return None

    n = len(text)
    if text.count("�") / n > MAX_REPLACEMENT:
        return None
    nonprint = sum(1 for ch in text if not ch.isprintable() and ch not in _KEEP_CONTROL)
    if nonprint / n > MAX_NONPRINT:
        return None

    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return None
    if len(lines) > 8:
        unique = len(set(lines))
        if 1.0 - unique / len(lines) > MAX_DUPLICATE_LINES:
            return None
    if source in PROSE and sum(len(ln) for ln in lines) / len(lines) < MIN_MEAN_LINE:
        return None
    if source in STRUCTURED and not balanced(text):
        return None
    return text


# --- the packer ------------------------------------------------------------
def pack(docs: Iterable[Sequence[int]], seq_len: int, *, add_eos: bool = True,
         keep_tail: bool = True) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Concatenate encoded documents into fixed rows, one segment id a token.

    Yields `(tokens, seg_ids)`, both `(seq_len,)` uint16. Segment ids start at 1
    and count up *within a row*; padding is token PAD_ID and segment 0, so a
    padded position can never satisfy `seg > 0` and therefore never contributes
    an attention row or a loss term. That is the whole contract with
    `make_packing_mask`, which turns these ids into the block-diagonal mask that
    stops document two attending to the end of document one.

    A document longer than a row is split across rows, and the continuation gets
    a fresh segment id rather than inheriting one. That is correct: whatever was
    left behind is in another row and no attention edge could reach it anyway,
    so pretending the two halves are one segment would only be a lie in the
    mask.

    Packing is worth doing at all because padding to the longest document wastes
    about a third of every batch. The hazard it brings is exactly the seam these
    ids close.
    """
    if seq_len < 2:
        raise ValueError(f"seq_len must be at least 2, got {seq_len}")
    row_t = np.full(seq_len, PAD_ID, dtype=np.uint16)
    row_s = np.zeros(seq_len, dtype=np.uint16)
    fill = seg = 0

    for doc in docs:
        ids = np.asarray(doc, dtype=np.uint16)
        if add_eos:
            ids = np.append(ids, np.uint16(EOS_ID))
        if ids.size == 0:
            continue
        pos = 0
        while pos < ids.size:
            take = min(seq_len - fill, ids.size - pos)
            seg += 1
            row_t[fill:fill + take] = ids[pos:pos + take]
            row_s[fill:fill + take] = seg
            fill += take
            pos += take
            if fill == seq_len:
                yield row_t.copy(), row_s.copy()
                row_t[:] = PAD_ID
                row_s[:] = 0
                fill = seg = 0
    if keep_tail and fill:
        yield row_t.copy(), row_s.copy()


# --- reading the shards ----------------------------------------------------
def _open(path: Path):
    """Open a shard, transparently ungzipping it. Undecodable bytes become
    U+FFFD, which `clean` then counts and drops on, rather than killing a build
    that is six hours in."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _field(record: Any) -> str | None:
    """The document inside one JSON record, or None if it holds no text."""
    if isinstance(record, str):
        return record
    if isinstance(record, dict):
        for key in TEXT_FIELDS:
            value = record.get(key)
            if isinstance(value, str) and value:
                return value
        return None
    return None


def read_shard(path: Path) -> Iterator[str]:
    """Documents out of one file, by extension.

    `.jsonl` is one record a line, `.json` is a list of records or a single one,
    and `.txt` is one document a *line* -- the same convention the tokenizer
    corpus uses, so the two stages can share a file.
    """
    name = path.name[:-3] if path.suffix == ".gz" else path.name
    with _open(path) as fh:
        if name.endswith(".jsonl"):
            for line in fh:
                if line.strip():
                    text = _field(json.loads(line))
                    if text:
                        yield text
        elif name.endswith(".json"):
            payload = json.load(fh)
            records = payload if isinstance(payload, list) else [payload]
            for record in records:
                text = _field(record)
                if text:
                    yield text
        else:
            for line in fh:
                if line.strip():
                    yield line.rstrip("\n")


def source_documents(directory: Path, source: str, *, min_chars: int = MIN_CHARS,
                     stats: dict[str, int] | None = None) -> Iterator[str]:
    """Every cleaned document under one source directory, shard by shard.

    Exact duplicates are dropped within a shard, which is the cheap nine tenths
    of the deduplication win: near-identical crawls of the same page usually
    arrive together. Global deduplication is a separate offline job and is not
    pretended at here, because a set over a hundred million documents does not
    fit beside the packer.
    """
    shards = sorted(p for p in directory.rglob("*") if p.is_file())
    if not shards:
        raise FileNotFoundError(
            f"no shards under {directory} for source {source!r}. Expected "
            f"*.jsonl, *.json or *.txt, optionally gzipped.")
    for shard in shards:
        seen: set[int] = set()
        for raw in read_shard(shard):
            kept = clean(raw, source=source, min_chars=min_chars)
            if stats is not None:
                stats["read"] = stats.get("read", 0) + 1
            if kept is None:
                continue
            key = hash(kept)
            if key in seen:
                if stats is not None:
                    stats["duplicate"] = stats.get("duplicate", 0) + 1
                continue
            seen.add(key)
            if stats is not None:
                stats["kept"] = stats.get("kept", 0) + 1
            yield kept


def source_dirs(root: str | Path) -> dict[str, Path]:
    """Locate one directory per key of MIX, or say exactly which is missing."""
    base = Path(root)
    found = {name: base / name for name in MIX}
    missing = [str(p) for p in found.values() if not p.is_dir()]
    if missing:
        raise FileNotFoundError(
            "winnow needs one directory per source, missing: "
            + ", ".join(missing)
            + f"\nExpected under {base}: " + ", ".join(MIX))
    return found


# --- mixing ----------------------------------------------------------------
class _Mixer:
    """Draw chunks of documents from the four sources, by remaining token budget.

    MIX is a share of *tokens*, not of documents, and the two are not the same:
    a web page is several times longer than a JSON schema, so drawing chunks at
    the fixed shares would hand the corpus far more web text than 62 percent.
    Each draw is therefore weighted by what is left of a source's budget, which
    starts proportional to its share and self-corrects as the real document
    lengths arrive.

    A source closes when its budget is spent or its shards run out, and the
    remaining weights renormalise by themselves, so an exhausted source degrades
    the mix instead of stalling the build. Token counts arrive from the encoder
    after the fact, so a source can overshoot by at most one look-ahead of
    chunks. On a 120 billion token run that is a rounding error; on a ten
    thousand token smoke test it is visible, and the report prints what was
    actually taken rather than what was asked for.
    """

    def __init__(self, streams: dict[str, Iterator[str]], budget: dict[str, int],
                 seed: int = 0) -> None:
        self.streams = dict(streams)
        self.budget = dict(budget)
        self.used: dict[str, int] = dict.fromkeys(streams, 0)
        self.rng = random.Random(seed)

    def _open_sources(self) -> list[str]:
        return [s for s in self.streams if self.used.get(s, 0) < self.budget[s]]

    def chunks(self, size: int) -> Iterator[tuple[str, list[str]]]:
        while True:
            live = self._open_sources()
            if not live:
                return
            weights = [self.budget[s] - self.used.get(s, 0) for s in live]
            source = self.rng.choices(live, weights=weights, k=1)[0]
            batch: list[str] = []
            for text in self.streams[source]:
                batch.append(text)
                if len(batch) >= size:
                    break
            if not batch:                        # this source has run dry
                del self.streams[source]
                continue
            yield source, batch


def _imap_ordered(pool: ThreadPoolExecutor, fn: Callable[[Any], Any],
                  items: Iterable[Any], ahead: int) -> Iterator[Any]:
    """`pool.map` with a bounded queue, so the corpus is never materialised.

    `ThreadPoolExecutor.map` drains its whole input up front, which for a 120
    billion token stream means reading the corpus into memory before encoding a
    single chunk of it. This keeps `ahead` chunks in flight and no more, and
    preserves order so a build is reproducible from its seed.
    """
    done = object()
    queue: deque = deque()
    it = iter(items)
    for item in itertools.islice(it, ahead):
        queue.append(pool.submit(fn, item))
    while queue:
        yield queue.popleft().result()
        item = next(it, done)
        if item is not done:
            queue.append(pool.submit(fn, item))


# --- writing ---------------------------------------------------------------
_HEADER_BYTES = 128


class _RowWriter:
    """Stream rows into one `.npy`, writing the header last.

    The row count is not known until the sources run out, and a `.npy` header
    carries the shape at the front of the file. Preallocating the requested size
    and trimming later means writing hundreds of gigabytes to throw some away,
    so instead the first 128 bytes are reserved, the rows are appended, and the
    real header is written into that reservation on close. 128 is the size numpy
    itself produces for a header this small, and the format only asks that it be
    a multiple of 64 so the data lands aligned.
    """

    def __init__(self, path: Path, seq_len: int) -> None:
        self.path = path
        self.seq_len = seq_len
        self.rows = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = path.open("wb")
        self.fh.write(b"\0" * _HEADER_BYTES)

    def write(self, tokens: np.ndarray, segments: np.ndarray) -> None:
        row = np.stack([tokens, segments], axis=-1).astype(np.uint16, copy=False)
        self.fh.write(row.tobytes(order="C"))
        self.rows += 1

    def close(self) -> Path:
        shape = (self.rows, self.seq_len, 2)
        text = (f"{{'descr': '{np.dtype(np.uint16).str}', 'fortran_order': False, "
                f"'shape': {shape}, }}")
        magic = b"\x93NUMPY\x01\x00"
        pad = _HEADER_BYTES - len(magic) - 2 - len(text) - 1
        if pad < 0:
            raise ValueError(f"npy header for {shape} does not fit {_HEADER_BYTES} bytes")
        header = magic + struct.pack("<H", _HEADER_BYTES - len(magic) - 2)
        self.fh.seek(0)
        self.fh.write(header + text.encode("latin-1") + b" " * pad + b"\n")
        self.fh.close()
        return self.path


def build(sources: str | Path = "data/sources", out: str | Path = "data/corpus",
          tokens: float = TOTAL_TOKENS, seq_len: int = 0, tokenizer: str = "",
          min_chars: int = MIN_CHARS, workers: int = 8, seed: int = 0,
          chunk: int = 512, report: bool = True) -> str:
    """Filter, mix, encode and pack the corpus into one memory-mappable file.

    Args:
        sources: a directory holding one subdirectory per key of MIX.
        out: a directory (the rows land in `corpus.npy` inside it) or a path
            ending in `.npy`.
        tokens: the total token budget, split across MIX by share.
        seq_len: row length. 0 takes `QuartzConfig().max_seq_len`.
        tokenizer: a fitted `scribe.model`; empty searches the usual places.
        min_chars: documents shorter than this are dropped before packing.
        workers: threads encoding chunks. SentencePiece releases the GIL inside
            its extension, so threads are the right shape here and processes
            would only add a copy of every chunk.
        seed: fixes the source draw, so two builds of the same shards agree.
        chunk: documents per encode call.
        report: print the mix table and what was actually taken.

    Returns:
        The path written, as a string.
    """
    from quartz.model.scribe import get_tokenizer, pre_tokenize

    tokens = int(tokens)
    seq_len = int(seq_len) or QuartzConfig().max_seq_len
    target_rows = tokens // seq_len
    if target_rows < 1:
        raise ValueError(
            f"a budget of {tokens:,} tokens does not fill one row of {seq_len}")

    dirs = source_dirs(sources)
    sp = get_tokenizer(tokenizer or None)
    budget = plan(tokens)
    if report:
        print(format_plan(tokens))

    stats: dict[str, dict[str, int]] = {name: {} for name in MIX}
    streams = {name: source_documents(path, name, min_chars=min_chars,
                                      stats=stats[name])
               for name, path in dirs.items()}
    mixer = _Mixer(streams, budget, seed=seed)

    def encode(job: tuple[str, list[str]]) -> tuple[str, list[list[int]]]:
        source, batch = job
        return source, list(sp.encode([pre_tokenize(t) for t in batch]))

    out_path = Path(out)
    if out_path.suffix != ".npy":
        out_path = out_path / "corpus.npy"
    writer = _RowWriter(out_path, seq_len)
    taken: dict[str, int] = dict.fromkeys(MIX, 0)

    def documents() -> Iterator[list[int]]:
        """Encoded documents, in mix order, charging each source as it lands."""
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            jobs = mixer.chunks(chunk)
            for source, encoded in _imap_ordered(pool, encode, jobs,
                                                 max(2, workers * 2)):
                counted = sum(len(ids) for ids in encoded) + len(encoded)
                mixer.used[source] = mixer.used.get(source, 0) + counted
                taken[source] += counted
                yield from encoded

    try:
        for row_tokens, row_segments in pack(documents(), seq_len):
            writer.write(row_tokens, row_segments)
            if writer.rows >= target_rows:
                break
    finally:
        path = writer.close()

    written = writer.rows * seq_len
    if report:
        total = sum(taken.values()) or 1
        print(f"  {'packed':<14} {writer.rows:,} rows x {seq_len} "
              f"= {written:,} tokens")
        for name in MIX:
            kept = stats[name].get("kept", 0)
            read = stats[name].get("read", 0)
            print(f"  {name:<14} {taken[name] / total:>6.1%} of the mix   "
                  f"{kept:,} of {read:,} documents kept")
        if writer.rows < target_rows:
            # A budget that is not a whole number of rows is not a shortfall,
            # so this only fires when the shards genuinely ran out.
            print(f"  {'short':<14} asked for {tokens:,} tokens, the shards held "
                  f"{written:,}")
    return str(path)


def load_rows(path: str | Path):
    """Memory-map a packed corpus back, as `(rows, seq_len, 2)` uint16.

    Plane 0 is the tokens and plane 1 the segment ids. Nothing is read until a
    batch is indexed, which is what lets a training run of any size open a
    corpus of any size.
    """
    file = Path(path)
    if file.is_dir():
        file = file / "corpus.npy"
    if not file.is_file():
        raise FileNotFoundError(
            f"no packed corpus at {file}. Build one with "
            f"`python -m quartz.cli winnow --sources <dir>`.")
    return np.load(file, mmap_mode="r")
