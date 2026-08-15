"""Ingot, the container: one file, no parsing, no names.

The model is a pile of numpy arrays and a tokenizer. It has to become one file a
device can map into memory and use, on a runtime with no JSON parser and no
allocator to spare. Three decisions carry the format.

Coded tensors are pre-transposed to [out, in], the axis the quantiser grouped on
running last, so one output row is contiguous along the axis a matrix-vector
product streams. Tensors are laid out layer major, so one layer's tensors sit in
one region and a loader touches each page once. And the directory has no names:
tensors are found by position in a fixed canonical order, so a runtime either
knows that order or has no business loading the file. A name field would be four
hundred odd strings of dead weight and a second way to be wrong.

Everything sub-byte is packed LSB first, because that is the order a shift and
mask extractor gets for free. Ternary is stored as two bit crumbs, three of the
four codes used: the fourth code costs nothing and a base three packer would
cost a division per weight on a device that has none.

Numpy only. Nothing here imports JAX.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import (
    CQ_BITS,
    CQ_GROUP,
    DEFAULT_BITS_MAP,
    KV_GROUP,
    TERNARY_BITS,
    QuartzConfig,
)
from .grist import (
    bits_for,
    codebook,
    cq_decode,
    cq_encode,
    family,
    flatten_params,
    group_axis,
    is_head,
    is_quantised,
    is_training_only,
    natural_key,
    parse_bits_map,
)

__all__ = [
    "ALIGN",
    "CODEBOOK_BYTES",
    "HEADER_SIZE",
    "REC_SIZE",
    "TAG",
    "Ingot",
    "IngotStats",
    "Record",
    "canonical_order",
    "codebook_block",
    "collect",
    "is_whole",
    "pack_lsb",
    "read_ingot",
    "stored_bits",
    "unpack_lsb",
    "write_ingot",
]

#: The magic word: 'Q', 'T', 'Z', 0 read as a big-endian quad, stored as one
#: little-endian uint32 like every other field.
TAG = 0x51545A00
#: Blob alignment. A cache line on every device this ships to, and enough for a
#: fp16 or fp32 view to be taken over the bytes without a copy.
ALIGN = 64
#: Bumped when a field moves. A reader refuses a version it does not know rather
#: than reading the wrong offsets and returning noise.
FORMAT_VERSION = 1

FP16, FP32, CQ, RAW = 1, 2, 3, 4

#: The header, one uint32 each, then rope_theta as a float. Order is frozen.
_HDR_FIELDS = (
    "tag", "version", "n_tensors", "dir_offset", "blob_offset", "file_bytes",
    "flags", "vocab_size", "d_model", "num_heads", "num_kv_heads", "num_layers",
    "max_seq_len", "hadamard_n", "pad_token_id", "lanes", "sinkhorn_iters",
    "imprint_slots", "imprint_tables", "imprint_sub_dim", "imprint_site_mask",
    "imprint_order_mask", "dowser_dim", "dowser_probes", "gauge_probes",
    "cq_group", "kv_window", "kv_group", "runtime_widths",
)
_HDR_FMT = f"<{len(_HDR_FIELDS)}If"          # 29 uints, then rope_theta
_REC_FMT = "<BBHIIIIQQII"                    # 44 bytes, no name field
HEADER_SIZE = struct.calcsize(_HDR_FMT)      # 120
REC_SIZE = struct.calcsize(_REC_FMT)         # 44
#: Four shape slots in a record. Nothing here is deeper: the widest tensor is a
#: memory table at (tables, slots, sub_dim).
MAX_RANK = 4
#: num_layers and the n-gram orders travel as bit masks in one uint32 each.
MASK_BITS = 32

_DTYPES = ("float32", "bfloat16", "float16")
_FLAG_TOKENIZER = 1

#: Families that are not per layer. A leading axis that happens to equal
#: num_layers is a coincidence on these, not a scan axis, and slicing a memory
#: table in half because a tiny model has two layers and two tables is exactly
#: the bug this list exists to prevent.
WHOLE_FAMILIES = ("embed", "imprint", "loom", "final_norm")


def is_whole(name: str) -> bool:
    """True for a tensor the exporter must not cut along its leading axis."""
    return family(name).startswith(WHOLE_FAMILIES) or is_head(name)


# --------------------------------------------------------------------------
# sub-byte packing
# --------------------------------------------------------------------------
def pack_lsb(idx: np.ndarray, bits: int) -> np.ndarray:
    """Eight indices into one uint64, LSB first, one bitstream per row.

    Eight at a time because eight indices at any width from one to eight bits is
    a whole number of bytes, so no group ever straddles a row boundary and a
    row's bytes can be handed to a device kernel on their own.
    """
    idx = np.asarray(idx)
    if not 1 <= bits <= 8:
        raise ValueError(f"pack_lsb handles 1 to 8 bits, got {bits}")
    n = idx.shape[-1]
    if n % 8:
        raise ValueError(f"a packed row needs a multiple of 8 indices, got {n}")
    lead = idx.shape[:-1]
    g = idx.reshape(*lead, n // 8, 8).astype(np.uint64)
    if g.size and int(g.max()) >> bits:
        raise ValueError(f"an index does not fit in {bits} bits")
    word = np.zeros(g.shape[:-1], dtype=np.uint64)
    for i in range(8):
        word |= g[..., i] << np.uint64(i * bits)
    packed = np.stack([(word >> np.uint64(8 * b)) & np.uint64(0xFF)
                       for b in range(bits)], axis=-1)
    return packed.astype(np.uint8).reshape(*lead, n * bits // 8)


def unpack_lsb(packed: np.ndarray, bits: int, n: int) -> np.ndarray:
    """The inverse of pack_lsb: `n` indices per row, back out of the bytes."""
    packed = np.asarray(packed)
    if not 1 <= bits <= 8:
        raise ValueError(f"unpack_lsb handles 1 to 8 bits, got {bits}")
    if n % 8:
        raise ValueError(f"a packed row holds a multiple of 8 indices, got {n}")
    if packed.shape[-1] != n * bits // 8:
        raise ValueError(f"{packed.shape[-1]} bytes cannot hold {n} indices at {bits} bits")
    lead = packed.shape[:-1]
    b = packed.reshape(*lead, n // 8, bits).astype(np.uint64)
    word = np.zeros(b.shape[:-1], dtype=np.uint64)
    for i in range(bits):
        word |= b[..., i] << np.uint64(8 * i)
    mask = np.uint64((1 << bits) - 1)
    idx = np.stack([(word >> np.uint64(i * bits)) & mask for i in range(8)], axis=-1)
    return idx.astype(np.uint8).reshape(*lead, n)


def stored_bits(bits: float) -> int:
    """The width on disk. Ternary rounds up from 1.58 to a two bit crumb."""
    return 2 if float(bits) == TERNARY_BITS else int(bits)


#: 4 + 8 + 16 levels of float32. A fixed 112 bytes, so a reader knows where the
#: directory starts before it has parsed anything.
CODEBOOK_BYTES = sum(1 << b for b in CQ_BITS) * 4


def codebook_block(group: int = CQ_GROUP, widths: tuple[int, ...] = CQ_BITS) -> bytes:
    """The shared codebooks, 4 + 8 + 16 levels of float32, 112 bytes.

    They ride in the file so a device never runs the Lloyd-Max fit, and so a
    file written today decodes the same way after the fit is ever touched.
    Ternary is absent on purpose: its three levels are a fixed constant, so
    storing them would be storing a number the reader already has.
    """
    return np.concatenate([codebook(b, group) for b in widths]).astype("<f4").tobytes()


# --------------------------------------------------------------------------
# the directory
# --------------------------------------------------------------------------
@dataclass
class Record:
    """One directory entry. Forty four bytes, and no name."""

    kind: int
    bits: float
    shape: tuple[int, ...]
    offset: int
    nbytes: int
    group: int
    axis: int


@dataclass
class _Entry:
    name: str
    layer: int                   # -1 when the tensor is not per layer
    kind: int
    bits: float
    group: int
    axis: int
    shape: tuple[int, ...]
    blob: bytes
    offset: int = 0


def _bucket(name: str, layer: int) -> int:
    """Which region of the file a tensor lives in. The order is the format."""
    if layer >= 0:
        return 1
    head = family(name)
    if head.startswith("embed"):
        return 0
    if head.startswith("loom"):
        return 2
    if head.startswith("imprint"):
        return 3
    if head.startswith("final_norm"):
        return 4
    if is_head(name):
        return 5
    return 6


def _plan(params, cfg: QuartzConfig) -> list[tuple[str, int, np.ndarray]]:
    """Flatten, drop the training heads, slice the layer axis, and order it.

    Everything scan gave a leading layer axis is cut back into per-layer tensors
    and laid out layer major, so a loader walking the directory in order walks
    the file in order.
    """
    out: list[tuple[str, int, np.ndarray]] = []
    for name, leaf in flatten_params(params):
        if is_training_only(name):
            continue             # Foresight never leaves the training host
        if leaf.ndim >= 1 and leaf.shape[0] == cfg.num_layers and not is_whole(name):
            out += [(name, i, leaf[i]) for i in range(cfg.num_layers)]
        else:
            out.append((name, -1, leaf))
    out.sort(key=lambda e: (_bucket(e[0], e[1]), max(e[1], 0), natural_key(e[0])))
    return out


def canonical_order(params, cfg: QuartzConfig) -> list[tuple[str, int]]:
    """The names behind the nameless directory, as (name, layer) in file order.

    The file does not carry these. A loader recomputes them from the same
    config, which is the whole point: the order is part of the format, not part
    of the payload. Layer is -1 for a tensor that is not per layer.
    """
    return [(name, layer) for name, layer, _ in _plan(params, cfg)]


def collect(params, cfg: QuartzConfig, bits_map: str = DEFAULT_BITS_MAP,
            group: int = CQ_GROUP) -> list[_Entry]:
    """Every shipped tensor, quantised, packed, and ready to be given an offset."""
    table, default = parse_bits_map(bits_map)
    entries: list[_Entry] = []
    for name, layer, leaf in _plan(params, cfg):
        arr = np.asarray(leaf)
        if arr.ndim > MAX_RANK:
            raise ValueError(f"{name} has rank {arr.ndim}; a record holds {MAX_RANK}")
        if arr.dtype.kind not in "fc":
            raise TypeError(f"{name} is {arr.dtype}, and only floats are exported")
        bits = bits_for(name, table, default)
        if bits < 16 and is_quantised(name, arr.shape, group):
            axis = group_axis(name, arr.ndim)
            codes, norm, shape = cq_encode(arr, bits, group, axis)
            blob = (pack_lsb(codes, stored_bits(bits)).tobytes()
                    + norm.astype("<f2").tobytes())
            entries.append(_Entry(name, layer, CQ, bits, group, axis, shape, blob))
        elif bits >= 32:
            entries.append(_Entry(name, layer, FP32, 32, 0, 0, tuple(arr.shape),
                                  arr.astype("<f4").tobytes()))
        else:
            entries.append(_Entry(name, layer, FP16, 16, 0, 0, tuple(arr.shape),
                                  arr.astype("<f2").tobytes()))
    return entries


# --------------------------------------------------------------------------
# the header
# --------------------------------------------------------------------------
def _mask(values, limit: int, what: str) -> int:
    out = 0
    for v in values:
        if not 0 <= int(v) < limit:
            raise ValueError(f"{what} {v} does not fit the {limit} bit header mask")
        out |= 1 << int(v)
    return out


def _unmask(mask: int) -> tuple[int, ...]:
    return tuple(i for i in range(MASK_BITS) if mask >> i & 1)


def _header(cfg: QuartzConfig, n_tensors: int, dir_offset: int, blob_offset: int,
            file_bytes: int, has_tokenizer: bool, group: int) -> bytes:
    orders, _, sub_dim = cfg.imprint_geometry
    dtype_code = _DTYPES.index(cfg.dtype) if cfg.dtype in _DTYPES else 0
    flags = (_FLAG_TOKENIZER if has_tokenizer else 0) | (dtype_code << 8)
    values = {
        "tag": TAG, "version": FORMAT_VERSION, "n_tensors": n_tensors,
        "dir_offset": dir_offset, "blob_offset": blob_offset,
        "file_bytes": file_bytes, "flags": flags,
        "vocab_size": cfg.vocab_size, "d_model": cfg.d_model,
        "num_heads": cfg.num_heads, "num_kv_heads": cfg.num_kv_heads,
        "num_layers": cfg.num_layers, "max_seq_len": cfg.max_seq_len,
        "hadamard_n": cfg.hadamard_n, "pad_token_id": cfg.pad_token_id,
        "lanes": cfg.lanes, "sinkhorn_iters": cfg.sinkhorn_iters,
        "imprint_slots": cfg.imprint_slots, "imprint_tables": cfg.imprint_tables,
        "imprint_sub_dim": sub_dim,
        "imprint_site_mask": _mask(cfg.imprint_sites, MASK_BITS, "imprint site"),
        "imprint_order_mask": _mask(orders, MASK_BITS, "n-gram order"),
        "dowser_dim": cfg.dowser_dim, "dowser_probes": cfg.dowser_probes,
        "gauge_probes": cfg.gauge_probes, "cq_group": group,
        "kv_window": cfg.effective_window(), "kv_group": KV_GROUP,
        "runtime_widths": (cfg.kv_bits & 0xFF) | ((cfg.act_bits & 0xFF) << 8),
    }
    return struct.pack(_HDR_FMT, *(values[k] for k in _HDR_FIELDS), cfg.rope_theta)


def _config_from(head: dict[str, int], rope_theta: float) -> QuartzConfig:
    """Rebuild the geometry from the header alone, which is what a device does."""
    orders = _unmask(head["imprint_order_mask"])
    tables = head["imprint_tables"]
    heads = max(1, tables // max(len(orders), 1))
    return QuartzConfig(
        vocab_size=head["vocab_size"], d_model=head["d_model"],
        num_heads=head["num_heads"], num_kv_heads=head["num_kv_heads"],
        num_layers=head["num_layers"], max_seq_len=head["max_seq_len"],
        pad_token_id=head["pad_token_id"], rope_theta=rope_theta,
        dtype=_DTYPES[((head["flags"] >> 8) & 0xFF) % len(_DTYPES)],
        lanes=head["lanes"], sinkhorn_iters=head["sinkhorn_iters"],
        imprint_sites=_unmask(head["imprint_site_mask"]), imprint_orders=orders,
        imprint_heads=heads, imprint_slots=head["imprint_slots"],
        dowser_dim=head["dowser_dim"], dowser_probes=head["dowser_probes"],
        gauge_probes=head["gauge_probes"], kv_window=head["kv_window"],
        kv_bits=head["runtime_widths"] & 0xFF,
        act_bits=(head["runtime_widths"] >> 8) & 0xFF,
    )


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------
@dataclass
class IngotStats:
    """What a build produced, for the one line the CLI prints."""

    path: str
    bytes: int
    tensors: int
    weight_bytes: int
    tokenizer_bytes: int
    directory_bytes: int
    first_blob: int
    bits_map: str


def _tokenizer_bytes(tokenizer) -> bytes | None:
    """Accept raw bytes, a path to a .model file, or a SentencePieceProcessor."""
    if tokenizer is None:
        return None
    if isinstance(tokenizer, bytes | bytearray):
        return bytes(tokenizer)
    if isinstance(tokenizer, str | Path):
        path = Path(tokenizer)
        if not path.is_file():
            raise FileNotFoundError(f"no tokenizer model at {path}")
        return path.read_bytes()
    proto = getattr(tokenizer, "serialized_model_proto", None)
    if callable(proto):
        return bytes(proto())
    raise TypeError("tokenizer must be bytes, a path, or a SentencePieceProcessor, "
                    f"got {type(tokenizer).__name__}")


def write_ingot(params, cfg: QuartzConfig, path: str | Path,
                bits_map: str | None = None, tokenizer=None,
                group: int = CQ_GROUP) -> IngotStats:
    """Cast a parameter tree, a geometry and a tokenizer into one file.

    `bits_map` defaults to whatever the config was stamped with by the
    quantisation-aware stage, and to the shipped scheme if it was never stamped.
    """
    spec = bits_map or cfg.weight_bits or DEFAULT_BITS_MAP
    entries = collect(params, cfg, spec, group)
    blob = _tokenizer_bytes(tokenizer)
    if blob is not None:
        entries.append(_Entry("tokenizer", -1, RAW, 0, 0, 0, (len(blob),), blob))

    codebooks = codebook_block(group)
    dir_offset = HEADER_SIZE + len(codebooks)
    pos = dir_offset + len(entries) * REC_SIZE
    for entry in entries:
        pos = (pos + ALIGN - 1) & ~(ALIGN - 1)       # 64-byte aligned
        entry.offset, pos = pos, pos + len(entry.blob)

    first_blob = entries[0].offset if entries else pos
    header = _header(cfg, len(entries), dir_offset, first_blob, pos,
                     blob is not None, group)

    path = Path(path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(codebooks)
        for entry in entries:
            dims = (*entry.shape, 0, 0, 0, 0)[:MAX_RANK]
            fh.write(struct.pack(
                _REC_FMT, entry.kind, _bits_code(entry.bits), len(entry.shape),
                *dims, entry.offset, len(entry.blob), entry.group, entry.axis))
        for entry in entries:
            gap = entry.offset - fh.tell()
            if gap < 0:
                raise RuntimeError(f"the blob for {entry.name} would land on the "
                                   f"previous one; the offset pass is wrong")
            fh.write(b"\x00" * gap)
            fh.write(entry.blob)

    weights = sum(len(e.blob) for e in entries if e.kind != RAW)
    return IngotStats(
        path=str(path), bytes=pos, tensors=len(entries), weight_bytes=weights,
        tokenizer_bytes=0 if blob is None else len(blob),
        directory_bytes=len(entries) * REC_SIZE, first_blob=first_blob,
        bits_map=spec)


def _bits_code(bits: float) -> int:
    """One byte for the width. 1 means ternary, which is 1.58 and not a byte."""
    return 1 if float(bits) == TERNARY_BITS else int(bits) & 0xFF


def _bits_value(code: int) -> float:
    return TERNARY_BITS if code == 1 else float(code)


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------
@dataclass
class Ingot:
    """A file, read back: the geometry, the tensors in order, the tokenizer."""

    config: QuartzConfig
    tensors: list[np.ndarray]
    tokenizer: bytes | None
    header: dict[str, int]
    records: list[Record] = field(default_factory=list)
    codebooks: dict[float, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, i: int) -> np.ndarray:
        return self.tensors[i]

    def named(self, order: list[tuple[str, int]]) -> dict[str, np.ndarray]:
        """Zip the positional tensors back onto names from canonical_order.

        Per-layer tensors are stacked back into one array with the layer axis in
        front, which is the shape the model was trained with.
        """
        if len(order) > len(self.tensors):
            raise ValueError(f"{len(order)} names against {len(self.tensors)} tensors")
        out: dict[str, list[tuple[int, np.ndarray]]] = {}
        for (name, layer), tensor in zip(order, self.tensors, strict=False):
            out.setdefault(name, []).append((layer, tensor))
        # the layer axis is decided by the entry, not by how many entries there
        # are, so a one layer model still gets its leading axis back
        return {name: (rows[0][1] if rows[0][0] < 0 else np.stack([t for _, t in rows]))
                for name, rows in out.items()}


def _read_codebooks(raw: bytes, group: int) -> dict[float, np.ndarray]:
    if len(raw) != CODEBOOK_BYTES:
        raise ValueError(f"the codebook block is {len(raw)} bytes, not {CODEBOOK_BYTES}")
    levels = np.frombuffer(raw, dtype="<f4")
    books, at = {}, 0
    for bits in CQ_BITS:
        n = 1 << bits
        books[float(bits)] = np.array(levels[at:at + n], dtype=np.float32)
        at += n
    # ternary is a constant, not a fit, so it is rebuilt rather than carried
    books[TERNARY_BITS] = codebook(TERNARY_BITS, group)
    return books


def read_ingot(path: str | Path, dtype=np.float32) -> Ingot:
    """Read a file back into arrays, in the order the directory lists them.

    Quantised tensors come back dequantised against the codebooks the file
    carries, so what you get is exactly what the device computes with, not the
    weights before the squeeze.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no ingot at {path}")
    raw = path.read_bytes()
    if len(raw) < HEADER_SIZE:
        raise ValueError(f"{path} is {len(raw)} bytes, too short to be an ingot")

    values = struct.unpack_from(_HDR_FMT, raw, 0)
    head = dict(zip(_HDR_FIELDS, values[:-1], strict=True))
    rope_theta = float(values[-1])
    if head["tag"] != TAG:
        raise ValueError(f"{path} is not an ingot: tag {head['tag']:#010x}")
    if head["version"] != FORMAT_VERSION:
        raise ValueError(f"{path} is format version {head['version']}, "
                         f"this build reads {FORMAT_VERSION}")
    if head["file_bytes"] != len(raw):
        raise ValueError(f"{path} says {head['file_bytes']} bytes and is {len(raw)}")

    group = head["cq_group"]
    books = _read_codebooks(raw[HEADER_SIZE:head["dir_offset"]], group)
    cfg = _config_from(head, rope_theta)
    if head["dir_offset"] + head["n_tensors"] * REC_SIZE > len(raw):
        raise ValueError(f"{path} claims {head['n_tensors']} tensors, which is more "
                         f"directory than there is file")

    tensors: list[np.ndarray] = []
    records: list[Record] = []
    tokenizer: bytes | None = None
    for i in range(head["n_tensors"]):
        fields = struct.unpack_from(_REC_FMT, raw, head["dir_offset"] + i * REC_SIZE)
        kind, code, ndim = fields[0], fields[1], fields[2]
        if ndim > MAX_RANK:
            raise ValueError(f"tensor {i} claims rank {ndim}, past the {MAX_RANK} "
                             f"shape slots a record has")
        shape = tuple(int(s) for s in fields[3:3 + ndim])
        offset, nbytes, rec_group, axis = (int(fields[7]), int(fields[8]),
                                           int(fields[9]), int(fields[10]))
        blob = raw[offset:offset + nbytes]
        if len(blob) != nbytes:
            raise ValueError(f"{path} is truncated inside tensor {i}")
        records.append(Record(kind, _bits_value(code), shape, offset, nbytes,
                              rec_group, axis))
        if kind == RAW:
            tokenizer = bytes(blob) if tokenizer is None else tokenizer
            tensors.append(np.frombuffer(blob, dtype=np.uint8).copy())
        elif kind == FP16:
            tensors.append(np.frombuffer(blob, dtype="<f2").reshape(shape).astype(dtype))
        elif kind == FP32:
            tensors.append(np.frombuffer(blob, dtype="<f4").reshape(shape).astype(dtype))
        elif kind == CQ:
            tensors.append(_read_cq(blob, shape, _bits_value(code), rec_group, axis,
                                    books).astype(dtype))
        else:
            raise ValueError(f"tensor {i} has unknown kind {kind}")

    if not head["flags"] & _FLAG_TOKENIZER:
        tokenizer = None
    return Ingot(config=cfg, tensors=tensors, tokenizer=tokenizer, header=head,
                 records=records, codebooks=books)


def _read_cq(blob: bytes, shape: tuple[int, ...], bits: float, group: int,
             axis: int, books: dict[float, np.ndarray]) -> np.ndarray:
    n = shape[axis]
    padded = -(-n // group) * group
    rows = math.prod(shape) // n
    width = stored_bits(bits)
    split = rows * padded * width // 8
    packed = np.frombuffer(blob[:split], dtype=np.uint8).reshape(rows, padded * width // 8)
    norm = np.frombuffer(blob[split:], dtype="<f2").reshape(rows, padded // group)
    codes = unpack_lsb(packed, width, padded)
    return cq_decode(codes, norm, bits, shape, group, axis, levels=books[float(bits)])


# --------------------------------------------------------------------------
# the proof
# --------------------------------------------------------------------------
def _demo() -> None:
    """Write a whole tiny model out and read it back, byte for byte."""
    import tempfile

    from .config import preset
    from .grist import shipped_tensors

    rng = np.random.default_rng(0)
    cfg = preset("tiny")

    # a parameter tree in the shape the trunk hands over: nested, layer stacked,
    # plus one Foresight tensor that must not reach the file
    params: dict = {}
    for name, shape in shipped_tensors(cfg):
        node = params
        parts = name.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = rng.standard_normal(shape, dtype=np.float32)
    params["fs_block"] = {"q_proj": {"kernel": rng.standard_normal(
        (cfg.d_model, cfg.d_model), dtype=np.float32)}}
    tokenizer = rng.integers(0, 256, 4096, dtype=np.uint8).tobytes()

    print("  indices [0, 1, 2, 3, 3, 2, 1, 0]")
    packed = pack_lsb(np.array([[0, 1, 2, 3, 3, 2, 1, 0]], dtype=np.uint8), 2)
    print(f"  packed  {[format(b, '08b') for b in packed[0]]}   "
          f"{packed.shape[1]} bytes for 8 indices at 2 bits")
    back = unpack_lsb(packed, 2, 8)
    if not np.array_equal(back[0], [0, 1, 2, 3, 3, 2, 1, 0]):
        raise AssertionError("pack_lsb and unpack_lsb disagree")

    spec = DEFAULT_BITS_MAP
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "quartz-tiny.ingot"
        stats = write_ingot(params, cfg, path, spec, tokenizer)
        got = read_ingot(path)

        print(f"  wrote     {path.name}  {stats.bytes / 1e6:.2f} MB  "
              f"{stats.tensors} tensors  mixed[{spec}]")
        print(f"  header    {HEADER_SIZE} B    codebooks {CODEBOOK_BYTES} B"
              f"    prelude ends at {HEADER_SIZE + CODEBOOK_BYTES}")
        print(f"  directory {stats.tensors} x {REC_SIZE} = {stats.directory_bytes} B"
              f"    first blob at {stats.first_blob}")
        print(f"  tokenizer {stats.tokenizer_bytes / 1e3:.1f} kB inline")

        if path.stat().st_size != stats.bytes:
            raise AssertionError("the file is not the size the header claims")
        if stats.first_blob % ALIGN:
            raise AssertionError("the first blob is not aligned")
        for rec in got.records:
            if rec.offset % ALIGN:
                raise AssertionError("a blob is not aligned")

        order = canonical_order(params, cfg)
        if len(order) + 1 != len(got):
            raise AssertionError("the directory lost a tensor")
        if got.tokenizer != tokenizer:
            raise AssertionError("the tokenizer did not survive the round trip")

        # the reference: grist's answer for each tensor, computed from the raw
        # leaves, with no container anywhere in the path
        table, default = parse_bits_map(spec)
        source = dict(flatten_params(params))
        exact = 0
        for (name, layer), tensor in zip(order, got.tensors, strict=False):
            leaf = source[name]
            leaf = leaf[layer] if layer >= 0 else leaf
            bits = bits_for(name, table, default)
            if bits < 16 and is_quantised(name, leaf.shape):
                axis = group_axis(name, leaf.ndim)
                codes, norm, shape = cq_encode(leaf, bits, CQ_GROUP, axis)
                want = cq_decode(codes, norm, bits, shape, CQ_GROUP, axis)
            else:
                want = leaf.astype(np.float16).astype(np.float32)
            if not np.array_equal(tensor, want):
                raise AssertionError(f"{name} layer {layer} came back different")
            exact += 1

        if any(is_training_only(name) for name, _ in order):
            raise AssertionError("Foresight reached the file")

        restored = got.config
        for attr in ("vocab_size", "d_model", "num_heads", "num_kv_heads",
                     "num_layers", "max_seq_len", "lanes", "sinkhorn_iters",
                     "imprint_slots", "imprint_sites", "imprint_orders",
                     "imprint_tables", "dowser_dim", "dowser_probes",
                     "gauge_probes", "kv_bits", "act_bits", "dtype",
                     "pad_token_id", "rope_theta"):
            if getattr(restored, attr) != getattr(cfg, attr):
                raise AssertionError(f"{attr} did not survive the header")
        if restored.imprint_geometry != cfg.imprint_geometry:
            raise AssertionError("the imprint geometry did not survive the header")

        print(f"  round trip {exact} tensors exact, geometry restored, "
              f"tokenizer {len(got.tokenizer or b'')} B")


if __name__ == "__main__":     # pragma: no cover - the round trip, run by hand
    _demo()
