"""Ingot: one file, no parsing, no names.

The container has three properties worth testing and one worth writing down.
It round-trips: what comes back is bit-for-bit what Grist would have computed,
because the reader dequantises against the codebooks the file carries. It is
positional: the directory has no name field, so a loader recomputes the order
from the same config or has no business reading the file. And it drops
Foresight, because a training head in a download is a megabyte and a half nobody
runs.
"""
from __future__ import annotations

import numpy as np
import pytest

from quartz.model.config import CQ_BITS, CQ_GROUP, DEFAULT_BITS_MAP, TERNARY_BITS
from quartz.model.grist import (
    bits_for,
    cq_decode,
    cq_encode,
    flatten_params,
    group_axis,
    is_quantised,
    is_training_only,
    parse_bits_map,
)
from quartz.model.ingot import (
    ALIGN,
    HEADER_SIZE,
    REC_SIZE,
    TAG,
    canonical_order,
    codebook_block,
    is_whole,
    pack_lsb,
    read_ingot,
    stored_bits,
    unpack_lsb,
    write_ingot,
)


@pytest.fixture
def written(tmp_path, tiny_params, tiny_cfg):
    """One ingot, written with the shipped scheme and a tokenizer inline."""
    path = tmp_path / "quartz-tiny.ingot"
    tokenizer = bytes(range(256)) * 8
    stats = write_ingot(tiny_params, tiny_cfg, path, DEFAULT_BITS_MAP, tokenizer)
    return path, stats, tokenizer


def reference(name: str, leaf: np.ndarray, bits: float) -> np.ndarray:
    """What Grist alone would produce for this tensor, with no container in the
    path. The file has to agree with it exactly, not approximately."""
    if bits < 16 and is_quantised(name, leaf.shape):
        axis = group_axis(name, leaf.ndim)
        codes, norm, shape = cq_encode(leaf, bits, CQ_GROUP, axis)
        return cq_decode(codes, norm, bits, shape, CQ_GROUP, axis)
    return leaf.astype(np.float16).astype(np.float32)


# --- sub-byte packing -------------------------------------------------------
def test_packing_is_lsb_first():
    """Eight two-bit indices in two bytes, low index in the low bits.

    That order is what a shift-and-mask extractor gets for free on a device
    with no bit-field instructions, so it is part of the format.
    """
    packed = pack_lsb(np.array([[0, 1, 2, 3, 3, 2, 1, 0]], np.uint8), 2)
    assert packed.shape == (1, 2)
    assert list(packed[0]) == [0xE4, 0x1B]
    assert list(unpack_lsb(packed, 2, 8)[0]) == [0, 1, 2, 3, 3, 2, 1, 0]


@pytest.mark.parametrize("bits", [1, 2, 3, 4, 8])
def test_pack_and_unpack_are_inverse(bits):
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 1 << bits, (3, 64), dtype=np.uint8)
    packed = pack_lsb(idx, bits)
    assert packed.shape == (3, 64 * bits // 8)
    assert np.array_equal(unpack_lsb(packed, bits, 64), idx)


def test_a_packed_row_is_a_whole_number_of_bytes():
    with pytest.raises(ValueError, match="multiple of 8"):
        pack_lsb(np.zeros((1, 12), np.uint8), 2)
    with pytest.raises(ValueError, match="multiple of 8"):
        unpack_lsb(np.zeros((1, 3), np.uint8), 2, 12)


def test_an_index_that_does_not_fit_is_an_error():
    with pytest.raises(ValueError, match="does not fit"):
        pack_lsb(np.array([[0, 1, 2, 4, 0, 0, 0, 0]], np.uint8), 2)
    with pytest.raises(ValueError, match="1 to 8 bits"):
        pack_lsb(np.zeros((1, 8), np.uint8), 9)


def test_ternary_is_stored_as_a_two_bit_crumb():
    """A base three packer would cost a division per weight on a device with
    none, so the fourth code is bought and left unused."""
    assert stored_bits(TERNARY_BITS) == 2
    assert stored_bits(4) == 4


def test_the_codebooks_ride_in_the_file():
    """So a device never runs Lloyd-Max, and a file written today still decodes
    after the fit is ever touched."""
    block = codebook_block(CQ_GROUP)
    assert len(block) == sum(1 << bits for bits in CQ_BITS) * 4 == 112


# --- the round trip ---------------------------------------------------------
def test_write_then_read_round_trips_shapes_and_values(written, tiny_cfg, tiny_params):
    path, stats, _ = written
    got = read_ingot(path)
    order = canonical_order(tiny_params, tiny_cfg)
    source = dict(flatten_params(tiny_params))
    table, default = parse_bits_map(stats.bits_map)

    assert len(order) + 1 == len(got)          # the tokenizer is the extra one
    for (name, layer), tensor in zip(order, got.tensors, strict=False):
        leaf = source[name]
        leaf = leaf[layer] if layer >= 0 else leaf
        assert tensor.shape == leaf.shape, name
        assert np.array_equal(tensor, reference(name, leaf, bits_for(name, table,
                                                                    default))), name

    named = got.named(order)
    for name, leaf in named.items():
        assert leaf.shape == source[name].shape, name


def test_the_values_come_back_within_the_quantisation_error(written, tiny_cfg,
                                                            tiny_params):
    """Two bits on a Gaussian costs about a tenth of the signal, and that is
    what a reader gets: the weights the device computes with, not the ones
    before the squeeze."""
    path, _, _ = written
    named = read_ingot(path).named(canonical_order(tiny_params, tiny_cfg))
    source = dict(flatten_params(tiny_params))

    for name in ("embedding", "attn.q_proj", "loom.phi_pre"):
        want, got = source[name], named[name]
        error = np.mean((got - want) ** 2) / np.mean(want ** 2)
        assert 0.0 < error < 0.3, f"{name} came back at relative error {error}"

    # the heads never went through the codebook at all
    assert np.allclose(named["gauge.probe"], source["gauge.probe"], atol=1e-3)


def test_the_geometry_survives_the_header(written, tiny_cfg):
    path, _, _ = written
    restored = read_ingot(path).config

    for attr in ("vocab_size", "d_model", "num_heads", "num_kv_heads", "num_layers",
                 "max_seq_len", "lanes", "sinkhorn_iters", "imprint_slots",
                 "imprint_sites", "imprint_orders", "imprint_tables", "dowser_dim",
                 "dowser_probes", "gauge_probes", "kv_bits", "act_bits", "dtype",
                 "pad_token_id", "rope_theta"):
        assert getattr(restored, attr) == getattr(tiny_cfg, attr), attr
    assert restored.imprint_geometry == tiny_cfg.imprint_geometry
    assert restored.kv_window == tiny_cfg.effective_window()


def test_the_tokenizer_travels_inline(written):
    path, stats, tokenizer = written
    got = read_ingot(path)
    assert got.tokenizer == tokenizer
    assert stats.tokenizer_bytes == len(tokenizer)


def test_a_file_without_a_tokenizer_says_so(tmp_path, tiny_params, tiny_cfg):
    path = tmp_path / "bare.ingot"
    write_ingot(tiny_params, tiny_cfg, path, "default=2")
    assert read_ingot(path).tokenizer is None


def test_a_processor_can_be_handed_over_instead_of_bytes(tmp_path, tiny_params,
                                                         tiny_cfg, fake_tokenizer):
    path = tmp_path / "sp.ingot"
    write_ingot(tiny_params, tiny_cfg, path, "default=2", fake_tokenizer)
    assert read_ingot(path).tokenizer == fake_tokenizer.serialized_model_proto()


def test_foresight_never_reaches_the_file(written, tiny_cfg, tiny_params):
    path, _, _ = written
    order = canonical_order(tiny_params, tiny_cfg)
    assert any(is_training_only(name) for name in dict(flatten_params(tiny_params)))
    assert not any(is_training_only(name) for name, _ in order)
    assert read_ingot(path).config.num_layers == tiny_cfg.num_layers


def test_the_directory_carries_no_names(written):
    """405 strings of dead weight and a second way to be wrong."""
    path, stats, _ = written
    raw = path.read_bytes()
    assert REC_SIZE == 44
    assert stats.directory_bytes == stats.tensors * REC_SIZE
    for probe in (b"q_proj", b"attn", b"embedding", b"loom"):
        assert probe not in raw, probe


def test_every_blob_is_cache_line_aligned(written):
    path, stats, _ = written
    got = read_ingot(path)
    assert stats.first_blob % ALIGN == 0
    assert all(record.offset % ALIGN == 0 for record in got.records)
    assert path.stat().st_size == stats.bytes
    assert stats.first_blob >= HEADER_SIZE + len(codebook_block())


def test_the_layer_axis_is_cut_and_the_rest_is_not(tiny_cfg, tiny_params):
    """A tensor whose leading axis happens to equal num_layers is not a scan
    axis, and slicing a memory table in half is exactly the bug this prevents."""
    order = canonical_order(tiny_params, tiny_cfg)
    layers = {name: [layer for other, layer in order if other == name]
              for name, _ in order}

    assert layers["attn.q_proj"] == list(range(tiny_cfg.num_layers))
    assert layers["embedding"] == [-1]
    assert layers["loom.phi_pre"] == [-1]
    assert all(is_whole(name) for name in ("embedding", "final_norm", "loom.phi_pre",
                                           "imprint.1.table", "dowser.probe"))
    assert not is_whole("attn.q_proj")


def test_the_file_is_laid_out_layer_major(tiny_cfg, tiny_params):
    """One layer's tensors sit in one region, so a loader touches each page once."""
    order = canonical_order(tiny_params, tiny_cfg)
    per_layer = [layer for _, layer in order if layer >= 0]
    assert per_layer == sorted(per_layer)


@pytest.mark.parametrize("bits", ["default=2", "default=3", "default=4",
                                  f"default={TERNARY_BITS}", "default=16"])
def test_every_width_writes_and_reads(tmp_path, tiny_params, tiny_cfg, bits):
    path = tmp_path / "w.ingot"
    stats = write_ingot(tiny_params, tiny_cfg, path, bits)
    got = read_ingot(path)
    order = canonical_order(tiny_params, tiny_cfg)
    source = dict(flatten_params(tiny_params))

    assert stats.bits_map == bits
    assert len(got) == len(order)
    for (name, layer), tensor in zip(order, got.tensors, strict=False):
        leaf = source[name]
        assert tensor.shape == (leaf[layer] if layer >= 0 else leaf).shape


def test_a_wider_scheme_is_a_bigger_file(tmp_path, tiny_params, tiny_cfg):
    sizes = []
    for spec in (f"default={TERNARY_BITS}", "default=2", "default=3", "default=4"):
        path = tmp_path / f"{spec}.ingot"
        sizes.append(write_ingot(tiny_params, tiny_cfg, path, spec).bytes)
    assert sizes == sorted(sizes)
    assert sizes[0] == sizes[1]        # ternary is stored in two-bit crumbs


def test_the_config_stamps_the_scheme_when_none_is_given(tmp_path, tiny_params,
                                                         tiny_cfg):
    """The quantisation-aware stage writes `weight_bits`, so the exporter ships
    exactly the scheme the model was adapted to and nobody has to remember."""
    from quartz.model.config import preset

    cfg = preset("tiny", vocab_size=tiny_cfg.vocab_size,
                 imprint_slots=tiny_cfg.imprint_slots, weight_bits="default=4")
    stats = write_ingot(tiny_params, cfg, tmp_path / "stamped.ingot")
    assert stats.bits_map == "default=4"


# --- refusals ---------------------------------------------------------------
def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="no ingot"):
        read_ingot(tmp_path / "absent.ingot")


def test_something_that_is_not_an_ingot_is_refused(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_bytes(b"x" * (HEADER_SIZE + 8))
    with pytest.raises(ValueError, match="not an ingot"):
        read_ingot(path)

    short = tmp_path / "short.ingot"
    short.write_bytes(b"x" * 8)
    with pytest.raises(ValueError, match="too short"):
        read_ingot(short)


def test_a_truncated_file_is_refused(written):
    path, stats, _ = written
    path.write_bytes(path.read_bytes()[: stats.bytes - ALIGN])
    with pytest.raises(ValueError, match="bytes and is"):
        read_ingot(path)


def test_the_tag_is_the_magic_word():
    assert TAG == 0x51545A00
    assert bytes([(TAG >> shift) & 0xFF for shift in (24, 16, 8, 0)]) == b"QTZ\x00"


def test_only_floats_are_exported(tmp_path, tiny_cfg):
    with pytest.raises(TypeError, match="only floats"):
        write_ingot({"tokens": np.arange(8, dtype=np.int32)}, tiny_cfg,
                    tmp_path / "ints.ingot", "default=2")
