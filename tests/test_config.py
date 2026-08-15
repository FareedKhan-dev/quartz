"""The geometry, and the four numbers everything else is quoted against.

45,211,383 parameters, 16,704 bytes of cache per position, a 704 position budget
window and a 256 position effective one. Three of those are derived and one is a
choice, and this file is where that distinction is written down: if a change to
the config moves any of them, the failure should say which.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from quartz.model.architecture import BASE_PARAM_COUNT, EXPORT_DROPPED, param_count
from quartz.model.config import (
    CHAT_MARKERS,
    FIRST_MARKER_ID,
    ISOLATED_CHARS,
    KV_BUDGET_BYTES,
    KV_GROUP,
    KV_WINDOW_MIN,
    PRESETS,
    QuartzConfig,
    preset,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


# --- the four numbers -------------------------------------------------------
def test_base_preset_is_45_211_383_parameters(base_cfg):
    """The headline count, summed from the components rather than quoted."""
    assert sum(param_count(base_cfg).values()) == 45_211_383
    assert BASE_PARAM_COUNT == 45_211_383


def test_kv_costs_16704_bytes_a_position(base_cfg):
    """Both cached streams, both group scales, and the memory sites too.

    The sites are the term that gets left out. Dropping them understates the
    cache by two layers' worth and overstates the window that fits in it.
    """
    assert base_cfg.kv_bytes_per_position() == 16_704

    per_layer = 2 * base_cfg.kv_dim + 2 * (base_cfg.kv_dim // KV_GROUP) * 4
    per_site = base_cfg.d_model + (base_cfg.d_model // KV_GROUP) * 4
    assert base_cfg.kv_bytes_per_position() == (
        base_cfg.num_layers * per_layer + len(base_cfg.imprint_sites) * per_site)
    assert base_cfg.kv_bytes_per_position() > base_cfg.num_layers * per_layer


def test_budget_window_is_704(base_cfg):
    """What the budget allows, rounded down to a whole cache group."""
    assert base_cfg.budget_window() == 704
    assert base_cfg.budget_window() % KV_GROUP == 0
    assert base_cfg.budget_window() * base_cfg.kv_bytes_per_position() <= KV_BUDGET_BYTES
    over = (base_cfg.budget_window() + KV_GROUP) * base_cfg.kv_bytes_per_position()
    assert over > KV_BUDGET_BYTES


def test_effective_window_is_256_and_is_a_choice(base_cfg):
    """704 fits; we run 256 because the sessions are 247 tokens long."""
    assert base_cfg.effective_window() == 256
    assert base_cfg.effective_window() == base_cfg.kv_window
    assert base_cfg.effective_window() < base_cfg.budget_window()

    unset = preset("base", kv_window=0)
    assert unset.effective_window() == unset.budget_window()

    greedy = preset("base", kv_window=100_000)
    assert greedy.effective_window() == greedy.budget_window()


def test_budget_window_never_falls_below_the_floor():
    """A geometry too heavy for the budget still gets a usable window."""
    heavy = QuartzConfig(num_layers=512, imprint_sites=(2, 15), kv_window=0)
    assert heavy.kv_bytes_per_position() * KV_WINDOW_MIN > KV_BUDGET_BYTES
    assert heavy.budget_window() == KV_WINDOW_MIN


def test_budget_window_is_capped_by_the_sequence_length():
    """You cannot cache more positions than the model was trained to see."""
    short = QuartzConfig(max_seq_len=256, kv_window=0)
    assert short.budget_window() == 256
    assert short.budget_window() < preset("base").budget_window()


# --- the parameter table ----------------------------------------------------
def test_param_count_is_derived_not_quoted():
    """Change the width and every component follows, or the table is a fiction."""
    small = preset("base", d_model=256, num_heads=8, num_kv_heads=4)
    counts = param_count(small)
    assert counts["embedding"] == small.vocab_size * small.d_model
    assert counts["final_norm"] == small.d_model
    assert sum(counts.values()) < BASE_PARAM_COUNT


def test_export_drops_only_foresight(base_cfg):
    counts = param_count(base_cfg)
    assert EXPORT_DROPPED == ("foresight",)
    assert set(EXPORT_DROPPED) <= set(counts)
    shipped = sum(counts.values()) - sum(counts[name] for name in EXPORT_DROPPED)
    assert 0 < shipped < sum(counts.values())


def test_the_tied_embedding_is_counted_once(base_cfg):
    """One table is both the input embedding and the output head.

    Untied, the same geometry would be 49,405,687 parameters, and the count
    below is what says the tie is real rather than a claim in a docstring.
    """
    counts = param_count(base_cfg)
    assert counts["embedding"] == base_cfg.vocab_size * base_cfg.d_model == 4_194_304
    assert not [name for name in counts if "head" in name or "unembed" in name]
    assert sum(counts.values()) + counts["embedding"] == 49_405_687


# --- derived geometry -------------------------------------------------------
def test_derived_geometry(base_cfg):
    assert base_cfg.head_dim == base_cfg.d_model // base_cfg.num_heads
    assert base_cfg.kv_dim == base_cfg.num_kv_heads * base_cfg.head_dim
    assert base_cfg.lane_dim == base_cfg.lanes * base_cfg.d_model
    assert base_cfg.head_dim % 2 == 0        # rope pairs the halves


@pytest.mark.parametrize("d_model", [128, 512, 513, 768, 1024])
def test_hadamard_width_is_the_next_power_of_two(d_model):
    """Spin works at next_pow2(d_model), because Walsh needs a power of two."""
    n = QuartzConfig(d_model=d_model, num_heads=1, num_kv_heads=1).hadamard_n
    assert n >= d_model
    assert n & (n - 1) == 0
    assert n // 2 < d_model


def test_imprint_geometry_tiles_d_model(base_cfg, tiny_cfg):
    """num_tables * sub_dim is d_model exactly, at every geometry."""
    for cfg in (base_cfg, tiny_cfg):
        orders, heads, sub_dim = cfg.imprint_geometry
        assert orders == cfg.imprint_orders
        assert cfg.imprint_tables == len(orders) * heads
        assert cfg.imprint_tables * sub_dim == cfg.d_model


# --- validation -------------------------------------------------------------
def test_unknown_keys_are_dropped_not_raised():
    """A checkpoint from a later version loads here, minus its new fields."""
    cfg = QuartzConfig(d_model=256, num_heads=8, num_kv_heads=4, invented_later=7)
    assert cfg.d_model == 256
    assert not hasattr(cfg, "invented_later")


def test_a_memory_site_past_the_last_layer_is_an_error():
    with pytest.raises(ValueError, match="imprint sites"):
        QuartzConfig(num_layers=4, imprint_sites=(2, 15))


def test_the_head_geometry_has_to_divide():
    with pytest.raises(ValueError, match="d_model must divide"):
        QuartzConfig(d_model=100, num_heads=8)
    with pytest.raises(ValueError, match="multiple of"):
        QuartzConfig(num_heads=8, num_kv_heads=3)


def test_imprint_sites_are_tuples_however_they_arrive():
    """YAML hands lists back, and a list would not hash or compare."""
    cfg = QuartzConfig(imprint_sites=[2, 15], imprint_orders=[2, 3])
    assert cfg.imprint_sites == (2, 15)
    assert cfg.imprint_orders == (2, 3)


# --- presets and files ------------------------------------------------------
def test_every_preset_builds_and_is_self_consistent():
    for name in PRESETS:
        cfg = preset(name)
        assert cfg.d_model % cfg.num_heads == 0
        assert all(site < cfg.num_layers for site in cfg.imprint_sites)
        assert 0 < cfg.effective_window() <= cfg.max_seq_len
        assert sum(param_count(cfg).values()) > 0


def test_unknown_preset_names_the_ones_that_exist():
    with pytest.raises(KeyError, match="unknown preset"):
        preset("enormous")


def test_overrides_beat_the_preset():
    cfg = preset("tiny", num_layers=3, vocab_size=64)
    assert (cfg.num_layers, cfg.vocab_size) == (3, 64)


def test_to_dict_round_trips(base_cfg):
    again = QuartzConfig(**base_cfg.to_dict())
    assert again.to_dict() == base_cfg.to_dict()
    assert again.effective_window() == base_cfg.effective_window()


def test_shipped_yaml_matches_the_base_preset(base_cfg):
    """The file people edit and the preset the code defaults to are one model."""
    path = CONFIG_DIR / "base.yaml"
    assert path.is_file(), f"the repository is missing {path}"
    from_file = QuartzConfig.from_yaml(path)
    assert sum(param_count(from_file).values()) == BASE_PARAM_COUNT
    assert from_file.kv_bytes_per_position() == base_cfg.kv_bytes_per_position()
    assert from_file.effective_window() == base_cfg.effective_window()


def test_from_yaml_says_which_file_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        QuartzConfig.from_yaml(tmp_path / "nothing.yaml")


# --- the constants other modules assert against -----------------------------
def test_chat_markers_are_ten_and_start_at_four():
    assert len(CHAT_MARKERS) == len(set(CHAT_MARKERS)) == 10
    assert FIRST_MARKER_ID == 4
    assert all(marker.startswith("<") and marker.endswith(">") for marker in CHAT_MARKERS)


def test_isolated_characters_are_the_eight_that_carry_json():
    assert len(ISOLATED_CHARS) == len(set(ISOLATED_CHARS)) == 8
    assert set(ISOLATED_CHARS) == set('{}[]",()')
