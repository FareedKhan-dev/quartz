"""The one geometry every other module imports.

Nothing in this package hard-codes a shape. Everything is derived from a
QuartzConfig, so a two-layer model on a laptop and the 45M model on four cards
run identical code paths.
"""
from dataclasses import dataclass, field, asdict
from typing import Tuple

# --- tokenizer -------------------------------------------------------------
PAD_ID, EOS_ID, BOS_ID, UNK_ID = 0, 1, 2, 3

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
THINK_START, THINK_END = "<think>", "</think>"
TOOLS_START, TOOLS_END = "<tools>", "</tools>"
CALL_START, CALL_END = "<tool_call>", "</tool_call>"
RESULT_START, RESULT_END = "<tool_result>", "</tool_result>"

#: Order is load bearing. These land on ids 4 through 13 and the rest of the
#: package asserts those positions rather than looking them up.
CHAT_MARKERS = [
    IM_START, IM_END, THINK_START, THINK_END,
    TOOLS_START, TOOLS_END, CALL_START, CALL_END,
    RESULT_START, RESULT_END,
]
FIRST_MARKER_ID = 4

#: The eight characters that carry JSON structure. A merge table that has eaten
#: one of these can never be made to give it back, so they are isolated before
#: the tokenizer is ever fitted. See quartz.model.scribe.pre_tokenize.
ISOLATED_CHARS = "({[\",]})"

# --- quantisation ----------------------------------------------------------
CQ_GROUP = 128
CQ_BITS = (2, 3, 4)
TERNARY_BITS = 1.58
TERNARY_CENTROID = 1.2240064

#: The shipped scheme. Two families are promoted off two bits because they are
#: the two that mind: the routers feed an exponential map, and out_proj is the
#: write path into the residual stream.
DEFAULT_BITS_MAP = "default=2,loom.phi=4,attn.out_proj=3"

# --- the KV budget ---------------------------------------------------------
KV_BUDGET_BYTES = 11 * 1024 * 1024 + 512 * 1024
KV_GROUP = 32
KV_WINDOW_MIN = 160

# --- Imprint ---------------------------------------------------------------
IMPRINT_SUB_DIM = 128
IMPRINT_TAPS = 4
IMPRINT_SEED = 0x9E3779B9        # golden ratio, so per-table seeds decorrelate
IMPRINT_PRIME = 0x01000193       # the FNV-1a 32-bit prime


@dataclass
class QuartzConfig:
    """Everything that defines a Quartz model.

    Unknown keys are dropped rather than raising, so a checkpoint written by a
    later version still loads on an earlier one with its extra fields ignored.
    """

    vocab_size: int = 8192
    d_model: int = 512
    num_heads: int = 8
    num_kv_heads: int = 4
    num_layers: int = 27
    max_seq_len: int = 2048
    pad_token_id: int = PAD_ID

    rope_theta: float = 1e5
    dtype: str = "bfloat16"

    # Loom
    lanes: int = 4
    sinkhorn_iters: int = 20

    # Imprint
    imprint_sites: Tuple[int, ...] = (2, 15)
    imprint_orders: Tuple[int, ...] = (2, 3)
    imprint_heads: int = 0            # 0 derives it from d_model
    imprint_slots: int = 8192

    # heads
    dowser_dim: int = 128
    dowser_probes: int = 4
    gauge_probes: int = 8

    # runtime and deployment
    kv_window: int = 256              # 0 means "size it from the budget alone"
    kv_bits: int = 8
    act_bits: int = 8
    weight_bits: str = ""             # a bits map, stamped by the qapt stage

    # training toggles
    remat: bool = True
    scan_unroll: int = 1

    def __init__(self, **kwargs):
        valid = {f.name for f in self.__dataclass_fields__.values()}
        for k, v in kwargs.items():
            if k in valid:
                setattr(self, k, v)
        self.imprint_sites = tuple(self.imprint_sites)
        self.imprint_orders = tuple(self.imprint_orders)
        if not all(s < self.num_layers for s in self.imprint_sites):
            raise ValueError(
                f"imprint sites {self.imprint_sites} must all be below "
                f"num_layers {self.num_layers}")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must divide by num_heads")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be a multiple of num_kv_heads")

    # --- derived geometry, never stored --------------------------------
    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def hadamard_n(self) -> int:
        """Spin's working width: the next power of two at or above d_model."""
        return 1 << (self.d_model - 1).bit_length()

    @property
    def lane_dim(self) -> int:
        """The flattened width the Loom routers read."""
        return self.lanes * self.d_model

    @property
    def imprint_geometry(self):
        """(orders, heads, sub_dim). num_tables * sub_dim == d_model always."""
        orders = self.imprint_orders
        heads = self.imprint_heads or max(
            1, self.d_model // (len(orders) * IMPRINT_SUB_DIM))
        return orders, heads, self.d_model // (len(orders) * heads)

    @property
    def imprint_tables(self) -> int:
        orders, heads, _ = self.imprint_geometry
        return len(orders) * heads

    def kv_bytes_per_position(self) -> int:
        """What one cached token costs, including the int8 group scales.

        The memory sites cache their value stream too, which is the term people
        leave out and then wonder why the arithmetic is short.
        """
        per_layer = 2 * self.kv_dim + 2 * (self.kv_dim // KV_GROUP) * 4
        per_site = self.d_model + (self.d_model // KV_GROUP) * 4
        return self.num_layers * per_layer + len(self.imprint_sites) * per_site

    def budget_window(self) -> int:
        """How many positions fit in the KV budget, rounded down to a group."""
        w = (KV_BUDGET_BYTES // self.kv_bytes_per_position()) // KV_GROUP * KV_GROUP
        return max(KV_WINDOW_MIN, min(w, self.max_seq_len))

    def effective_window(self) -> int:
        """The window we actually run at, which is a choice, not a derivation."""
        return min(self.budget_window(), self.kv_window) if self.kv_window \
            else self.budget_window()

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_yaml(cls, path):
        import yaml
        with open(path) as fh:
            return cls(**(yaml.safe_load(fh) or {}))


#: Named geometries. `base` is the 45,211,383 parameter model the post builds.
PRESETS = {
    "base": dict(d_model=512, num_heads=8, num_kv_heads=4, num_layers=27,
                 imprint_sites=(2, 15)),
    "wide": dict(d_model=768, num_heads=12, num_kv_heads=6, num_layers=27,
                 imprint_sites=(2, 15)),
    "tiny": dict(d_model=128, num_heads=4, num_kv_heads=2, num_layers=2,
                 imprint_sites=(1,), imprint_slots=256, lanes=2,
                 max_seq_len=256, kv_window=128, remat=False),
}


def preset(name: str, **overrides) -> QuartzConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}, have {sorted(PRESETS)}")
    return QuartzConfig(**{**PRESETS[name], **overrides})
