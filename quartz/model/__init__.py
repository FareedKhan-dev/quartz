"""The model: geometry, words, architecture, decoder, quantiser, container.

Importing this package imports none of its modules. Every name below is
resolved on first use, through the module `__getattr__`, and that is not a
micro-optimisation: `architecture`, `decode`, `graft` and half of `grist` need
JAX and Flax, which are optional extras. A tool that reads an Ingot header, or a
test that checks the tokenizer isolation rule, must not pay for a framework it
never calls, and a laptop with no accelerator must still be able to
`import quartz`.

So both of these work, and only the second one imports JAX::

    from quartz.model import QuartzConfig, preset          # free
    from quartz.model import QuartzNetwork                 # loads flax now

The modules, in the order the build uses them:

- `config`        the single source of every shape and constant
- `scribe`        the tokenizer, the isolation rule, and the turn format
- `architecture`  ZCRMSNorm, Spin, Porthole, Imprint, Loom, Dowser, Gauge
- `trellis`       the constrained decoder: tries, a state machine, a mask
- `decode`        the KV cache and the generation loop
- `grist`         rotate, index the direction, keep the length
- `ingot`         405 tensors and a tokenizer in one mmap-able file
- `graft`         LoRA over the scanned layer axis
- `run`           checkpoint save, load and averaging, shared by every stage
"""
from __future__ import annotations

import importlib
from typing import Any

#: Public names per submodule. Written out rather than discovered by importing
#: and reading `__all__`, because discovering it would mean importing all of it,
#: which is the one thing this file exists to avoid. A typo here surfaces as an
#: AttributeError at the edge of the package instead of as a silent import of
#: something else halfway through a training run.
_EXPORTS: dict[str, tuple[str, ...]] = {
    "config": ("BOS_ID", "CALL_END", "CALL_START", "CHAT_MARKERS", "CQ_BITS",
               "CQ_GROUP", "DEFAULT_BITS_MAP", "EOS_ID", "FIRST_MARKER_ID",
               "IMPRINT_PRIME", "IMPRINT_SEED", "IMPRINT_SUB_DIM", "IMPRINT_TAPS",
               "IM_END", "IM_START", "ISOLATED_CHARS", "KV_BUDGET_BYTES",
               "KV_GROUP", "KV_WINDOW_MIN", "PAD_ID", "PRESETS", "QuartzConfig",
               "RESULT_END", "RESULT_START", "TERNARY_BITS", "TERNARY_CENTROID",
               "THINK_END", "THINK_START", "TOOLS_END", "TOOLS_START", "UNK_ID",
               "preset"),
    "scribe": ("Scribe", "get_tokenizer", "pre_tokenize", "render",
               "train_tokenizer"),
    "architecture": ("Block", "Dowser", "Gauge", "Imprint", "Porthole",
                     "QuartzNetwork", "Spin", "Trunk", "ZCRMSNorm", "apply_rope",
                     "imprint_indices", "make_causal_mask", "make_packing_mask",
                     "param_count", "rope_tables", "walsh", "window_mask"),
    "trellis": ("ConstrainedDecoder", "SchemaConstraints", "State", "StateMachine",
                "TokenIndex", "ToolConstraints", "Trie", "apply_constraints",
                "build_token_strings", "digit_allowed"),
    "grist": ("codebook", "cq_quantize", "lloyd_max_gaussian", "mixed_quantise",
              "model_bytes", "parse_bits_map", "quant_leaf_names",
              "quantise_params", "walsh_np"),
    "ingot": ("ALIGN", "TAG", "pack_lsb", "read_ingot", "unpack_lsb",
              "write_ingot"),
    "decode": ("DecodeCfg", "decode_cfg", "forward_cached", "generate",
               "init_kv_cache"),
    "run": ("CHECKPOINT_FORMAT_VERSION", "average_checkpoints", "load_checkpoint",
            "save_checkpoint"),
    "graft": ("graft", "init_lora", "lora_target_paths", "merge_lora"),
}

#: symbol -> the submodule that defines it
_SYMBOLS: dict[str, str] = {
    name: module for module, names in _EXPORTS.items() for name in names
}

#: `from quartz.model import *` pulls every submodule, and therefore JAX. That
#: is what a star import asks for; every named import stays lazy.
__all__ = sorted([*_EXPORTS, *_SYMBOLS])


def __getattr__(name: str) -> Any:
    """Resolve a submodule, or one of its public names, on first access.

    PEP 562. This is what keeps JAX out of `import quartz`, and a test enforces
    it, because the rule is one convenience import at the top of this file away
    from being broken and impossible to notice until someone installs the
    package on a machine with no GPU.
    """
    if name in _EXPORTS:
        return importlib.import_module(f"{__name__}.{name}")
    module = _SYMBOLS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{module}"), name)


def __dir__() -> list[str]:
    return __all__
