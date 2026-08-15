"""The half of the package that is about tools rather than tensors.

Two modules, and neither imports JAX:

- `tools`  every way of declaring a tool (a function, a pydantic model, a raw
           dict) reduced to the one JSON schema Trellis compiles into tries.
- `fetch`  where the weights live and how they get there.

Both are pure Python over the standard library, so `from quartz.agent import
tool` costs nothing on a machine that has no accelerator and never will.
"""
from .fetch import DEFAULT_RELEASE, RELEASES, fetch_weights, weights_path
from .tools import Field, build_schema, is_pydantic_model, pydantic_schema, tool

__all__ = [
    "DEFAULT_RELEASE",
    "RELEASES",
    "Field",
    "build_schema",
    "fetch_weights",
    "is_pydantic_model",
    "pydantic_schema",
    "tool",
    "weights_path",
]
