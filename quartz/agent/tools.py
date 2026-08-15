"""Every way of declaring a tool, reduced to one JSON schema.

A caller declares tools as plain functions, as pydantic models, or as raw dicts
they already had. The model sees the same thing in all three cases: a name, a
description, and a flat JSON Schema object with `properties` and `required`.

That object is not decoration. Trellis compiles `properties` into a trie and
masks the decoder against it, so a key that is missing here is a key the model
physically cannot spell, and a key that is here but wrong is one it will happily
emit. The schema is the guarantee.

Nothing in this module imports pydantic. A package whose whole point is fitting
in fourteen megabytes should not pull in a validation library to read a class
attribute, so pydantic models are recognised by the attributes they have carried
since v1 and read through those.
"""
from __future__ import annotations

import enum
import inspect
import json
import re
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

__all__ = ["Field", "build_schema", "is_pydantic_model", "pydantic_schema", "tool"]

#: Sentinel for "no value", distinct from None, which is a legal default.
_MISSING = object()

#: Python types to JSON Schema types. `bool` precedes `int` because bool is a
#: subclass of int and the subclass walk below takes the first match.
_JSON_TYPES: dict[type, str] = {
    bool: "boolean",
    int: "integer",
    float: "number",
    str: "string",
    bytes: "string",
    list: "array",
    tuple: "array",
    set: "array",
    frozenset: "array",
    dict: "object",
    type(None): "null",
}

_ARRAY_ORIGINS = (list, tuple, set, frozenset)
_OBJECT_ORIGINS = (dict,)

#: Type names we trust in a Google docstring, for `room (str): the room` on a
#: parameter with no annotation. Nothing here is evaluated; a name we do not
#: know leaves the argument a string.
_DOC_TYPES = {
    "str": "string", "string": "string", "text": "string",
    "int": "integer", "integer": "integer",
    "float": "number", "number": "number", "double": "number",
    "bool": "boolean", "boolean": "boolean",
    "list": "array", "array": "array", "sequence": "array", "tuple": "array",
    "dict": "object", "object": "object", "mapping": "object",
}

#: Constraint attribute names to their JSON Schema keys. Both spellings are
#: accepted so a declaration written against pydantic (`ge`, `le`) and one
#: written against JSON Schema (`minimum`, `maximum`) both land.
_CONSTRAINTS = {
    "ge": "minimum", "minimum": "minimum",
    "le": "maximum", "maximum": "maximum",
    "gt": "exclusiveMinimum", "exclusive_minimum": "exclusiveMinimum",
    "lt": "exclusiveMaximum", "exclusive_maximum": "exclusiveMaximum",
    "min_length": "minLength", "max_length": "maxLength",
    "min_items": "minItems", "max_items": "maxItems",
    "pattern": "pattern", "multiple_of": "multipleOf",
}

_ARG_SECTIONS = {"args", "arguments", "parameters", "params",
                 "keyword args", "keyword arguments"}
_OTHER_SECTIONS = {"returns", "return", "yields", "yield", "raises", "example",
                   "examples", "note", "notes", "attributes", "warns",
                   "warnings", "references", "see also", "todo"}
_ARG_LINE = re.compile(r"^(?P<name>\*{0,2}\w+)\s*(?:\((?P<type>[^)]*)\))?\s*:\s?(?P<text>.*)$")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class Field:
    """Constraints for one argument, carried inside `Annotated[...]`.

    Named after pydantic's Field and accepting the same spellings, so a
    declaration can move here without being rewritten, and so this package never
    has to import pydantic to read one::

        def set_lights(room: str,
                       brightness: Annotated[int, Field("percent", ge=0, le=100)]):

    Only the constraints Trellis can enforce are worth setting: `enum` becomes a
    value trie and `ge`/`le` become a digit walker. The rest is documentation the
    model reads as text, and every character of it costs window.
    """

    description: str = ""
    default: Any = _MISSING
    enum: Sequence[Any] | None = None
    ge: float | None = None
    le: float | None = None
    gt: float | None = None
    lt: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None
    multiple_of: float | None = None


def is_pydantic_model(obj: Any) -> bool:
    """True for a pydantic model class, decided without importing pydantic.

    Detection is by the two attributes a model class has always exposed: a field
    map and a validating constructor. Importing pydantic to run an isinstance
    check would make a validation library a hard dependency of an inference
    package that otherwise needs numpy and a tokenizer.
    """
    if not isinstance(obj, type):
        return False
    v2 = hasattr(obj, "model_fields") and hasattr(obj, "model_validate")
    v1 = hasattr(obj, "__fields__") and hasattr(obj, "parse_obj")
    return bool(v2 or v1)


def pydantic_schema(model: type, *, name: str | None = None,
                    description: str | None = None) -> dict[str, Any]:
    """Turn a pydantic model class into a tool schema.

    Built field by field rather than through `model_json_schema()`, because that
    emits `$defs` and `$ref` for every nested model, and Trellis compiles tries
    straight off `properties`. A reference the decoder would have to resolve at
    decode time is a guarantee it cannot make, so nested models are inlined here
    instead.
    """
    if not is_pydantic_model(model):
        raise TypeError(f"{model!r} is not a pydantic model class")
    # __doc__ rather than getdoc(), which walks the MRO and would hand back the
    # base class's own documentation for a model that declares none.
    doc = inspect.cleandoc(model.__doc__ or "")
    return {
        "name": name or _snake(model.__name__),
        "description": description if description is not None else _summary(doc),
        "parameters": _model_object(model),
    }


def build_schema(obj: Any, *, name: str | None = None,
                 description: str | None = None) -> dict[str, Any]:
    """Build the tool schema for a callable, a pydantic model or a raw dict.

    The returned shape is fixed, because everything downstream reads it::

        {"name": str,
         "description": str,
         "parameters": {"type": "object", "properties": {...}, "required": [...]}}

    `properties` and `required` are always present, even when empty, so no
    consumer has to guess whether a tool takes no arguments or forgot to say.
    """
    if isinstance(obj, Mapping):
        return _dict_schema(obj, name=name, description=description)
    cached = getattr(obj, "tool_schema", None)
    if isinstance(cached, Mapping):
        # decorated with @tool: reuse the parse, allow this call to rename it
        return _dict_schema(cached, name=name, description=description)
    if is_pydantic_model(obj):
        return pydantic_schema(obj, name=name, description=description)
    if callable(obj):
        return _callable_schema(obj, name=name, description=description)
    raise TypeError(
        f"cannot build a tool schema from {type(obj).__name__}; declare a tool as "
        f"a function, a pydantic model, or a dict with 'name' and 'parameters'")


def tool(fn: Any = None, *, name: str | None = None,
         description: str | None = None) -> Any:
    """Attach a JSON schema to a function and leave the function alone.

    Usable bare or with arguments::

        @tool
        def start_timer(minutes: int) -> str: ...

        @tool(name="lights", description="set one room's brightness")
        def set_lights(room: str, brightness: int = 100) -> str: ...

    The schema is built once, at decoration time, and parked on
    `fn.tool_schema`. Building it per turn would re-parse the same docstring for
    every request the device ever serves, and the function is returned unwrapped
    so calling it stays a plain call with no interposed frame.
    """
    def wrap(f: Any) -> Any:
        f.tool_schema = build_schema(f, name=name, description=description)
        return f

    return wrap if fn is None else wrap(fn)


# --- annotations -----------------------------------------------------------
def _annotation_schema(ann: Any, seen: tuple = ()) -> tuple[dict[str, Any], bool]:
    """Return (schema, allows_none) for one annotation.

    `allows_none` is separate from the schema because it does not describe the
    value, it decides whether the argument lands in `required`.
    """
    if ann is inspect.Parameter.empty or ann is Any or ann is None:
        # An undeclared argument is a string. That is what a text model copies
        # out of a query, and declaring nothing should not silently widen to
        # anything at all.
        return {"type": "string"}, False

    meta = getattr(ann, "__metadata__", None)
    if meta is not None:                      # Annotated[T, ...]
        schema, allows_none = _annotation_schema(ann.__origin__, seen)
        for item in meta:
            schema.update(_read_constraints(item))
        return schema, allows_none

    origin, args = get_origin(ann), get_args(ann)

    if origin is Literal:
        values = list(args)
        return _enum_schema(values), any(v is None for v in values)

    if origin is Union or origin is getattr(types, "UnionType", ()):
        inner = [a for a in args if a is not type(None)]
        allows_none = len(inner) != len(args)
        if not inner:
            return {"type": "null"}, True
        if len(inner) == 1:
            return _annotation_schema(inner[0], seen)[0], allows_none
        return {"anyOf": [_annotation_schema(a, seen)[0] for a in inner]}, allows_none

    if origin in _ARRAY_ORIGINS or (isinstance(origin, type)
                                    and issubclass(origin, (list, tuple, set))):
        schema: dict[str, Any] = {"type": "array"}
        items = [a for a in args if a is not Ellipsis]
        if items:
            schema["items"] = _annotation_schema(items[0], seen)[0]
        return schema, False

    if origin in _OBJECT_ORIGINS or (isinstance(origin, type)
                                     and issubclass(origin, dict)):
        schema = {"type": "object"}
        if len(args) == 2 and args[1] is not Any:
            schema["additionalProperties"] = _annotation_schema(args[1], seen)[0]
        return schema, False

    if isinstance(ann, type):
        if issubclass(ann, enum.Enum):
            return _enum_schema([member.value for member in ann]), False
        if is_pydantic_model(ann):
            return _model_object(ann, seen), False
        exact = _JSON_TYPES.get(ann)
        if exact is not None:
            return {"type": exact}, ann is type(None)
        for base, json_type in _JSON_TYPES.items():
            if base is not type(None) and issubclass(ann, base):
                return {"type": json_type}, False

    # A generic we do not model (a Callable, a third-party alias). String is the
    # honest answer: the model will copy text into it either way.
    return {"type": "string"}, False


def _enum_schema(values: list[Any]) -> dict[str, Any]:
    """An enum with a type when every member shares one, and bare when not."""
    kinds = {_JSON_TYPES.get(type(v)) for v in values if v is not None}
    if len(kinds) == 1 and None not in kinds:
        return {"type": kinds.pop(), "enum": values}
    return {"enum": values}


def _read_constraints(meta: Any) -> dict[str, Any]:
    """Read JSON Schema constraints off one piece of Annotated metadata.

    Duck-typed on purpose. It reads our own Field, pydantic's FieldInfo, and the
    annotated-types objects pydantic hides constraints inside, without importing
    any of them. A bare string is taken as the description, because
    `Annotated[int, "0 to 100"]` is what people actually write.
    """
    if isinstance(meta, str):
        return {"description": _WS.sub(" ", meta).strip()} if meta.strip() else {}

    out: dict[str, Any] = {}
    desc = getattr(meta, "description", None)
    if isinstance(desc, str) and desc.strip():
        out["description"] = _WS.sub(" ", desc).strip()

    choices = getattr(meta, "enum", None)
    if choices is None:
        choices = getattr(meta, "choices", None)
    if choices and not isinstance(choices, type):
        out["enum"] = list(choices)

    for attr, key in _CONSTRAINTS.items():
        value = getattr(meta, attr, None)
        if value is not None and not callable(value):
            out[key] = value

    default = getattr(meta, "default", None)
    if _usable_default(default):
        out["default"] = default

    for inner in getattr(meta, "metadata", ()) or ():
        out.update(_read_constraints(inner))
    return out


# --- pydantic --------------------------------------------------------------
def _model_object(model: type, seen: tuple = ()) -> dict[str, Any]:
    """The `parameters` half of a model's schema, with nested models inlined."""
    if model in seen:
        # A self-referential model has no finite flat schema, and the decoder
        # cannot mask against a shape it cannot enumerate.
        return {"type": "object", "properties": {}, "required": []}
    seen = (*seen, model)

    props: dict[str, Any] = {}
    required: list[str] = []
    for fname, info in _model_fields(model).items():
        schema, allows_none = _annotation_schema(info["annotation"], seen)
        schema.update(info["constraints"])
        if info["description"]:
            schema.setdefault("description", info["description"])
        props[fname] = schema
        if info["required"] and not allows_none:
            # The model's own answer about requiredness is the authority. A
            # sentinel sitting in `default` because the field has none is not a
            # value, and must not reach the window as though it were.
            schema.pop("default", None)
            required.append(fname)
        elif _usable_default(info["default"]):
            schema.setdefault("default", info["default"])
    return {"type": "object", "properties": props, "required": required}


def _model_fields(model: type) -> dict[str, dict[str, Any]]:
    """Normalise a v1 or v2 model's field map into one shape.

    The two versions disagree about where every one of these four facts lives,
    which is exactly the reason to flatten them here once instead of branching
    at each use.
    """
    out: dict[str, dict[str, Any]] = {}
    v2_fields = getattr(model, "model_fields", None)
    if isinstance(v2_fields, Mapping) and hasattr(model, "model_validate"):
        for fname, info in v2_fields.items():
            is_required = getattr(info, "is_required", None)
            required = bool(is_required()) if callable(is_required) else True
            default = _MISSING if required else getattr(info, "default", _MISSING)
            out[fname] = {
                "annotation": getattr(info, "annotation", Any),
                "description": getattr(info, "description", "") or "",
                "required": required,
                "default": default,
                "constraints": _read_constraints(info),
            }
        return out

    for fname, mf in (getattr(model, "__fields__", {}) or {}).items():
        field_info = getattr(mf, "field_info", None)
        required = bool(getattr(mf, "required", True))
        out[fname] = {
            "annotation": getattr(mf, "outer_type_", getattr(mf, "annotation", Any)),
            "description": getattr(field_info, "description", "") or "",
            "required": required,
            "default": _MISSING if required else getattr(mf, "default", _MISSING),
            "constraints": _read_constraints(field_info) if field_info else {},
        }
    return out


# --- callables -------------------------------------------------------------
def _callable_schema(fn: Any, *, name: str | None = None,
                     description: str | None = None) -> dict[str, Any]:
    """Read a function's signature and its Google docstring into a schema."""
    doc = inspect.getdoc(fn) or ""
    summary, arg_docs = _parse_docstring(doc)
    try:
        # Resolves string annotations, which is not optional: a module with
        # `from __future__ import annotations` hands back nothing but strings.
        hints = get_type_hints(fn, include_extras=True)
    except Exception:
        hints = dict(getattr(fn, "__annotations__", {}))

    props: dict[str, Any] = {}
    required: list[str] = []
    for pname, param in inspect.signature(fn).parameters.items():
        if pname in ("self", "cls"):
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            # *args and **kwargs cannot be declared. The decoder masks argument
            # keys against a trie of names, and a key it cannot name it cannot
            # guarantee, so an undeclarable parameter is dropped rather than
            # quietly widened to anything.
            continue

        doc_type, doc_optional, doc_text = arg_docs.get(pname, (None, False, ""))
        annotation = hints.get(pname, param.annotation)
        schema, allows_none = _annotation_schema(annotation)
        if annotation is param.empty and doc_type:
            schema = {"type": doc_type}
        if doc_text:
            schema.setdefault("description", doc_text)
        if param.default is not param.empty and _usable_default(param.default):
            schema.setdefault("default", param.default)
        props[pname] = schema

        optional = (param.default is not param.empty or allows_none
                    or doc_optional or "default" in schema)
        if not optional:
            required.append(pname)

    return {
        "name": name or getattr(fn, "__name__", type(fn).__name__),
        "description": description if description is not None else summary,
        "parameters": {"type": "object", "properties": props, "required": required},
    }


def _parse_docstring(doc: str) -> tuple[str, dict[str, tuple[str | None, bool, str]]]:
    """Split a Google-style docstring into a summary and per-argument text.

    Returns `(summary, {name: (json_type, optional, description)})`. The type in
    `room (str, optional):` is used only when the parameter carries no
    annotation, and `optional` there is honoured, because a docstring that says
    optional and a signature that gives no default disagree about the same fact
    and the docstring is the one a person wrote on purpose.
    """
    args: dict[str, tuple[str | None, bool, str]] = {}
    if not doc:
        return "", args

    section: str | None = None
    head_lines: list[str] = []
    body: list[str] = []
    for raw in doc.expandtabs().splitlines():
        stripped = raw.strip()
        label = stripped[:-1].strip().lower() if stripped.endswith(":") else None
        if label in _ARG_SECTIONS:
            section = "args"
            continue
        if label in _OTHER_SECTIONS:
            section = "other"
            continue
        if section is None:
            head_lines.append(stripped)
        elif section == "args":
            body.append(raw)

    entries = [line for line in body if line.strip()]
    if entries:
        base = min(len(line) - len(line.lstrip()) for line in entries)
        current: str | None = None
        for line in entries:
            indent = len(line) - len(line.lstrip())
            match = _ARG_LINE.match(line.strip()) if indent == base else None
            if match:
                current = match["name"].lstrip("*")
                parts = [p.strip().lower() for p in (match["type"] or "").split(",")]
                first = parts[0].split(" ")[0] if parts and parts[0] else ""
                args[current] = (_DOC_TYPES.get(first), "optional" in parts,
                                 match["text"].strip())
            elif current is not None:
                doc_type, optional, text = args[current]
                args[current] = (doc_type, optional,
                                 f"{text} {line.strip()}".strip())

    return _summary("\n".join(head_lines)), args


def _summary(text: str) -> str:
    """The leading paragraph, whitespace collapsed.

    Only the first paragraph, because every character of a description is spent
    inside a 256 token window at inference time. The paragraph after the summary
    is written for people reading the source, and they are not paying for it.
    """
    para: list[str] = []
    for line in text.strip().splitlines():
        if not line.strip():
            break
        para.append(line.strip())
    return _WS.sub(" ", " ".join(para)).strip()


# --- dicts -----------------------------------------------------------------
def _dict_schema(spec: Mapping[str, Any], *, name: str | None = None,
                 description: str | None = None) -> dict[str, Any]:
    """Normalise a schema someone already had into the shape used here.

    Accepts the bare form, the `{"type": "function", "function": {...}}` wrapper,
    an `input_schema` key in place of `parameters`, and a naked JSON Schema
    object given a name. `$ref` is left exactly as written: this is the escape
    hatch, and rewriting someone's schema behind their back is worse than a
    constraint that declines to compile.
    """
    spec = dict(spec)
    inner = spec.get("function")
    if isinstance(inner, Mapping):
        spec = dict(inner)

    tool_name = name or spec.get("name")
    desc = description if description is not None else spec.get("description", "")

    params = spec.get("parameters", spec.get("input_schema"))
    if params is None and spec.get("type") == "object":
        params = spec                      # a naked JSON Schema object
        tool_name = name or spec.get("title")
    if not tool_name:
        raise ValueError(
            "a dict tool needs a 'name', either in the dict or as name=; a bare "
            "JSON Schema object carries none")

    params = dict(params or {})
    params.pop("title", None)
    params.pop("$schema", None)
    props = dict(params.get("properties") or {})
    # A required key with no property has no trie, so the decoder would be asked
    # to force a name it cannot spell. Dropping it here keeps that impossible.
    required = [k for k in (params.get("required") or []) if k in props]
    params.update(type="object", properties=props, required=required)
    return {"name": str(tool_name), "description": str(desc or ""),
            "parameters": params}


# --- small helpers ---------------------------------------------------------
def _snake(name: str) -> str:
    """CamelCase to snake_case, because that is how the training data spells a
    tool name and the model spells what it saw."""
    return _CAMEL.sub("_", name).lower()


def _jsonable(value: Any) -> bool:
    """Whether a value survives the trip into the prompt as JSON."""
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def _usable_default(value: Any) -> bool:
    """Whether a default is worth putting in the window.

    None is dropped: absence from `required` already says the argument may be
    left out, and a 256 token window should not carry the same fact twice.
    Ellipsis and anything unserialisable are dropped too, which is how
    pydantic's undefined sentinel disappears without this file naming it.
    """
    return (value is not None and value is not _MISSING and value is not Ellipsis
            and _jsonable(value))
