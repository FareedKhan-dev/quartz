"""Every way of declaring a tool, reduced to one JSON schema.

The schema is not documentation. Trellis compiles `properties` into a trie and
masks the decoder against it, so a key missing here is a key the model
physically cannot spell, and a key that is here but wrong is one it will emit
happily. That makes three things worth testing: the type table, which decides
what a value is allowed to look like; the required computation, which decides
when the closing brace is legal; and the constraints, which are the only part of
a description the decoder can actually enforce.
"""
from __future__ import annotations

import enum
from typing import Annotated, Any, Literal, Optional, Union

import pytest

# _JSON_TYPES is the table under test. Imported rather than restated, so the
# parametrisation below fails the day a row is added and left untested.
from quartz.agent.tools import (
    _JSON_TYPES,
    Field,
    build_schema,
    is_pydantic_model,
    pydantic_schema,
    tool,
)

#: A default that cannot be serialised, so it must never reach the window.
UNSERIALISABLE = object()

#: One parameter per row of _JSON_TYPES, so the coverage assertion below can be
#: an equality rather than a hope.
ROWS: dict[str, type] = {
    "flag": bool,
    "count": int,
    "ratio": float,
    "text": str,
    "blob": bytes,
    "items": list,
    "pair": tuple,
    "unique": set,
    "frozen": frozenset,
    "mapping": dict,
    "nothing": None,
}


class Mode(enum.Enum):
    """Declared at module level, because a string annotation is resolved
    against the module's namespace and never against a function's locals."""

    ECO = "eco"
    AUTO = "auto"


def every_type(flag: bool, count: int, ratio: float, text: str, blob: bytes,
               items: list, pair: tuple, unique: set, frozen: frozenset,
               mapping: dict, nothing: None) -> str:
    """One argument per row of the type table."""
    return ""


def properties_of(obj: Any, **kwargs) -> dict[str, Any]:
    return build_schema(obj, **kwargs)["parameters"]["properties"]


def required_of(obj: Any) -> list[str]:
    return build_schema(obj)["parameters"]["required"]


# --- the type table ---------------------------------------------------------
def test_every_row_of_the_type_table():
    """bool before int, because bool is a subclass of int and the walk takes
    the first match. Get that pair backwards and every flag becomes an integer."""
    props = properties_of(every_type)
    tested = {type(None) if py is None else py for py in ROWS.values()}
    assert tested == set(_JSON_TYPES), "a row of the table has no test"

    for name, py in ROWS.items():
        want = _JSON_TYPES[type(None) if py is None else py]
        assert props[name]["type"] == want, name
    assert props["flag"]["type"] == "boolean"


def test_an_undeclared_argument_is_a_string():
    """That is what a text model copies out of a query, and declaring nothing
    should not silently widen to anything at all."""
    def fn(room):
        return room

    assert properties_of(fn)["room"] == {"type": "string"}


def test_containers_carry_their_element_type():
    def fn(names: list[str], counts: dict[str, int], pair: tuple[int, ...]):
        return names, counts, pair

    props = properties_of(fn)
    assert props["names"] == {"type": "array", "items": {"type": "string"}}
    assert props["counts"] == {"type": "object",
                               "additionalProperties": {"type": "integer"}}
    assert props["pair"] == {"type": "array", "items": {"type": "integer"}}


def test_a_literal_becomes_an_enum_with_a_type():
    def fn(mode: Literal["eco", "auto"]):
        return mode

    assert properties_of(fn)["mode"] == {"type": "string", "enum": ["eco", "auto"]}


def test_a_mixed_enum_keeps_its_members_and_drops_the_type():
    def fn(mode: Literal["eco", 1]):
        return mode

    assert properties_of(fn)["mode"] == {"enum": ["eco", 1]}


def test_an_enum_class_becomes_its_values():
    def fn(mode: Mode):
        return mode

    assert properties_of(fn)["mode"] == {"type": "string", "enum": ["eco", "auto"]}


def test_a_union_becomes_any_of():
    def fn(value: Union[int, str]):  # noqa: UP007 - the spelling is the point
        return value

    assert properties_of(fn)["value"] == {
        "anyOf": [{"type": "integer"}, {"type": "string"}]}


def test_a_type_we_do_not_model_is_a_string():
    """The honest answer: the model will copy text into it either way."""
    from collections.abc import Callable

    def fn(hook: Callable[[int], int]):
        return hook

    assert properties_of(fn)["hook"] == {"type": "string"}


# --- required ---------------------------------------------------------------
def test_required_is_everything_without_a_way_out():
    def fn(room: str, brightness: int = 100, tint: int | None = None,
           legacy: Optional[str] = None):  # noqa: UP045 - both spellings
        return room, brightness, tint, legacy

    assert required_of(fn) == ["room"]
    props = properties_of(fn)
    assert props["brightness"]["default"] == 100
    # None is dropped: absence from `required` already says it may be left out
    assert "default" not in props["tint"]
    assert "default" not in props["legacy"]


def test_an_optional_type_is_not_required_even_with_no_default():
    def fn(room: str | None):
        return room

    assert required_of(fn) == []


def test_a_docstring_can_declare_a_type_and_an_optional():
    """A docstring that says optional and a signature that gives no default
    disagree about the same fact, and a person wrote the docstring on purpose."""
    def fn(room, brightness, note):
        """Set a room.

        Args:
            room (str): which room.
            brightness (int, optional): 0 to 100.
            note: free text
                continued on a second line.
        """
        return room, brightness, note

    schema = build_schema(fn)
    props = schema["parameters"]["properties"]
    assert schema["description"] == "Set a room."
    assert props["room"] == {"type": "string", "description": "which room."}
    assert props["brightness"]["type"] == "integer"
    assert props["note"]["description"] == "free text continued on a second line."
    assert schema["parameters"]["required"] == ["room", "note"]


def test_star_args_are_dropped_not_widened():
    """The decoder masks keys against a trie of names, and a key it cannot name
    it cannot guarantee."""
    def fn(room: str, *extra: int, **rest: str):
        return room, extra, rest

    assert list(properties_of(fn)) == ["room"]


def test_self_and_cls_are_not_arguments():
    def method(self, room: str):
        return self, room

    assert list(properties_of(method)) == ["room"]


def test_an_unserialisable_default_never_reaches_the_window():
    def fn(when=UNSERIALISABLE):
        return when

    assert "default" not in properties_of(fn)["when"]
    assert required_of(fn) == []


# --- Field ------------------------------------------------------------------
def test_field_constraints_become_json_schema():
    """Only the constraints Trellis can enforce are worth setting: `enum`
    becomes a value trie and ge/le become a digit walker."""
    def fn(brightness: Annotated[int, Field("percent", ge=0, le=100)],
           mode: Annotated[str, Field("how", enum=["eco", "auto"])]):
        return brightness, mode

    props = properties_of(fn)
    assert props["brightness"] == {"type": "integer", "description": "percent",
                                   "minimum": 0, "maximum": 100}
    assert props["mode"] == {"type": "string", "description": "how",
                             "enum": ["eco", "auto"]}


def test_field_carries_every_spelling_a_declaration_might_use():
    def fn(a: Annotated[int, Field(gt=0, lt=10, multiple_of=2)],
           b: Annotated[str, Field(min_length=1, max_length=8, pattern="^[a-z]+$")]):
        return a, b

    props = properties_of(fn)
    assert props["a"] == {"type": "integer", "exclusiveMinimum": 0,
                          "exclusiveMaximum": 10, "multipleOf": 2}
    assert props["b"] == {"type": "string", "minLength": 1, "maxLength": 8,
                          "pattern": "^[a-z]+$"}


def test_a_field_default_makes_an_argument_optional():
    def fn(room: Annotated[str, Field("which room", default="kitchen")]):
        return room

    assert properties_of(fn)["room"]["default"] == "kitchen"
    assert required_of(fn) == []


def test_a_bare_string_annotation_is_the_description():
    """Because `Annotated[int, "0 to 100"]` is what people actually write."""
    def fn(brightness: Annotated[int, "  0 to 100  "]):
        return brightness

    assert properties_of(fn)["brightness"]["description"] == "0 to 100"


def test_field_is_a_value_object():
    assert Field("percent", ge=0).ge == 0
    assert Field("percent") == Field("percent")
    with pytest.raises(Exception, match="assign"):
        Field("percent").ge = 1


# --- the decorator ----------------------------------------------------------
def test_tool_attaches_a_schema_and_leaves_the_function_alone():
    @tool
    def set_lights(room: str, brightness: int = 100) -> str:
        """Set one room's brightness."""
        return f"{room} at {brightness}"

    assert set_lights("kitchen") == "kitchen at 100"      # still a plain call
    assert set_lights.tool_schema["name"] == "set_lights"
    assert set_lights.tool_schema["description"] == "Set one room's brightness."
    assert set_lights.tool_schema["parameters"]["required"] == ["room"]


def test_tool_takes_a_name_and_a_description():
    @tool(name="lights", description="set one room")
    def set_lights(room: str) -> str:
        return room

    assert set_lights.tool_schema["name"] == "lights"
    assert set_lights.tool_schema["description"] == "set one room"


def test_build_schema_reuses_the_decorated_parse():
    """Built once at decoration time. Re-parsing the docstring for every request
    a device ever serves is a cost with no benefit."""
    @tool
    def set_lights(room: str) -> str:
        return room

    again = build_schema(set_lights, name="lights")
    assert again["name"] == "lights"
    assert again["parameters"] == set_lights.tool_schema["parameters"]


def test_only_the_first_paragraph_reaches_the_window():
    """Every character of a description is spent inside a 256 token window."""
    @tool
    def fn() -> None:
        """The summary line.

        The paragraph after it is written for people reading the source, and
        they are not paying for it.
        """

    assert fn.tool_schema["description"] == "The summary line."


# --- dicts ------------------------------------------------------------------
def test_a_dict_is_normalised_not_rewritten():
    spec = {"name": "set_lights", "description": "d",
            "parameters": {"type": "object", "title": "drop me",
                           "properties": {"room": {"type": "string"}},
                           "required": ["room", "ghost"]}}
    schema = build_schema(spec)

    assert schema["name"] == "set_lights"
    assert "title" not in schema["parameters"]
    # a required key with no property has no trie, so the decoder would be asked
    # to force a name it cannot spell
    assert schema["parameters"]["required"] == ["room"]


def test_the_function_wrapper_and_input_schema_spellings_are_accepted():
    wrapped = build_schema({"type": "function",
                            "function": {"name": "a", "parameters": {}}})
    other = build_schema({"name": "b", "input_schema": {"type": "object"}})
    for schema in (wrapped, other):
        assert schema["parameters"]["type"] == "object"
        assert schema["parameters"]["properties"] == {}
        assert schema["parameters"]["required"] == []


def test_a_naked_json_schema_object_needs_a_name():
    naked = {"type": "object", "properties": {"room": {"type": "string"}}}
    assert build_schema(naked, name="set_lights")["name"] == "set_lights"
    with pytest.raises(ValueError, match="needs a 'name'"):
        build_schema(naked)


def test_a_reference_is_left_exactly_as_written():
    """The escape hatch. Rewriting someone's schema behind their back is worse
    than a constraint that declines to compile."""
    spec = {"name": "t", "parameters": {"type": "object",
                                        "properties": {"x": {"$ref": "#/$defs/X"}},
                                        "$defs": {"X": {"type": "string"}}}}
    schema = build_schema(spec)
    assert schema["parameters"]["properties"]["x"] == {"$ref": "#/$defs/X"}
    assert "$defs" in schema["parameters"]


def test_properties_and_required_are_always_present():
    """So no consumer has to guess whether a tool takes no arguments or forgot
    to say."""
    def fn():
        return None

    schema = build_schema(fn)
    assert schema["parameters"] == {"type": "object", "properties": {}, "required": []}


def test_something_that_is_not_a_tool_is_refused():
    with pytest.raises(TypeError, match="cannot build a tool schema"):
        build_schema(42)


# --- pydantic, without importing pydantic to decide ------------------------
def test_is_pydantic_model_is_duck_typed():
    class Fake:
        model_fields: dict = {}

        @classmethod
        def model_validate(cls, value):
            return value

    assert is_pydantic_model(Fake)
    assert not is_pydantic_model(Fake())
    assert not is_pydantic_model(dict)
    with pytest.raises(TypeError, match="not a pydantic model"):
        pydantic_schema(dict)


def test_a_pydantic_model_becomes_a_flat_schema():
    pydantic = pytest.importorskip("pydantic")

    class Invoice(pydantic.BaseModel):
        """Pull the totals off an invoice."""

        supplier: str
        total: float = pydantic.Field(description="in pounds", ge=0)
        note: str = ""

    schema = build_schema(Invoice)
    props = schema["parameters"]["properties"]

    assert schema["name"] == "invoice"
    assert schema["description"] == "Pull the totals off an invoice."
    assert props["supplier"] == {"type": "string"}
    assert props["total"]["type"] == "number"
    assert props["total"]["description"] == "in pounds"
    assert props["total"]["minimum"] == 0
    assert schema["parameters"]["required"] == ["supplier", "total"]
    assert props["note"]["default"] == ""


def test_a_nested_model_is_inlined_rather_than_referenced():
    """Trellis compiles tries straight off `properties`. A reference it would
    have to resolve at decode time is a guarantee it cannot make."""
    pydantic = pytest.importorskip("pydantic")

    class Address(pydantic.BaseModel):
        city: str

    class Order(pydantic.BaseModel):
        ship_to: Address

    params = build_schema(Order)["parameters"]
    assert params["properties"]["ship_to"] == {
        "type": "object", "properties": {"city": {"type": "string"}},
        "required": ["city"]}
    assert "$defs" not in str(params)


def test_a_self_referential_model_terminates():
    """It has no finite flat schema, and the decoder cannot mask against a shape
    it cannot enumerate."""
    pydantic = pytest.importorskip("pydantic")

    class Node(pydantic.BaseModel):
        name: str
        child: Node | None = None

    # the forward reference resolves once `Node` is bound in this frame
    Node.model_rebuild()

    params = build_schema(Node)["parameters"]
    assert params["properties"]["child"] == {
        "type": "object", "properties": {}, "required": []}
    assert params["required"] == ["name"]
