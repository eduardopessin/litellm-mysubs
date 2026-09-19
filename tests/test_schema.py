"""Schema sanitisation for Cloud Code Assist.

The values expected here were established by differential testing against
``normalizeSchemaForCCA`` from `@oh-my-pi/pi-ai` 18.2.6 running under Node (3167 cases,
including random fuzz, all identical). They are not what one assumes OMP does.
"""

from __future__ import annotations

from typing import Any

from litellm_mysubs.wire.schema import CCA_FALLBACK_SCHEMA, normalize_for_cca


def test_simple_schema_passes_through_unchanged() -> None:
    """If a trivial schema does not pass through intact, every tool loses the typing of
    its arguments because of sanitisation that had nothing to sanitise."""
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Path"}},
        "required": ["path"],
    }
    assert normalize_for_cca(schema) == schema


def test_type_array_with_null_becomes_a_scalar_type() -> None:
    """``type: ["string","null"]`` has no representation in the CCA proto: the ``type``
    field is scalar. Letting it through as a list would drop it into the fallback and the
    tool would lose all typing because of a single optional argument."""
    out = normalize_for_cca(
        {
            "type": "object",
            "properties": {"a": {"type": ["string", "null"]}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        }
    )
    assert out["properties"]["a"] == {"type": "string"}


def test_anyof_with_a_bare_null_branch_frees_the_field_from_required() -> None:
    """Zod's idiomatic form for "optional" is ``anyOf: [T, {type: null}]``.

    CCA cannot express nullability, but it can express *absence*: the ``null`` branch
    disappears and the field leaves ``required``. Keeping it mandatory would force the
    model to invent a value for an argument the client declared may be missing.
    """
    out = normalize_for_cca(
        {
            "type": "object",
            "properties": {
                "a": {"anyOf": [{"type": "string", "enum": ["x", "y"]}, {"type": "null"}]}
            },
            "required": ["a", "b"],
        }
    )
    assert out["properties"]["a"] == {"type": "string", "enum": ["x", "y"]}
    assert out["required"] == ["b"]


def test_anyof_of_same_typed_enums_unions_the_members() -> None:
    """Collapsing to the first branch would lose the members of the rest: the model would
    stop knowing that ``"b"`` is a legal value and would never propose it."""
    out = normalize_for_cca(
        {
            "anyOf": [
                {"type": "string", "enum": ["a", "b"]},
                {"type": "string", "enum": ["b", "c"]},
            ]
        }
    )
    assert out == {"type": "string", "enum": ["a", "b", "c"]}


def test_anyof_of_mixed_types_collapses_and_dumps_each_branch_constraint() -> None:
    """The proto's ``type`` field is scalar, so the union has to collapse. The constraints
    of the discarded branches go into the description — silently deleting them would leave
    the model violating a ``minLength`` it has no way of knowing about."""
    out = normalize_for_cca(
        {
            "anyOf": [
                {"type": "string", "minLength": 1},
                {"type": "integer", "minimum": 0},
            ]
        }
    )
    assert out["type"] == "string"
    assert "minLength" not in out
    assert "minLength: 1" in out["description"]
    assert "minimum: 0" in out["description"]


def test_anyof_with_a_branch_without_enum_does_not_narrow_to_the_other_members() -> None:
    """A branch without ``enum`` accepts everything the branch with ``enum`` accepts and
    more. Collapsing to the enum branch would reject arguments the original schema
    admits."""
    out = normalize_for_cca(
        {
            "type": "object",
            "properties": {"s": {"anyOf": [{"type": "string", "enum": ["a"]}, {"type": "string"}]}},
        }
    )
    assert out["properties"]["s"] == {"type": "string"}


def test_anyof_of_objects_unions_properties_and_keeps_only_the_common_required() -> None:
    """``anyOf`` accepts any one branch. Keeping mandatory what only one branch requires
    would reject valid instances; discarding the other branch's properties would hide
    legitimate arguments from the model."""
    out = normalize_for_cca(
        {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                    "required": ["a", "b"],
                },
                {
                    "type": "object",
                    "properties": {"a": {"type": "string"}, "c": {"type": "string"}},
                    "required": ["a"],
                },
            ]
        }
    )
    assert set(out["properties"]) == {"a", "b", "c"}
    assert out["required"] == ["a"]


def test_residual_not_falls_into_the_fallback() -> None:
    """``not`` has no field in the proto and there is no projection that expresses it.
    Sending it would make the backend return 400 and the whole request — not just this
    tool — would fail."""
    out = normalize_for_cca({"type": "object", "properties": {"x": {"not": {"type": "string"}}}})
    assert out == CCA_FALLBACK_SCHEMA


def test_unresolvable_external_ref_is_removed_without_dragging_the_schema_down() -> None:
    """A ``$ref`` pointing outside the document cannot be inlined. The field would show up
    as an unknown name in protojson, so it disappears — the tool survives with an untyped
    argument instead of the whole schema going down."""
    out = normalize_for_cca(
        {
            "type": "object",
            "properties": {"x": {"$ref": "https://example.com/s.json"}, "y": {"type": "string"}},
        }
    )
    assert out["properties"] == {"x": {}, "y": {"type": "string"}}


def test_local_ref_is_inlined_from_defs() -> None:
    """``$defs`` is one of the fields the proto does not have. Without inlining before
    removing it, the property that pointed at it would lose all its structure."""
    out = normalize_for_cca(
        {
            "$defs": {"Node": {"type": "object", "properties": {"v": {"type": "string"}}}},
            "type": "object",
            "properties": {"n": {"$ref": "#/$defs/Node"}},
        }
    )
    assert out["properties"]["n"] == {"type": "object", "properties": {"v": {"type": "string"}}}
    assert "$defs" not in out


def test_removed_constraints_go_to_the_description_preserving_the_existing_one() -> None:
    """``pattern`` and ``minLength`` do not exist in the proto. If they vanished without a
    trace, the model would generate arguments the tool rejects and would have no way to
    tell why; and if they replaced the description, the author's text would be lost."""
    out = normalize_for_cca(
        {
            "type": "object",
            "properties": {
                "p": {"type": "string", "pattern": "^a+$", "minLength": 3, "description": "Name"}
            },
        }
    )
    p = out["properties"]["p"]
    assert p["type"] == "string"
    assert "pattern" not in p and "minLength" not in p
    assert p["description"] == 'Name\n\n{pattern: "^a+$", minLength: 3}'


def test_snake_case_field_name_is_renamed_before_being_interpreted() -> None:
    """``any_of`` is the form python-genai emits. Without renaming it, it would not be
    recognised as a combinator: it would pass through raw as an unknown field and the
    request would take a 400."""
    out = normalize_for_cca(
        {
            "type": "object",
            "properties": {"a": {"any_of": [{"type": "string"}, {"type": "number"}]}},
        }
    )
    assert out["properties"]["a"] == {"type": "string"}


def test_object_typed_node_gets_properties_even_if_empty() -> None:
    """A ``type: object`` without ``properties`` is read as an opaque object, and the tool
    would end up receiving arguments it never declared."""
    assert normalize_for_cca({"type": "object"}) == {"type": "object", "properties": {}}


def test_anyof_of_const_becomes_an_enum_with_the_inferred_type() -> None:
    """CCA has neither ``const`` nor combinators, but it does have ``enum``. Without this
    translation a union of literals — the way Zod writes enums — always fell into the
    fallback."""
    out = normalize_for_cca(
        {
            "type": "object",
            "properties": {"mode": {"anyOf": [{"const": "read"}, {"const": "write"}]}},
        }
    )
    assert out["properties"]["mode"] == {"enum": ["read", "write"], "type": "string"}


def test_property_name_equal_to_a_keyword_is_not_treated_as_a_combinator() -> None:
    """Inside ``properties`` the keys are the user's names. Treating a property called
    ``not`` as the keyword would send a perfectly valid schema into the fallback."""
    schema = {
        "type": "object",
        "properties": {"not": {"type": "number"}, "anyOf": {"type": "string"}},
    }
    assert normalize_for_cca(schema) == schema


def test_reference_cycle_terminates_instead_of_recursing_forever() -> None:
    """A self-referential schema coming from an MCP server would hang the proxy process —
    not just the request — if the traversal did not cut the cycle."""
    cyclic: dict[str, Any] = {"type": "object", "properties": {}}
    cyclic["properties"]["self"] = cyclic
    assert normalize_for_cca(cyclic) == {"type": "object", "properties": {"self": {}}}


def test_cycle_guard_does_not_confuse_distinct_nodes_through_address_reuse() -> None:
    """CPython's ``id()`` is the address: a temporary node freed mid-traversal returns the
    same ``id`` to the next one, and a guard that stores only integers declares that
    sibling already visited, truncating it to ``{}``.

    Here that would make the residual ``not`` disappear before the residue check, and a
    schema CCA rejects with 400 would go out on the wire instead of falling into the
    fallback. The original's ``WeakMap`` does not have the problem because the key is the
    live object.
    """
    out = normalize_for_cca({"properties": {"p0": {"properties": {"p0": {"not": {}}, "p1": {}}}}})
    assert out == CCA_FALLBACK_SCHEMA


def test_input_schema_is_not_mutated() -> None:
    """The schema belongs to the caller and is reused on every request. Mutating it would
    make the second request sanitise an already sanitised schema, accumulating spill in
    the description."""
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string", "minLength": 2}},
        "required": ["a"],
    }
    before = repr(schema)
    normalize_for_cca(schema)
    assert repr(schema) == before
