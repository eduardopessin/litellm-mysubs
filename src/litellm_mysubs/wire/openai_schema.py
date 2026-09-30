"""Tool JSON Schema normalization for the OpenAI Responses wire (the Codex backend).

Port of the path omp runs on every Codex function tool
(``providers/openai-codex-responses.ts :: convertOpenAICodexResponsesTools``):

    adaptSchemaForStrict(sanitizeSchemaForOpenAIResponses(toolWireSchema(tool)), strict)

``toolWireSchema`` (draft 2020-12 upgrade plus the wire post-processing) is shared with
the Cloud Code Assist path and lives in `wire.schema`; this module holds the Responses half.
The backend rejects ``oneOf`` in a tool schema even without strict mode, and every
``type: "object"`` node that has no ``properties`` — sending the client's schema raw made
the whole request fail, not just the tool.
"""

from __future__ import annotations

from typing import Any, Final

from .schema import tool_wire_schema, upgrade_to_2020_12

JsonObject = dict[str, Any]

# omp: utils/schema/normalize.ts :: OPENAI_RESPONSES_SCHEMA_ARRAY_KEYS
_ARRAY_KEYS: Final = frozenset({"anyOf", "oneOf", "allOf", "prefixItems"})

#: ``dependencies`` is the draft-04..07 schema-valued form older MCP servers still emit;
#: its string-array entries pass through untouched because non-objects return as-is.
# omp: utils/schema/normalize.ts :: OPENAI_RESPONSES_SCHEMA_MAP_KEYS
_MAP_KEYS: Final = frozenset(
    {"properties", "patternProperties", "dependencies", "dependentSchemas", "$defs", "definitions"}
)

# omp: utils/schema/normalize.ts :: OPENAI_RESPONSES_SCHEMA_VALUE_KEYS
_VALUE_KEYS: Final = frozenset(
    {
        "items",
        "additionalItems",
        "contains",
        "contentSchema",
        "propertyNames",
        "if",
        "then",
        "else",
        "not",
        "additionalProperties",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)

# omp: utils/schema/normalize.ts :: OPENAI_UNSUPPORTED_REGEX_LOOKAROUNDS
_UNSUPPORTED_LOOKAROUNDS: Final = frozenset({"=", "!", "<=", "<!"})

# omp: utils/schema/normalize.ts :: OPENAI_RESPONSES_PATTERN_PROPERTIES_FALLBACK
# omp= OPENAI_RESPONSES_PATTERN_PROPERTIES_FALLBACK = ".*"
_PATTERN_PROPERTIES_FALLBACK: Final = ".*"


# omp: utils/schema/normalize.ts :: hasOpenAIUnsupportedRegexLookaround
def _has_unsupported_lookaround(pattern: str) -> bool:
    """Whether an unescaped ``(?=``, ``(?!``, ``(?<=`` or ``(?<!`` group occurs."""
    group_start = pattern.find("(?")
    while group_start != -1:
        escapes = 0
        index = group_start - 1
        while index >= 0 and pattern[index] == "\\":
            escapes += 1
            index -= 1
        if escapes % 2 == 0:
            after = pattern[group_start + 2 : group_start + 3]
            operator = pattern[group_start + 2 : group_start + 4] if after == "<" else after
            if operator in _UNSUPPORTED_LOOKAROUNDS:
                return True
        group_start = pattern.find("(?", group_start + 2)
    return False


# omp: utils/schema/normalize.ts :: declaresObjectType
def _declares_object_type(type_value: object) -> bool:
    """``"object"``, or a draft 2020-12 type array that includes it."""
    return type_value == "object" or (isinstance(type_value, list) and "object" in type_value)


# omp: utils/schema/normalize.ts :: normalizeOpenAIResponsesSchemaNode
def _normalize_node(value: object, cache: dict[int, object]) -> object:
    """One schema node, rewritten only where it has to be; the input itself when not.

    ``{}`` is ``true`` (draft 2020-12 §4.3.1). The cache is seeded with the node under
    construction before recursing, so a cycle resolves to the partial instead of looping.
    """
    if not isinstance(value, dict):
        return value
    if not value:
        return True
    cached = cache.get(id(value))
    if cached is not None:
        return cached

    output: JsonObject = {}
    cache[id(value)] = output
    changed = False
    for key, child in value.items():
        # A well-formed `oneOf` is re-emitted as `anyOf` after the loop, so a neighbouring
        # `anyOf` can be concatenated with it; a malformed one is kept verbatim.
        if key == "oneOf" and isinstance(child, list):
            changed = True
            continue
        if key == "pattern" and isinstance(child, str) and _has_unsupported_lookaround(child):
            changed = True
            continue
        rewritten: object = child
        if key == "patternProperties" and isinstance(child, dict):
            rewritten = _normalize_map(child, cache, strip_unsupported_regex_keys=True)
        elif key in _MAP_KEYS and isinstance(child, dict):
            rewritten = _normalize_map(child, cache, strip_unsupported_regex_keys=False)
        elif key in _ARRAY_KEYS and isinstance(child, list):
            rewritten = _normalize_array(child, cache)
        elif key in _VALUE_KEYS and isinstance(child, dict):
            rewritten = _normalize_node(child, cache)
        if rewritten is not child:
            changed = True
        output[key] = rewritten

    one_of = value.get("oneOf")
    if isinstance(one_of, list):
        rewritten_one_of = _normalize_array(one_of, cache)
        existing = output.get("anyOf")
        output["anyOf"] = (
            [*existing, *rewritten_one_of] if isinstance(existing, list) else rewritten_one_of
        )

    if _declares_object_type(value.get("type")) and "properties" not in value:
        output["properties"] = {}
        changed = True

    result: object = (output or True) if changed else value
    cache[id(value)] = result
    return result


# omp: utils/schema/normalize.ts :: normalizeOpenAIResponsesSchemaArray
def _normalize_array(value: list[Any], cache: dict[int, object]) -> list[Any]:
    rewritten = [_normalize_node(item, cache) for item in value]
    changed = any(new is not old for new, old in zip(rewritten, value, strict=True))
    return rewritten if changed else value


# omp: utils/schema/normalize.ts :: normalizeOpenAIResponsesSchemaMap
def _normalize_map(
    schema_map: JsonObject, cache: dict[int, object], *, strip_unsupported_regex_keys: bool
) -> JsonObject:
    """A ``{name: schema}`` map. Under ``patternProperties`` a key with a lookaround the
    backend cannot compile is folded into the ``.*`` fallback rather than dropped."""
    changed = False
    output: JsonObject = {}
    for key, child in schema_map.items():
        rewritten = _normalize_node(child, cache)
        if rewritten is not child:
            changed = True
        if strip_unsupported_regex_keys and _has_unsupported_lookaround(key):
            changed = True
            _append_fallback_pattern_property(output, rewritten)
            continue
        output[key] = rewritten
    return output if changed else schema_map


# omp: utils/schema/normalize.ts :: appendOpenAIResponsesFallbackPatternProperty
def _append_fallback_pattern_property(output: JsonObject, schema: object) -> None:
    if _PATTERN_PROPERTIES_FALLBACK not in output:
        output[_PATTERN_PROPERTIES_FALLBACK] = schema
        return
    existing = output[_PATTERN_PROPERTIES_FALLBACK]
    any_of = existing.get("anyOf") if isinstance(existing, dict) else None
    if isinstance(existing, dict) and isinstance(any_of, list) and len(existing) == 1:
        existing["anyOf"] = [*any_of, schema]
        return
    output[_PATTERN_PROPERTIES_FALLBACK] = {"anyOf": [existing, schema]}


# omp: utils/schema/normalize.ts :: sanitizeSchemaForOpenAIResponses
def sanitize_schema_for_openai_responses(schema: object) -> object:
    """``oneOf`` becomes ``anyOf`` (concatenated onto an existing one), a ``pattern`` with a
    lookaround goes, an object node without ``properties`` gets an empty one, and ``{}``
    becomes ``true`` — in schema-valued positions only, so literal payloads under ``enum``,
    ``const``, ``default`` and ``examples`` stay as the client wrote them."""
    return _normalize_node(schema, {})


# omp: utils/schema/adapt.ts :: adaptSchemaForStrict
def adapt_schema_for_strict(schema: object) -> object:
    """The non-strict branch: the schema upgraded to draft 2020-12, passed through.

    Strict mode is never requested on this path: omp's chat server drops a client's
    ``strict`` when it builds the tool (``openai-chat-server.ts :: buildTools``), so the
    Codex provider sees ``tool.strict`` unset and sends no ``strict`` at all.
    """
    return upgrade_to_2020_12(schema)


# omp: providers/openai-codex-responses.ts :: convertOpenAICodexResponsesTools
def codex_tool_parameters(parameters: object) -> object:
    """A function tool's ``parameters`` as the Codex backend receives them."""
    wire = tool_wire_schema(parameters)
    return adapt_schema_for_strict(sanitize_schema_for_openai_responses(wire))
