"""Tool JSON Schema sanitization for Cloud Code Assist.

Port of ``utils/schema/normalize.ts`` — only the CCA path (``normalizeSchemaForCCA``) and
the transitive closure of what it uses.

The Cloud Code Assist backend runs protojson over a closed `Schema` proto: any field that
does not fit the proto returns 400 with "Cannot find field", and the composition forms
(``anyOf``/``oneOf``/``allOf``/``not``/``$ref``) and ``type: ["string","null"]`` have no
representation at all. Sending the client's schema raw makes the whole request fail, not
just the tool — hence sanitizing before writing to the wire.

The strategy is always to **widen**, never to narrow: an over-permissive schema lets the
model produce an argument the tool then rejects; an over-narrowed one stops it from even
trying. When widening is not enough either, it falls back to the empty schema
``{"type": "object", "properties": {}}`` — typing is lost, but the call goes through.
"""

from __future__ import annotations

import json
import re
from typing import Any, Final

JsonObject = dict[str, Any]

#: Fields Google's Schema proto does not have. Draft 2020-12 features (``$dynamicRef``),
#: pointers (``$ref``), annotations (``deprecated``, ``readOnly``) and all fine-grained
#: validation — protojson rejects the unknown name before it even looks at the value.
# omp: utils/schema/fields.ts :: UNSUPPORTED_SCHEMA_FIELDS
UNSUPPORTED_SCHEMA_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "$schema",
        "$ref",
        "$defs",
        "$dynamicRef",
        "$dynamicAnchor",
        "examples",
        "prefixItems",
        "unevaluatedProperties",
        "unevaluatedItems",
        "patternProperties",
        "additionalProperties",
        "propertyNames",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "pattern",
        "format",
        "dependencies",
        "dependentSchemas",
        "dependentRequired",
        "x-mcp-header",
        "deprecated",
        "readOnly",
        "writeOnly",
        "$comment",
    }
)

#: Of the removed fields, these say something to the model in natural language. They move
#: into ``description`` because a silently dropped ``minLength: 3`` costs a rejected call
#: the model has no way to predict.
# omp: utils/schema/fields.ts :: LIFTABLE_TO_DESCRIPTION_FIELDS
LIFTABLE_TO_DESCRIPTION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "pattern",
        "format",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minProperties",
        "maxProperties",
        "default",
        "examples",
    }
)

# omp: utils/schema/fields.ts :: CLOUD_CODE_ASSIST_TYPE_SPECIFIC_KEYS
CLOUD_CODE_ASSIST_TYPE_SPECIFIC_KEYS: Final[dict[str, frozenset[str]]] = {
    "array": frozenset(
        {
            "items",
            "prefixItems",
            "contains",
            "minContains",
            "maxContains",
            "minItems",
            "maxItems",
            "uniqueItems",
            "unevaluatedItems",
        }
    ),
    "object": frozenset(
        {
            "properties",
            "required",
            "additionalProperties",
            "patternProperties",
            "propertyNames",
            "minProperties",
            "maxProperties",
            "dependentRequired",
            "dependentSchemas",
            "unevaluatedProperties",
        }
    ),
    "string": frozenset(
        {
            "minLength",
            "maxLength",
            "pattern",
            "format",
            "contentEncoding",
            "contentMediaType",
        }
    ),
    "number": frozenset(
        {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"}
    ),
    "integer": frozenset(
        {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"}
    ),
    "boolean": frozenset(),
    "null": frozenset(),
}

# omp: utils/schema/fields.ts :: ALL_CCA_TYPE_SPECIFIC_KEYS
ALL_CCA_TYPE_SPECIFIC_KEYS: Final[frozenset[str]] = frozenset().union(
    *CLOUD_CODE_ASSIST_TYPE_SPECIFIC_KEYS.values()
)

# omp: utils/schema/fields.ts :: CLOUD_CODE_ASSIST_SHARED_SCHEMA_KEYS
CLOUD_CODE_ASSIST_SHARED_SCHEMA_KEYS: Final[frozenset[str]] = frozenset(
    {
        "title",
        "description",
        "default",
        "examples",
    }
)

#: The schema a non-sanitizable one is replaced with. An object with no properties: the
#: model loses argument typing but the tool still exists in the catalog.
# omp: utils/schema/normalize.ts :: CLOUD_CODE_ASSIST_CLAUDE_FALLBACK_SCHEMA
CCA_FALLBACK_SCHEMA: Final[JsonObject] = {"type": "object", "properties": {}}

#: python-genai renames these keys before serializing; the proto only knows the camelCase
#: form, so an ``any_of`` passed raw would be silently discarded.
# omp: utils/schema/normalize.ts :: SNAKE_TO_CAMEL_RENAMES
SNAKE_TO_CAMEL_RENAMES: Final[dict[str, str]] = {
    "additional_properties": "additionalProperties",
    "any_of": "anyOf",
    "prefix_items": "prefixItems",
    "property_ordering": "propertyOrdering",
}

# omp: utils/schema/normalize.ts :: JSON_SCHEMA_COMBINERS
JSON_SCHEMA_COMBINERS: Final[tuple[str, str]] = ("anyOf", "oneOf")

# omp: utils/schema/normalize.ts :: SCHEMA_COMPOSITION_COMBINERS
SCHEMA_COMPOSITION_COMBINERS: Final[tuple[str, str, str]] = ("allOf", "anyOf", "oneOf")

#: Keywords whose value is *one* subschema. A raw ``True``/``False`` here is a draft 2020-12
#: boolean subschema, not a literal value.
# omp: utils/schema/normalize.ts :: SUBSCHEMA_VALUE_KEYS
SUBSCHEMA_VALUE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "items",
        "additionalItems",
        "unevaluatedItems",
        "not",
        "if",
        "then",
        "else",
        "contains",
        "propertyNames",
        "contentSchema",
    }
)

# omp: utils/schema/normalize.ts :: BOOLEAN_OR_SCHEMA_VALUE_KEYS
BOOLEAN_OR_SCHEMA_VALUE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "additionalProperties",
        "unevaluatedProperties",
    }
)

# omp: utils/schema/normalize.ts :: SUBSCHEMA_ARRAY_KEYS
SUBSCHEMA_ARRAY_KEYS: Final[frozenset[str]] = frozenset({"anyOf", "oneOf", "allOf", "prefixItems"})

# omp: utils/schema/normalize.ts :: SUBSCHEMA_MAP_KEYS
SUBSCHEMA_MAP_KEYS: Final[frozenset[str]] = frozenset(
    {
        "properties",
        "patternProperties",
        "dependencies",
        "dependentSchemas",
        "$defs",
        "definitions",
    }
)

# ---------------------------------------------------------------------------
# Draft-07 -> draft 2020-12 (utils/schema/draft.ts)
# ---------------------------------------------------------------------------

JSON_SCHEMA_DRAFT_2020_12_URI: Final = "https://json-schema.org/draft/2020-12/schema"

#: Both the canonical form with ``#`` and the one without — Zod emits one, MCP servers the other.
# omp: utils/schema/draft.ts :: DRAFT_07_SCHEMA_URIS
DRAFT_07_SCHEMA_URIS: Final[frozenset[str]] = frozenset(
    {
        "http://json-schema.org/draft-07/schema#",
        "https://json-schema.org/draft-07/schema#",
        "http://json-schema.org/draft-07/schema",
        "https://json-schema.org/draft-07/schema",
    }
)

# omp: utils/schema/draft.ts :: SCHEMA_MAP_KEYS
DRAFT_SCHEMA_MAP_KEYS: Final[frozenset[str]] = frozenset(
    {
        "properties",
        "patternProperties",
        "dependentSchemas",
    }
)

#: Descending through these would corrupt the payload: ``type: ["string","null"]`` is a
#: value, not a subschema, and an ``enum`` of objects would be rewritten as if they were schemas.
# omp: utils/schema/draft.ts :: NON_SCHEMA_VALUE_KEYS
NON_SCHEMA_VALUE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "const",
        "default",
        "enum",
        "example",
        "examples",
        "required",
        "dependentRequired",
        "type",
    }
)


# omp: utils/schema/equality.ts :: areJsonValuesEqual
def _json_equal(left: object, right: object) -> bool:
    """Structural equality. Python's ``==`` treats ``True == 1`` as true, which would merge
    an ``enum: [true]`` with an ``enum: [1]`` and lose a member."""
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, list) or isinstance(right, list):
        if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
            return False
        return all(_json_equal(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            return False
        if left.keys() != right.keys():
            return False
        return all(_json_equal(left[k], right[k]) for k in left)
    return left == right


# omp: utils/schema/draft.ts :: convertRef
def _convert_ref(value: str) -> str:
    prefix = "#/definitions/"
    return f"#/$defs/{value[len(prefix) :]}" if value.startswith(prefix) else value


# omp: utils/schema/draft.ts :: combineSchemas
def _combine_schemas(left: object, right: object) -> object:
    """Intersection. ``allOf`` is the only way to keep both constraints when the keys of
    ``dependencies`` collide — discarding one would lose validation."""
    if left is None or left is True:
        return right
    if right is None or right is True:
        return left
    if left is False or right is False:
        return False
    if _json_equal(left, right):
        return left
    return {"allOf": [left, right]}


# omp: utils/schema/draft.ts :: mergeArrayValues
def _merge_array_values(left: list[Any], right: list[Any]) -> list[Any]:
    merged = list(left)
    for value in right:
        if not any(_json_equal(existing, value) for existing in merged):
            merged.append(value)
    return merged


# omp: utils/schema/draft.ts :: mergePrefixItems
def _merge_prefix_items(existing: object, converted: list[Any]) -> list[Any]:
    if not isinstance(existing, list):
        return converted
    merged = list(existing)
    for index, item in enumerate(converted):
        if index < len(merged):
            merged[index] = _combine_schemas(merged[index], item)
        else:
            merged.append(item)
    return merged


# omp: utils/schema/draft.ts :: hasNullType
def _has_null_type(type_value: object) -> bool:
    return type_value == "null" or (isinstance(type_value, list) and "null" in type_value)


# omp: utils/schema/draft.ts :: makeNullable
def _make_nullable(schema: JsonObject) -> JsonObject:
    """OpenAPI 3.0's ``nullable: true`` does not exist in 2020-12; the equivalent form
    depends on what the node already declares, and always wrapping in ``anyOf`` would create
    a combiner where widening ``type`` was enough."""
    type_value = schema.get("type")
    if isinstance(type_value, str):
        if type_value != "null":
            schema["type"] = [type_value, "null"]
        return schema
    if isinstance(type_value, list):
        if "null" not in type_value:
            schema["type"] = [*type_value, "null"]
        return schema
    any_of = schema.get("anyOf")
    if isinstance(any_of, list):
        has_null = any(isinstance(v, dict) and _has_null_type(v.get("type")) for v in any_of)
        if not has_null:
            schema["anyOf"] = [*any_of, {"type": "null"}]
        return schema
    return {"anyOf": [schema, {"type": "null"}]}


# omp: utils/schema/draft.ts :: upgradeJsonSchemaTo202012Impl
def _upgrade_node(value: object, cache: dict[int, Any]) -> object:
    """The cache is seeded **before** the recursion so that a back edge in a cyclic graph
    resolves to the reference under construction instead of looping forever."""
    if isinstance(value, list):
        cached_list = cache.get(id(value))
        if cached_list is not None:
            return cached_list
        result_list: list[Any] = []
        cache[id(value)] = result_list
        result_list.extend(_upgrade_node(entry, cache) for entry in value)
        return result_list
    if not isinstance(value, dict):
        return value

    cached = cache.get(id(value))
    if cached is not None:
        return cached

    result: JsonObject = {}
    cache[id(value)] = result
    for key, entry in value.items():
        if key in ("definitions", "$defs"):
            if isinstance(entry, dict):
                defs = result.setdefault("$defs", {})
                if isinstance(defs, dict):
                    for name, sub in entry.items():
                        defs[name] = _upgrade_node(sub, cache)
            continue
        if key in DRAFT_SCHEMA_MAP_KEYS:
            if not isinstance(entry, dict):
                result[key] = entry
                continue
            target = result.setdefault(key, {})
            if isinstance(target, dict):
                for name, sub in entry.items():
                    target[name] = _upgrade_node(sub, cache)
            continue
        if key in NON_SCHEMA_VALUE_KEYS:
            result[key] = entry
            continue
        if key in ("dependencies", "additionalItems", "nullable"):
            continue
        if key == "$schema":
            result["$schema"] = (
                JSON_SCHEMA_DRAFT_2020_12_URI
                if isinstance(entry, str) and entry in DRAFT_07_SCHEMA_URIS
                else entry
            )
            continue
        if key == "$ref" and isinstance(entry, str):
            result["$ref"] = _convert_ref(entry)
            continue
        if key == "items" and isinstance(entry, list):
            continue
        result[key] = _upgrade_node(entry, cache)

    raw_items = value.get("items")
    if isinstance(raw_items, list):
        converted = _upgrade_node(raw_items, cache)
        if isinstance(converted, list):
            result["prefixItems"] = _merge_prefix_items(result.get("prefixItems"), converted)
        additional = value.get("additionalItems")
        if additional is not None and additional is not True:
            result["items"] = _upgrade_node(additional, cache)
        else:
            result.pop("items", None)

    _convert_dependencies(value, result, cache)

    if value.get("nullable") is True:
        nullable = _make_nullable(result)
        if nullable is not result:
            cache[id(value)] = nullable
        return nullable

    return result


# omp: utils/schema/draft.ts :: convertDependencies
def _convert_dependencies(source: JsonObject, target: JsonObject, cache: dict[int, Any]) -> None:
    """Draft-07 mixes array and schema dependencies under one key; 2020-12 splits them into
    ``dependentRequired`` and ``dependentSchemas``."""
    dependencies = source.get("dependencies")
    if not isinstance(dependencies, dict):
        return
    for key, dependency in dependencies.items():
        converted = _upgrade_node(dependency, cache)
        if isinstance(converted, list):
            required_map = target.setdefault("dependentRequired", {})
            if not isinstance(required_map, dict):
                continue
            existing = required_map.get(key)
            if existing is None:
                required_map[key] = converted
            elif isinstance(existing, list):
                required_map[key] = _merge_array_values(existing, converted)
        else:
            schema_map = target.setdefault("dependentSchemas", {})
            if isinstance(schema_map, dict):
                schema_map[key] = _combine_schemas(schema_map.get(key), converted)


# omp: utils/schema/draft.ts :: upgradeJsonSchemaTo202012
def upgrade_to_2020_12(schema: object) -> object:
    return _upgrade_node(schema, {})


# ---------------------------------------------------------------------------
# $ref inlining (utils/schema/dereference.ts)
# ---------------------------------------------------------------------------

_LOCAL_REF_RE: Final = re.compile(r"^#/(\$defs|definitions)/(.+)$")


# omp: utils/schema/dereference.ts :: resolveLocalRef
def _resolve_local_ref(ref: str, root: JsonObject) -> JsonObject | None:
    match = _LOCAL_REF_RE.match(ref)
    if match is None:
        return None
    defs = root.get(match.group(1))
    if not isinstance(defs, dict):
        return None
    resolved = defs.get(match.group(2))
    return resolved if isinstance(resolved, dict) else None


# omp: utils/schema/dereference.ts :: dereferenceNode
def _dereference_node(node: object, root: JsonObject, visiting: set[str]) -> object:
    if isinstance(node, list):
        return [_dereference_node(item, root, visiting) for item in node]
    if not isinstance(node, dict):
        return node

    ref = node.get("$ref")
    if isinstance(ref, str):
        # An inlined `$ref` cycle never terminates; `{}` cuts it while keeping the node valid.
        if ref in visiting:
            return {}
        resolved = _resolve_local_ref(ref, root)
        if resolved is None:
            return node
        visiting.add(ref)
        inlined = _dereference_node(resolved, root, visiting)
        visiting.discard(ref)
        siblings = [k for k in node if k != "$ref"]
        if not siblings or not isinstance(inlined, dict):
            return inlined
        # In 2020-12 keys sibling to `$ref` are valid and more specific than the definition
        # pointed at, so they win.
        merged: JsonObject = {**inlined, **node}
        merged.pop("$ref", None)
        return merged

    result: JsonObject = {}
    for key, value in node.items():
        if key in ("$defs", "definitions"):
            continue
        if isinstance(value, list):
            result[key] = [_dereference_node(item, root, visiting) for item in value]
        elif isinstance(value, dict):
            result[key] = _dereference_node(value, root, visiting)
        else:
            result[key] = value
    return result


# omp: utils/schema/dereference.ts :: dereferenceJsonSchema
def dereference_schema(schema: object) -> object:
    if not isinstance(schema, dict):
        return schema
    if "$defs" not in schema and "definitions" not in schema:
        return schema
    return _dereference_node(schema, schema, set())


# ---------------------------------------------------------------------------
# Spill into description (utils/schema/spill.ts)
# ---------------------------------------------------------------------------


# omp: utils/schema/spill.ts :: spillToDescription
def _spill_to_description(node: JsonObject, entries: list[tuple[str, Any]]) -> None:
    """Appends the removed constraints to the end of ``description``.

    "Spill" format: one JSON object per node, separated from the existing text by a blank
    line. Concatenating without a separator would make the constraint indistinguishable from
    the prose.
    """
    if not entries:
        return
    existing = node.get("description")
    existing_text = existing if isinstance(existing, str) else ""
    # Compact separators: that is what `JSON.stringify` produces, and the description counts
    # against the context window of every call.
    body = ", ".join(f"{key}: {json.dumps(value, separators=(',', ':'))}" for key, value in entries)
    formatted = f"{{{body}}}"
    node["description"] = f"{existing_text}\n\n{formatted}" if existing_text else formatted


# ---------------------------------------------------------------------------
# Combiner collapse (utils/schema/normalize.ts, utils/schema/equality.ts)
# ---------------------------------------------------------------------------


# omp: utils/schema/normalize.ts :: classifySchemaChild
def _classify_schema_child(key: str, value: object, inside_schema_map: bool) -> str | None:
    """Only the children that really are JSON Schema; an instance payload stays opaque.

    Inside a map (``properties``) the keys are user-chosen names: a property called
    ``items`` is a schema because it is a map value, not because of the word.
    """
    if inside_schema_map:
        return "schema"
    normalized_key = SNAKE_TO_CAMEL_RENAMES.get(key, key)
    if normalized_key in SUBSCHEMA_MAP_KEYS:
        return "map"
    if normalized_key in SUBSCHEMA_VALUE_KEYS or normalized_key in SUBSCHEMA_ARRAY_KEYS:
        return "schema"
    if normalized_key in BOOLEAN_OR_SCHEMA_VALUE_KEYS and isinstance(value, dict):
        return "schema"
    return None


# omp: utils/schema/normalize.ts :: copySchemaWithout
def _copy_without(schema: JsonObject, key: str) -> JsonObject:
    return {k: v for k, v in schema.items() if k != key}


# omp: utils/schema/equality.ts :: mergeCompatibleEnumSchemas
def _merge_compatible_enum_schemas(existing: object, incoming: object) -> JsonObject | None:
    """Union of the members only when the branches agree on ``type`` and on everything that
    is not ``enum``. Merging despite disagreement would swap one branch's description for
    the other's."""
    if not isinstance(existing, dict) or not isinstance(incoming, dict):
        return None
    existing_enum = existing.get("enum")
    incoming_enum = incoming.get("enum")
    if not isinstance(existing_enum, list) or not isinstance(incoming_enum, list):
        return None
    if not _json_equal(existing.get("type"), incoming.get("type")):
        return None
    existing_keys = {k for k in existing if k != "enum"}
    incoming_keys = {k for k in incoming if k != "enum"}
    if existing_keys != incoming_keys:
        return None
    for key in existing_keys:
        if not _json_equal(existing[key], incoming[key]):
            return None
    merged_enum = list(existing_enum)
    for value in incoming_enum:
        if not any(_json_equal(seen, value) for seen in merged_enum):
            merged_enum.append(value)
    return {**existing, "enum": merged_enum}


# omp: utils/schema/equality.ts :: mergePropertySchemas
def _merge_property_schemas(existing: object, incoming: object) -> object:
    """Two definitions of the same property in different branches: accept both. Picking one
    would reject valid arguments from the other branch."""
    if _json_equal(existing, incoming):
        return existing
    merged_enum = _merge_compatible_enum_schemas(existing, incoming)
    if merged_enum is not None:
        return merged_enum
    variants: list[Any] = []
    for schema in (existing, incoming):
        branch = (
            schema["anyOf"]
            if isinstance(schema, dict) and isinstance(schema.get("anyOf"), list)
            else [schema]
        )
        for variant in branch:
            if not any(_json_equal(seen, variant) for seen in variants):
                variants.append(variant)
    return variants[0] if len(variants) == 1 else {"anyOf": variants}


# omp: utils/schema/normalize.ts :: mergeSchemaDescriptions
def _merge_descriptions(existing: object, incoming: object) -> str:
    if not isinstance(existing, str):
        return incoming if isinstance(incoming, str) else ""
    if not isinstance(incoming, str) or not incoming or existing == incoming:
        return existing
    if not existing:
        return incoming
    return f"{existing}\n\n{incoming}"


# omp: utils/schema/normalize.ts :: mergeObjectCombinerVariants
def _merge_object_combiner_variants(schema: JsonObject, combiner: str) -> JsonObject:
    """Branches that are all object-shaped merge into a union of ``properties``.

    It is the only projection that loses no properties: collapsing to the first branch would
    leave the model unaware that the other branches' fields exist. It is ``required`` that
    has to shrink — see below.
    """
    variants_raw = schema.get(combiner)
    if not isinstance(variants_raw, list) or not variants_raw:
        return schema

    variants: list[JsonObject] = []
    for entry in variants_raw:
        if not isinstance(entry, dict):
            return schema
        variant_type = entry.get("type")
        has_object_shape = (
            isinstance(entry.get("properties"), dict)
            or isinstance(entry.get("required"), list)
            or "additionalProperties" in entry
        )
        if variant_type is None and not has_object_shape:
            return schema
        if variant_type is not None and variant_type != "object":
            return schema
        if "properties" in entry and not isinstance(entry["properties"], dict):
            return schema
        if "required" in entry and not isinstance(entry["required"], list):
            return schema
        variants.append(entry)

    own_properties = schema["properties"] if isinstance(schema.get("properties"), dict) else {}
    merged_properties: JsonObject = dict(own_properties)
    for variant in variants:
        properties = variant["properties"] if isinstance(variant.get("properties"), dict) else {}
        for name, property_schema in properties.items():
            existing = merged_properties.get(name)
            merged_properties[name] = (
                property_schema
                if name not in merged_properties
                else _merge_property_schemas(existing, property_schema)
            )

    next_schema = _copy_without(schema, combiner)
    next_schema["type"] = "object"
    next_schema["properties"] = merged_properties

    branch_required = [
        [r for r in variant["required"] if isinstance(r, str)]
        if isinstance(variant.get("required"), list)
        else []
        for variant in variants
    ]
    if combiner == "allOf":
        # `allOf` requires every branch, so the union does not narrow acceptance.
        combined_required: list[str] = []
        for required in branch_required:
            for name in required:
                if name not in combined_required:
                    combined_required.append(name)
    else:
        # `anyOf`/`oneOf` accept any one branch: only what ALL of them require stays
        # required, otherwise the projection would reject instances the original accepts.
        intersection: list[str] | None = None
        for required in branch_required:
            if intersection is None:
                intersection = list(required)
            else:
                allowed = set(required)
                intersection = [r for r in intersection if r in allowed]
        combined_required = intersection or []

    parent_required = (
        [r for r in schema["required"] if isinstance(r, str)]
        if isinstance(schema.get("required"), list)
        else []
    )
    safe_required = {name for name in combined_required if name in merged_properties}
    safe_required.update(
        name for name in parent_required if name in own_properties and name in merged_properties
    )
    required_in_property_order = [n for n in merged_properties if n in safe_required]
    if required_in_property_order:
        next_schema["required"] = required_in_property_order
    else:
        next_schema.pop("required", None)
    return next_schema


# omp: utils/schema/normalize.ts :: collapseMixedTypeCombinerVariants
def _collapse_mixed_type_combiner_variants(schema: JsonObject, combiner: str) -> JsonObject:
    """A union of different types (``string`` or ``number``) collapses to a single type.

    The CCA proto has a scalar ``type`` field: there is no way to express the union. It only
    collapses when the branches contradict each other in nothing else, otherwise constraint
    would be lost without warning — in that case the schema is returned intact and the
    residual test sends it to the fallback.
    """
    variants_raw = schema.get(combiner)
    if not isinstance(variants_raw, list) or not variants_raw:
        return schema

    seen_types: set[str] = set()
    variant_types: list[str] = []
    merged_variant_fields: JsonObject = {}
    for entry in variants_raw:
        if not isinstance(entry, dict) or not isinstance(entry.get("type"), str):
            return schema
        variant_type = entry["type"]
        if variant_type in seen_types:
            return schema
        allowed_keys = CLOUD_CODE_ASSIST_TYPE_SPECIFIC_KEYS.get(variant_type)
        if allowed_keys is None:
            return schema

        for key, variant_value in entry.items():
            if key == "type":
                continue
            if key not in allowed_keys and key not in CLOUD_CODE_ASSIST_SHARED_SCHEMA_KEYS:
                return schema
            existing = merged_variant_fields.get(key)
            if key in merged_variant_fields and not _json_equal(existing, variant_value):
                if key != "description":
                    return schema
                # Descriptions are annotation, not structure: merging does not change acceptance.
                merged_variant_fields[key] = _merge_descriptions(existing, variant_value)
                continue
            merged_variant_fields[key] = variant_value

        seen_types.add(variant_type)
        variant_types.append(variant_type)

    if len(variant_types) < 2 or all(t == "object" for t in variant_types):
        return schema
    next_schema = _copy_without(schema, combiner)
    non_null_types = [t for t in variant_types if t != "null"]
    chosen_type = non_null_types[0] if non_null_types else variant_types[0]
    next_schema["type"] = chosen_type
    chosen_allowed = CLOUD_CODE_ASSIST_TYPE_SPECIFIC_KEYS.get(chosen_type, frozenset())

    # An `items` inherited from the parent on a node now typed `string` is a field the proto
    # does not accept in that position — protojson 400s even though it is valid in itself.
    for key in list(next_schema):
        if key == "type":
            continue
        if (
            key in ALL_CCA_TYPE_SPECIFIC_KEYS
            and key not in chosen_allowed
            and key not in CLOUD_CODE_ASSIST_SHARED_SCHEMA_KEYS
        ):
            del next_schema[key]

    for key, value in merged_variant_fields.items():
        if key not in chosen_allowed and key not in CLOUD_CODE_ASSIST_SHARED_SCHEMA_KEYS:
            continue
        existing = next_schema.get(key)
        if key in next_schema and not _json_equal(existing, value):
            if key != "description":
                return schema
            next_schema[key] = _merge_descriptions(existing, value)
            continue
        if key not in next_schema:
            next_schema[key] = value
    return next_schema


# omp: utils/schema/normalize.ts :: collapseSameTypeCombinerVariants
def _collapse_same_type_combiner_variants(schema: JsonObject, combiner: str) -> JsonObject:
    """Branches that are all of the same ``type`` collapse into a single node."""
    variants_raw = schema.get(combiner)
    if not isinstance(variants_raw, list) or not variants_raw:
        return schema
    common_type: str | None = None
    variants: list[JsonObject] = []
    for entry in variants_raw:
        if not isinstance(entry, dict) or not isinstance(entry.get("type"), str):
            return schema
        if common_type is None:
            common_type = entry["type"]
        elif entry["type"] != common_type:
            return schema
        variants.append(entry)
    first_entry = variants[0]

    enum_variant_count = sum(1 for v in variants if isinstance(v.get("enum"), list))

    collapsed: JsonObject
    if enum_variant_count == len(variants):
        # Stopping at the first branch would erase the other branches' members: an `anyOf` of
        # two string enums collapsed to half the legal values.
        merged: JsonObject | None = first_entry
        for variant in variants[1:]:
            if merged is None:
                break
            merged = _merge_compatible_enum_schemas(merged, variant)
        if merged is None:
            return schema
        collapsed = merged
    elif enum_variant_count > 0:
        # There is a branch with no `enum`, hence wider. Collapsing to it preserves
        # acceptance; collapsing to an enum branch would narrow to its members.
        collapsed = next((v for v in variants if not isinstance(v.get("enum"), list)), first_entry)
    else:
        collapsed = first_entry

    next_schema = _copy_without(schema, combiner)
    for key, value in collapsed.items():
        if key not in next_schema:
            next_schema[key] = value
    return next_schema


class _Seen:
    """Identity-keyed visited set that **retains** every object it marks.

    CPython's ``id()`` is the address: a temporary dict freed mid-traversal hands the same
    ``id`` to the next one, and a bare ``set[int]`` declared it already visited. In practice
    that truncated properties to ``{}`` in an allocator-dependent way — differential testing
    against the TS caught 18 cases out of 600 before this retention. The original's
    ``WeakMap`` does not have the problem because the key is the live object.
    """

    __slots__ = ("_ids", "_keep")

    def __init__(self) -> None:
        self._ids: set[int] = set()
        self._keep: list[object] = []

    def first(self, value: object) -> bool:
        """True the first time ``value`` is seen in this traversal."""
        key = id(value)
        if key in self._ids:
            return False
        self._ids.add(key)
        self._keep.append(value)
        return True


# omp: utils/schema/normalize.ts :: stripResidualCombinersNode
def _strip_residual_combiners_node(value: object, seen: _Seen, inside_schema_map: bool) -> object:
    if isinstance(value, list):
        if not seen.first(value):
            return []
        return [_strip_residual_combiners_node(entry, seen, False) for entry in value]
    if not isinstance(value, dict):
        return value
    if not seen.first(value):
        return {}
    result: JsonObject = {}
    for key, entry in value.items():
        child_kind = _classify_schema_child(key, entry, inside_schema_map)
        result[key] = (
            _strip_residual_combiners_node(entry, seen, child_kind == "map")
            if child_kind
            else entry
        )
    if inside_schema_map:
        return result

    current = result
    changed = True
    while changed:
        changed = False
        for combiner in JSON_SCHEMA_COMBINERS:
            same_type = _collapse_same_type_combiner_variants(current, combiner)
            if same_type is not current:
                current = same_type
                changed = True
            mixed = _collapse_mixed_type_combiner_variants(current, combiner)
            if mixed is not current:
                current = mixed
                changed = True
    return current


# omp: utils/schema/normalize.ts :: stripResidualCombiners
def strip_residual_combiners(value: object) -> object:
    """Fixed point. Merging object combiners can synthesize a new ``anyOf`` on a shared
    property (see ``_merge_property_schemas``) after the recursion over the children has
    already run — a single pass let it escape to the wire."""
    return _strip_residual_combiners_node(value, _Seen(), False)


# ---------------------------------------------------------------------------
# Nullable and residual incompatibilities
# ---------------------------------------------------------------------------


# omp: utils/schema/normalize.ts :: extractNullableUnionSchema
def _extract_nullable_union_schema(schema: object) -> tuple[object, bool]:
    """Returns ``(schema_without_null, was_nullable)``.

    The CCA has neither ``nullable`` nor a union with ``null``. Nullability survives by
    being translated into *optional*: the caller drops the field from ``required``. Erasing
    it outright would make required a field the client declared could be absent.
    """
    if not isinstance(schema, dict):
        return schema, False

    if schema.get("nullable") is True:
        next_schema = {k: v for k, v in schema.items() if k != "nullable"}
        return next_schema, True

    type_value = schema.get("type")
    if isinstance(type_value, list):
        type_variants = [t for t in type_value if isinstance(t, str)]
        non_null = [t for t in type_variants if t != "null"]
        if "null" in type_variants and len(non_null) == 1:
            return {**schema, "type": non_null[0]}, True

    for combiner in JSON_SCHEMA_COMBINERS:
        variants_raw = schema.get(combiner)
        if not isinstance(variants_raw, list):
            continue

        has_null_variant = False
        non_null_variants: list[Any] = []
        for variant in variants_raw:
            # Only a bare `{type: "null"}` counts: `{type: "null", description: …}` carries
            # information that would be lost by discarding the branch.
            if isinstance(variant, dict) and variant.get("type") == "null" and len(variant) == 1:
                has_null_variant = True
                continue
            non_null_variants.append(variant)

        if (
            not has_null_variant
            or len(non_null_variants) != 1
            or not isinstance(non_null_variants[0], dict)
        ):
            continue

        next_schema = _copy_without(schema, combiner)
        for key, value in non_null_variants[0].items():
            if key in next_schema and not _json_equal(next_schema[key], value):
                # Parent and branch disagree: merging would arbitrarily pick one of them.
                return schema, False
            if key not in next_schema:
                next_schema[key] = value
        return next_schema, True

    return schema, False


# omp: utils/schema/normalize.ts :: normalizeNullablePropertiesForCloudCodeAssist
def _normalize_nullable_properties(
    value: object,
    is_property_schema: bool,
    seen: _Seen,
    inside_schema_map: bool = False,
) -> tuple[object, bool]:
    if isinstance(value, list):
        if not seen.first(value):
            return [], False
        return [_normalize_nullable_properties(entry, False, seen)[0] for entry in value], False
    if not isinstance(value, dict):
        return value, False
    if not seen.first(value):
        return {}, False

    normalized: JsonObject = {}
    for key, entry in value.items():
        child_kind = _classify_schema_child(key, entry, inside_schema_map)
        normalized[key] = (
            _normalize_nullable_properties(entry, False, seen, child_kind == "map")[0]
            if child_kind
            else entry
        )
    if inside_schema_map:
        return normalized, False

    properties = normalized.get("properties")
    if isinstance(properties, dict):
        raw_required = normalized.get("required")
        required = (
            [r for r in raw_required if isinstance(r, str)]
            if isinstance(raw_required, list)
            else []
        )
        next_properties: JsonObject = {}
        for name, property_schema in properties.items():
            next_properties[name], nullable = _normalize_nullable_properties(
                property_schema, True, seen
            )
            if nullable:
                required = [r for r in required if r != name]
        normalized["properties"] = next_properties
        if isinstance(raw_required, list):
            normalized["required"] = required

    if not is_property_schema:
        return normalized, False

    return _extract_nullable_union_schema(normalized)


# omp: utils/schema/normalize.ts :: createResidualIncompatibilityChecks
#: The five residues the CCA rejects with 400. Checked after all sanitization: whatever is
#: still here has no possible representation, so the schema goes to the fallback.
RESIDUAL_INCOMPATIBILITIES: Final[frozenset[str]] = frozenset(
    {
        "type-array",
        "type-null",
        "nullable",
        "combiners",
        "not",
    }
)


# omp: utils/schema/normalize.ts :: hasResidualSchemaIncompatibilities
def has_residual_incompatibilities(
    value: object, seen: _Seen | None = None, inside_schema_map: bool = False
) -> bool:
    if seen is None:
        seen = _Seen()
    if isinstance(value, list):
        if not seen.first(value):
            return False
        return any(has_residual_incompatibilities(entry, seen, False) for entry in value)
    if not isinstance(value, dict):
        return False
    if not seen.first(value):
        return False

    if not inside_schema_map:
        if isinstance(value.get("type"), list):
            return True
        if value.get("type") == "null":
            return True
        if "nullable" in value or "not" in value:
            return True
        if any(isinstance(value.get(c), list) for c in SCHEMA_COMPOSITION_COMBINERS):
            return True
    for key, entry in value.items():
        child_kind = _classify_schema_child(key, entry, inside_schema_map)
        if child_kind and has_residual_incompatibilities(entry, seen, child_kind == "map"):
            return True
    return False


# ---------------------------------------------------------------------------
# Main pass (utils/schema/normalize.ts)
# ---------------------------------------------------------------------------


# omp: utils/schema/normalize.ts :: applySnakeCaseRenames
def _apply_snake_case_renames(obj: JsonObject) -> JsonObject:
    """A collision resolves in favour of the snake_case one (python-genai's ``pop(from)`` →
    ``set(to)``), which is the form the client wrote on purpose."""
    if not any(k in SNAKE_TO_CAMEL_RENAMES for k in obj):
        return obj
    out: JsonObject = {}
    for key, value in obj.items():
        renamed = SNAKE_TO_CAMEL_RENAMES.get(key)
        if renamed is not None:
            out[renamed] = value
        elif key not in out:
            out[key] = value
    return out


# omp: utils/schema/normalize.ts :: inferJsonSchemaTypeFromValue
def _infer_type_from_value(value: object) -> str | None:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, dict):
        return "object"
    return None


# omp: utils/schema/normalize.ts :: pushEnumValue
def _push_enum_value(values: list[Any], value: object) -> None:
    if not any(_json_equal(existing, value) for existing in values):
        values.append(value)


# omp: utils/schema/normalize.ts :: applyNodePostProcessing
def _apply_node_post_processing(schema: JsonObject) -> JsonObject:
    current = schema
    for combiner in JSON_SCHEMA_COMBINERS:
        current = _merge_object_combiner_variants(current, combiner)
        current = _collapse_mixed_type_combiner_variants(current, combiner)
        current = _collapse_same_type_combiner_variants(current, combiner)
    return current


# omp: utils/schema/normalize.ts :: normalizeSchemaNode
def _normalize_schema_node(
    value: object, path: set[int], inside_schema_map: bool, boolean_is_subschema: bool
) -> object:
    if isinstance(value, list):
        if id(value) in path:
            return []
        path.add(id(value))
        try:
            return [
                _normalize_schema_node(entry, path, inside_schema_map, boolean_is_subschema)
                for entry in value
            ]
        finally:
            path.discard(id(value))
    if isinstance(value, bool):
        # A boolean is only a subschema in a subschema position; in `nullable: true` or in an
        # `enum` member it is a literal value and coercing it would destroy the schema.
        if not boolean_is_subschema:
            return value
        # "Standard" mode: `false` is the impossible schema, which only `not: {}` expresses.
        return {} if value else {"not": {}}
    if not isinstance(value, dict):
        return value
    # Path trail, not a visited set: subtrees shared in a DAG are normalized at each
    # occurrence; only true cycles short-circuit.
    if id(value) in path:
        return {}
    path.add(id(value))
    try:
        return _normalize_schema_object_node(value, path, inside_schema_map)
    finally:
        path.discard(id(value))


def _walk_child(key: str, entry: object, path: set[int], inside_schema_map: bool) -> object:
    child_kind = _classify_schema_child(key, entry, inside_schema_map)
    if child_kind is None:
        return entry
    return _normalize_schema_node(entry, path, child_kind == "map", child_kind == "schema")


# omp: utils/schema/normalize.ts :: normalizeSchemaObjectNode
def _normalize_schema_object_node(
    value: JsonObject, path: set[int], inside_schema_map: bool
) -> object:
    obj = value if inside_schema_map else _apply_snake_case_renames(value)
    result: JsonObject = {}
    spill: list[tuple[str, Any]] = []

    def strip_or_keep(key: str, entry: object) -> bool:
        """True when the key was removed (and possibly spilled into the description)."""
        if not inside_schema_map and key in UNSUPPORTED_SCHEMA_FIELDS:
            if key in LIFTABLE_TO_DESCRIPTION_FIELDS:
                spill.append((key, entry))
            return True
        return key == "nullable"

    for combiner in JSON_SCHEMA_COMBINERS:
        variants = obj.get(combiner)
        if not isinstance(variants, list) or not variants:
            continue
        if not all(isinstance(v, dict) and "const" in v for v in variants):
            continue

        # An `anyOf` of `const` is an `enum` spelled out; the CCA has `enum` but neither
        # `const` nor combiners, so this is the translation that loses nothing.
        deduped_enum: list[Any] = []
        for variant in variants:
            _push_enum_value(deduped_enum, variant["const"])
        result["enum"] = deduped_enum

        explicit_types = [v["type"] for v in variants if isinstance(v.get("type"), str)]
        if len(explicit_types) == len(variants) and len(set(explicit_types)) == 1:
            result["type"] = explicit_types[0]
        else:
            inferred = [t for t in map(_infer_type_from_value, deduped_enum) if t is not None]
            if len(set(inferred)) == 1:
                result["type"] = inferred[0]
            else:
                non_null = [t for t in inferred if t != "null"]
                if "null" in inferred and len(set(non_null)) == 1:
                    result["type"] = non_null[0]

        for key, entry in obj.items():
            if key == combiner or key in result or strip_or_keep(key, entry):
                continue
            result[key] = _walk_child(key, entry, path, inside_schema_map)
        _spill_to_description(result, spill)
        return _apply_node_post_processing(result)

    const_value: Any = None
    has_const = False
    for key, entry in obj.items():
        if strip_or_keep(key, entry):
            continue
        if key == "const":
            const_value = entry
            has_const = True
            continue
        result[key] = _walk_child(key, entry, path, inside_schema_map)

    if isinstance(result.get("type"), list):
        types = [t for t in result["type"] if isinstance(t, str)]
        non_null = [t for t in types if t != "null"]
        result["type"] = non_null[0] if non_null else (types[0] if types else None)

    if has_const:
        existing_enum = result["enum"] if isinstance(result.get("enum"), list) else []
        _push_enum_value(existing_enum, const_value)
        result["enum"] = existing_enum
        if not result.get("type"):
            result["type"] = _infer_type_from_value(const_value)

    if (
        not result.get("type")
        and not isinstance(result.get("anyOf"), list)
        and not isinstance(result.get("oneOf"), list)
        and isinstance(result.get("enum"), list)
        and result["enum"]
    ):
        # Without `type` the proto cannot deserialize the members; inferring from the enum is
        # the only source available and only applies when they all agree.
        enum_types = [_infer_type_from_value(v) for v in result["enum"]]
        if all(t is not None for t in enum_types) and len(set(enum_types)) == 1:
            result["type"] = enum_types[0]

    if result.get("type") == "object" and "properties" not in result:
        # The proto requires the field; a `type: object` without it is read as an opaque
        # object and the tool receives arguments it never validated.
        result["properties"] = {}

    _spill_to_description(result, spill)
    return _apply_node_post_processing(result)


# omp: utils/schema/normalize.ts :: normalizeSchema, normalizeSchemaForCCA
def normalize_for_cca(schema: object) -> dict[str, Any]:
    """Sanitizes a tool JSON Schema for Cloud Code Assist.

    Returns ``{"type": "object", "properties": {}}`` when the result still carries a form
    the backend rejects or stopped being valid JSON Schema — a request with an invalid
    schema fails in its entirety, not just on that tool.
    """
    upgraded = upgrade_to_2020_12(schema)
    dereferenced = dereference_schema(upgraded)
    normalized = _normalize_schema_node(dereferenced, set(), False, True)
    normalized = strip_residual_combiners(normalized)
    normalized = _normalize_nullable_properties(normalized, False, _Seen())[0]
    if has_residual_incompatibilities(normalized):
        return dict(CCA_FALLBACK_SCHEMA)
    if not is_valid_json_schema(normalized) or not isinstance(normalized, dict):
        return dict(CCA_FALLBACK_SCHEMA)
    return normalized


# ---------------------------------------------------------------------------
# Meta-validator (utils/schema/meta-validator.ts)
# ---------------------------------------------------------------------------

_TYPE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "string",
        "number",
        "integer",
        "boolean",
        "object",
        "array",
        "null",
    }
)


def _is_non_negative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


# omp: utils/schema/meta-validator.ts :: hasUniqueJsonValues
def _has_unique_json_values(values: list[Any]) -> bool:
    for i, left in enumerate(values):
        if any(_json_equal(left, right) for right in values[i + 1 :]):
            return False
    return True


# omp: utils/schema/meta-validator.ts :: checkTypeKeyword
def _check_type_keyword(value: object) -> bool:
    if isinstance(value, str):
        return value in _TYPE_NAMES
    if not isinstance(value, list) or not value:
        return False
    seen: set[str] = set()
    for entry in value:
        if not isinstance(entry, str) or entry not in _TYPE_NAMES or entry in seen:
            return False
        seen.add(entry)
    return True


# omp: utils/schema/meta-validator.ts :: checkNode
def _check_node(node: object, seen: _Seen) -> bool:
    """Unknown keywords pass (forward compatibility); the known ones are checked so that a
    malformed payload lands on the fallback instead of going to the wire."""
    if node is True or node is False:
        return True
    if not isinstance(node, dict):
        return False
    if not seen.first(node):
        return True

    if "type" in node and not _check_type_keyword(node["type"]):
        return False

    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        if key in node:
            entry = node[key]
            if not isinstance(entry, list) or not all(_check_node(e, seen) for e in entry):
                return False
    for key in ("not", "if", "then", "else", "propertyNames", "contains"):
        if key in node and not _check_node(node[key], seen):
            return False
    for key in ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas"):
        if key not in node:
            continue
        entry = node[key]
        if not isinstance(entry, dict) or not all(_check_node(v, seen) for v in entry.values()):
            return False

    if "required" in node:
        entry = node["required"]
        if not isinstance(entry, list):
            return False
        if not all(isinstance(r, str) for r in entry) or len(set(entry)) != len(entry):
            return False

    if "items" in node:
        items = node["items"]
        if isinstance(items, list) or not _check_node(items, seen):
            return False
    # Draft-07 forms the upgrade should have eliminated: their survival means the pass
    # failed, and sending them anyway would be a 400 that is harder to diagnose.
    if "additionalItems" in node or "dependencies" in node:
        return False

    for key in ("additionalProperties", "unevaluatedProperties", "unevaluatedItems"):
        if key in node and not isinstance(node[key], bool) and not _check_node(node[key], seen):
            return False

    if "dependentRequired" in node:
        entry = node["dependentRequired"]
        if not isinstance(entry, dict):
            return False
        for deps in entry.values():
            if not isinstance(deps, list) or not all(isinstance(d, str) for d in deps):
                return False

    if "enum" in node:
        entry = node["enum"]
        if not isinstance(entry, list) or not entry or not _has_unique_json_values(entry):
            return False

    for key in ("minimum", "maximum", "multipleOf"):
        if key in node and not _is_number(node[key]):
            return False
    multiple_of = node.get("multipleOf")
    if (
        isinstance(multiple_of, int | float)
        and not isinstance(multiple_of, bool)
        and multiple_of <= 0
    ):
        return False
    for key in ("exclusiveMinimum", "exclusiveMaximum"):
        if key in node and not _is_number(node[key]) and not isinstance(node[key], bool):
            return False
    for key in (
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "minProperties",
        "maxProperties",
        "minContains",
        "maxContains",
    ):
        if key in node and not _is_non_negative_int(node[key]):
            return False
    for key in ("uniqueItems", "nullable", "readOnly", "writeOnly", "deprecated"):
        if key in node and not isinstance(node[key], bool):
            return False
    if "pattern" in node:
        if not isinstance(node["pattern"], str):
            return False
        try:
            re.compile(node["pattern"])
        except re.error:
            return False
    return not ("format" in node and not isinstance(node["format"], str))


# omp: utils/schema/meta-validator.ts :: isValidJsonSchema
def is_valid_json_schema(schema: object) -> bool:
    return _check_node(schema, _Seen())
