"""Saneamento do JSON Schema das ferramentas para o Cloud Code Assist.

Porte de ``utils/schema/normalize.ts`` — só o caminho CCA (``normalizeSchemaForCCA``) e o
fecho transitivo do que ela usa.

O backend do Cloud Code Assist faz protojson sobre um `Schema` proto fechado: qualquer
campo que não caiba no proto devolve 400 com "Cannot find field", e as formas de
composição (``anyOf``/``oneOf``/``allOf``/``not``/``$ref``) e ``type: ["string","null"]``
não têm sequer representação. Mandar o schema do cliente cru faz o pedido inteiro falhar,
não só a ferramenta — daí sanear antes de escrever no fio.

A estratégia é sempre **alargar**, nunca estreitar: um schema demasiado permissivo deixa o
modelo produzir um argumento que a ferramenta rejeita depois; um schema estreitado de mais
impede-o de sequer o tentar. Quando nem alargar chega, cai-se no schema vazio
``{"type": "object", "properties": {}}`` — perde-se a tipagem, mas a chamada passa.
"""

from __future__ import annotations

import json
import re
from typing import Any, Final

JsonObject = dict[str, Any]

#: Campos que o proto Schema da Google não tem. Recursos de draft 2020-12 (``$dynamicRef``),
#: pointers (``$ref``), anotações (``deprecated``, ``readOnly``) e toda a validação fina —
#: protojson rejeita o nome desconhecido antes sequer de olhar para o valor.
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

#: Dos campos removidos, estes dizem algo ao modelo em linguagem natural. Vão para a
#: ``description`` porque um ``minLength: 3`` apagado em silêncio custa uma chamada
#: rejeitada que o modelo não tem como prever.
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

#: O schema com que se substitui um que não dê para sanear. Objecto sem propriedades: o
#: modelo perde a tipagem dos argumentos mas a ferramenta continua a existir no catálogo.
# omp: utils/schema/normalize.ts :: CLOUD_CODE_ASSIST_CLAUDE_FALLBACK_SCHEMA
CCA_FALLBACK_SCHEMA: Final[JsonObject] = {"type": "object", "properties": {}}

#: python-genai renomeia estas chaves antes de serializar; o proto só conhece a forma
#: camelCase, por isso um ``any_of`` que passasse cru seria descartado em silêncio.
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

#: Palavras-chave cujo valor é *um* subschema. Um ``True``/``False`` cru aqui é um boolean
#: subschema de draft 2020-12, não um valor literal.
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

#: Tanto a forma canónica com ``#`` como a sem — o Zod emite uma, os servidores MCP a outra.
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

#: Descer por aqui corromperia a carga: ``type: ["string","null"]`` é um valor, não um
#: subschema, e um ``enum`` de objectos seria reescrito como se fossem schemas.
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
    """Igualdade estrutural. ``==`` do Python trata ``True == 1`` como verdadeiro, o que
    fundiria um ``enum: [true]`` com um ``enum: [1]`` e perderia um membro."""
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
    """Intersecção. ``allOf`` é o único modo de manter ambas as restrições quando as
    chaves de ``dependencies`` colidem — descartar uma perderia validação."""
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
    """``nullable: true`` do OpenAPI 3.0 não existe em 2020-12; a forma equivalente
    depende do que o nó já declara, e envolver sempre em ``anyOf`` criaria um combinador
    onde bastava alargar o ``type``."""
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
    """A cache é semeada **antes** da recursão para que uma aresta de retorno num grafo
    cíclico resolva para a referência em construção em vez de entrar em ciclo infinito."""
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
    """Draft-07 mistura dependências de array e de schema sob uma chave; 2020-12 separa-as
    em ``dependentRequired`` e ``dependentSchemas``."""
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
        # Um ciclo de `$ref` inlinado nunca termina; `{}` corta-o mantendo o nó válido.
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
        # Em 2020-12 as chaves irmãs de `$ref` são válidas e mais específicas que a
        # definição apontada, por isso ganham.
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
# Spill para description (utils/schema/spill.ts)
# ---------------------------------------------------------------------------


# omp: utils/schema/spill.ts :: spillToDescription
def _spill_to_description(node: JsonObject, entries: list[tuple[str, Any]]) -> None:
    """Junta as restrições removidas ao fim da ``description``.

    Formato "spill": um objecto JSON por nó, separado por linha em branco do texto
    existente. Concatenar sem separador tornaria a restrição indistinguível da prosa.
    """
    if not entries:
        return
    existing = node.get("description")
    existing_text = existing if isinstance(existing, str) else ""
    # Separadores compactos: é o que `JSON.stringify` produz, e a descrição vai contar
    # para a janela de contexto de cada chamada.
    body = ", ".join(f"{key}: {json.dumps(value, separators=(',', ':'))}" for key, value in entries)
    formatted = f"{{{body}}}"
    node["description"] = f"{existing_text}\n\n{formatted}" if existing_text else formatted


# ---------------------------------------------------------------------------
# Colapso de combinadores (utils/schema/normalize.ts, utils/schema/equality.ts)
# ---------------------------------------------------------------------------


# omp: utils/schema/normalize.ts :: classifySchemaChild
def _classify_schema_child(key: str, value: object, inside_schema_map: bool) -> str | None:
    """Só os filhos que são de facto JSON Schema; uma carga de instância fica opaca.

    Dentro de um mapa (``properties``) as chaves são nomes escolhidos pelo utilizador: uma
    propriedade chamada ``items`` é um schema por ser valor do mapa, não pela palavra.
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
    """União dos membros só quando os ramos concordam em ``type`` e em tudo o que não é
    ``enum``. Discordar e fundir à mesma trocaria a descrição de um ramo pela do outro."""
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
    """Duas definições da mesma propriedade em ramos diferentes: aceitar ambas. Escolher
    uma rejeitaria argumentos válidos do outro ramo."""
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
    """Ramos todos de forma objecto fundem-se numa união de ``properties``.

    É a única projecção que não perde propriedades: colapsar para o primeiro ramo deixaria
    o modelo sem saber que os campos dos outros existem. O ``required`` é que tem de
    encolher — ver abaixo.
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
        # `allOf` exige todos os ramos, por isso a união não estreita a aceitação.
        combined_required: list[str] = []
        for required in branch_required:
            for name in required:
                if name not in combined_required:
                    combined_required.append(name)
    else:
        # `anyOf`/`oneOf` aceitam um ramo qualquer: só o que TODOS exigem se mantém
        # obrigatório, senão a projecção rejeitaria instâncias que o original aceita.
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
    """União de tipos diferentes (``string`` ou ``number``) colapsa para um só tipo.

    O proto do CCA tem um campo ``type`` escalar: não há forma de exprimir a união. Só se
    colapsa quando os ramos não se contradizem em mais nada, senão perder-se-ia restrição
    sem aviso — nesse caso devolve-se o schema intacto e o teste de resíduos manda-o para
    o fallback.
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
                # Descrições são anotação, não estrutura: juntar não muda a aceitação.
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

    # Um `items` herdado do pai num nó agora tipado `string` é um campo que o proto não
    # aceita naquela posição — protojson 400a mesmo sendo ele próprio válido.
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
    """Ramos todos do mesmo ``type`` colapsam num só nó."""
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
        # Ficar-se pelo primeiro ramo apagaria os membros dos outros: um `anyOf` de dois
        # enums de string colapsava para metade dos valores legais.
        merged: JsonObject | None = first_entry
        for variant in variants[1:]:
            if merged is None:
                break
            merged = _merge_compatible_enum_schemas(merged, variant)
        if merged is None:
            return schema
        collapsed = merged
    elif enum_variant_count > 0:
        # Há um ramo sem `enum`, logo mais lato. Colapsar para ele mantém a aceitação;
        # colapsar para um ramo enum estreitaria aos seus membros.
        collapsed = next((v for v in variants if not isinstance(v.get("enum"), list)), first_entry)
    else:
        collapsed = first_entry

    next_schema = _copy_without(schema, combiner)
    for key, value in collapsed.items():
        if key not in next_schema:
            next_schema[key] = value
    return next_schema


class _Seen:
    """Conjunto de visitados por identidade, que **retém** cada objecto marcado.

    ``id()`` do CPython é o endereço: um dicionário temporário libertado a meio da travessia
    devolve o mesmo ``id`` ao seguinte, e um ``set[int]`` cru declarava-o já visitado. Na
    prática isso truncava propriedades para ``{}`` de forma dependente do alocador — a
    diferenciação contra o TS apanhou 18 casos em 600 antes desta retenção. O
    ``WeakMap`` do original não tem o problema porque a chave é o objecto vivo.
    """

    __slots__ = ("_ids", "_keep")

    def __init__(self) -> None:
        self._ids: set[int] = set()
        self._keep: list[object] = []

    def first(self, value: object) -> bool:
        """True na primeira vez que ``value`` é visto nesta travessia."""
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
    """Ponto fixo. Fundir combinadores de objecto pode sintetizar um ``anyOf`` novo numa
    propriedade partilhada (ver ``_merge_property_schemas``) depois de a recursão sobre os
    filhos já ter corrido — uma só passagem deixava-o escapar para o fio."""
    return _strip_residual_combiners_node(value, _Seen(), False)


# ---------------------------------------------------------------------------
# Nullable e incompatibilidades residuais
# ---------------------------------------------------------------------------


# omp: utils/schema/normalize.ts :: extractNullableUnionSchema
def _extract_nullable_union_schema(schema: object) -> tuple[object, bool]:
    """Devolve ``(schema_sem_null, era_nullable)``.

    O CCA não tem ``nullable`` nem união com ``null``. A nulabilidade sobrevive ao ser
    traduzida para *opcional*: o chamador tira o campo do ``required``. Apagá-la sem mais
    tornaria obrigatório um campo que o cliente declarou poder faltar.
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
            # Só um `{type: "null"}` pelado conta: `{type: "null", description: …}` leva
            # informação que se perderia ao descartar o ramo.
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
                # Pai e ramo discordam: fundir escolheria arbitrariamente um deles.
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
#: Os cinco resíduos que o CCA rejeita com 400. Verificados depois de todo o saneamento:
#: o que ainda cá estiver não tem representação possível, logo o schema vai para fallback.
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
# Passagem principal (utils/schema/normalize.ts)
# ---------------------------------------------------------------------------


# omp: utils/schema/normalize.ts :: applySnakeCaseRenames
def _apply_snake_case_renames(obj: JsonObject) -> JsonObject:
    """Colisão resolve-se a favor do snake_case (``pop(from)`` → ``set(to)`` do
    python-genai), que é a forma que o cliente escreveu de propósito."""
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
        # Um booleano só é subschema numa posição de subschema; em `nullable: true` ou num
        # membro de `enum` é valor literal e coagi-lo destruiria o schema.
        if not boolean_is_subschema:
            return value
        # Modo "standard": `false` é o schema impossível, que só `not: {}` exprime.
        return {} if value else {"not": {}}
    if not isinstance(value, dict):
        return value
    # Rasto do caminho, não conjunto de visitados: subárvores partilhadas num DAG são
    # normalizadas em cada ocorrência; só ciclos verdadeiros curto-circuitam.
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
        """True quando a chave foi removida (e talvez despejada na descrição)."""
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

        # Um `anyOf` de `const` é um `enum` escrito por extenso; o CCA tem `enum` mas não
        # `const` nem combinadores, por isso esta é a tradução que não perde nada.
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
        # Sem `type` o proto não sabe desserializar os membros; inferir do enum é a única
        # fonte disponível e só se aplica quando todos concordam.
        enum_types = [_infer_type_from_value(v) for v in result["enum"]]
        if all(t is not None for t in enum_types) and len(set(enum_types)) == 1:
            result["type"] = enum_types[0]

    if result.get("type") == "object" and "properties" not in result:
        # O proto exige o campo; um `type: object` sem ele é lido como objecto opaco e a
        # ferramenta recebe argumentos que nunca validou.
        result["properties"] = {}

    _spill_to_description(result, spill)
    return _apply_node_post_processing(result)


# omp: utils/schema/normalize.ts :: normalizeSchema, normalizeSchemaForCCA
def normalize_for_cca(schema: object) -> dict[str, Any]:
    """Saneia um JSON Schema de ferramenta para o Cloud Code Assist.

    Devolve ``{"type": "object", "properties": {}}`` quando o resultado ainda traz uma
    forma que o backend rejeita ou deixou de ser JSON Schema válido — um pedido com schema
    inválido falha por inteiro, não só naquela ferramenta.
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
# Meta-validador (utils/schema/meta-validator.ts)
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
    """Palavras-chave desconhecidas passam (compatibilidade futura); as conhecidas são
    verificadas para que uma carga malformada caia no fallback em vez de ir para o fio."""
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
    # Formas de draft-07 que o upgrade devia ter eliminado: sobreviverem significa que a
    # passagem falhou, e mandá-las na mesma seria um 400 mais difícil de diagnosticar.
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
