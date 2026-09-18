"""Wire protocol do Google Antigravity (Cloud Code API).

Extraído sem alteração de comportamento do ``sitecustomize.py`` original. Constrói o
envelope de ``:streamGenerateContent``; o transporte (SSE, failover de host, catálogo)
fica fora.

Duas dependências injectadas em vez de globais: a busca de media por URL
(``fetch_url``) e o catálogo (``ModelCatalog``). É o que permite construir o payload
inteiro sem rede — o original chamava ``httpx.get`` a meio da conversão.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import re
import urllib.parse
from collections.abc import Callable, Mapping
from typing import Any, Final, NamedTuple

from .antigravity_models import ModelCatalog, base_family, map_model, supports_function_ids

# Limite de bytes para inlinar media. O backend aceita bem além disto, mas um pedido que
# arraste dezenas de MB por turno é um problema de latência e de janela, não de capacidade.
INLINE_MAX_BYTES: Final = 12 * 1024 * 1024
FETCH_TIMEOUT_S: Final = 20.0
FETCH_USER_AGENT: Final = "Mozilla/5.0 (X11; Linux x86_64) litellm-mysubs-antigravity/1.0"

DATA_URI_RE: Final = re.compile(r"^data:([^;,]+)(;[^,]*)?,(.*)$", re.S)

# URIs que o `fileData` aceita: Files API do Gemini e GCS. Um URL da web não serve —
# medido: `fileData` com https://upload.wikimedia.org/... devolve "404 Requested entity was
# not found", logo esses têm de ser buscados e inlinados por nós.
FILE_URI_PREFIXES: Final[tuple[str, ...]] = (
    "gs://",
    "https://generativelanguage.googleapis.com/",
)

# omp: stream.ts :: mapEffortToGoogleThinkingLevel
# Effort -> thinkingLevel do Gemini 3 (o dialecto 2.x usa thinkingBudget).
THINKING_LEVEL: Final[dict[str, str]] = {
    "minimal": "MINIMAL",
    "low": "LOW",
    "medium": "MEDIUM",
    "high": "HIGH",
    "xhigh": "HIGH",
    "max": "HIGH",
}

#: Nível usado para **suprimir** o raciocínio. O OMP manda `{level: "MINIMAL"}` ou
#: `{budget: 0}`; mandar `LOW` ou o `minThinkingBudget` do catálogo (que para várias
#: variantes não é zero) continua a gastar orçamento — e com `includeThoughts: false` os
#: tokens são facturados sem o texto voltar.
SUPPRESSED_THINKING_LEVEL: Final = "MINIMAL"

DEFAULT_MAX_OUTPUT_TOKENS: Final = 64000

# omp: providers/google-shared.ts :: SKIP_THOUGHT_SIGNATURE
#: O CCA exige a sentinela quando a **primeira** chamada de um turno assistant vai sem
#: assinatura; chamadas seguintes do mesmo turno ficam nuas.
SIGNATURE_SENTINEL: Final = "skip_thought_signature_validator"

#: Texto de um tool result que só traz imagem.
IMAGE_ONLY_RESULT: Final = "(see attached image)"

#: Assinaturas de raciocínio são base64 com padding. Uma string que não case dá 400 do
#: CCA — e como é truthy, impedia a sentinela de salvar o pedido.
_BASE64_SIGNATURE = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")


# omp: providers/google-shared.ts :: isValidThoughtSignature
def is_valid_signature(signature: object) -> bool:
    text = str(signature or "")
    return bool(text) and len(text) % 4 == 0 and _BASE64_SIGNATURE.match(text) is not None


TEXT_PART_TYPES: Final[tuple[str, ...]] = ("text", "input_text", "output_text")


class MediaTooLargeError(Exception):
    """Media acima do limite de inline."""


class MediaFetchError(Exception):
    """Não foi possível obter a media que o backend não aceita por URL."""


class FetchedMedia(NamedTuple):
    mime: str
    content: bytes


#: Assinatura de quem vai buscar um URL. Injectada para o payload ser construível sem rede.
UrlFetcher = Callable[[str], FetchedMedia]


def _reject_fetch(url: str) -> FetchedMedia:
    raise MediaFetchError(
        f"Google Antigravity: a media em {url[:120]} teria de ser buscada e inlinada "
        f"(o backend não aceita URLs da web em fileData), mas não foi fornecido um fetcher"
    )


def normalize_effort(value: object) -> tuple[str | None, str | None]:
    """Devolve ``(effort, summary)``; ver a mesma função em ``wire.anthropic``."""
    if isinstance(value, dict):
        effort = value.get("effort")
        summary = value.get("summary")
    else:
        effort = value
        summary = None
    return (
        str(effort or "").strip().lower() or None,
        str(summary).strip().lower() if summary else None,
    )


# -- media ---------------------------------------------------------------------


def inline_part(mime: str | None, raw: bytes) -> dict[str, Any] | None:
    if not raw:
        return None
    if len(raw) > INLINE_MAX_BYTES:
        raise MediaTooLargeError(
            f"Google Antigravity: media de {len(raw)} bytes excede o limite "
            f"de {INLINE_MAX_BYTES} para inlinar"
        )
    return {
        "inlineData": {
            "mimeType": str(mime or "application/octet-stream"),
            "data": base64.b64encode(raw).decode("ascii"),
        }
    }


def media_from_url(
    url: object, mime_hint: str | None = None, fetch: UrlFetcher | None = None
) -> dict[str, Any] | None:
    """``inlineData`` a partir de um data URI, ``fileData`` de um URI aceite, ou fetch.

    Medido no backend: ``inlineData.data`` tem de ser base64 nu (o prefixo
    ``data:...;base64,`` dá 400 "Invalid value at ... inline_data.data"), e o ``mimeType``
    é respeitado — um PDF inlinado com ``application/pdf`` foi lido (devolveu a palavra
    que estava na página).
    """
    text = str(url or "")

    if match := DATA_URI_RE.match(text):
        mime, params, payload = match.group(1), match.group(2) or "", match.group(3)
        if "base64" in params:
            return inline_part(mime, base64.b64decode(payload))
        return inline_part(mime, urllib.parse.unquote_to_bytes(payload))

    if text.startswith(FILE_URI_PREFIXES):
        return {
            "fileData": {
                "mimeType": str(mime_hint or "application/octet-stream"),
                "fileUri": text,
            }
        }

    if text.startswith(("http://", "https://")):
        fetched = (fetch or _reject_fetch)(text)
        return inline_part(fetched.mime or mime_hint or "application/octet-stream", fetched.content)

    # Base64 nu, que alguns clientes enviam sem prefixo.
    if len(text) > 64 and re.fullmatch(r"[A-Za-z0-9+/=\s]+", text):
        try:
            return inline_part(mime_hint or "image/png", base64.b64decode(text, validate=False))
        except Exception:
            return None
    return None


def media_part(part: dict[str, Any], fetch: UrlFetcher | None = None) -> dict[str, Any] | None:
    """Converte uma parte multimodal do shape OpenAI.

    Sem isto, um pedido com imagem chegava ao modelo apenas com o texto e a resposta falava
    de uma imagem que ele nunca viu. A ponte do Codex já tratava isto, logo a assimetria
    não era intencional.
    """
    kind = part.get("type")

    if kind in ("image_url", "input_image"):
        image = part.get("image_url") or part.get("image") or part.get("url")
        if isinstance(image, dict):
            return media_from_url(image.get("url"), image.get("mime_type"), fetch)
        return media_from_url(image, None, fetch)

    if kind in ("file", "input_file", "input_document", "document"):
        nested = part.get("file")
        spec: dict[str, Any] = nested if isinstance(nested, dict) else part
        mime = spec.get("mime_type") or spec.get("mimeType")
        if not mime:
            name = str(spec.get("filename") or "")
            mime = mimetypes.guess_type(name)[0] if name else None
        if data := (spec.get("file_data") or spec.get("data")):
            return media_from_url(data, mime, fetch)
        if uri := (spec.get("file_uri") or spec.get("fileUri") or spec.get("file_id")):
            return media_from_url(uri, mime, fetch)

    if kind in ("input_audio", "audio"):
        nested_audio = part.get("input_audio")
        spec = nested_audio if isinstance(nested_audio, dict) else part
        fmt = str(spec.get("format") or "wav").lower()
        if data := spec.get("data"):
            return media_from_url(data, f"audio/{fmt}", fetch)
    return None


def content_parts(content: object, fetch: UrlFetcher | None = None) -> list[dict[str, Any]]:
    """Partes de um turno, com texto e media preservados pela ordem de entrada."""
    if not isinstance(content, list):
        return [{"text": str(content)}] if content is not None and str(content) else []

    parts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") in TEXT_PART_TYPES:
            if part.get("text"):
                parts.append({"text": str(part["text"])})
            continue
        if media := media_part(part, fetch):
            parts.append(media)
    return parts


# -- tools ---------------------------------------------------------------------


def tools_to_declarations(
    model: str, tools: list[Any] | None
) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]]]:
    """Declarações de função no dialecto do Antigravity.

    ``parametersJsonSchema`` aceita OpenAPI 3.0 completo; o campo antigo ``parameters`` é
    o dialecto reduzido, e só a família Claude servida por esta API o exige.
    """
    if not tools:
        return None, []

    declarations: list[dict[str, Any]] = []
    legacy = model.split("/")[-1].lower().startswith("claude-")
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        nested = tool.get("function")
        function: dict[str, Any] = nested if isinstance(nested, dict) else tool
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        schema = function.get("parameters") or {"type": "object", "properties": {}}
        declaration: dict[str, Any] = {
            "name": name,
            "description": str(function.get("description") or ""),
        }
        declaration["parameters" if legacy else "parametersJsonSchema"] = schema
        declarations.append(declaration)

    return ([{"functionDeclarations": declarations}] if declarations else None), declarations


def tool_config(choice: object, declarations: list[dict[str, Any]]) -> dict[str, Any]:
    """``VALIDATED`` por default: o backend valida a chamada contra o schema antes de a
    emitir."""
    if choice in (None, "auto"):
        return {"functionCallingConfig": {"mode": "VALIDATED"}}
    if choice == "none":
        return {"functionCallingConfig": {"mode": "NONE"}}
    if choice in ("required", "any"):
        return {"functionCallingConfig": {"mode": "ANY"}}
    if isinstance(choice, dict):
        nested = choice.get("function")
        function = nested if isinstance(nested, dict) else choice
        name = function.get("name") if isinstance(function, dict) else None
        if name and any(d.get("name") == name for d in declarations):
            return {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [name]}}
    return {"functionCallingConfig": {"mode": "VALIDATED"}}


# omp: providers/google-shared.ts :: pendingToolImageParts
def tool_result_value(
    message: dict[str, Any], fetch: UrlFetcher | None = None
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Texto do resultado e a media que vai à parte, em ``functionResponse.parts``.

    Medido: uma imagem dentro de ``functionResponse.parts`` é vista pelo modelo em todas as
    gerações que esta conta serve — gemini-3.8-flash, 3.1-pro, 3.1-flash-lite, 2.5-flash,
    2.5-flash-lite e pro-agent responderam todos "Azul" a uma captura azul devolvida por
    uma tool. O omp só usa a forma inline no Gemini 3+ e nos anteriores manda a imagem num
    turno user seguinte, porque a API pública antiga rejeita-a; no Antigravity não é
    preciso, e é um turno sintético a menos no histórico.
    """
    content = message.get("content")
    if isinstance(content, list):
        # Separador entre partes: sem ele a última palavra de uma cola-se à primeira da
        # seguinte e o modelo lê duas frases como uma.
        text = "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in (None, *TEXT_PART_TYPES)
        )
        media = [
            built
            for built in (media_part(x, fetch) for x in content if isinstance(x, dict))
            if built
        ]
    else:
        text = str(content or "")
        media = []

    # Um resultado só com imagem tem de dizer alguma coisa: `output: ""` é lido como tool
    # sem resultado, e o modelo tende a repetir a chamada.
    if not text and media:
        text = IMAGE_ONLY_RESULT
    value = {"error" if message.get("is_error") else "output": text}
    return value, media


# -- envelope ------------------------------------------------------------------


def _thinking_config(effort: str, info: Mapping[str, Any]) -> dict[str, Any]:
    """Omitir ``thinkingConfig`` faz o CCA reaplicar os defaults do servidor e facturar
    thinking tokens sem devolver o texto.

    O Antigravity usa transporte por *budget*; o ``thinkingLevel`` é o dialecto do
    gemini-cli. Com catálogo usa-se o ``thinkingBudget`` anunciado para a variante
    (-low 1000, -medium 4000, -high -1 = dinâmico, pro-agent 10001) e o
    ``minThinkingBudget`` para desligar. Sem catálogo cai-se no ``thinkingLevel``, que
    também é aceite.
    """
    budget = info.get("thinkingBudget")

    if effort == "none":
        # Suprimir é orçamento zero, não o mínimo do catálogo: com `includeThoughts: False`
        # um orçamento positivo é facturado sem devolver texto nenhum.
        config: dict[str, Any] = {"includeThoughts": False}
        if isinstance(budget, int):
            config["thinkingBudget"] = 0
        else:
            config["thinkingLevel"] = SUPPRESSED_THINKING_LEVEL
        return config

    config = {"includeThoughts": True}
    if isinstance(budget, int) and budget > 0:
        config["thinkingBudget"] = budget
    elif not isinstance(budget, int):
        config["thinkingLevel"] = THINKING_LEVEL.get(effort, "MEDIUM")
    return config


def build_payload(
    model: str,
    messages: list[Any],
    project_id: str,
    request_id: str,
    tools: list[Any] | None = None,
    extra: dict[str, Any] | None = None,
    catalog: ModelCatalog | None = None,
    fetch: UrlFetcher | None = None,
    thought_signatures: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Envelope de ``:streamGenerateContent``.

    ``request_id`` entra por argumento: tem o formato ``agent/<id>/<ts>/<traj>/<passo>`` e
    é estado de sessão, não algo que a conversão deva inventar.
    """
    extra = extra or {}
    signatures = thought_signatures or {}
    effort = normalize_effort(extra.get("reasoning_effort"))[0] or ""
    mapped_model = map_model(model, effort or None, catalog)
    supports_ids = supports_function_ids(model)

    # O nome da função não viaja no tool result do formato OpenAI; recolhe-se das chamadas.
    tool_names: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            call_id = str(tool_call.get("id") or "").split("|", 1)[0]
            if call_id:
                tool_names[call_id] = function.get("name") or "tool"

    contents: list[dict[str, Any]] = []
    system_parts: list[dict[str, Any]] = []
    pending_tool_responses: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal pending_tool_responses
        if pending_tool_responses:
            contents.append({"role": "user", "parts": pending_tool_responses})
            pending_tool_responses = []

    for message in messages:
        role = message.get("role", "user") if isinstance(message, dict) else "user"
        if role != "tool":
            flush()
        content = message.get("content") if isinstance(message, dict) else None

        if role == "tool":
            call_id = str(message.get("tool_call_id") or "").split("|", 1)[0]
            value, media = tool_result_value(message, fetch)
            function_response: dict[str, Any] = {
                "name": message.get("name") or tool_names.get(call_id) or "tool",
                "response": value,
            }
            if media:
                function_response["parts"] = media
            if supports_ids and call_id:
                function_response["id"] = call_id
            pending_tool_responses.append({"functionResponse": function_response})
            continue

        parts = content_parts(content, fetch)

        if role == "system":
            system_parts.extend(parts)
            continue

        if role == "assistant":
            # A sentinela é por **turno**, não por pedido: o CCA exige-a sempre que a
            # primeira chamada de um turno assistant vai sem assinatura. Marcá-la uma vez
            # só deixava os turnos seguintes com chamadas nuas e 400 na validação.
            first_tool_call = True
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function") or {}
                call_id, _, encoded_signature = str(tool_call.get("id") or "").partition("|")
                arguments = function.get("arguments") or {}
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"__raw": arguments}

                function_call: dict[str, Any] = {
                    "name": function.get("name") or "",
                    "args": arguments,
                }
                if supports_ids and call_id:
                    function_call["id"] = call_id
                part: dict[str, Any] = {"functionCall": function_call}

                # Só se reenvia uma assinatura que seja base64 válido: uma string
                # arbitrária dá 400 e, por ser truthy, impedia a sentinela de a salvar.
                candidate = next(
                    (
                        value
                        for value in (
                            tool_call.get("thoughtSignature"),
                            tool_call.get("thought_signature"),
                            encoded_signature,
                            signatures.get(call_id),
                        )
                        if is_valid_signature(value)
                    ),
                    None,
                )
                if candidate:
                    part["thoughtSignature"] = candidate
                elif first_tool_call:
                    part["thoughtSignature"] = SIGNATURE_SENTINEL
                first_tool_call = False
                parts.append(part)
            if parts:
                contents.append({"role": "model", "parts": parts})
            continue

        if parts:
            contents.append({"role": "user", "parts": parts})
    flush()

    # O OMP envia max_completion_tokens (estilo OpenAI); aceitar ambas as grafias, senão o
    # tecto de output que o cliente pediu é substituído em silêncio pelo default.
    max_tokens = (
        extra.get("max_tokens") or extra.get("max_completion_tokens") or DEFAULT_MAX_OUTPUT_TOKENS
    )
    info = (catalog.info.get(mapped_model) if catalog else None) or {}
    request: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "thinkingConfig": _thinking_config(effort, info),
        },
    }

    # O campo nativo é aceite com role "user" e sem limite prático de tamanho — verificado
    # com 2520 chars: HTTP 200.
    if system_parts:
        request["systemInstruction"] = {"role": "user", "parts": system_parts}

    antigravity_tools, declarations = tools_to_declarations(model, tools)
    if antigravity_tools:
        request["tools"] = antigravity_tools
        request["toolConfig"] = tool_config(extra.get("tool_choice"), declarations)

    return {
        "project": project_id,
        "requestId": request_id,
        "model": mapped_model,
        "userAgent": "antigravity",
        "requestType": "agent",
        "request": request,
    }


__all__ = [
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "INLINE_MAX_BYTES",
    "SIGNATURE_SENTINEL",
    "FetchedMedia",
    "MediaFetchError",
    "MediaTooLargeError",
    "base_family",
    "build_payload",
    "content_parts",
    "inline_part",
    "media_from_url",
    "media_part",
    "normalize_effort",
    "tool_config",
    "tool_result_value",
    "tools_to_declarations",
]
