"""Google Antigravity (Cloud Code API) wire protocol.

Extracted from the original ``sitecustomize.py`` with no behaviour change. Builds the
``:streamGenerateContent`` envelope; transport (SSE, host failover, catalog) stays out.

Two injected dependencies instead of globals: media fetching by URL (``fetch_url``) and the
catalog (``ModelCatalog``). That is what makes the whole payload buildable without the
network — the original called ``httpx.get`` in the middle of the conversion.
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
from .schema import normalize_for_cca

# Byte limit for inlining media. The backend accepts well beyond this, but a request that
# drags tens of MB per turn is a latency and context-window problem, not a capacity one.
INLINE_MAX_BYTES: Final = 12 * 1024 * 1024
FETCH_TIMEOUT_S: Final = 20.0
FETCH_USER_AGENT: Final = "Mozilla/5.0 (X11; Linux x86_64) litellm-mysubs-antigravity/1.0"

DATA_URI_RE: Final = re.compile(r"^data:([^;,]+)(;[^,]*)?,(.*)$", re.S)

# URIs that `fileData` accepts: the Gemini Files API and GCS. A web URL does not work —
# measured: `fileData` with https://upload.wikimedia.org/... returns "404 Requested entity
# was not found", so those have to be fetched and inlined by us.
FILE_URI_PREFIXES: Final[tuple[str, ...]] = (
    "gs://",
    "https://generativelanguage.googleapis.com/",
)

# omp: stream.ts :: mapEffortToGoogleThinkingLevel
# Effort -> Gemini 3 thinkingLevel (the 2.x dialect uses thinkingBudget).
THINKING_LEVEL: Final[dict[str, str]] = {
    "minimal": "MINIMAL",
    "low": "LOW",
    "medium": "MEDIUM",
    "high": "HIGH",
    "xhigh": "HIGH",
    "max": "HIGH",
}

#: Level used to **suppress** reasoning. omp sends `{level: "MINIMAL"}` or `{budget: 0}`;
#: sending `LOW` or the catalog's `minThinkingBudget` (which for several variants is not
#: zero) still spends budget — and with `includeThoughts: false` the tokens are billed
#: without the text coming back.
SUPPRESSED_THINKING_LEVEL: Final = "MINIMAL"

DEFAULT_MAX_OUTPUT_TOKENS: Final = 64000

# omp: providers/google-shared.ts :: SKIP_THOUGHT_SIGNATURE
#: The CCA requires the sentinel when the **first** call of an assistant turn goes without
#: a signature; later calls in the same turn go bare.
SIGNATURE_SENTINEL: Final = "skip_thought_signature_validator"

#: Text of a tool result that carries nothing but an image.
IMAGE_ONLY_RESULT: Final = "(see attached image)"

#: Reasoning signatures are padded base64. A string that does not match gets a 400 from the
#: CCA — and being truthy, it kept the sentinel from rescuing the request.
_BASE64_SIGNATURE = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")


# omp: providers/google-shared.ts :: isValidThoughtSignature
def is_valid_signature(signature: object) -> bool:
    text = str(signature or "")
    return bool(text) and len(text) % 4 == 0 and _BASE64_SIGNATURE.match(text) is not None


TEXT_PART_TYPES: Final[tuple[str, ...]] = ("text", "input_text", "output_text")

# omp: providers/vision-guard.ts :: NON_VISION_IMAGE_PLACEHOLDER
#: Sending an image to a model without vision gives a 400. Dropping it silently was worse:
#: the model answered about content it never received. The placeholder says so in text.
NON_VISION_IMAGE_PLACEHOLDER: Final = "[image omitted: model does not support vision]"

#: A surrogate does not encode as UTF-8 and blows up the payload.
#:
#: Language difference that matters: in JavaScript a `\ud83d\ude00` pair **is** one
#: character (😀) and `toWellFormed()` preserves it, replacing only the lone ones. In Python
#: they are two separate code points and neither encodes — a "valid" pair still blows up
#: `str.encode("utf-8")` and the `json.dumps(..., ensure_ascii=False)` that many HTTP
#: clients use. Here **all** of them are replaced, not just the lone ones.
_SURROGATE = re.compile(r"[\ud800-\udfff]")


# omp: providers/google-shared.ts :: convertMessages
def well_formed(text: object) -> str:
    """UTF-8-encodable text, ready for the wire.

    Equivalent to omp's ``toWellFormed()``, adapted to Python semantics: there the pair
    survives because it forms one character, here it forms none and would have to blow up
    during serialization.
    """
    return _SURROGATE.sub("\ufffd", str(text))


class MediaTooLargeError(Exception):
    """Media above the inlining limit."""


class MediaFetchError(Exception):
    """The media the backend does not accept by URL could not be fetched."""


#: Markers of the notice a retired model returns **instead** of an answer. Measured on the
#: real account, with HTTP 200 and `finishReason: STOP`:
#:
#:     "Gemini 3.5 Flash is no longer available. Please switch to Gemini 3.7 Flash in the
#:      latest version of Antigravity."
#:
#: Stored lowercase and compared lowercase: upstream has already changed the casing of the
#: sentence between client versions, and a case-sensitive `in` would stop catching the
#: notice without anything failing visibly.
RETIREMENT_MARKERS: Final[tuple[str, ...]] = ("is no longer available", "please switch to")

#: Counts the CCA returns in `usageMetadata`. A retired model comes back with all of them at
#: zero (measured: `total_tokens=0`) because no model ran — not even the prompt was billed.
_USAGE_COUNTS: Final[tuple[str, ...]] = (
    "totalTokenCount",
    "promptTokenCount",
    "candidatesTokenCount",
    "thoughtsTokenCount",
    "cachedContentTokenCount",
)


class ModelRetiredError(Exception):
    """Upstream accepted the request but the model no longer exists.

    Its own type so that whoever catches it can tell this apart from a transport failure: a
    network failure is worth retrying, a retired model never answers again — what you do is
    migrate to the name the notice itself points at.
    """

    def __init__(self, wire_model: str, notice: str) -> None:
        super().__init__(
            f"Google Antigravity: {wire_model} has been retired — the upstream returned "
            f"200 with a notice and zero tokens instead of running the model. "
            f"Upstream: {notice.strip()!r}"
        )
        #: Name that went on the wire, for anyone wanting to mark it bad in the registry.
        self.wire_model = wire_model
        #: Upstream text exactly as it came: it is what says where to migrate.
        self.notice = notice


def is_retirement_notice(text: object) -> bool:
    """The text has the shape of the retirement notice.

    This alone is **not** proof: a legitimate answer discussing retired models would match
    the same markers. See `is_retired_response`.
    """
    lowered = str(text or "").lower()
    return all(marker in lowered for marker in RETIREMENT_MARKERS)


def usage_is_zero(meta: Mapping[str, Any] | None) -> bool:
    """No tokens counted — the request never got to run on a model.

    This alone is not proof either: a turn that returns only a tool call, or an empty
    answer, can arrive without counts.
    """
    if not meta:
        return True
    return not any(int(meta.get(key) or 0) > 0 for key in _USAGE_COUNTS)


def is_retired_response(text: object, usage_meta: Mapping[str, Any] | None) -> bool:
    """Retired-model response: notice text **and** zero usage.

    Both conditions are required because each alone errs in a different direction: by text
    you would catch a genuine answer about retired models (which came with billed tokens),
    by usage you would catch any legitimate empty answer. It is the conjunction that
    separates the guard from the false positive.

    Note the proof is always the **response**, never the name. Measured on the same account:
    `gemini-3.5-flash-lite` answers ("2 + 2 = 4", 12 tokens) while `-low` and `-extra-low`
    are dead; `tab_flash_lite_preview` answers and `tab_jump_flash_lite_preview` gives 400.
    A prefix rule killed good models and let the dead ones through.
    """
    return is_retirement_notice(text) and usage_is_zero(usage_meta)


def raise_if_retired(text: object, usage_meta: Mapping[str, Any] | None, wire_model: str) -> None:
    """Raises `ModelRetiredError` if the response is the retirement notice.

    It has to run **before** the text is emitted: accepted as an answer, the notice enters
    the conversation history as if the model had spoken. That is worse than an error — an
    error is at least visible.
    """
    if is_retired_response(text, usage_meta):
        raise ModelRetiredError(wire_model, str(text))


class FetchedMedia(NamedTuple):
    mime: str
    content: bytes


#: Signature of whoever fetches a URL. Injected so the payload is buildable without network.
UrlFetcher = Callable[[str], FetchedMedia]


def _reject_fetch(url: str) -> FetchedMedia:
    raise MediaFetchError(
        f"Google Antigravity: the media at {url[:120]} would have to be fetched and "
        f"inlined (the backend does not accept web URLs in fileData), but no fetcher "
        f"was provided"
    )


def normalize_effort(value: object) -> tuple[str | None, str | None]:
    """Returns ``(effort, summary)``; see the same function in ``wire.anthropic``."""
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
            f"Google Antigravity: media of {len(raw)} bytes exceeds the "
            f"{INLINE_MAX_BYTES} limit for inlining"
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
    """``inlineData`` from a data URI, ``fileData`` from an accepted URI, or a fetch.

    Measured on the backend: ``inlineData.data`` has to be bare base64 (the
    ``data:...;base64,`` prefix gives 400 "Invalid value at ... inline_data.data"), and the
    ``mimeType`` is honoured — a PDF inlined with ``application/pdf`` was read (it returned
    the word that was on the page).
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

    # Bare base64, which some clients send with no prefix.
    if len(text) > 64 and re.fullmatch(r"[A-Za-z0-9+/=\s]+", text):
        try:
            return inline_part(mime_hint or "image/png", base64.b64decode(text, validate=False))
        except Exception:
            return None
    return None


#: Block types that carry an image — the only ones the vision guard filters. A PDF or audio
#: does not go through the backend's vision path.
IMAGE_PART_TYPES: Final[tuple[str, ...]] = ("image_url", "input_image")


# omp: providers/google-shared.ts :: convertGoogleImagePart
def media_part(
    part: dict[str, Any], fetch: UrlFetcher | None = None, *, supports_images: bool = True
) -> dict[str, Any] | None:
    """Converts a multimodal part in the OpenAI shape.

    Without this, a request with an image reached the model with only the text and the
    answer talked about an image it never saw. The Codex bridge already handled this, so the
    asymmetry was not intentional.
    """
    kind = part.get("type")

    if kind in IMAGE_PART_TYPES:
        if not supports_images:
            return None
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


# omp: providers/google-shared.ts :: convertMessages
def content_parts(
    content: object, fetch: UrlFetcher | None = None, *, supports_images: bool = True
) -> list[dict[str, Any]]:
    """Parts of a turn, with text and media preserved in input order.

    A text block that is empty or whitespace-only produces no ``part``: the source says it
    "can cause issues with some models (e.g. Claude via Antigravity)", and a
    ``{"text": " "}`` carries no information to pay for that risk.
    """
    if not isinstance(content, list):
        text = well_formed(content) if content is not None else ""
        return [{"text": text}] if text.strip() else []

    parts: list[dict[str, Any]] = []
    omitted_image = False
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") in TEXT_PART_TYPES:
            text = well_formed(part.get("text") or "")
            if text.strip():
                parts.append({"text": text})
            continue
        if media := media_part(part, fetch, supports_images=supports_images):
            parts.append(media)
        elif not supports_images and part.get("type") in IMAGE_PART_TYPES:
            omitted_image = True
    if omitted_image:
        parts.append({"text": NON_VISION_IMAGE_PLACEHOLDER})
    return parts


# -- tools ---------------------------------------------------------------------


# omp: providers/google-gemini-cli.ts :: normalizeAntigravityTools
def tools_to_declarations(
    model: str, tools: list[Any] | None
) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]]]:
    """Function declarations in the Antigravity dialect.

    Every declaration goes in ``parameters`` with the sanitized schema — ``model`` is there
    only for error context. ``parametersJsonSchema`` never reaches this backend's wire:
    Cloud Code Assist refuses with 400 the constructs full JSON Schema allows (``anyOf``,
    ``oneOf``, ``not``, ``$ref``, ``type: ["string", "null"]``, ``const``), and sending them
    raw made the request fail instead of the schema being normalized.
    """
    if not tools:
        return None, []

    declarations: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        nested = tool.get("function")
        function: dict[str, Any] = nested if isinstance(nested, dict) else tool
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        schema = function.get("parameters") or {"type": "object", "properties": {}}
        declarations.append(
            {
                "name": name,
                "description": str(function.get("description") or ""),
                "parameters": normalize_for_cca(schema),
            }
        )

    return ([{"functionDeclarations": declarations}] if declarations else None), declarations


def tool_config(choice: object, declarations: list[dict[str, Any]]) -> dict[str, Any]:
    """``VALIDATED`` by default: the backend validates the call against the schema before
    emitting it."""
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
    message: dict[str, Any], fetch: UrlFetcher | None = None, *, supports_images: bool = True
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Result text and the media that rides along, in ``functionResponse.parts``.

    Measured: an image inside ``functionResponse.parts`` is seen by the model on every
    generation this account serves — gemini-3.8-flash, 3.1-pro, 3.1-flash-lite, 2.5-flash,
    2.5-flash-lite and pro-agent all answered "Azul" (blue) to a blue screenshot returned by
    a tool. omp only uses the inline form on Gemini 3+ and on earlier ones sends the image
    in a following user turn, because the old public API rejects it; on Antigravity that is
    unnecessary, and it is one synthetic turn less in the history.
    """
    content = message.get("content")
    omitted_image = False
    if isinstance(content, list):
        # Separator between parts: without it the last word of one sticks to the first of
        # the next and the model reads two sentences as one.
        text = "\n".join(
            well_formed(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in (None, *TEXT_PART_TYPES)
        )
        media = [
            built
            for built in (
                media_part(x, fetch, supports_images=supports_images)
                for x in content
                if isinstance(x, dict)
            )
            if built
        ]
        omitted_image = not supports_images and any(
            isinstance(x, dict) and x.get("type") in IMAGE_PART_TYPES for x in content
        )
    else:
        text = well_formed(content) if content else ""
        media = []

    if omitted_image:
        # Without the note the model reads a result that omits the image the tool returned,
        # and answers as if it did not exist.
        text = "\n".join(x for x in (text, NON_VISION_IMAGE_PLACEHOLDER) if x)
    elif not text and media:
        # A result with nothing but an image has to say something: `output: ""` is read as a
        # tool with no result, and the model tends to repeat the call.
        text = IMAGE_ONLY_RESULT
    value = {"error" if message.get("is_error") else "output": text}
    return value, media


# -- envelope ------------------------------------------------------------------


def _thinking_config(
    effort: str, info: Mapping[str, Any], max_output_tokens: int | None = None
) -> dict[str, Any]:
    """Omitting ``thinkingConfig`` makes the CCA reapply the server defaults and bill
    thinking tokens without returning the text.

    Antigravity uses *budget* transport; ``thinkingLevel`` is the gemini-cli dialect. With a
    catalog, the ``thinkingBudget`` advertised for the variant is used (-low 1000,
    -medium 4000, -high -1 = dynamic, pro-agent 10001) plus ``minThinkingBudget`` to turn it
    off. Without a catalog it falls back to ``thinkingLevel``, which is also accepted.

    ``max_output_tokens`` caps the budget because the two are not independent on the
    Anthropic backend::

        HTTP 400 `max_tokens` must be greater than `thinking.budget_tokens`

    Reproduced on the live gateway with ``max_tokens: 1024`` against a variant whose
    catalog budget is larger: the request failed on the first turn, while the same call
    with no ceiling succeeded. The budget is what gives way — it is this plugin's own
    choice, while the ceiling belongs to the caller, and raising it would bill for output
    nobody asked for. A dynamic budget (-1) is left alone: the backend picks it itself.
    """
    budget = info.get("thinkingBudget")

    if effort == "none":
        # Suppressing means zero budget, not the catalog minimum: with
        # `includeThoughts: False` a positive budget is billed without returning any text.
        config: dict[str, Any] = {"includeThoughts": False}
        if isinstance(budget, int):
            config["thinkingBudget"] = 0
        else:
            config["thinkingLevel"] = SUPPRESSED_THINKING_LEVEL
        return config

    config = {"includeThoughts": True}
    if isinstance(budget, int) and budget > 0:
        fitted = _fit_budget(budget, max_output_tokens)
        if fitted is None:
            # No budget can satisfy both bounds: serve the turn without reasoning rather
            # than fail it. The caller asked for a ceiling, not for thinking.
            return {"includeThoughts": False, "thinkingBudget": 0}
        config["thinkingBudget"] = fitted
    elif not isinstance(budget, int):
        config["thinkingLevel"] = THINKING_LEVEL.get(effort, "MEDIUM")
    return config


#: Anthropic refuses any positive budget below this — `thinking.enabled.budget_tokens:
#: Input should be greater than or equal to 1024`. Measured on the live gateway.
MIN_THINKING_BUDGET: Final = 1024


def _fit_budget(budget: int, max_output_tokens: int | None) -> int | None:
    """The budget that fits under the caller's ceiling, or ``None`` if none does.

    Two bounds apply at once, and they close on each other::

        max_tokens      > budget_tokens      (the ceiling has to leave room)
        budget_tokens  >= 1024               (Anthropic's own minimum)

    Together they mean a ceiling of 1024 or less admits **no** valid budget. The first
    attempt here capped the budget at three quarters of the ceiling, which turned one
    rejection into another: ``max_tokens: 1024`` produced 768 and the backend answered
    `Input should be greater than or equal to 1024`. Both were measured on the live
    gateway, in that order.

    So a ceiling that cannot host reasoning returns ``None`` and the caller drops thinking
    for that turn. The budget is what gives way — it is this plugin's own choice, while the
    ceiling belongs to the caller and raising it would bill for output nobody asked for.
    """
    if max_output_tokens is None or budget < max_output_tokens:
        return budget
    if max_output_tokens <= MIN_THINKING_BUDGET:
        return None
    return max(MIN_THINKING_BUDGET, max_output_tokens - 1)


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
    supports_images: bool = True,
) -> dict[str, Any]:
    """``:streamGenerateContent`` envelope.

    ``request_id`` comes in as an argument: it has the form ``agent/<id>/<ts>/<traj>/<step>``
    and is session state, not something the conversion should invent.
    """
    extra = extra or {}
    signatures = thought_signatures or {}
    effort = normalize_effort(extra.get("reasoning_effort"))[0] or ""
    mapped_model = map_model(model, effort or None, catalog)
    supports_ids = supports_function_ids(model)

    # The function name does not travel in the OpenAI-shaped tool result; collect it from
    # the calls.
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
            value, media = tool_result_value(message, fetch, supports_images=supports_images)
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

        parts = content_parts(content, fetch, supports_images=supports_images)

        if role == "system":
            system_parts.extend(parts)
            continue

        if role == "assistant":
            # The sentinel is per **turn**, not per request: the CCA requires it whenever the
            # first call of an assistant turn goes without a signature. Marking it only once
            # left later turns with bare calls and a 400 at validation.
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

                # Only a signature that is valid base64 is resent: an arbitrary string gives
                # a 400 and, being truthy, kept the sentinel from rescuing it.
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

    # omp sends max_completion_tokens (OpenAI style); accept both spellings, otherwise the
    # output ceiling the client asked for is silently replaced by the default.
    max_tokens = (
        extra.get("max_tokens") or extra.get("max_completion_tokens") or DEFAULT_MAX_OUTPUT_TOKENS
    )
    info = (catalog.info.get(mapped_model) if catalog else None) or {}
    request: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "thinkingConfig": _thinking_config(effort, info, max_tokens),
        },
    }

    # The native field is accepted with role "user" and with no practical size limit —
    # verified with 2520 chars: HTTP 200.
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
    "NON_VISION_IMAGE_PLACEHOLDER",
    "RETIREMENT_MARKERS",
    "SIGNATURE_SENTINEL",
    "FetchedMedia",
    "MediaFetchError",
    "MediaTooLargeError",
    "ModelRetiredError",
    "base_family",
    "build_payload",
    "content_parts",
    "inline_part",
    "is_retired_response",
    "is_retirement_notice",
    "media_from_url",
    "media_part",
    "normalize_effort",
    "raise_if_retired",
    "tool_config",
    "tool_result_value",
    "tools_to_declarations",
    "usage_is_zero",
    "well_formed",
]
