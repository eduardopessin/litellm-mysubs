"""Google Antigravity (Cloud Code API) wire protocol.

Extracted from the original ``sitecustomize.py`` with no behaviour change. Builds the
``:streamGenerateContent`` envelope; transport (SSE, host failover, catalog) stays out.

Two injected dependencies instead of globals: media fetching by URL (``fetch_url``) and the
catalog (``ModelCatalog``). That is what makes the whole payload buildable without the
network — the original called ``httpx.get`` in the middle of the conversion.
"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import re
import secrets
import time
import urllib.parse
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final, NamedTuple

from .anthropic import supports_sampling_params
from .antigravity_models import (
    ModelCatalog,
    base_family,
    is_claude,
    map_model,
    requires_first_call_signature,
    supports_function_ids,
)
from .schema import normalize_for_cca, normalize_for_google, tool_wire_schema

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

# omp: model-thinking.ts :: mapEffortToGoogleThinkingLevel
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

# omp: providers/google-shared.ts :: SKIP_THOUGHT_SIGNATURE
#: The CCA requires the sentinel when the **first** call of an assistant turn of a Gemini 3+
#: model goes without a signature; later calls in the same turn go bare.
SIGNATURE_SENTINEL: Final = "skip_thought_signature_validator"

# omp: providers/google-gemini-cli.ts :: buildRequest
#: ``google-antigravity-forced-tool.md``, which ``buildRequest`` imports as text — the file's
#: trailing newline included. Appended as a user turn when a Gemini request forces a tool
#: call. The drift checker hashes TypeScript and KDL only, so this anchor tracks the function
#: that sends the text, not the markdown file itself.
FORCED_TOOL_DIRECTIVE: Final = (
    "TOOL-ONLY TURN. This turn accepts a tool call and nothing else; a text reply here is "
    "discarded unread and you will be re-prompted. Emit the tool call now.\n"
)

#: ``toolConfig`` when nothing else applies. omp: "Antigravity's default tool mode is
#: VALIDATED (verified for Gemini and Claude)" — the backend checks each call against the
#: declared schema before emitting it.
VALIDATED_MODE: Final = "VALIDATED"

# omp: providers/google-shared.ts :: convertMessages
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


_ASCII_SPACE: Final = " \t\n\v\f\r"


# omp: dialect/rendering.ts :: findDelimitedThinkingClose
def _find_thinking_close(open_: str, close: str, text: str, start: int, end: int) -> int:
    depth = 1
    cursor = start
    while cursor < end:
        next_close = text.find(close, cursor)
        if next_close < 0 or next_close >= end:
            return -1
        next_open = text.find(open_, cursor)
        if 0 <= next_open < next_close:
            depth += 1
            cursor = next_open + len(open_)
            continue
        depth -= 1
        if depth == 0:
            return next_close
        cursor = next_close + len(close)
    return -1


# omp: dialect/rendering.ts :: unwrapDelimitedThinking
def _unwrap_thinking(open_: str, close: str, text: str) -> str:
    """Reasoning already wrapped in the target's tags, unwrapped so it is not wrapped twice."""
    end = len(text.rstrip(_ASCII_SPACE))
    cursor = len(text) - len(text.lstrip(_ASCII_SPACE))
    if cursor >= end or not text.startswith(open_, cursor):
        return text
    segments: list[str] = []
    while cursor < end:
        if not text.startswith(open_, cursor):
            return text
        inner_start = cursor + len(open_)
        inner_end = _find_thinking_close(open_, close, text, inner_start, end)
        if inner_end < 0:
            return text
        inner = text[inner_start:inner_end].strip(_ASCII_SPACE)
        segments.append(_unwrap_thinking(open_, close, inner))
        rest = text[inner_end + len(close) : end]
        cursor = end - len(rest.lstrip(_ASCII_SPACE))
    return "\n".join(segments)


# omp: dialect/demotion.ts :: renderDemotedThinking
# omp: dialect/gemini.ts :: renderThinking
# omp: dialect/xml.ts :: renderThinking
def demoted_thinking(model: str, text: str) -> str:
    """Prior-turn reasoning as text in the target model's own thinking form.

    omp's reasoning: a replayed unsigned ``thought`` part is accepted and silently discarded
    — "verified end-to-end against Gemini 3" — so reasoning a client sends back survives
    only as text, wrapped the way the model writes its own so it reads as reasoning and not
    as prose to imitate. Claude gets it bare: Anthropic's classifier refuses or leaks
    reasoning replayed inside thinking tags. ``gpt-oss`` (Harmony) gets a plain ``<think>``
    block, Gemini its ``thinking`` code fence, anything else omp's XML fallback.
    """
    if not text:
        return ""
    text = well_formed(text)
    name = str(model).split("/")[-1].lower()
    if is_claude(name):
        return text
    if name.startswith("gpt-oss"):
        return f"<think>\n{text}\n</think>"
    if name.startswith("gemini-"):
        return f"```thinking\n{text}\n```"
    return f"<thinking>\n{_unwrap_thinking('<thinking>', '</thinking>', text)}\n</thinking>"


# -- tools ---------------------------------------------------------------------


# omp: providers/openai-chat-server.ts :: buildTools
# omp: providers/google-shared.ts :: convertTools
# omp: providers/google-gemini-cli.ts :: normalizeAntigravityTools
def tools_to_declarations(model: str, tools: list[Any] | None) -> list[dict[str, Any]] | None:
    """Function declarations in the Antigravity dialect, the schema built as omp builds it.

    Every declaration goes in ``parameters`` normalized for Cloud Code Assist: the CCA
    refuses with 400 the constructs full JSON Schema allows (``anyOf``, ``oneOf``, ``not``,
    ``$ref``, ``type: ["string", "null"]``, ``const``), and sending them raw made the
    request fail instead of the schema being widened. The road there is omp's:
    ``toolWireSchema`` first; Claude (``ccaLegacyParametersSchema``) straight to the CCA
    pass; every other model through the Google dialect before it — ``convertTools`` builds
    ``parametersJsonSchema`` with it and ``normalizeAntigravityTools`` moves that into
    ``parameters`` through the CCA pass. The Google pass is what adds ``propertyOrdering``
    and reads a null branch or a ``type: "null"`` field as nullable instead of giving up on
    the whole schema.

    A tool with no ``parameters`` declares ``{}``, as omp's ``buildTools`` does — except on
    Claude, where the backend hands ``parameters`` to Anthropic as ``input_schema`` and
    Anthropic requires its ``type``. Measured on the live backend on 2026-09-30 with a
    parameterless tool on claude-sonnet-4-6: ``{}`` answered 400
    "tools.0.custom.input_schema.type: Field required",
    ``{"type": "object", "properties": {}}`` answered 200; gemini-3-flash took both.
    """
    if not tools:
        return None

    legacy_parameters = is_claude(model)
    declarations: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        nested = tool.get("function")
        function: dict[str, Any] = nested if isinstance(nested, dict) else tool
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        wire = tool_wire_schema(function.get("parameters") or {})
        parameters = normalize_for_cca(wire if legacy_parameters else normalize_for_google(wire))
        if legacy_parameters and "type" not in parameters:
            parameters = {"type": "object", "properties": {}, **parameters}
        declarations.append(
            {
                "name": name,
                "description": str(function.get("description") or ""),
                "parameters": parameters,
            }
        )

    return [{"functionDeclarations": declarations}] if declarations else None


# omp: providers/openai-chat-server.ts :: normalizeToolChoice
# omp: providers/openai-responses-server.ts :: mapToolChoice
# omp: stream.ts :: mapGoogleToolChoice
# omp: providers/google-shared.ts :: mapToolChoice
def function_calling_config(choice: object) -> dict[str, Any] | None:
    """The ``functionCallingConfig`` a chat or Responses ``tool_choice`` asks for, or
    ``None`` when it asks for nothing beyond the default.

    omp's chain, end to end: the server parser keeps ``auto``/``none``/``required`` and
    reduces a named choice (chat ``{"function": {"name"}}``, Responses or Anthropic-style
    ``{"type": ..., "name"}``) to its name; ``mapGoogleToolChoice`` turns ``required`` into
    ``ANY`` and a name into an ``ANY`` allow-list of one; ``auto`` and everything else set
    nothing. The name is not checked against the declared tools: omp sends it as asked.
    """
    if choice == "none":
        return {"mode": "NONE"}
    if choice == "required":
        return {"mode": "ANY"}
    if not isinstance(choice, dict):
        return None
    function = choice.get("function")
    if function:
        name = function.get("name") if isinstance(function, dict) else None
    elif choice.get("type") in ("tool", "function", "custom"):
        name = choice.get("name")
    else:
        return None
    if isinstance(name, str) and name:
        return {"mode": "ANY", "allowedFunctionNames": [name]}
    return None


# omp: providers/openai-chat-server.ts :: stringifyContent
def text_of(content: object) -> str:
    """A system or assistant message's text: its text parts joined with nothing between.

    omp's chat server reads both roles this way — ``systemInstruction`` is text, and an
    assistant turn replays as one text block — so the media either carries is dropped.
    """
    if content is None:
        return ""
    if isinstance(content, list):
        return "".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") in TEXT_PART_TYPES
        )
    return str(content)


_UNSAFE_CALL_ID = re.compile(r"[^a-zA-Z0-9_-]")


# omp: utils.ts :: normalizeToolCallId
# omp: providers/google-shared.ts :: convertMessages
def wire_call_id(call_id: str) -> str:
    """A tool call id the backend takes: ``[a-zA-Z0-9_-]``, at most 64 characters.

    Anthropic, behind Claude on this host, refuses anything else in ``tool_use.id``. The call
    and its result are rewritten alike, so they still pair.
    """
    return _UNSAFE_CALL_ID.sub("_", call_id)[:64]


# omp: providers/openai-chat-server.ts :: buildAssistantMessage
def call_arguments(raw: object) -> object:
    """A replayed call's arguments as the object ``functionCall.args`` must be.

    What does not parse into an object is kept under ``__raw`` rather than thrown away:
    losing it lost the intent of the call.
    """
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    if not isinstance(raw, str):
        return {"__raw": raw}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"__raw": raw}
    return parsed if isinstance(parsed, dict) else {"__raw": raw}


# omp: stream.ts :: mapOptionsForApi
# omp: providers/google-gemini-cli.ts :: buildRequest
#: The caller's sampling fields omp carries into ``generationConfig`` (besides
#: ``temperature``, which takes the slot ahead of ``maxOutputTokens``).
SAMPLING_FIELDS: Final[tuple[tuple[str, str], ...]] = (
    ("top_p", "topP"),
    ("top_k", "topK"),
    ("presence_penalty", "presencePenalty"),
)

#: Claude refuses a lower ``top_p`` while thinking ("`top_p` must be greater than or equal
#: to 0.95 or unset when thinking is enabled").
CLAUDE_THINKING_MIN_TOP_P: Final = 0.95


# omp: stream.ts :: withSupportedSamplingParams
def refused_sampling(model: str, field: str, value: object, *, thinking: bool) -> bool:
    """A sampling field the backend refuses for this model — left out instead of failing
    the turn. ``field`` is the wire name (``temperature``, ``topP``, ``topK``,
    ``presencePenalty``).

    omp 18.8.6 drops every sampling field before any provider builds its payload when the
    model's compat says ``supportsSamplingParams: false`` (`withSupportedSamplingParams`;
    the axis now covers the ``google`` APIs). On Antigravity that resolves false only for
    adaptive Claude — ``claude-opus-5-5`` and ``claude-sonnet-5-5`` in pi-catalog 18.8.6's
    ``models.json`` — the same rule `anthropic.supports_sampling_params` ports. Not
    measured here: the account does not serve Claude 5.5 through Antigravity; leaving a
    field out cannot cause a 400, sending one is what omp's catalog records as a 400.

    The rest are measured on the live backend on 2026-09-30, one field at a time over a
    request that otherwise answered 200 (``temperature 0.2``, ``topP 0.9``, ``topK 40``,
    ``presencePenalty 0.5``, ``frequencyPenalty 0.5``) — omp sends them all as given:

    - every Gemini (gemini-3-flash, gemini-3.1-pro-low, gemini-3.8-flash-medium,
      gemini-2.5-flash) refused both penalties — "Penalty is not enabled for this model" —
      and took the other three;
    - claude-sonnet-4-6, thinking, refused ``topP 0.9`` and took the other four;
    - gpt-oss-120b-medium took all five.

    ``frequencyPenalty`` is not in the list at all: omp never sends it.
    """
    name = str(model).split("/")[-1].lower()
    if is_claude(name) and not supports_sampling_params(name):
        return True
    if field == "presencePenalty":
        return name.startswith("gemini-")
    if field == "topP" and thinking and is_claude(name):
        return isinstance(value, int | float) and value < CLAUDE_THINKING_MIN_TOP_P
    return False


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


# omp: providers/google-gemini-cli.ts :: buildRequest
def _thinking_config(
    effort: str,
    info: Mapping[str, Any],
    max_output_tokens: int | None = None,
    *,
    claude: bool = False,
) -> dict[str, Any]:
    """Omitting ``thinkingConfig`` makes the CCA reapply the server defaults and bill
    thinking tokens without returning the text.

    Antigravity uses *budget* transport; ``thinkingLevel`` is the gemini-cli dialect. With a
    catalog, the ``thinkingBudget`` advertised for the variant is used (-low 1000,
    -medium 4000, -high -1 = dynamic, pro-agent 10001) plus ``minThinkingBudget`` to turn it
    off. Without a catalog it falls back to ``thinkingLevel``, which is also accepted.

    ``max_output_tokens`` is the request's total ceiling (`output_ceiling`) — the caller's
    own; a budget that does not fit under it gives way as `_fit_budget` decides.
    """
    budget = info.get("thinkingBudget")

    if effort == "none":
        if isinstance(budget, int) and not refuses_zero_budget(info, claude=claude):
            # Suppressing means zero budget, not the catalog minimum: with
            # `includeThoughts: False` a positive budget is billed without returning text.
            return {"includeThoughts": False, "thinkingBudget": 0}
        if isinstance(budget, int):
            return thinking_floor(budget, info)
        return {"includeThoughts": False, "thinkingLevel": SUPPRESSED_THINKING_LEVEL}

    config: dict[str, Any] = {"includeThoughts": True}
    if isinstance(budget, int) and budget > 0:
        fitted = _fit_budget(
            budget, max_output_tokens, claude=claude, floor=info.get("minThinkingBudget")
        )
        if fitted is None:
            # No budget fits under the ceiling: serve the turn without reasoning, omp's
            # "budget clamped to zero — fall through to the thinking-off path".
            return {"includeThoughts": False, "thinkingBudget": 0}
        config["thinkingBudget"] = fitted
    elif not isinstance(budget, int):
        config["thinkingLevel"] = THINKING_LEVEL.get(effort, "MEDIUM")
    return config


#: The ``minThinkingBudget`` of every model measured to accept a budget of 0.
ZERO_BUDGET_MIN_THINKING: Final = 32


def refuses_zero_budget(info: Mapping[str, Any], *, claude: bool) -> bool:
    """Whether this model answers ``thinkingBudget: 0`` with 400.

    Live 2026-09-30, ``maxOutputTokens: 64``, ``includeThoughts: false``, budget 0:

    - refused — gemini-3.1-pro-low and gemini-pro-agent ("Budget 0 is invalid. This model
      only works in thinking mode."), gemini-2.5-flash, gemini-2.5-flash-lite,
      gemini-3.5-flash-lite and gpt-oss-120b-medium ("Request contains an invalid
      argument"). Their catalog ``minThinkingBudget`` is 128, or absent (gpt-oss);
    - accepted — Claude (sonnet-4-6, opus-4-6-thinking) and the gemini-3 / 3.6 / 3.7 / 3.8
      flash ids, whose catalog minimum is 32.

    The catalog minimum is what separates them, so it is what is read: Claude accepts 0,
    anything else only with a minimum of at most 32. An id not measured is placed by the
    same reading of its catalog entry.
    """
    if claude:
        return False
    floor = info.get("minThinkingBudget")
    if isinstance(floor, bool) or not isinstance(floor, int) or floor <= 0:
        return True
    return floor > ZERO_BUDGET_MIN_THINKING


# omp: stream.ts :: normalizeMandatoryReasoningOptions
def thinking_floor(budget: int, info: Mapping[str, Any]) -> dict[str, Any]:
    """The least thinking a model that cannot turn it off accepts.

    omp answers "no reasoning" on a model whose thinking is mandatory by raising the request
    to the model's lowest effort, not by switching thinking off. Here the lowest is the
    catalog's ``minThinkingBudget`` — measured 200 at 128 on gemini-3.1-pro-low,
    gemini-pro-agent, gemini-2.5-flash, gemini-2.5-flash-lite and gemini-3.5-flash-lite —
    or, with no minimum (gpt-oss-120b-medium), the catalog budget itself (8192 measured 200;
    nothing lower was measured). The thoughts come back: with ``includeThoughts: false`` a
    positive budget is billed without the text.
    """
    floor = info.get("minThinkingBudget")
    if isinstance(floor, int) and not isinstance(floor, bool) and floor > 0:
        return {"includeThoughts": True, "thinkingBudget": floor}
    if budget > 0:
        return {"includeThoughts": True, "thinkingBudget": budget}
    return {"includeThoughts": True}


# omp: stream.ts :: MIN_OUTPUT_TOKENS
#: Room omp keeps for the visible answer when a ceiling forces the budget down.
MIN_OUTPUT_TOKENS: Final = 1024

#: Anthropic refuses any positive budget below this — `thinking.enabled.budget_tokens:
#: Input should be greater than or equal to 1024`. Measured on the live gateway.
MIN_THINKING_BUDGET: Final = 1024


# omp: stream.ts :: mapOptionsForApi
def _fit_budget(
    budget: int, max_output_tokens: int | None, *, claude: bool, floor: object = None
) -> int | None:
    """The budget for a total ceiling that does not exceed it, or ``None`` for no thinking.

    omp's rule (``stream.ts``, the ``google-gemini-cli`` case) is kept where it holds: the
    budget becomes ``ceiling - MIN_OUTPUT_TOKENS``. Where that leaves nothing, omp turns
    thinking off with a budget of 0 — and that is only right on Claude, whose own minimum
    is 1024 and whose budget must stay under the ceiling. Everything else keeps thinking,
    at the catalog's ``minThinkingBudget`` (or the budget itself when there is none): on
    this backend ``maxOutputTokens`` bounds thoughts and answer together and a budget above
    it is accepted, while a budget of 0 is refused by the thinking-only models.

    Measured on the live backend (2026-09-30), ``maxOutputTokens: 64``, an essay prompt:

    - budget 0 refused — gemini-3.1-pro-low and gemini-pro-agent ("Budget 0 is invalid.
      This model only works in thinking mode."), gemini-2.5-flash, gemini-2.5-flash-lite,
      gemini-3.5-flash-lite and gpt-oss-120b-medium ("Request contains an invalid
      argument"); accepted by the gemini-3 / 3.6 / 3.7 / 3.8 flash and 3.1-flash-lite ids;
    - a budget above the ceiling accepted everywhere but Claude — 1001 and 128 on
      gemini-3.1-pro-low, 10001 on gemini-pro-agent, 4000 on gemini-3.8-flash-medium, 8192
      on gpt-oss — each ending ``MAX_TOKENS`` within the 64;
    - at ``minThinkingBudget`` (32) the flash ids answered 47-53 words, where their own
      4000 left no answer at all (every token went to thinking).
    """
    if max_output_tokens is None or budget < max_output_tokens:
        return budget
    fitted = max_output_tokens - MIN_OUTPUT_TOKENS
    if claude:
        return fitted if fitted >= MIN_THINKING_BUDGET else None
    minimum = 0
    if isinstance(floor, int) and not isinstance(floor, bool) and floor > 0:
        minimum = floor
    if fitted > 0 and fitted >= minimum:
        return fitted
    return minimum or budget


# omp: wire/gemini-headers.ts :: ANTIGRAVITY_MODEL_WIRE_PROFILES
def declared_output_tokens(entry: Any) -> int | None:
    """The output ceiling the catalog declares for a model, or ``None`` if it declares none.

    ``maxOutputTokens`` is per model and not uniform — measured on 2026-09-28 against the
    account's ``:fetchAvailableModels``: 65536 for the gemini 3.x variants, 65535 for 2.5
    and 3.1, 64000 for the two `claude-*`, 32768 for `gpt-oss-120b-medium`, 4096 for the
    `tab_*` ones, and absent for `chat_20706`, `chat_23310` and `gemini-3.1-flash-image`.
    Absent stays absent: a ceiling filled in here would be one the provider never set.
    """
    if not isinstance(entry, Mapping):
        return None
    value = entry.get("maxOutputTokens")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


# omp: providers/google-gemini-cli.ts :: buildRequest
def output_ceiling(requested: Any, entry: Any) -> int | None:
    """``maxOutputTokens`` for one request: the caller's, never above what the model accepts.

    The caller's ``max_tokens`` is the total — omp's "add thinking budget on top" is not
    followed. Measured on the live backend (2026-09-30): claude-sonnet-4-6 with no reasoning
    asked, ``max_tokens: 64`` and the 1024 budget on top went out as 1088, thought for 61
    characters and answered 623 words (844 completion tokens), ``STOP``; with 64 as the
    total it stopped at ``MAX_TOKENS`` within 64. The budget only bounds thinking, so the
    answer takes whatever thinking leaves of the sum.

    - the caller asked for a ceiling: it is sent, lowered only to the declared one — Claude
      on this backend answers ``maxOutputTokens > 64000`` with 400 (omp,
      ``ANTIGRAVITY_MODEL_WIRE_PROFILES``);
    - the caller asked for none and the catalog declares one: the declared one is sent;
    - neither: the field is omitted and the backend applies its own. A flat 64000 used to
      fill that gap: below the 65536 the gemini 3.x variants accept, and above the 4096 and
      32768 of the `tab_*` models and `gpt-oss-120b-medium`.
    """
    declared = declared_output_tokens(entry)
    try:
        asked = int(requested) if requested else None
    except (TypeError, ValueError):
        asked = None
    if asked is None or asked <= 0:
        return declared
    return asked if declared is None else min(asked, declared)


# omp: providers/transform-messages.ts :: transformMessages
#: Result omp synthesizes for a call the history never answered.
MISSING_TOOL_RESULT: Final = "No result provided"


def _pairing_key(call_id: object) -> str:
    """The call component of an id: what pairs a call with its result."""
    return str(call_id or "").split("|", 1)[0]


def _stale_tool_result(message: dict[str, Any]) -> dict[str, Any] | None:
    """A result whose call is gone, kept as a user note so the model still sees it."""
    content = message.get("content")
    if isinstance(content, list):
        texts = [
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") in (None, *TEXT_PART_TYPES)
        ]
    else:
        texts = [str(content)] if content else []
    texts = [text for text in texts if text.strip()]
    if not texts:
        return None
    error = ' is-error="true"' if message.get("is_error") else ""
    body = "\n".join(texts)
    return {
        "role": "user",
        "content": (
            f'<stale-tool-result tool="{message.get("name") or ""}" '
            f'id="{message.get("tool_call_id") or ""}"{error}>\n{body}\n</stale-tool-result>'
        ),
    }


# omp: providers/transform-messages.ts :: transformMessages
def pair_tool_results(messages: list[Any]) -> list[Any]:
    """The history with every assistant tool call followed by exactly one result.

    omp's ``transformMessages`` second pass, which runs before ``convertMessages``: a
    result that arrives late is pulled up behind its call, a duplicate is dropped, a call
    left unanswered gets an error result (``No result provided``), and a result whose call
    is nowhere in the history becomes a ``<stale-tool-result>`` user note — or is dropped
    while other calls still wait for theirs. Cloud Code refuses a turn whose function
    responses do not match its function calls, so the whole request used to fail.

    System messages pass through without closing a result window: omp's chat server lifts
    them out of the history before this pass runs.
    """
    results: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    declared: set[str] = set()
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "tool":
            key = _pairing_key(message.get("tool_call_id"))
            results.setdefault(key, []).append((index, message))
        elif message.get("role") == "assistant":
            declared.update(
                _pairing_key(call.get("id"))
                for call in message.get("tool_calls") or []
                if isinstance(call, dict)
            )
    consumed: set[int] = set()

    def take(key: str, after: int) -> dict[str, Any] | None:
        for index, message in results.get(key, ()):
            if index in consumed or index <= after:
                continue
            consumed.add(index)
            return message
        return None

    paired: list[Any] = []
    pending: list[dict[str, Any]] = []
    pending_from = -1
    resolved: set[str] = set()

    def flush() -> None:
        nonlocal pending
        for call in pending:
            key = _pairing_key(call.get("id"))
            if key in resolved:
                continue
            real = take(key, pending_from)
            paired.append(
                real
                if real is not None
                else {
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "content": MISSING_TOOL_RESULT,
                    "is_error": True,
                }
            )
            resolved.add(key)
        pending = []

    for index, message in enumerate(messages):
        role = message.get("role") if isinstance(message, dict) else None
        if role == "system":
            paired.append(message)
        elif role == "assistant":
            flush()
            calls = [c for c in message.get("tool_calls") or [] if isinstance(c, dict)]
            if calls:
                pending, pending_from = calls, index
            paired.append(message)
        elif role == "tool":
            key = _pairing_key(message.get("tool_call_id"))
            if key in resolved:
                continue
            if any(_pairing_key(call.get("id")) == key for call in pending):
                resolved.add(key)
                paired.append(message)
                continue
            if key in declared:
                # Its call is elsewhere; `flush` pulls it into that call's window.
                continue
            if any(_pairing_key(call.get("id")) not in resolved for call in pending):
                continue
            if note := _stale_tool_result(message):
                paired.append(note)
        else:
            flush()
            paired.append(message)
    flush()
    return paired


# -- request envelope ------------------------------------------------------------

# omp: wire/gemini-headers.ts :: ANTIGRAVITY_MODEL_WIRE_PROFILES
#: ``labels.model_enum`` per routed wire id — "the opaque token the client tags each request
#: with". Only the ``modelEnum`` half of omp's profiles: the fixed ``maxOutputTokens`` half
#: is the output-ceiling divergence recorded in ``docs/OMP.md``. The Claude ids have none.
ANTIGRAVITY_MODEL_ENUMS: Final[Mapping[str, str]] = {
    "gemini-3.5-flash-extra-low": "MODEL_PLACEHOLDER_M187",
    "gemini-3.5-flash-low": "MODEL_PLACEHOLDER_M20",
    "gemini-3-flash-agent": "MODEL_PLACEHOLDER_M132",
    "gemini-3.1-pro-low": "MODEL_PLACEHOLDER_M36",
    "gemini-pro-agent": "MODEL_PLACEHOLDER_M16",
}

# omp: providers/google-gemini-cli.ts :: INT63_MASK
_INT63_MASK: Final = (1 << 63) - 1
# omp: providers/google-gemini-cli.ts :: ANTIGRAVITY_RANDOM_BOUND
_RANDOM_BOUND: Final = 9_000_000_000_000_000_000


# omp: providers/google-gemini-cli.ts :: deriveSignedDecimalFromHash
def _signed_decimal_from_hash(text: str) -> str:
    # A lone surrogate does not encode; Bun hashes it as U+FFFD, which `well_formed` gives.
    digest = hashlib.sha256(well_formed(text).encode("utf-8")).digest()
    return f"-{int.from_bytes(digest[:8], 'big') & _INT63_MASK}"


# omp: providers/google-gemini-cli.ts :: randomBoundedInt63
# omp: providers/google-gemini-cli.ts :: randomSignedDecimalSessionId
def _random_signed_decimal() -> str:
    while True:
        value = int.from_bytes(secrets.token_bytes(8), "big") & _INT63_MASK
        if value < _RANDOM_BOUND:
            return f"-{value}"


# omp: providers/google-gemini-cli.ts :: getFirstUserTextForAntigravitySession
def _first_user_text(messages: list[Any]) -> str | None:
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return next(
                (
                    str(part.get("text") or "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") in TEXT_PART_TYPES
                ),
                None,
            )
        return None
    return None


# omp: providers/google-gemini-cli.ts :: deriveAntigravitySessionId
def derive_session_id(messages: list[Any]) -> str:
    """``sessionId`` with no session state: the first user text hashed, else random."""
    text = _first_user_text(messages)
    return _signed_decimal_from_hash(text) if text and text.strip() else _random_signed_decimal()


# omp: providers/google-gemini-cli.ts :: AntigravityProviderSessionState
@dataclass(slots=True)
class AntigravitySession:
    """One conversation's request identity, as the real ``antigravity/hub`` client keeps it.

    ``agent_id``/``trajectory_id`` are UUIDs, ``session_id`` a signed decimal, ``step_index``
    the monotonic step counter and ``last_execution_id`` the previous successful response's
    ``responseId``, echoed back as ``labels.last_execution_id``.
    """

    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trajectory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = field(default_factory=_random_signed_decimal)
    step_index: int = 1
    last_execution_id: str | None = None


#: omp's gateway keeps provider state per conversation in a bounded store; a proxy never
#: sees a conversation end, so the least recently used one is dropped — it restarts its
#: step count and trajectory, which the backend reads as a new agent session.
SESSION_LIMIT: Final = 4096

_sessions: OrderedDict[str, AntigravitySession] = OrderedDict()


# omp: providers/google-gemini-cli.ts :: getAntigravityProviderSessionState
# omp: auth-gateway/session-state.ts :: AuthGatewaySessionStateStore
def antigravity_session(key: str) -> AntigravitySession:
    """The state of the conversation ``key`` names, created on first use."""
    session = _sessions.get(key)
    if session is None:
        session = _sessions[key] = AntigravitySession()
        while len(_sessions) > SESSION_LIMIT:
            _sessions.popitem(last=False)
    else:
        _sessions.move_to_end(key)
    return session


class RequestEnvelope(NamedTuple):
    session_id: str
    request_id: str
    labels: dict[str, str]


# omp: providers/google-gemini-cli.ts :: buildAntigravityRequestEnvelope
def request_envelope(
    model: str, messages: list[Any], wire_model: str, state: AntigravitySession | None
) -> RequestEnvelope:
    """``sessionId``, ``requestId`` and ``labels`` for one request, advancing ``state``.

    ``requestId`` is ``agent/<agentId>/<ms>/<trajectoryId>/<step>`` and
    ``labels.last_step_index`` trails its step by one. Without state (a direct call) the
    ids are ephemeral and the session id comes from the first user text, as in omp.
    """
    if state is not None:
        state.step_index += 1
    agent_id = state.agent_id if state else str(uuid.uuid4())
    trajectory_id = state.trajectory_id if state else str(uuid.uuid4())
    session_id = state.session_id if state else derive_session_id(messages)
    step = state.step_index if state else 2
    request_id = f"agent/{agent_id}/{int(time.time() * 1000)}/{trajectory_id}/{step}"
    labels: dict[str, str] = {}
    if state is not None and state.last_execution_id:
        labels["last_execution_id"] = state.last_execution_id
    labels["last_step_index"] = str(step - 1)
    if (model_enum := ANTIGRAVITY_MODEL_ENUMS.get(wire_model)) is not None:
        labels["model_enum"] = model_enum
    labels["trajectory_id"] = trajectory_id
    # `antigravityUsageLabel ?? String(isClaude)`: the catalog sets "true" on the Claude
    # class only, so the fallback and the label agree.
    usage_label = "true" if is_claude(model) else "false"
    labels["used_claude"] = usage_label
    labels["used_claude_conservative"] = usage_label
    return RequestEnvelope(session_id, request_id, labels)


def build_payload(
    model: str,
    messages: list[Any],
    project_id: str,
    tools: list[Any] | None = None,
    extra: dict[str, Any] | None = None,
    catalog: ModelCatalog | None = None,
    fetch: UrlFetcher | None = None,
    thought_signatures: Mapping[str, str] | None = None,
    supports_images: bool = True,
    session: AntigravitySession | None = None,
) -> dict[str, Any]:
    """``:streamGenerateContent`` envelope.

    ``session`` is the conversation's state (`antigravity_session`); the request's ids and
    labels come from it, and building the payload advances its step.
    """
    extra = extra or {}
    messages = pair_tool_results(messages)
    signatures = thought_signatures or {}
    effort = normalize_effort(extra.get("reasoning_effort"))[0] or ""
    mapped_model = map_model(model, effort or None, catalog)
    supports_ids = supports_function_ids(model)
    first_call_sentinel = requires_first_call_signature(model)
    claude = is_claude(model)

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
    system_texts: list[str] = []
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
                # The name of the call it answers wins over the one the result carries, so
                # the pair always matches.
                "name": tool_names.get(call_id) or message.get("name") or "tool",
                "response": value,
            }
            if media:
                function_response["parts"] = media
            if supports_ids and call_id:
                function_response["id"] = wire_call_id(call_id)
            pending_tool_responses.append({"functionResponse": function_response})
            continue

        if role == "system":
            if text := text_of(content):
                system_texts.append(text)
            continue

        if role == "assistant":
            # Whitespace-only text is left out: the source says it "can cause issues with
            # some models (e.g. Claude via Antigravity)".
            text = well_formed(text_of(content))
            parts = [{"text": text}] if text.strip() else []
        else:
            parts = content_parts(content, fetch, supports_images=supports_images)

        if role == "assistant":
            # omp: providers/openai-chat-server.ts :: buildAssistantMessage
            # omp: providers/transform-messages.ts :: transformMessages
            # Reasoning the client sends back leads the turn as text (`demoted_thinking`).
            reasoning = message.get("reasoning_content")
            demoted = isinstance(reasoning, str) and bool(reasoning.strip())
            if demoted:
                parts.insert(0, {"text": demoted_thinking(model, str(reasoning))})
            # The sentinel is per **turn**, not per request: the CCA requires it whenever the
            # first call of an assistant turn goes without a signature. Marking it only once
            # left later turns with bare calls and a 400 at validation.
            first_tool_call = True
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function") or {}
                call_id, _, encoded_signature = str(tool_call.get("id") or "").partition("|")
                function_call: dict[str, Any] = {
                    "name": function.get("name") or "",
                    "args": call_arguments(function.get("arguments")),
                }
                if supports_ids and call_id:
                    function_call["id"] = wire_call_id(call_id)
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
                elif first_tool_call and first_call_sentinel:
                    part["thoughtSignature"] = SIGNATURE_SENTINEL
                first_tool_call = False
                parts.append(part)
            if demoted and len(parts) == 1:
                # As the turn's last block it loses its trailing whitespace: Anthropic
                # refuses a final assistant text that ends in whitespace.
                parts[0]["text"] = parts[0]["text"].rstrip()
            if parts:
                contents.append({"role": "model", "parts": parts})
            continue

        if parts:
            contents.append({"role": "user", "parts": parts})
    flush()

    request: dict[str, Any] = {"contents": contents}

    # omp: utils.ts :: normalizeSystemPrompts
    # omp's chat server joins every system message into one prompt. The native field is
    # accepted with role "user" and with no practical size limit — verified with 2520
    # chars: HTTP 200.
    system_prompt = well_formed("\n\n".join(system_texts))
    if system_prompt.strip():
        request["systemInstruction"] = {"role": "user", "parts": [{"text": system_prompt}]}

    antigravity_tools = tools_to_declarations(model, tools)
    if antigravity_tools:
        request["tools"] = antigravity_tools
        calling = function_calling_config(extra.get("tool_choice"))
        if calling is not None:
            request["toolConfig"] = {"functionCallingConfig": calling}
            # omp: "Cloud Code Assist drops `toolConfig` on Antigravity's Gemini routes: the
            # backend answers in text under `mode: "ANY"`" — so a Gemini request that forces
            # a call restates it in the transcript. Claude honours the config itself.
            if not claude and calling["mode"] == "ANY":
                contents.append({"role": "user", "parts": [{"text": FORCED_TOOL_DIRECTIVE}]})
        request.setdefault("toolConfig", {"functionCallingConfig": {"mode": VALIDATED_MODE}})
    if claude:
        # `antigravity-claude-tool-mode`: Claude on Antigravity always goes out VALIDATED,
        # with or without tools and over any explicit choice — the framing omp sends.
        request["toolConfig"] = {"functionCallingConfig": {"mode": VALIDATED_MODE}}

    envelope = request_envelope(model, messages, mapped_model, session)
    # Inside `request`, where omp puts them. Measured on the live backend (2026-09-30):
    # HTTP 200 on gemini-3-flash and claude-sonnet-4-6; the 400 "Unknown name" once
    # recorded was for the top level of the body.
    request["labels"] = envelope.labels

    # omp sends max_completion_tokens (OpenAI style); accept both spellings, otherwise the
    # output ceiling the client asked for is silently replaced.
    info = (catalog.info.get(mapped_model) if catalog else None) or {}
    max_tokens = output_ceiling(
        extra.get("max_tokens") or extra.get("max_completion_tokens"), info
    )
    # The caller's sampling knobs, in the slots omp gives them: `temperature` ahead of the
    # ceiling, which keeps its place ahead of `thinkingConfig`.
    thinking_config = _thinking_config(effort, info, max_tokens, claude=claude)
    thinking = bool(thinking_config.get("includeThoughts"))
    generation: dict[str, Any] = {}
    temperature = extra.get("temperature")
    if temperature is not None and not refused_sampling(
        mapped_model, "temperature", temperature, thinking=thinking
    ):
        generation["temperature"] = temperature
    if max_tokens is not None:
        generation["maxOutputTokens"] = max_tokens
    for source, target in SAMPLING_FIELDS:
        sampled = extra.get(source)
        if sampled is not None and not refused_sampling(
            mapped_model, target, sampled, thinking=thinking
        ):
            generation[target] = sampled
    generation["thinkingConfig"] = thinking_config
    request["generationConfig"] = generation
    request["sessionId"] = envelope.session_id

    return {
        "project": project_id,
        "requestId": envelope.request_id,
        "model": mapped_model,
        "userAgent": "antigravity",
        "requestType": "agent",
        "request": request,
    }


__all__ = [
    "ANTIGRAVITY_MODEL_ENUMS",
    "CLAUDE_THINKING_MIN_TOP_P",
    "FORCED_TOOL_DIRECTIVE",
    "INLINE_MAX_BYTES",
    "MIN_OUTPUT_TOKENS",
    "MISSING_TOOL_RESULT",
    "NON_VISION_IMAGE_PLACEHOLDER",
    "RETIREMENT_MARKERS",
    "SAMPLING_FIELDS",
    "SESSION_LIMIT",
    "SIGNATURE_SENTINEL",
    "ZERO_BUDGET_MIN_THINKING",
    "AntigravitySession",
    "FetchedMedia",
    "MediaFetchError",
    "MediaTooLargeError",
    "ModelRetiredError",
    "RequestEnvelope",
    "antigravity_session",
    "base_family",
    "build_payload",
    "call_arguments",
    "content_parts",
    "declared_output_tokens",
    "demoted_thinking",
    "derive_session_id",
    "function_calling_config",
    "inline_part",
    "is_retired_response",
    "is_retirement_notice",
    "media_from_url",
    "media_part",
    "normalize_effort",
    "output_ceiling",
    "pair_tool_results",
    "raise_if_retired",
    "refused_sampling",
    "refuses_zero_budget",
    "request_envelope",
    "text_of",
    "thinking_floor",
    "tool_result_value",
    "tools_to_declarations",
    "usage_is_zero",
    "well_formed",
    "wire_call_id",
]
