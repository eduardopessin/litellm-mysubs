"""OpenAI Codex wire protocol (Responses API) over a ChatGPT Plus subscription.

Extracted from the original ``sitecustomize.py`` with no behaviour change. Everything in
this module is payload construction — pure and testable without a network. Transport (SSE,
quota, token refresh) stays outside.

Structural difference against the Anthropic bridge: here the request is not a LiteLLM
kwargs dict that gets adjusted, it is a Responses API body built from scratch. Messages in
chat completions format are translated into ``input`` items.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import uuid
from typing import Any, Final, NamedTuple

# The ChatGPT account rejects the 5.4 family with "The 'gpt-5.4' model is not supported
# when using Codex with a ChatGPT account".
#
# Family aliases, not version aliases: "codex"/"gpt-5"/"gpt-6" do not promise a concrete
# version, so resolving them to the served one is honest. `gpt-5.4` and `gpt-5.4-mini` were
# here once, pointing at gpt-5.5 — they name a version this account does not serve, and the
# client was billed and logged against a model that never ran. Whoever asks for them gets
# the upstream refusal.
WIRE_ALIASES: Final[dict[str, str]] = {
    "gpt-6": "gpt-6-astra",
    "gpt6": "gpt-6-astra",
    "gpt-5": "gpt-5.5",
    "gpt5": "gpt-5.5",
    "codex": "gpt-5.5",
}

# With reasoning disabled, GPT-5.6+ Responses still reserve "juice"; omp pins it with a
# developer item at the end of the input (getJuiceValue).
JUICE: Final[dict[str, int]] = {
    "none": 0,
    "minimal": 2,
    "low": 4,
    "medium": 8,
    "high": 48,
    "xhigh": 112,
    "max": 960,
}

#: From this generation on, the juice item is needed to really disable reasoning.
JUICE_MIN_GENERATION: Final = 5.6

#: ``original`` is a valid API value; some Responses backends (GitHub Copilot, for
#: instance) refuse it with 400, and there it degrades to "auto" — the closest fidelity
#: that passes. Always forcing "auto" lost detail on screenshots against hosts that serve
#: it.
IMAGE_DETAILS: Final[tuple[str, ...]] = ("auto", "low", "high", "original")

# Tools hosted by the backend (web search, image generation, shell…) have no `function`:
# they travel with their own spec and were discarded before this existed.
HOSTED_TOOL_TYPES: Final[tuple[str, ...]] = (
    "web_search",
    "web_search_preview",
    "image_generation",
    "code_interpreter",
    "local_shell",
    "computer",
    "computer_use_preview",
    "custom",
    "mcp",
    "file_search",
)

TEXT_PART_TYPES: Final[tuple[str, ...]] = ("text", "input_text", "output_text")


def is_codex_model(model: str) -> bool:
    lowered = str(model).lower()
    return "gpt-" in lowered or "codex" in lowered or lowered.startswith("gpt")


def wire_generation(model: str) -> float:
    """Numeric generation of a model name: ``gpt-5.6-terra`` -> ``5.6``."""
    pieces = str(model).lower().split("-")
    try:
        return float(pieces[1]) if len(pieces) > 1 else 0.0
    except ValueError:
        return 0.0


def resolve_model(model: str, unsupported: dict[str, str] | None = None) -> str:
    """Name that goes on the wire, after aliases and learned refusals."""
    name = str(model).split("/")[-1]
    name = WIRE_ALIASES.get(name.lower(), name)
    if unsupported:
        name = unsupported.get(name.lower(), name)
    return name


# -- token identity ------------------------------------------------------------


def token_claims(token: str) -> dict[str, Any]:
    """Claims of a JWT, without verifying the signature.

    We do not validate because we do not issue: the token comes from the OAuth flow and the
    backend is the one that verifies it. Here we only read the account id and the residency.
    """
    try:
        parts = str(token).split(".")
        if len(parts) != 3:
            return {}
        padding = len(parts[1]) % 4
        padded = parts[1] + ("=" * (4 - padding) if padding else "")
        decoded = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
        claims = json.loads(decoded)
        return claims if isinstance(claims, dict) else {}
    except Exception:
        return {}


def account_id(token: str) -> str | None:
    auth = token_claims(token).get("https://api.openai.com/auth") or {}
    return auth.get("chatgpt_account_id")


# The Codex wire constants live in `pi-catalog`, not in `pi-ai`. The initial audit declared
# them unverifiable because only the latter was at hand — they are on npm, and that is where
# these values come from.

# omp: wire/codex.ts :: ORIGINATOR_CODEX
#: The port emitted "pi". The backend uses this value to identify the client.
ORIGINATOR: Final = "omp"

# omp: wire/codex.ts :: CODEX_CLIENT_VERSION
#: The backend gates model availability against this version, both on `/models` and on
#: `/responses` — `gpt-6-astra` requires >= 0.153.0. An old version silently hides new SKUs
#: from discovery.
CLIENT_VERSION: Final = "0.153.0"

# omp: wire/codex.ts :: OPENAI_HEADER_VALUES
BETA_RESPONSES: Final = "responses=experimental"

# omp: dirs.ts :: USER_AGENT
#: `omp/<version>`, not `codex/<version>`: it is OMP's own user agent, shared by every
#: provider, not a value from the Codex dialect. It was written wrong by analogy with
#: `claude-cli/…` on the Anthropic path, where the CLI *is* the client; here it is not. The
#: constant lives in a third package (`@oh-my-pi/pi-utils`), which neither `pi-ai` nor
#: `pi-catalog` contained.
OMP_VERSION: Final = "18.2.6"
USER_AGENT: Final = f"omp/{OMP_VERSION}"

# omp: providers/openai-codex-responses.ts :: OpenAICodexRequestKind
#: Closed vocabulary: "turn" | "prewarm" | "compaction". The port emitted "chat", which
#: does not belong to the set.
REQUEST_KIND_TURN: Final = "turn"


# omp: wire/codex.ts :: codexRoutingHint
def routing_hint(model: str, service_tier: str | None = None) -> str:
    """Value of ``x-codex-routing-hint``: the requested model and, when present, the tier."""
    return f"model={model};tier={service_tier}" if service_tier else f"model={model}"


def build_headers(
    token: str,
    *,
    window_id: str,
    session_id: str | None = None,
    turn_id: str | None = None,
    turn_state: str | None = None,
    model: str | None = None,
    service_tier: str | None = None,
) -> dict[str, str]:
    """Headers of a request to the Codex backend.

    The transport ids come in as arguments instead of from global state: they are
    per-process, and injecting them is what lets us assert the shape without guessing them.
    """
    claims = token_claims(token)
    auth = claims.get("https://api.openai.com/auth") or {}
    resolved_session = session_id or str(claims.get("session_id") or window_id)

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "accept": "text/event-stream",
        "originator": ORIGINATOR,
        "OpenAI-Beta": BETA_RESPONSES,
        "version": CLIENT_VERSION,
        "User-Agent": USER_AGENT,
        "conversation_id": resolved_session,
        "session_id": resolved_session,
        "session-id": resolved_session,
        "x-client-request-id": resolved_session,
        "x-codex-window-id": window_id,
        # The `installation_id` travels only in the metadata envelope; OMP explicitly
        # deletes it from the headers before sending.
        "x-codex-turn-metadata": json.dumps(
            {
                "session_id": resolved_session,
                "thread_id": resolved_session,
                "turn_id": turn_id or str(uuid.uuid4()),
                "window_id": window_id,
                "request_kind": REQUEST_KIND_TURN,
            }
        ),
    }

    account = auth.get("chatgpt_account_id")
    if account:
        headers["chatgpt-account-id"] = account

    # Routing hint: the backend uses it to pick the model's route. It travels on every
    # ChatGPT-OAuth request; API key traffic never carries it.
    if model:
        headers["x-codex-routing-hint"] = routing_hint(model, service_tier)

    # The backend returns x-codex-turn-state and expects it back on the next turn: it is
    # the session's transport state.
    if turn_state:
        headers["x-codex-turn-state"] = turn_state

    # Enterprise workspaces with pinned residency answer 401 "Workspace is not authorized
    # in this region" to requests from another region. The header only travels when the
    # token carries the claim: personal accounts do not have it.
    residency = auth.get("chatgpt_data_residency") or auth.get("chatgpt_compute_residency")
    if residency and str(residency) != "no_constraint":
        headers["x-openai-internal-codex-residency"] = str(residency)
    return headers


# -- content -------------------------------------------------------------------


def content_to_text(content: object) -> str:
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in TEXT_PART_TYPES
        )
    return str(content) if content is not None else ""


PROMPT_CACHE_KEY_MAX_CHARS: Final = 64


# omp: providers/openai-shared.ts :: getOpenAIPromptCacheKey
def prompt_cache_key(session_id: str | None, *, cache_retention: str | None = None) -> str | None:
    """Prompt cache key, derived from the **session identity**.

    It is not derived from the content: two distinct conversations sharing the system
    prompt and the first message would collide on the same key — across sessions and across
    users — and a conversation whose head was edited would lose the hit for no reason.

    ``cache_retention="none"`` disables the cache; without it there was no way for the
    caller to opt out.
    """
    if cache_retention == "none" or not session_id:
        return None
    if len(session_id) <= PROMPT_CACHE_KEY_MAX_CHARS:
        return session_id
    return f"pc_{_stable_hash(session_id)}"


# omp: providers/openai-shared.ts :: clampResponsesImageDetail
def clamp_image_detail(detail: object, *, supports_detail_original: bool = True) -> str:
    """Normalize ``detail``, degrading ``original`` only where the host refuses it."""
    resolved = str(detail or "auto").lower()
    if resolved not in IMAGE_DETAILS:
        return "auto"
    if resolved == "original" and not supports_detail_original:
        return "auto"
    return resolved


# omp: providers/openai-shared.ts :: convertResponsesInputImage
def image_part(
    part: dict[str, Any], *, supports_detail_original: bool = True
) -> dict[str, str] | None:
    """chat completions ``image_url`` -> Responses ``input_image``.

    An image already uploaded to the backend travels by ``file_id`` and has no ``url``:
    without this branch we returned ``None`` and the image was silently discarded.
    """
    image = part.get("image_url")
    spec: dict[str, Any] = image if isinstance(image, dict) else part
    detail = clamp_image_detail(
        spec.get("detail") or part.get("detail"),
        supports_detail_original=supports_detail_original,
    )
    if file_id := spec.get("file_id"):
        return {"type": "input_image", "detail": detail, "file_id": str(file_id)}
    url = image.get("url") if isinstance(image, dict) else image
    if not url:
        return None
    return {"type": "input_image", "image_url": str(url), "detail": detail}


def file_part(part: dict[str, Any]) -> dict[str, str] | None:
    """chat completions ``file`` -> Responses ``input_file``."""
    nested = part.get("file")
    spec: dict[str, Any] = nested if isinstance(nested, dict) else part
    data = spec.get("file_data") or spec.get("data")
    file_id = spec.get("file_id")
    if not data and not file_id:
        return None
    item: dict[str, str] = {"type": "input_file"}
    if spec.get("filename"):
        item["filename"] = str(spec["filename"])
    if file_id:
        item["file_id"] = str(file_id)
    else:
        item["file_data"] = str(data)
    return item


#: An image or file that fails to convert is discarded, but it never takes the rest of the
#: turn with it.
IMAGE_PART_TYPES: Final[tuple[str, ...]] = ("image_url", "input_image")
FILE_PART_TYPES: Final[tuple[str, ...]] = ("file", "input_file")


def content_to_parts(
    content: object, *, assistant: bool = False, supports_detail_original: bool = True
) -> list[dict[str, str]]:
    """Preserve images and files instead of dropping them.

    Before this, any multimodal request reached the model with the text only — and the
    answer talked about an image it had never seen.
    """
    text_type = "output_text" if assistant else "input_text"
    if not isinstance(content, list):
        text = str(content) if content is not None else ""
        return [{"type": text_type, "text": text}] if text else []

    parts: list[dict[str, str]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        kind = str(part.get("type"))
        if kind in TEXT_PART_TYPES:
            if part.get("text"):
                parts.append({"type": text_type, "text": str(part["text"])})
            continue
        built: dict[str, str] | None = None
        if kind in IMAGE_PART_TYPES:
            built = image_part(part, supports_detail_original=supports_detail_original)
        elif kind in FILE_PART_TYPES:
            built = file_part(part)
        if built:
            parts.append(built)
    return parts


# -- tool calls ----------------------------------------------------------------


# omp: providers/transform-messages.ts :: normalizeResponsesToolCallId
def composite_call_id(call_id: str | None, item_id: str | None) -> str:
    """Join ``(call_id, item_id)`` into a single identifier.

    Responses identifies each tool call by the pair. Joining them makes replay reconstruct
    the exact pair — without that, parallel calls get out of alignment.
    """
    if call_id and item_id and call_id != item_id:
        return f"{call_id}|{item_id}"
    return call_id or item_id or f"call_{uuid.uuid4().hex[:8]}"


#: The backend refuses ids outside this set or above this length.
CALL_ID_MAX_CHARS: Final = 64
_INVALID_CALL_ID_CHARS: Final = re.compile(r"[^a-zA-Z0-9_-]")
_TRAILING_UNDERSCORES: Final = re.compile(r"_+$")
#: Separators: `|` is our composite one, `\n` shows up in ids forwarded from another
#: provider.
_CALL_ID_SEPARATOR: Final = re.compile(r"[\n|]")


def _stable_hash(text: str) -> str:
    """Short deterministic hash, in base36 like OMP's."""
    digest = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    out = ""
    while digest:
        digest, remainder = divmod(digest, 36)
        out = alphabet[remainder] + out
    return out or "0"


# omp: providers/openai-codex/request-transformer.ts :: sanitizeCodexCallId
def split_call_id(value: object) -> str:
    """Call id sanitized for the Codex wire.

    An id coming from another provider frequently carries characters the backend refuses,
    or goes past 64 characters; letting it through raw gives 400. When the id has to be
    altered, a hash is appended so that two different ids do not collapse into the same one.
    """
    raw = str(value or "")
    if not raw:
        return f"call_{_stable_hash('empty')}"

    match = _CALL_ID_SEPARATOR.search(raw)
    if match is None:
        base = raw
    elif match.start() == 0:
        base = raw[1:]
    else:
        base = raw[: match.start()]

    sanitized = _TRAILING_UNDERSCORES.sub("", _INVALID_CALL_ID_CHARS.sub("_", base))
    if 0 < len(sanitized) <= CALL_ID_MAX_CHARS and sanitized == base:
        return sanitized

    digest = _stable_hash(base or raw)
    effective = sanitized or "call"
    prefix_length = max(0, CALL_ID_MAX_CHARS - 1 - len(digest))
    return f"{effective[:prefix_length]}_{digest}"[:CALL_ID_MAX_CHARS]


# omp: providers/openai-codex/request-transformer.ts :: CODEX_ORPHAN_OUTPUT_LIMIT
#: A huge orphan result (the read of a 2 MB file, for instance) blew past the request body
#: limit instead of being cut.
ORPHAN_OUTPUT_LIMIT: Final = 16_000

# omp: providers/openai-codex/request-transformer.ts :: CODEX_INTERRUPTED_TOOL_OUTPUT
INTERRUPTED_TOOL_OUTPUT: Final = (
    "[No tool output recorded: the tool call was interrupted before it produced a result.]"
)


def _orphan_output_text(item: dict[str, Any]) -> str:
    """Text of a result whose call was lost, truncated."""
    output = item.get("output")
    if isinstance(output, str):
        text = output
    else:
        try:
            text = json.dumps(output)
        except (TypeError, ValueError):
            text = str(output if output is not None else "")
    if len(text) > ORPHAN_OUTPUT_LIMIT:
        text = f"{text[:ORPHAN_OUTPUT_LIMIT]}\n...[truncated]"
    return text


#: Literal text from the source (there it is inline in the `computer` branch of
#: `repairToolCallPairs`, with no name of its own). A `computer_call` has no synthesizable
#: output: the missing screenshot cannot be invented, so the call becomes the note the model
#: reads.
INTERRUPTED_COMPUTER_CALL: Final = (
    "[Computer call interrupted before a screenshot was recorded; call_id={call_id}]"
)

#: Call item -> tool type. The pair only closes between items of the **same** type:
#: Responses refuses a ``custom_tool_call_output`` closing a ``function_call``.
_CALL_KINDS: Final[dict[str, str]] = {
    "function_call": "function",
    "custom_tool_call": "custom",
    "computer_call": "computer",
}
_OUTPUT_KINDS: Final[dict[str, str]] = {
    "function_call_output": "function",
    "custom_tool_call_output": "custom",
    "computer_call_output": "computer",
}


# omp: providers/openai-codex/request-transformer.ts :: repairToolCallPairs, toolCallKind
# omp: providers/openai-codex/request-transformer.ts :: toolOutputKind
# omp: providers/openai-codex/request-transformer.ts :: orphanFunctionOutputToMessage
def repair_tool_pairs(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Close loose halves of a tool exchange, indexed by tool **type**.

    Responses rejects with 400 both an output without its call and a call without its
    output. A history truncated by the client (or a turn aborted after the call had been
    emitted) brings exactly that, and repairing is preferable to a 400 over something the
    model interprets. Indexing by ``call_id`` alone paired different types — a
    ``custom_tool_call_output`` "closing" a ``function_call`` gives 400 again.
    """
    call_kinds: dict[str, str] = {}
    output_kinds: dict[str, str] = {}
    for item in items:
        call_id = item.get("call_id")
        if not isinstance(call_id, str):
            continue
        item_type = str(item.get("type"))
        if kind := _CALL_KINDS.get(item_type):
            call_kinds[call_id] = kind
        if kind := _OUTPUT_KINDS.get(item_type):
            output_kinds[call_id] = kind

    repaired: list[dict[str, Any]] = []
    for item in items:
        call_id = item.get("call_id")
        call_id = call_id if isinstance(call_id, str) else None
        item_type = str(item.get("type"))
        call_kind = _CALL_KINDS.get(item_type)
        output_kind = _OUTPUT_KINDS.get(item_type)

        if output_kind and call_id is not None and call_kinds.get(call_id) != output_kind:
            # The tool name comes from the item itself: without it the model does not know
            # what produced the orphan result.
            tool_name = item.get("name") if isinstance(item.get("name"), str) else "tool"
            repaired.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": (
                        f"[Previous {tool_name} result; call_id={call_id}]: "
                        f"{_orphan_output_text(item)}"
                    ),
                }
            )
            continue
        if call_kind and call_id is not None and output_kinds.get(call_id) != call_kind:
            if call_kind == "computer":
                repaired.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": INTERRUPTED_COMPUTER_CALL.format(call_id=call_id),
                    }
                )
                continue
            repaired.append(item)
            repaired.append(
                {
                    "type": (
                        "custom_tool_call_output"
                        if call_kind == "custom"
                        else "function_call_output"
                    ),
                    "call_id": call_id,
                    "output": INTERRUPTED_TOOL_OUTPUT,
                }
            )
            continue
        repaired.append(item)
    return repaired


class CodexInput(NamedTuple):
    """``instructions`` and ``input`` are distinct request fields, not a single one."""

    instructions: str | None
    items: list[dict[str, Any]]


def _last_developer_text(items: list[dict[str, Any]]) -> str | None:
    """Last developer text in the input, scanning from the end backwards."""
    for item in reversed(items):
        if item.get("role") != "developer":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in reversed(content):
            if not isinstance(part, dict) or part.get("type") != "input_text":
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                return text
    return None


# omp: providers/openai-codex-responses.ts :: buildTransformedCodexRequestBody
# omp: providers/openai-codex/request-transformer.ts :: transformRequestBody
# omp: utils.ts :: normalizeSystemPrompts
def messages_to_input(messages: list[Any], *, supports_detail_original: bool = True) -> CodexInput:
    """Translate chat completions messages into ``instructions`` + ``input`` items.

    The **first** system prompt goes into ``instructions``, which the backend treats as a
    cacheable base prompt; sending it as a developer item lost that treatment. The rest do
    not fit there (the field is a string) and travel as developer items at the top of the
    input, before the conversation.
    """
    instructions: str | None = None
    developer_items: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content")

        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": split_call_id(message.get("tool_call_id")),
                    "output": content_to_text(content),
                }
            )
            continue

        if role == "system":
            text = content_to_text(content)
            if not text.strip():
                continue
            if instructions is None:
                instructions = text
            else:
                developer_items.append(
                    {
                        "type": "message",
                        "role": "developer",
                        "content": [{"type": "input_text", "text": text}],
                    }
                )
            continue

        codex_role = role if role in ("user", "assistant", "developer") else "user"
        parts = content_to_parts(
            content,
            assistant=codex_role == "assistant",
            supports_detail_original=supports_detail_original,
        )
        if parts:
            items.append({"type": "message", "role": codex_role, "content": parts})

        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            arguments = function.get("arguments") or "{}"
            items.append(
                {
                    "type": "function_call",
                    "call_id": split_call_id(tool_call.get("id")),
                    "name": function.get("name") or "",
                    "arguments": json.dumps(arguments)
                    if isinstance(arguments, dict)
                    else arguments,
                }
            )

    repaired = repair_tool_pairs([*developer_items, *items])

    # An input with developer items only (a system prompt with no user turn) makes the
    # backend return an empty response: the last instruction is promoted to a `user` turn so
    # that there is something to answer.
    if not any(item.get("role") != "developer" for item in repaired):
        final = _last_developer_text(developer_items) or _last_developer_text(repaired)
        final = final or (instructions if instructions and instructions.strip() else None)
        if final is not None:
            repaired.append(
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": final}],
                }
            )
    return CodexInput(instructions, repaired)


def tools_to_codex_tools(tools: list[Any] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") in HOSTED_TOOL_TYPES:
            converted.append(dict(tool))
            continue
        nested = tool.get("function")
        function: dict[str, Any] = nested if isinstance(nested, dict) else tool
        if not function.get("name"):
            continue
        converted.append(
            {
                "type": "function",
                "name": function["name"],
                "description": function.get("description") or "",
                "parameters": function.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return converted or None


def tool_choice(choice: object) -> object:
    """Responses uses ``{"type": "function", "name": …}``, without the ``function`` level."""
    if not isinstance(choice, dict):
        return choice
    function = choice.get("function")
    if choice.get("type") == "function" and isinstance(function, dict) and function.get("name"):
        return {"type": "function", "name": function["name"]}
    return choice


# -- request body --------------------------------------------------------------


def normalize_effort(value: object) -> tuple[str | None, str | None]:
    """Return ``(effort, summary)``; see the same function in ``wire.anthropic``."""
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


# omp: providers/openai-responses.ts :: getJuiceValue
def juice_for(effort: str | None) -> int:
    """Reasoning budget reserved when thinking is disabled.

    The value is that of the effort **requested by the client**, not zero: disabling
    reasoning does not mean the model should be left with no budget at all. An unknown
    effort falls back to the ``medium`` default.
    """
    return JUICE.get(str(effort or "medium").strip().lower(), JUICE["medium"])


def build_request_body(
    model: str,
    messages: list[Any],
    tools: list[Any] | None = None,
    extra: dict[str, Any] | None = None,
    unsupported: dict[str, str] | None = None,
    session_id: str | None = None,
    supports_detail_original: bool = True,
) -> dict[str, Any]:
    """Body of a request to the Responses API."""
    req_model = resolve_model(model, unsupported)
    extra = extra or {}
    instructions, items = messages_to_input(
        messages, supports_detail_original=supports_detail_original
    )
    body: dict[str, Any] = {
        "model": req_model,
        "store": False,
        "stream": True,
        "input": items,
        # Without this the backend does not return the encrypted reasoning, and on a
        # stateless history (`store: false`) the model starts reasoning over on every turn.
        "include": ["reasoning.encrypted_content"],
    }
    if instructions is not None:
        body["instructions"] = instructions

    cache_key = prompt_cache_key(session_id, cache_retention=extra.get("cache_retention"))
    if cache_key:
        body["prompt_cache_key"] = cache_key
    if codex_tools := tools_to_codex_tools(tools):
        body["tools"] = codex_tools

    choice = tool_choice(extra.get("tool_choice"))
    if choice is not None:
        body["tool_choice"] = choice

    # The backend only returns reasoning text when the request carries the `reasoning`
    # object (verified: without it, zero response.reasoning_summary_text.delta events). omp
    # always sends an effort, so the default here is "medium" instead of omitting.
    effort, summary = normalize_effort(extra.get("reasoning_effort"))
    summary = summary if summary in ("auto", "detailed", "concise") else "auto"

    if effort == "none":
        # Disabling reasoning does not make the item unnecessary: recent generations still
        # reserve juice, and it is the item that pins it to the requested value.
        if wire_generation(req_model) >= JUICE_MIN_GENERATION:
            body["input"] = [
                *body["input"],
                {
                    "type": "message",
                    "role": "developer",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                f"# Juice: "
                                f"{juice_for(extra.get('juice_effort') or effort)} !important"
                            ),
                        }
                    ],
                },
            ]
    else:
        body["reasoning"] = {"effort": effort or "medium", "summary": summary}

    if extra.get("service_tier") is not None:
        body["service_tier"] = extra["service_tier"]
    return body
