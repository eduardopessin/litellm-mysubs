"""OpenAI Codex wire protocol (Responses API) over a ChatGPT Plus subscription.

Extracted from the original ``sitecustomize.py``. Everything in this module is payload
construction — testable without a network. Transport (SSE, quota, token refresh) stays
outside.

Structural difference against the Anthropic bridge: here the request is not a LiteLLM
kwargs dict that gets adjusted, it is a Responses API body built from scratch. Messages in
chat completions format are translated into ``input`` items.

The omp analogue of this proxy is its auth gateway: ``providers/openai-chat-server.ts``
parses a chat completions request into omp's canonical context, and the Codex provider
(``providers/openai-codex-responses.ts`` + ``openai-codex/request-transformer.ts``) turns
that into the wire request. Each function below names the omp symbol whose behaviour it
reproduces; the two hops are folded into one here because LiteLLM already hands us the
chat completions shape.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import time
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final, NamedTuple

from .openai_schema import codex_tool_parameters

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

#: ``original`` is a valid API value; some Responses backends (GitHub Copilot, for
#: instance) refuse it with 400, and there it degrades to "auto" — the closest fidelity
#: that passes. Always forcing "auto" lost detail on screenshots against hosts that serve
#: it. The closed set is ours: omp's ``ImageContent.detail`` is typed, a LiteLLM client's is
#: free text, and an unknown value reaching the wire is a 400.
IMAGE_DETAILS: Final[tuple[str, ...]] = ("auto", "low", "high", "original")

# Tools hosted by the backend (web search, image generation, shell…) have no `function`:
# they travel with their own spec and were discarded before this existed. omp has no such
# passthrough — its canonical `Tool` cannot express a hosted tool, and both of its servers
# drop them (`openai-chat-server.ts :: buildTools`, `openai-responses-server.ts ::
# buildTools`). Kept because a LiteLLM client can name them and the Codex backend serves
# them; the list is ours, a superset of omp's hosted `tool_choice` vocabulary
# (`openai-responses-server-schema.ts :: hostedToolType`).
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

# omp: providers/openai-chat-server.ts :: isReasoningEffort
#: Efforts a client may request. Anything else leaves ``reasoning`` off the body, as omp's
#: chat server does, instead of forwarding a value the backend answers with 400.
REASONING_EFFORTS: Final[tuple[str, ...]] = ("minimal", "low", "medium", "high", "xhigh", "max")

# omp: providers/openai-codex/request-transformer.ts :: ReasoningConfig
REASONING_SUMMARIES: Final[tuple[str, ...]] = ("auto", "concise", "detailed")

# omp: providers/openai-chat-server.ts :: isServiceTier
SERVICE_TIERS: Final[tuple[str, ...]] = ("auto", "default", "flex", "scale", "priority")


def is_codex_model(model: str) -> bool:
    lowered = str(model).lower()
    return "gpt-" in lowered or "codex" in lowered or lowered.startswith("gpt")


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
#: from discovery. Measured against a `plus` account on 2026-09-25, asking
#: `/backend-api/codex/models?client_version=<v>` and counting what came back:
#:
#:     0.153.0 -> 7 models (no gpt-6-luna, no gpt-6-sol)
#:     0.155.1 -> 9 models (both present)
#:
#: The gate is on the catalog, not only on inference: at 0.153.0 the two names are absent
#: from the listing, so no amount of probing finds them. Reported in #2.
#:
#: omp 18.4.4 raised it to 0.159.0: the gate ignores a model's own
#: `minimal_client_version` — GPT-6.1 Sol declares 0.153.0, yet `/models` omits it at
#: 0.155.1 and lists it at 0.159.0 (pi-catalog `wire/codex.ts`).
# omp= wire/codex.ts :: CODEX_CLIENT_VERSION = "0.159.0"
CLIENT_VERSION: Final = "0.159.0"

# omp: wire/codex.ts :: OPENAI_HEADER_VALUES
BETA_RESPONSES: Final = "responses=experimental"

# omp: dirs.ts :: USER_AGENT
#: `omp/<version>`, not `codex/<version>`: it is OMP's own user agent, shared by every
#: provider, not a value from the Codex dialect. It was written wrong by analogy with
#: `claude-cli/…` on the Anthropic path, where the CLI *is* the client; here it is not. The
#: constant lives in a third package (`@oh-my-pi/pi-utils`), which neither `pi-ai` nor
#: `pi-catalog` contained.
OMP_VERSION: Final = "18.4.4"
USER_AGENT: Final = f"omp/{OMP_VERSION}"

# omp: providers/openai-codex-responses.ts :: OpenAICodexRequestKind
#: Closed vocabulary: "turn" | "prewarm" | "compaction". The port emitted "chat", which
#: does not belong to the set.
REQUEST_KIND_TURN: Final = "turn"


# -- session identity ------------------------------------------------------------
#
# omp scopes every Codex identity — prompt cache key, `conversation_id`/`session_id`
# headers, and the thread/window/turn ids below — to one *session*. Its agent has a session
# object; a proxy request does not, so omp's own gateway resolves one per request from what
# the client sent, and derives a stable one from the conversation when the client sent
# nothing. That derivation is ported here as is, because it is the one omp applies to
# exactly our situation: a chat completions request arriving at a proxy.

# omp: auth-gateway/http.ts :: CACHE_KEY_HEADERS
SESSION_KEY_HEADERS: Final[tuple[str, ...]] = (
    "x-prompt-cache-key",
    "session_id",
    "conversation_id",
    "x-session-id",
    "x-conversation-id",
)

#: Keys omp reads from the body's ``metadata`` bag, in order.
_METADATA_SESSION_FIELDS: Final[tuple[str, ...]] = (
    "prompt_cache_key",
    "session_id",
    "conversation_id",
)


# omp: auth-gateway/http.ts :: readBodyCacheKey
def _body_session_key(extra: Mapping[str, Any]) -> str | None:
    """The body's own key: ``prompt_cache_key``, then the ``metadata`` bag.

    LiteLLM keeps the client's ``metadata`` under ``metadata`` on chat routes and under
    ``litellm_metadata`` on the Messages route; both are read, in that order.
    """
    direct = extra.get("prompt_cache_key")
    if isinstance(direct, str) and direct:
        return direct
    for bag_name in ("metadata", "litellm_metadata"):
        bag = extra.get(bag_name)
        if not isinstance(bag, Mapping):
            continue
        for name in _METADATA_SESSION_FIELDS:
            value = bag.get(name)
            if isinstance(value, str) and value:
                return value
    return None


# omp: auth-gateway/http.ts :: resolvePromptCacheKey
def _client_session_key(extra: Mapping[str, Any]) -> str | None:
    """The session the client named, by omp's precedence, then LiteLLM's own.

    Headers come from ``proxy_server_request``, the copy LiteLLM's proxy keeps of the
    inbound request. ``litellm_session_id`` goes last: it is LiteLLM's reading of the same
    client intent (``x-litellm-session-id``, any ``x-*-session-id``, Anthropic
    ``metadata.user_id``, W3C baggage), and only fires where omp's sources found nothing.
    ``user`` is deliberately not a source: it names a person, not a conversation, and every
    chat of one user would share a window and a thread.
    """
    if key := _body_session_key(extra):
        return key
    request = extra.get("proxy_server_request")
    headers = request.get("headers") if isinstance(request, Mapping) else None
    if isinstance(headers, Mapping):
        lowered = {str(k).lower(): v for k, v in headers.items()}
        for name in SESSION_KEY_HEADERS:
            value = lowered.get(name)
            if isinstance(value, str) and value:
                return value
    session = extra.get("litellm_session_id")
    return session if isinstance(session, str) and session else None


# omp: utils/deterministic-id.ts :: deterministicUuid
def deterministic_uuid(seed: str) -> str:
    """The first 128 bits of the seed's SHA-256, laid out as a UUID."""
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def _compact_json(value: object) -> str:
    """``JSON.stringify`` output: no spaces after separators, non-ASCII verbatim."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


# omp: auth-gateway/server.ts :: deriveSessionId
def _derive_session_id(model: str, messages: list[Any], tools: list[Any] | None) -> str:
    """A key that stays put across the turns of one conversation and differs between two.

    Built from what a client re-sends unchanged on every turn: the model, the system
    prompt, the tools, and the first message — the conversation's seed. Two chats sharing
    the system prompt still part ways on their first message.
    """
    parts = [str(model)]
    system = _system_prompt(messages)
    if system:
        parts.append(system)
    if tools:
        parts.append(_compact_json(tools))
    first = next(
        (m for m in messages if isinstance(m, Mapping) and m.get("role") != "system"), None
    )
    if first is not None:
        parts.append(_compact_json({"role": first.get("role"), "content": first.get("content")}))
    return deterministic_uuid("\x00".join(parts))


# omp: auth-gateway/server.ts :: buildStreamOptions
# omp: auth-gateway/dispatch.ts :: normalizeClientSessionKey
def session_key(
    model: str, messages: list[Any], tools: list[Any] | None, extra: Mapping[str, Any]
) -> str:
    """The session a Codex request belongs to.

    The client's own key wins; a blank one counts as none (honouring it would put every
    caller that sends an empty key into one shared session). Without one, the key is
    derived from the conversation.
    """
    client = _client_session_key(extra)
    if client is not None and client.strip():
        return client
    return _derive_session_id(model, messages, tools)


PROMPT_CACHE_KEY_MAX_CHARS: Final = 64


# omp: providers/openai-shared.ts :: normalizeOpenAIPromptCacheKey, normalizeOpenAIStableId
def normalize_session_id(session_id: str | None) -> str | None:
    """A session id within the 64 characters the backend accepts.

    omp hashes with ``Bun.hash`` (wyhash); this uses SHA-256 truncated to 64 bits, in the
    same base36. The value is opaque to the backend, so only stability matters.
    """
    if not session_id:
        return None
    if len(session_id) <= PROMPT_CACHE_KEY_MAX_CHARS:
        return session_id
    return f"pc_{_stable_hash(session_id)}"


# omp: providers/openai-shared.ts :: getOpenAIPromptCacheKey
def prompt_cache_key(session_id: str | None, *, cache_retention: str | None = None) -> str | None:
    """Prompt cache key, derived from the **session identity**.

    Not from the content of the turn: a conversation whose tail changes every turn must
    keep its key, or the cache never hits. ``cache_retention="none"`` disables it.
    """
    if cache_retention == "none":
        return None
    return normalize_session_id(session_id)


# -- request identity ------------------------------------------------------------

#: One per process. omp persists its install id to ``~/.omp/install-id`` (pi-utils
#: ``getInstallId``); this module builds payloads and owns no disk state, and the backend
#: only reads the value as telemetry inside the turn metadata.
INSTALLATION_ID: Final = str(uuid.uuid4())

# omp: wire/codex.ts :: OPENAI_HEADERS
HEADER_INSTALLATION_ID: Final = "x-codex-installation-id"
HEADER_WINDOW_ID: Final = "x-codex-window-id"
HEADER_TURN_METADATA: Final = "x-codex-turn-metadata"

# omp: providers/openai-codex-responses.ts :: X_CODEX_TURN_STATE_HEADER
# omp= X_CODEX_TURN_STATE_HEADER = "x-codex-turn-state"
HEADER_TURN_STATE: Final = "x-codex-turn-state"

# omp: providers/openai-codex-responses.ts :: X_MODELS_ETAG_HEADER
# omp= X_MODELS_ETAG_HEADER = "x-models-etag"
HEADER_MODELS_ETAG: Final = "x-models-etag"

# omp: wire/codex.ts :: CODEX_BASE_URL
# omp= CODEX_BASE_URL = "https://chatgpt.com/backend-api"
BASE_URL: Final = "https://chatgpt.com/backend-api"

#: omp keeps one identity per agent session and drops it with the session. A proxy never
#: sees a session end, so the table is bounded: the least recently used conversation loses
#: its thread and window ids — the cost is one cache miss for a conversation idle long
#: enough to fall off the end.
METADATA_SESSION_LIMIT: Final = 4096


# omp: providers/openai-codex-responses.ts :: CodexTurnStateCell
@dataclass(slots=True)
class TurnState:
    """The backend's sticky-routing token for the turn in progress.

    The backend answers a turn with ``x-codex-turn-state`` and expects it back on every
    request that continues the same turn — the tool-result follow-ups. The first value a
    turn receives is the one kept.
    """

    value: str | None = None


# omp: providers/openai-codex-responses.ts :: CodexMetadataSessionState
@dataclass(slots=True)
class MetadataSession:
    """Thread and window ids of one session, the turn it is on, and what the backend
    handed back for it.

    ``turn_states`` and ``models_etags`` are keyed by `compatibility_key`. omp holds the
    etag on its per-key transport session, which is scoped by the same session id, so
    keeping it here scopes it identically.
    """

    session_id: str
    thread_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    window_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    turn_id: str | None = None
    turn_started_at_unix_ms: int | None = None
    turn_states: dict[str, TurnState] = field(default_factory=dict)
    models_etags: dict[str, str] = field(default_factory=dict)


_metadata_sessions: OrderedDict[str, MetadataSession] = OrderedDict()


# omp: providers/openai-codex-responses.ts :: getOrCreateCodexMetadataSessionState
# omp: providers/openai-codex-responses.ts :: createCodexMetadataSessionState
def metadata_session(session_id: str) -> MetadataSession:
    session = _metadata_sessions.get(session_id)
    if session is None:
        session = _metadata_sessions[session_id] = MetadataSession(session_id)
        while len(_metadata_sessions) > METADATA_SESSION_LIMIT:
            _metadata_sessions.popitem(last=False)
    else:
        _metadata_sessions.move_to_end(session_id)
    return session


# omp: providers/openai-codex-responses.ts :: getCodexWebSocketSessionKey
def compatibility_key(
    session_id: str | None, model: str, token: str, base_url: str = BASE_URL
) -> str | None:
    """The credential + backend + model + session a turn-state token belongs to.

    A token minted for one account or model must not ride another's request. Without a
    session there is nothing to continue, and no key. Responses Lite is never requested
    here, so the key carries no ``:lite`` suffix.
    """
    if not session_id:
        return None
    account = account_id(token)
    credential = f"account:{account}" if account else f"token:{_stable_hash(token)}"
    return f"{credential}:{base_url}:{model}:{session_id}"


# omp: providers/openai-codex-responses.ts :: getOrCreateCodexTurnState
def turn_state(session: MetadataSession, key: str | None) -> TurnState:
    """The session's cell for ``key``; a throwaway one when there is no key."""
    if not key:
        return TurnState()
    cell = session.turn_states.get(key)
    if cell is None:
        cell = session.turn_states[key] = TurnState()
    return cell


# omp: providers/openai-codex-responses.ts :: clearCodexTurnStatesForNewTurn
def clear_turn_states_for_new_turn(session: MetadataSession, start_new_turn: bool) -> None:
    """A fresh logical turn drops every sticky-routing token, as codex-rs's per-turn
    ``OnceLock`` does."""
    if start_new_turn:
        session.turn_states.clear()


# omp: providers/openai-codex-responses.ts :: updateCodexSessionMetadataFromHeaders
def update_session_from_headers(
    session: MetadataSession, key: str | None, cell: TurnState, headers: Mapping[str, str]
) -> None:
    """Keep what a successful response handed back: the turn's first ``x-codex-turn-state``
    and the latest ``x-models-etag``. ``headers`` must be case-insensitive (httpx's are)."""
    turn = headers.get(HEADER_TURN_STATE)
    if cell.value is None and turn:
        cell.value = turn
    etag = headers.get(HEADER_MODELS_ETAG)
    if key and etag:
        session.models_etags[key] = etag


class RequestMetadata(NamedTuple):
    """One request's identity, shared by the body's ``client_metadata`` and the headers."""

    installation_id: str
    session_id: str
    thread_id: str
    window_id: str
    turn_id: str
    turn_metadata_json: str
    client_metadata: dict[str, str]


# omp: providers/openai-codex-responses.ts :: toAsciiJsonString
def _ascii_json(value: object) -> str:
    """Compact JSON with everything past ASCII escaped: it travels in a header."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


# omp: providers/openai-codex-responses.ts :: createCodexRequestMetadata
def request_metadata(
    session: MetadataSession,
    *,
    start_new_turn: bool,
    turn_started_at_unix_ms: int | None = None,
    request_kind: str = REQUEST_KIND_TURN,
) -> RequestMetadata:
    """The turn id changes only when a new turn starts; a continuation keeps it."""
    if start_new_turn or not session.turn_id:
        session.turn_id = str(uuid.uuid4())
        session.turn_started_at_unix_ms = turn_started_at_unix_ms
    turn_metadata: dict[str, Any] = {
        "installation_id": INSTALLATION_ID,
        "session_id": session.session_id,
        "thread_id": session.thread_id,
        "turn_id": session.turn_id,
        "window_id": session.window_id,
        "request_kind": request_kind,
    }
    if session.turn_started_at_unix_ms is not None:
        turn_metadata["turn_started_at_unix_ms"] = session.turn_started_at_unix_ms
    turn_metadata_json = _ascii_json(turn_metadata)
    return RequestMetadata(
        installation_id=INSTALLATION_ID,
        session_id=session.session_id,
        thread_id=session.thread_id,
        window_id=session.window_id,
        turn_id=session.turn_id,
        turn_metadata_json=turn_metadata_json,
        client_metadata={
            HEADER_INSTALLATION_ID: INSTALLATION_ID,
            "session_id": session.session_id,
            "thread_id": session.thread_id,
            HEADER_WINDOW_ID: session.window_id,
            "turn_id": session.turn_id,
            HEADER_TURN_METADATA: turn_metadata_json,
        },
    )


def _turn_role(message: object) -> str | None:
    """Role of a message for turn accounting; a raw Responses item has only a ``type``."""
    if not isinstance(message, Mapping):
        return None
    if role := message.get("role"):
        return str(role)
    item_type = str(message.get("type"))
    if item_type in _OUTPUT_KINDS:
        return "tool"
    if item_type in _CALL_KINDS or item_type == "reasoning":
        return "assistant"
    return None


# omp: providers/openai-codex-responses.ts :: isCodexWithinTurnContinuation
def is_within_turn_continuation(messages: list[Any]) -> bool:
    """True when everything after the last assistant message is tool results.

    System messages are skipped too: omp holds them apart from the message list
    (``Context.systemPrompt``), so they never end a turn there.
    """
    for message in reversed(messages):
        role = _turn_role(message)
        if role in ("tool", "system"):
            continue
        return role == "assistant"
    return False


class RequestContext(NamedTuple):
    """What one request carries of its session, and where the response's state goes."""

    metadata: RequestMetadata
    session: MetadataSession
    #: `compatibility_key` of this request; ``None`` without a session.
    key: str | None
    turn_state: TurnState

    @property
    def models_etag(self) -> str | None:
        return self.session.models_etags.get(self.key) if self.key else None

    def on_response(self, _status: int, headers: Mapping[str, str]) -> None:
        """`RequestSpec.on_response`: keep what the successful response handed back."""
        update_session_from_headers(self.session, self.key, self.turn_state, headers)


# omp: providers/openai-codex-responses.ts :: createCodexRequestContext
# omp: providers/openai-codex-responses.ts :: resolveCodexStartNewTurn, getCodexTurnStartedAtUnixMs
def request_context(
    session_id: str | None, messages: list[Any], *, model: str, token: str
) -> RequestContext:
    """Identity and turn state of a ``turn`` request in ``session_id``, for wire ``model``.

    A request without a session gets a throwaway identity, as omp does when it has no
    session id (``crypto.randomUUID()``), and a throwaway turn-state cell. Chat completions
    messages carry no timestamp, so the turn starts now — omp's own fallback when the last
    user message has none. A new turn drops the tokens the previous one collected.
    """
    transport_session = normalize_session_id(session_id)
    session = (
        metadata_session(transport_session)
        if transport_session
        else MetadataSession(str(uuid.uuid4()))
    )
    start_new_turn = not is_within_turn_continuation(messages)
    clear_turn_states_for_new_turn(session, start_new_turn)
    key = compatibility_key(transport_session, model, token)
    cell = turn_state(session, key)
    metadata = request_metadata(
        session,
        start_new_turn=start_new_turn,
        turn_started_at_unix_ms=int(time.time() * 1000),
    )
    return RequestContext(metadata=metadata, session=session, key=key, turn_state=cell)


# omp: wire/codex.ts :: codexRoutingHint
def routing_hint(model: str, service_tier: str | None = None) -> str:
    """Value of ``x-codex-routing-hint``: the requested model and, when present, the tier."""
    return f"model={model};tier={service_tier}" if service_tier else f"model={model}"


# omp: wire/codex.ts :: getCodexResidency
def residency(token: str) -> str | None:
    """The workspace's pinned region: ``chatgpt_data_residency``, else the compute one."""
    auth = token_claims(token).get("https://api.openai.com/auth") or {}
    for claim in (auth.get("chatgpt_data_residency"), auth.get("chatgpt_compute_residency")):
        if isinstance(claim, str) and claim.strip():
            return claim.strip()
    return None


# omp: providers/openai-codex-responses.ts :: createCodexHeaders, applyCodexCompatibilityHeaders
# omp: wire/codex.ts :: applyCodexResidencyHeader
def build_headers(
    token: str,
    *,
    session_id: str | None = None,
    metadata: RequestMetadata | None = None,
    window_id: str | None = None,
    turn_state: str | None = None,
    models_etag: str | None = None,
    model: str | None = None,
    service_tier: str | None = None,
) -> dict[str, str]:
    """Headers of a request to the Codex backend.

    ``metadata`` is the request's identity (see ``request_context``); the body carries the
    same one in ``client_metadata``. Without it — the discovery probes — a throwaway
    identity is built, on ``window_id`` when the caller pins one. ``model`` and
    ``service_tier`` are the body's, so the routing hint names what is actually requested.
    ``turn_state`` and ``models_etag`` are what the session's last successful response
    handed back (`RequestContext`).
    """
    transport_session = normalize_session_id(session_id)
    if metadata is None:
        session = MetadataSession(transport_session or str(uuid.uuid4()))
        if window_id:
            session.window_id = window_id
        metadata = request_metadata(
            session, start_new_turn=True, turn_started_at_unix_ms=int(time.time() * 1000)
        )

    headers = {"Authorization": f"Bearer {token}"}
    if account := account_id(token):
        headers["chatgpt-account-id"] = account
    # Routing hint: the backend uses it to pick the model's route. It travels on every
    # ChatGPT-OAuth request; API key traffic never carries it.
    if model:
        headers["x-codex-routing-hint"] = routing_hint(model, service_tier)
    # Enterprise workspaces with pinned residency answer 401 "Workspace is not authorized
    # in this region" to requests from another region. The token carries the claim.
    if region := residency(token):
        headers["x-openai-internal-codex-residency"] = region
    headers["OpenAI-Beta"] = BETA_RESPONSES
    headers["originator"] = ORIGINATOR
    headers["version"] = CLIENT_VERSION
    headers["User-Agent"] = USER_AGENT
    if transport_session:
        headers["conversation_id"] = transport_session
        headers["session_id"] = transport_session
        headers["x-client-request-id"] = transport_session
    # The installation id travels only inside the turn metadata; omp deletes the header.
    headers["session-id"] = metadata.session_id
    headers["thread-id"] = metadata.thread_id
    headers[HEADER_WINDOW_ID] = metadata.window_id
    headers[HEADER_TURN_METADATA] = metadata.turn_metadata_json
    # The backend returns x-codex-turn-state and expects it back on every request that
    # continues the turn; x-models-etag names the catalog the session last saw.
    if turn_state:
        headers[HEADER_TURN_STATE] = turn_state
    if models_etag:
        headers[HEADER_MODELS_ETAG] = models_etag
    headers["accept"] = "text/event-stream"
    headers["Content-Type"] = "application/json"
    return headers


# -- content -------------------------------------------------------------------


# omp: providers/openai-chat-server.ts :: stringifyContent
def content_to_text(content: object) -> str:
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in TEXT_PART_TYPES
        )
    return str(content) if content is not None else ""


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
    return {"type": "input_image", "detail": detail, "image_url": str(url)}


def file_part(part: dict[str, Any]) -> dict[str, str] | None:
    """chat completions ``file`` -> Responses ``input_file``.

    omp drops files here (its canonical content has no file block,
    ``openai-chat-server.ts :: parseUserLikeContent``); a document the user attached is not
    something to lose silently, and the backend accepts ``input_file``.
    """
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


def _media_parts(content: object, *, supports_detail_original: bool) -> list[dict[str, str]]:
    """The images and files of a content list, in order."""
    if not isinstance(content, list):
        return []
    parts: list[dict[str, str]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        kind = str(part.get("type"))
        built: dict[str, str] | None = None
        if kind in IMAGE_PART_TYPES:
            built = image_part(part, supports_detail_original=supports_detail_original)
        elif kind in FILE_PART_TYPES:
            built = file_part(part)
        if built:
            parts.append(built)
    return parts


# omp: providers/openai-codex-responses.ts :: normalizeInputMessageContent
# omp: providers/openai-shared.ts :: convertResponsesInputContent
# omp: providers/vision-guard.ts :: partitionVisionContent
def content_to_parts(
    content: object, *, supports_detail_original: bool = True
) -> list[dict[str, str]]:
    """User or developer content as Responses input parts: text first, then media.

    Blank text is dropped — a whitespace-only part or message says nothing. The order is
    omp's: its converter partitions text blocks ahead of images.
    """
    if not isinstance(content, list):
        text = str(content) if content is not None else ""
        return [{"type": "input_text", "text": text}] if text.strip() else []
    texts = [
        {"type": "input_text", "text": part["text"]}
        for part in content
        if isinstance(part, dict)
        and part.get("type") in TEXT_PART_TYPES
        and isinstance(part.get("text"), str)
        and part["text"].strip()
    ]
    return [*texts, *_media_parts(content, supports_detail_original=supports_detail_original)]


# -- tool calls ----------------------------------------------------------------


# omp: utils.ts :: normalizeResponsesToolCallId
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


# omp: providers/openai-chat-server.ts :: parseRequest
# omp: utils.ts :: normalizeSystemPrompts
def _system_prompt(messages: list[Any]) -> str | None:
    """Every system message, joined: omp's chat server folds them into one prompt."""
    parts = [
        text
        for message in messages
        if isinstance(message, Mapping) and message.get("role") == "system"
        if (text := content_to_text(message.get("content")))
    ]
    joined = "\n\n".join(parts)
    return joined if joined.strip() else None


# omp: providers/openai-chat-server.ts :: buildAssistantMessage
def _tool_call_arguments(arguments: object) -> str:
    """Arguments as omp replays them: parsed, then serialized compactly.

    What does not parse into an object travels under ``__raw`` rather than as a broken
    string the backend would reject.
    """
    if isinstance(arguments, dict):
        return _compact_json(arguments)
    raw = arguments if isinstance(arguments, str) else ""
    if not raw:
        return "{}"
    try:
        parsed = json.loads(raw)
    except ValueError:
        return _compact_json({"__raw": raw})
    return _compact_json(parsed if isinstance(parsed, dict) else {"__raw": raw})


def _is_malformed_tool_call(tool_call: Mapping[str, Any]) -> bool:
    function = tool_call.get("function")
    name = function.get("name") if isinstance(function, Mapping) else None
    call_id = tool_call.get("id")
    return not (isinstance(call_id, str) and call_id.strip()) or not (
        isinstance(name, str) and name.strip()
    )


# omp: providers/openai-codex-responses.ts :: convertMessages
# omp: providers/openai-shared.ts :: convertResponsesAssistantMessage
def _assistant_items(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    """An assistant turn: its text as one completed message, then its function calls."""
    items: list[dict[str, Any]] = []
    text = content_to_text(message.get("content"))
    if text:
        items.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
                "status": "completed",
            }
        )
    for tool_call in message.get("tool_calls") or []:
        if not isinstance(tool_call, Mapping) or tool_call.get("type") not in (None, "function"):
            continue
        if _is_malformed_tool_call(tool_call):
            continue
        function = tool_call["function"]
        items.append(
            {
                "type": "function_call",
                "call_id": split_call_id(tool_call["id"]),
                "name": function["name"],
                "arguments": _tool_call_arguments(function.get("arguments")),
            }
        )
    return items


# omp: providers/openai-chat-server.ts :: pushToolResultMessages
# omp: providers/openai-shared.ts :: encodeResponsesToolResultOutput
# omp: providers/openai-shared.ts :: appendResponsesToolResultMessages
def _tool_result_items(
    message: Mapping[str, Any], *, supports_detail_original: bool
) -> list[dict[str, Any]]:
    """A tool result, with its images hoisted into a user message right after it.

    The text parts are joined by newlines. Images ride in a follow-up ``user`` item so they
    still reach the model.
    """
    content = message.get("content")
    if isinstance(content, list):
        output = "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in TEXT_PART_TYPES
        )
    else:
        output = str(content) if content is not None else ""
    items: list[dict[str, Any]] = [
        {
            "type": "function_call_output",
            "call_id": split_call_id(message.get("tool_call_id")),
            "output": output,
        }
    ]
    if media := _media_parts(content, supports_detail_original=supports_detail_original):
        items.append({"role": "user", "content": media})
    return items


def _final_instruction(items: list[dict[str, Any]], instructions: str | None) -> str | None:
    """Last developer text in the input, else the instructions."""
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
    return instructions if instructions and instructions.strip() else None


# omp: providers/openai-codex-responses.ts :: convertMessages
# omp: providers/openai-codex/request-transformer.ts :: transformRequestBody
# omp: providers/transform-messages.ts :: sanitizeMalformedToolCalls
def messages_to_input(messages: list[Any], *, supports_detail_original: bool = True) -> CodexInput:
    """Translate chat completions messages into ``instructions`` + ``input`` items.

    The system messages become ``instructions``, which the backend treats as a cacheable
    base prompt. User and developer turns are plain ``{role, content}`` items, as omp
    sends them. A tool call with a blank id or name is dropped together with its result:
    replaying it is a 400 on every provider.
    """
    items: list[dict[str, Any]] = []
    #: Per tool-call id, whether each occurrence since the last assistant turn was dropped.
    dropped: dict[str, list[bool]] = {}

    for message in messages:
        if not isinstance(message, Mapping):
            continue
        role = message.get("role", "user")
        if role == "system":
            continue
        if role == "assistant":
            dropped = {}
            for tool_call in message.get("tool_calls") or []:
                if isinstance(tool_call, Mapping):
                    dropped.setdefault(str(tool_call.get("id")), []).append(
                        _is_malformed_tool_call(tool_call)
                    )
            items.extend(_assistant_items(message))
            continue
        if role == "tool":
            queue = dropped.get(str(message.get("tool_call_id")))
            if queue and queue.pop(0):
                continue
            items.extend(
                _tool_result_items(message, supports_detail_original=supports_detail_original)
            )
            continue
        codex_role = role if role in ("user", "developer") else "user"
        dropped = {}
        if parts := content_to_parts(
            message.get("content"), supports_detail_original=supports_detail_original
        ):
            items.append({"role": codex_role, "content": parts})

    instructions = _system_prompt(messages)
    repaired = repair_tool_pairs(items)

    # An input with developer items only (a system prompt with no user turn) makes the
    # backend return an empty response: the last instruction is promoted to a `user` turn so
    # that there is something to answer.
    final = _final_instruction(repaired, instructions)
    if final is not None and all(item.get("role") == "developer" for item in repaired):
        repaired.append(
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": final}],
            }
        )
    return CodexInput(instructions, repaired)


# omp: providers/openai-chat-server.ts :: buildTools
# omp: providers/openai-codex-responses.ts :: convertOpenAICodexResponsesTools
def tools_to_codex_tools(tools: list[Any] | None) -> list[dict[str, Any]] | None:
    """Function tools as the Codex backend takes them; hosted tools with their own spec.

    ``parameters`` go through omp's normalization (`openai_schema.codex_tool_parameters`):
    the backend rejects ``oneOf`` and an object node without ``properties`` even outside
    strict mode. A tool with no parameters, or ``{}``, keeps sending the empty object
    schema this package has always sent. omp's gateway would send ``{}``, which its
    normalization turns into ``parameters: true``. Measured on the live Codex backend on
    2026-09-30: ``true`` answers 400 ``invalid_type`` ("Invalid type for
    'tools[0].parameters': expected an object, but got a boolean instead"), while the
    empty object schema answers 200 — so the object schema stays (docs/OMP.md).
    """
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
        parameters = function.get("parameters") or {"type": "object", "properties": {}}
        converted.append(
            {
                "type": "function",
                "name": function["name"],
                "description": function.get("description") or "",
                "parameters": codex_tool_parameters(parameters),
            }
        )
    return converted or None


# omp: providers/openai-codex-responses.ts :: normalizeCodexToolChoice
def tool_choice(choice: object, tools: list[dict[str, Any]]) -> object:
    """Responses ``tool_choice`` for the offered ``tools``, or ``None`` to omit it.

    A string passes as is. A named choice — chat completions ``{"type": "function",
    "function": {"name": …}}``, the flat Responses form, or Anthropic's ``{"type": "tool",
    "name": …}`` — becomes ``{"type": "function", "name": …}`` only when that function is
    offered: forcing a tool the request does not carry is a 400.
    """
    if choice is None or isinstance(choice, str):
        return choice or None
    if not isinstance(choice, Mapping):
        return None
    kind = choice.get("type")
    name: object = None
    if kind == "function":
        function = choice.get("function")
        name = function.get("name") if isinstance(function, Mapping) else choice.get("name")
    elif kind == "tool":
        name = choice.get("name")
    offered = {t.get("name") for t in tools if t.get("type") == "function"}
    if isinstance(name, str) and name and name in offered:
        return {"type": "function", "name": name}
    return None


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


#: omp: the `# Juice: N` developer item is what turns reasoning off from this wire
#: generation on; the rolling Daybreak aliases ride it too.
JUICE_OFF_MIN_GENERATION: Final = 5.6


# omp: pi-catalog compat/rules/classes/openai.kdl — revision ">=5.6" sets
# requires-reasoning-off-juice-instruction; providers/openai-codex.kdl — "Rolling Daybreak
# aliases ride the 5.6 wire generation."
def requires_juice_to_turn_reasoning_off(model: str) -> bool:
    """Whether ``none`` is sent as the Juice item instead of ``{"effort": "none"}``."""
    name = resolve_model(model).lower()
    if "daybreak" in name:
        return True
    match = re.match(r"gpt-(\d+(?:\.\d+)?)", name)
    return match is not None and float(match.group(1)) >= JUICE_OFF_MIN_GENERATION


# omp: providers/openai-shared.ts :: getJuiceValue
def juice_off_item() -> dict[str, Any]:
    """Developer item that pins the reasoning budget to zero (``JUICE_EFFORT_MAP.none``)."""
    return {
        "type": "message",
        "role": "developer",
        "content": [{"type": "input_text", "text": "# Juice: 0 !important"}],
    }


# omp: providers/openai-chat-server.ts :: parseRequest
# omp: providers/openai-codex/request-transformer.ts :: getReasoningConfig
def reasoning_config(value: object) -> dict[str, str] | None:
    """The body's ``reasoning``, or ``None`` to leave it out.

    ``none`` turns reasoning off with ``{"effort": "none"}`` — omp's ``reasoningOff``, with
    no summary (see `build_request_body` for the 5.6+ exception). A known effort carries the
    summary (``auto`` unless the client picked one): without ``reasoning.summary`` the
    backend streams no reasoning text at all. No effort, or one outside the vocabulary,
    sends no ``reasoning`` and the backend applies the model's default.
    """
    effort, summary = normalize_effort(value)
    if effort == "none":
        return {"effort": "none"}
    if effort not in REASONING_EFFORTS:
        return None
    return {"effort": effort, "summary": summary if summary in REASONING_SUMMARIES else "auto"}


#: Tier ids each model's `/models` entry advertises (`service_tiers`), by wire slug, as the
#: last Codex discovery read them. A slug that is absent was not reported: omp's model
#: carries no list then, and the provider-level answer stands.
_advertised_service_tiers: dict[str, tuple[str, ...]] = {}


def remember_service_tiers(slug: str, tiers: tuple[str, ...] | None) -> None:
    """Record what discovery read for ``slug``; ``None`` forgets it (no list reported)."""
    if tiers is None:
        _advertised_service_tiers.pop(slug.lower(), None)
    else:
        _advertised_service_tiers[slug.lower()] = tiers


def advertised_service_tiers(model: str) -> tuple[str, ...] | None:
    return _advertised_service_tiers.get(model.lower())


# omp: types.ts :: shouldSendServiceTier
def service_tier(value: object, advertised: tuple[str, ...] | None = None) -> str | None:
    """The tier to send for a model whose discovery advertised ``advertised``.

    ``auto`` is never sent: omitting it means the same, and the Codex endpoint rejects it
    outright. ``priority``/``scale`` are dropped only when the model reports a non-empty
    list that omits them (codex-rs ``service_tier_for_request``); an empty or missing list
    counts as "not reported". ``flex`` and ``default`` are never gated. ``ultrafast``, which
    omp sends only when advertised, never gets here: omp's chat server does not parse it
    (``isServiceTier``), and neither does ``SERVICE_TIERS``.
    """
    if value not in SERVICE_TIERS or value == "auto":
        return None
    if value not in ("flex", "default") and advertised and value not in advertised:
        return None
    return str(value)


# omp: providers/openai-codex-responses.ts :: buildTransformedCodexRequestBody
# omp: providers/openai-codex/request-transformer.ts :: transformRequestBody
def build_request_body(
    model: str,
    messages: list[Any],
    tools: list[Any] | None = None,
    extra: dict[str, Any] | None = None,
    unsupported: dict[str, str] | None = None,
    session_id: str | None = None,
    metadata: RequestMetadata | None = None,
    supports_detail_original: bool = True,
) -> dict[str, Any]:
    """Body of a request to the Responses API.

    No sampling controls and no output caps: the Codex backend rejects every one of them
    with 400 ``Unsupported parameter``, so they are never read from ``extra``.
    """
    extra = extra or {}
    instructions, items = messages_to_input(
        messages, supports_detail_original=supports_detail_original
    )
    body: dict[str, Any] = {
        "model": resolve_model(model, unsupported),
        "input": items,
        "stream": True,
    }
    cache_key = prompt_cache_key(session_id, cache_retention=extra.get("cache_retention"))
    if cache_key:
        body["prompt_cache_key"] = cache_key
    if tier := service_tier(extra.get("service_tier"), advertised_service_tiers(body["model"])):
        body["service_tier"] = tier
    if codex_tools := tools_to_codex_tools(tools):
        body["tools"] = codex_tools
        choice = tool_choice(extra.get("tool_choice"), codex_tools)
        if choice is not None:
            body["tool_choice"] = choice
    if instructions is not None:
        body["instructions"] = instructions
    body["store"] = False
    reasoning = reasoning_config(extra.get("reasoning_effort"))
    if reasoning == {"effort": "none"} and requires_juice_to_turn_reasoning_off(model):
        # Diverges from omp's Codex path, which sends `{"effort": "none"}` here and has no
        # effort-fallback retry: the backend answers 400 "'none' is not supported" for
        # gpt-6-astra. omp's Responses path adds the Juice item, with the value of the
        # requested effort (medium, 8, when reasoning is forced off). Measured on the live
        # backend: without `reasoning` and with Juice 0, gpt-6-astra, gpt-5.6-sol and
        # gpt-5.5 spent 0 reasoning tokens; with Juice 8, 12-30.
        body["input"] = [*items, juice_off_item()]
    elif reasoning:
        body["reasoning"] = reasoning
    # Without this the backend does not return the encrypted reasoning, and on a stateless
    # history (`store: false`) the model starts reasoning over on every turn.
    body["include"] = ["reasoning.encrypted_content"]
    if metadata is not None:
        body["client_metadata"] = metadata.client_metadata
    return body
