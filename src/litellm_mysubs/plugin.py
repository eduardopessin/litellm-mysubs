"""Entry point: intercepts LiteLLM and serves the subscription models.

This module is the only part of the package that knows about LiteLLM. It translates
OpenAI ⇄ provider and delegates the HTTP to `transport/client.py`; body construction lives
entirely in `wire/*`.

Why monkey-patching and not `custom_provider_map`: the official map requires the model name
to carry a provider prefix (``mysubs/gpt-5.5``). Clients ask for ``gpt-5.5``, and rewriting
the name along the way would make the spend log record a model nobody asked for. The patch
catches the name exactly as it arrives.

Decisions taken where the contract leaves room
----------------------------------------------

``RemapRequired`` **propagates**. The transport raises it when the account refuses the
model name and signals that an alias may exist. But `codex.resolve_model` has already
applied the alias table *before* sending: if the upstream refused the result, there is no
name left to try, and retrying would send exactly the same request. Substituting another
model is what the README forbids — the response would come back with the ``model`` field
echoing the request and the billing would start lying. Since `RemapRequired` derives from
`UpstreamError`, the real status and body reach the client.

``RedeemRequired`` **propagates** for the same order of reasons: redeeming a reset credit
spends the user's balance, and the plugin has no mandate to do that unasked. An honest 429
is better than a silent charge.

The synchronous path (`litellm.main.completion`) runs the same asynchronous routine in a
private loop: the transport is natively async by contract, and duplicating the logic in a
synchronous version is exactly what made the two drift apart in the original.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import json
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import TYPE_CHECKING, Any, Final, Protocol

import httpx
import litellm
import litellm.main
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.types.utils import Delta, ModelResponse, ModelResponseStream, StreamingChoices

from .credentials.store import CredentialStore, ProviderId
from .transport import hosts
from .transport.client import RequestSpec, Transport
from .wire import anthropic, antigravity, antigravity_models, codex, planning_leak, thinking_loop
from .wire.usage import Usage, codex_finish_reason, codex_usage, google_finish_reason, google_usage

if TYPE_CHECKING:
    # Imported lazily at the call sites: `litellm.types.llms.openai` pulls the OpenAI SDK
    # response models, and this module has to stay importable without the proxy extras.
    from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse

#: Responses API endpoint served by the ChatGPT subscription.
CODEX_URL: Final = "https://chatgpt.com/backend-api/codex/responses"

ANTIGRAVITY_USER_AGENT: Final = (
    "antigravity/hub/2.8.0 (aidev_client; os_type=darwin; arch=arm64; cl=963137146)"
)

#: Reasoning signatures from Gemini tool calls, to send back on the next turn. The cap
#: exists because a long session would accumulate one entry per call until the process
#: ends.
_SIGNATURE_LIMIT: Final = 512

#: This instance's transport identity. Per process, as in the real client: a new
#: `window_id` on every request invalidated the backend's prompt cache.
_WINDOW_ID: Final = str(uuid.uuid4())
_AGENT_ID: Final = uuid.uuid4().hex[:16]
_TRAJECTORY_ID: Final = uuid.uuid4().hex[:16]


class _State:
    """Module state, in a single object so that `uninstall` leaves no loose ends."""

    __slots__ = (
        "catalog",
        "original_acompletion",
        "original_completion",
        "original_router_acompletion",
        "rebound_routers",
        "signatures",
        "step",
        "store",
        "transport",
    )

    def __init__(self) -> None:
        self.original_acompletion: Callable[..., Any] | None = None
        self.original_completion: Callable[..., Any] | None = None
        #: The proxy routes through `Router.acompletion`, not the module functions.
        self.original_router_acompletion: Callable[..., Any] | None = None
        #: Routers whose `aresponses` this module replaced, with the bound attribute it
        #: replaced. Keyed by `id()` because `Router` is unhashable, and held weakly in
        #: spirit only: the proxy builds one Router and keeps it for the process.
        self.rebound_routers: dict[int, tuple[Any, Any]] = {}
        self.store: CredentialStore | None = None
        self.transport: Transport | None = None
        self.signatures: OrderedDict[str, str] = OrderedDict()
        self.step = 0
        #: The Antigravity catalog, with `ModelCatalog`'s own TTL. Without it `map_model`
        #: falls back to the curated static map, which only knows the Gemini family —
        #: measured: `claude-sonnet-4-6`, `gpt-oss-120b-medium`, `chat_23310` and eight
        #: more raised `ModelNotServedError` with the message "is not served by this
        #: account", when all eleven were in the account's real catalog.
        self.catalog = antigravity_models.ModelCatalog()


_state = _State()


def configure(*, store: CredentialStore | None = None, transport: Transport | None = None) -> None:
    """Wires the dependencies. Call before `install`.

    Both are injected rather than discovered: that is what allows the whole dispatch to be
    exercised without touching the network or the disk.
    """
    if store is not None:
        _state.store = store
    if transport is not None:
        _state.transport = transport


def _transport() -> Transport:
    """The transport in use; creates the production one on first need."""
    if _state.transport is None:
        _state.transport = Transport(refresh=_refresh, rotation=hosts.HostRotation())
    return _state.transport


async def _refresh(provider: str) -> str | None:
    """Renews the credential after a 401.

    Two steps, in this order:

    1. **Re-read the source.** Another process — another proxy worker, the dashboard — may
       have rotated the token in the meantime. If the read already brings a token different
       from the one that failed, it is done, and the refresh token is not spent.
    2. **Renew**, but only if this store is the owner. Single-use rotating tokens do not
       tolerate two renewers: the rule is at the top of `credentials/store.py`, and a store
       with `owns_refresh=False` reads and never exchanges.

    A failure here returns `None`, which the transport translates into the upstream's real
    error. It does not raise: the original 401 is more informative than "I failed to
    renew".
    """
    store = _state.store
    if store is None:
        return None

    provider_id = _PROVIDER_IDS[provider]
    store.reload()
    credential = store.get(provider_id)
    if credential is None:
        return None
    if not credential.is_expired():
        # The re-read brought something still usable: another process already renewed.
        return credential.access_token

    if not getattr(store, "owns_refresh", False) or not credential.refresh_token:
        return None

    try:
        from .credentials import oauth

        async with httpx.AsyncClient() as client:
            renewed = await oauth.refresh(credential, client=client, store=store)
    except Exception:
        return None

    store.set(provider_id, renewed)
    return renewed.access_token


_PROVIDER_IDS: Final[dict[str, ProviderId]] = {
    "codex": "openai-codex",
    "antigravity": "google-antigravity",
    "anthropic": "anthropic",
}


async def _access_token(provider: str) -> str:
    """The token to use on the request, renewed before it expires if needed.

    Renewing here instead of waiting for the 401 saves one round trip to the upstream per
    expiring token, and keeps a streaming request from failing halfway — the transport only
    retries what it has not yet delivered, and a 401 after the first event is not
    recoverable.

    `Credential.is_expired` already carries 60 seconds of slack: the token is renewed while
    it still works, so there is no window between the check and the request.
    """
    store = _state.store
    if store is None:
        return ""
    credential = store.get(_PROVIDER_IDS[provider])
    if credential is None:
        return ""
    if credential.is_expired():
        renewed = await _refresh(provider)
        if renewed:
            return renewed
    return credential.access_token


def is_gemini_model(model: str) -> bool:
    """Models served by the Google Antigravity subscription.

    No OMP anchor on purpose: there the distinction is made by ``model.provider`` in a
    typed catalog (`google-gemini-cli.ts`), not by a predicate over the name. Here the name
    is all that arrives from the client. The shape comes from the original,
    `sitecustomize.py:1549`.
    """
    lowered = str(model).lower()
    return "gemini" in lowered or "antigravity" in lowered


#: Provider declared on the deployment, when the request comes from the Router.
#:
#: Guessing the provider from the name fails on Antigravity, which serves models from
#: **three** families. Measured on the account's real catalog: of the 32 models served,
#: seven have no "gemini" in the name — `claude-opus-4-6-thinking`, `claude-sonnet-4-6`,
#: `chat_23310`, `chat_20706`, `tab_flash_lite_preview`, `tab_jump_flash_lite_preview` fell
#: through to native LiteLLM (which has no credential and blows up), and
#: `gpt-oss-120b-medium` was dispatched to **Codex** — another subscription, another wire,
#: another account being charged.
#:
#: `ModelRegistry` already writes `model_info.mysubs_provider` on every entry it injects.
#: Reading that mark is the difference between knowing and assuming.
_PROVIDER_KEY: Final = "mysubs_provider"

#: Wire model carried from the Router to the streaming path, for cost calculation.
#:
#: Private to this hop: it is popped before anything is delegated upstream, because an
#: unknown kwarg reaches the provider client and ``acompletion() got an unexpected keyword
#: argument`` is the whole request lost, not a missing cost.
_WIRE_MODEL_KEY: Final = "mysubs_wire_model"


def provider_of_deployment(router: Any, model: str) -> ProviderId | None:
    """The provider declared for `model`, or `None` if it is not one of our entries.

    Looks up `model_name` in the Router's list. A name that is not there — or that is there
    without the mark — returns `None`, and dispatch falls back to the name heuristic, which
    is what serves callers of `litellm.acompletion` directly, without a Router.
    """
    for deployment in getattr(router, "model_list", None) or []:
        if not isinstance(deployment, dict):
            continue
        if deployment.get("model_name") != model:
            continue
        info = deployment.get("model_info") or {}
        declared = info.get(_PROVIDER_KEY)
        if declared in _PROVIDER_IDS.values():
            return declared
    return None


def wire_model_of_deployment(router: Any, model: str) -> str | None:
    """`litellm_params.model` of our deployment for `model`, or `None`.

    Only entries carrying our mark are read: taking the wire name off someone else's
    deployment would price a call this plugin never served.
    """
    for deployment in getattr(router, "model_list", None) or []:
        if not isinstance(deployment, dict):
            continue
        if deployment.get("model_name") != model:
            continue
        info = deployment.get("model_info") or {}
        if info.get(_PROVIDER_KEY) not in _PROVIDER_IDS.values():
            continue
        wire = (deployment.get("litellm_params") or {}).get("model")
        return str(wire) if wire else None
    return None


def _remember_signature(call_id: str, signature: str) -> None:
    signatures = _state.signatures
    signatures[call_id] = signature
    signatures.move_to_end(call_id)
    while len(signatures) > _SIGNATURE_LIMIT:
        signatures.popitem(last=False)


def _request_id() -> str:
    """``agent/<id>/<ts>/<traj>/<step>`` — the format the CCA expects."""
    _state.step += 1
    return f"agent/{_AGENT_ID}/{int(time.time() * 1000)}/{_TRAJECTORY_ID}/{_state.step}"


async def _codex_spec(model: str, messages: list[Any], extra: dict[str, Any]) -> RequestSpec:
    # No output caps to strip: `build_request_body` builds the body from scratch and does
    # not read `max_tokens`/`max_output_tokens`/`max_completion_tokens` from the kwargs. The
    # original had to delete them because it passed the kwargs on; here they never reach
    # the wire.
    token = await _access_token("codex")
    body = codex.build_request_body(
        model,
        messages,
        tools=extra.get("tools"),
        extra=extra,
        session_id=extra.get("litellm_session_id") or extra.get("user"),
    )
    headers = codex.build_headers(
        token,
        window_id=_WINDOW_ID,
        session_id=extra.get("litellm_session_id") or extra.get("user"),
        model=str(body.get("model") or model),
        service_tier=extra.get("service_tier"),
    )
    return RequestSpec(url=CODEX_URL, headers=headers, body=body, provider="codex", model=model)


async def _refresh_catalog(token: str, project_id: str) -> antigravity_models.ModelCatalog:
    """The Antigravity catalog, with `ModelCatalog`'s own TTL.

    A failure here returns whatever there is — empty the first time. That is deliberate:
    `map_model` then falls back to the static map, which serves the Gemini family, and the
    request goes through instead of dying because the catalog did not answer. The opposite
    would let an outage of the catalog endpoint disable every model at once.
    """
    catalog = _state.catalog
    if catalog.is_fresh():
        return catalog
    with contextlib.suppress(Exception):
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                hosts.HOSTS[0] + hosts.MODELS_PATH,
                json={"project": project_id} if project_id else {},
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "User-Agent": ANTIGRAVITY_USER_AGENT,
                },
            )
            if response.status_code == 200:
                catalog.update(response.json())
    return catalog


async def _antigravity_spec(model: str, messages: list[Any], extra: dict[str, Any]) -> RequestSpec:
    token = await _access_token("antigravity")
    store = _state.store
    credential = store.get("google-antigravity") if store else None
    project_id = credential.project_id if credential else ""
    body = antigravity.build_payload(
        model,
        messages,
        project_id=project_id,
        request_id=_request_id(),
        tools=extra.get("tools"),
        extra=extra,
        thought_signatures=_state.signatures,
        catalog=await _refresh_catalog(token, project_id),
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "User-Agent": ANTIGRAVITY_USER_AGENT,
    }
    return RequestSpec(
        url=hosts.HOSTS[0] + hosts.STREAM_PATH,
        headers=headers,
        body=body,
        provider="antigravity",
        model=model,
    )


# -- event interpretation ------------------------------------------------------


class _Turn:
    """Accumulator for what a stream of events produced.

    The same object serves both paths: in the non-streaming one it is read at the end, in
    the streaming one it is emitted as it goes. Having two interpretation routines is what
    made the original's synchronous and asynchronous versions diverge.
    """

    __slots__ = (
        "finish_raw",
        "reasoning",
        "response_payload",
        "terminal",
        "text",
        "tool_calls",
        "usage_meta",
    )

    def __init__(self) -> None:
        self.text: list[str] = []
        self.reasoning: list[str] = []
        self.tool_calls: list[dict[str, Any]] = []
        self.usage_meta: dict[str, Any] = {}
        self.finish_raw: object = None
        self.terminal = False
        #: The terminal event's own `response` object, kept verbatim.
        #:
        #: Codex speaks the Responses API natively, so `/v1/responses` is served by handing
        #: this back rather than rebuilding it from the accumulated text — see
        #: `_codex_responses_turn`. The chat path ignores it.
        self.response_payload: dict[str, Any] = {}


class StreamError(RuntimeError):
    """Failure inside a stream with HTTP 200.

    Both Codex (``response.failed``) and the CCA (in-band ``error``) report errors in the
    body of a successful response. Swallowing them delivered an empty turn as success.
    """


class _CodexReader:
    """Translates Responses API events into OpenAI chunks, updating a `_Turn`.

    A ``feed``/``close`` interface instead of a generator over an iterable: the same object
    serves the streaming path (what ``feed`` returns is emitted) and the non-streaming one
    (it is discarded), without an event that produces several chunks being held back.
    """

    __slots__ = ("_active", "_index_of", "_turn", "_ws_bytes", "_ws_events")

    def __init__(self, turn: _Turn) -> None:
        self._turn = turn
        self._active: dict[str, dict[str, Any]] = {}
        self._index_of: dict[str, int] = {}
        self._ws_events = 0
        self._ws_bytes = 0

    def feed(self, event: dict[str, Any]) -> list[ModelResponseStream]:
        kind = event.get("type")
        turn = self._turn

        if kind == "response.output_item.added":
            item = event.get("item") or {}
            if item.get("type") != "function_call":
                return []
            item_id = str(item.get("id") or "")
            call_id = codex.composite_call_id(item.get("call_id"), item.get("id"))
            name = str(item.get("name") or "")
            index = len(self._index_of)
            self._index_of[item_id] = index
            self._active[item_id] = {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": ""},
            }
            return [_tool_open_chunk(index, call_id, name)]

        if kind == "response.function_call_arguments.delta":
            item_id = str(event.get("item_id") or "")
            delta = str(event.get("delta") or "")
            if not delta:
                return []
            # The backend sometimes enters a loop emitting only whitespace in the
            # arguments; with no brake the stream never closes. OMP limits: 256 events /
            # 16 KB.
            if not delta.strip():
                self._ws_events += 1
                self._ws_bytes += len(delta)
                if self._ws_events > 256 or self._ws_bytes > 16384:
                    raise StreamError("Codex: whitespace loop in tool call arguments")
            if item_id in self._active:
                self._active[item_id]["function"]["arguments"] += delta
            return [_tool_delta_chunk(self._index_of.get(item_id, 0), delta)]

        if kind == "response.output_item.done":
            item = event.get("item") or {}
            item_id = str(item.get("id") or "")
            if item.get("type") == "function_call" and item_id in self._active:
                call = self._active.pop(item_id)
                if item.get("arguments"):
                    call["function"]["arguments"] = str(item["arguments"])
                turn.tool_calls.append(call)
            return []

        if kind in ("response.reasoning_text.delta", "response.reasoning_summary_text.delta"):
            delta = str(event.get("delta") or "")
            if not delta:
                return []
            turn.reasoning.append(delta)
            return [_delta_chunk(Delta(reasoning_content=delta))]

        if kind in ("response.output_text.delta", "response.refusal.delta"):
            # OMP treats `refusal` as visible text; without this branch a refused turn
            # reached the client with empty content and a clean stop.
            delta = str(event.get("delta") or "")
            if not delta:
                return []
            turn.text.append(delta)
            return [_delta_chunk(Delta(content=delta))]

        if kind in ("response.completed", "response.incomplete"):
            payload = event.get("response") or {}
            turn.terminal = True
            turn.usage_meta = payload.get("usage") or {}
            turn.finish_raw = payload.get("status") or (
                "incomplete" if kind == "response.incomplete" else "completed"
            )
            if isinstance(payload, dict):
                turn.response_payload = payload
            return []

        if kind in ("response.failed", "error"):
            payload = event.get("response") or {}
            detail = payload.get("error") or event.get("message") or "unknown error"
            raise StreamError(f"Codex: {detail}")

        return []

    def close(self) -> list[ModelResponseStream]:
        """Only `response.completed`/`response.incomplete` close the response.

        A stream cut before that is a transport failure: returning it as success delivered
        truncated output as if it were complete.
        """
        if not self._turn.terminal:
            raise StreamError("Codex: stream ended without response.completed/response.incomplete")
        return []


def _raise_in_band(event: dict[str, Any]) -> None:
    """The CCA returns errors inside the stream with HTTP 200."""
    error = event.get("error")
    if isinstance(error, dict) and int(error.get("code") or 0) >= 400:
        raise StreamError(f"Antigravity {error.get('code')}: {error.get('message') or error}")
    feedback = (event.get("response") or {}).get("promptFeedback")
    if isinstance(feedback, dict) and feedback.get("blockReason"):
        raise StreamError(f"Antigravity: content blocked ({feedback['blockReason']})")


class _AntigravityReader:
    """Translates ``:streamGenerateContent`` events, updating a `_Turn`."""

    __slots__ = ("_guard", "_leak", "_tool_index", "_turn", "_wire_model")

    def __init__(self, turn: _Turn, *, wire_model: str) -> None:
        self._turn = turn
        self._wire_model = wire_model
        self._guard = thinking_loop.guard_for(wire_model)
        self._leak = (
            planning_leak.PlanningLeakFilter()
            if planning_leak.is_flash_leak_model(wire_model)
            else None
        )
        self._tool_index = 0

    def feed(self, event: dict[str, Any]) -> list[ModelResponseStream]:
        _raise_in_band(event)
        turn = self._turn
        payload = event.get("response") or {}
        turn.usage_meta = payload.get("usageMetadata") or turn.usage_meta
        candidates = payload.get("candidates") or []
        if not candidates:
            return []
        turn.finish_raw = candidates[0].get("finishReason") or turn.finish_raw

        parts = (candidates[0].get("content") or {}).get("parts") or []
        if candidates[0].get("finishReason"):
            # The event that closes the turn is the only one carrying the full
            # `usageMetadata`, and it is the one the retirement notice comes in (measured:
            # a single event, with text, `finishReason: STOP` and `total_tokens=0`).
            # Guarding here, **before** emitting what this event carries, is what keeps the
            # notice from going out as content.
            self._guard_retired("".join(str(part.get("text") or "") for part in parts))

        chunks: list[ModelResponseStream] = []
        for part in parts:
            text = str(part.get("text") or "")
            if text:
                chunks.extend(self._text(text, thought=bool(part.get("thought"))))
            call = part.get("functionCall")
            if call:
                chunks.extend(self._call(call, part.get("thoughtSignature")))
        return chunks

    def _guard_retired(self, pending: str = "") -> None:
        """A retired model answers 200 with a notice; accepting it put it in the history.

        The check is over the turn's **accumulated** text plus what has not been emitted
        yet: the notice can arrive split across parts, and neither half alone matches the
        markers.
        """
        antigravity.raise_if_retired(
            "".join(self._turn.text) + pending, self._turn.usage_meta, self._wire_model
        )

    def _text(self, text: str, *, thought: bool) -> list[ModelResponseStream]:
        turn = self._turn
        if thought:
            if self._guard is not None and (reason := self._guard.feed(text)):
                raise thinking_loop.ThinkingLoopError(
                    f"Antigravity: reasoning loop ({reason}) after "
                    f"{self._guard.chars} chars on {self._wire_model}; aborted instead of "
                    "billing the rest"
                )
            turn.reasoning.append(text)
            return [_delta_chunk(Delta(reasoning_content=text))]
        visible = self._leak.feed(text) if self._leak is not None else text
        if not visible:
            return []
        turn.text.append(visible)
        return [_delta_chunk(Delta(content=visible))]

    def _call(self, call: dict[str, Any], signature: object) -> list[ModelResponseStream]:
        call_id = str(call.get("id") or f"call_{uuid.uuid4().hex[:8]}")
        if signature:
            _remember_signature(call_id, str(signature))
        name = str(call.get("name") or "")
        arguments = json.dumps(call.get("args") or {})
        self._turn.tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
        index = self._tool_index
        self._tool_index += 1
        return [_tool_open_chunk(index, call_id, name), _tool_delta_chunk(index, arguments)]

    def close(self) -> list[ModelResponseStream]:
        """Flushes what the leak filter held back and turned out not to be planning."""
        chunks: list[ModelResponseStream] = []
        if self._leak is not None and (tail := self._leak.flush()):
            self._turn.text.append(tail)
            chunks.append(_delta_chunk(Delta(content=tail)))
        # Safety net: if the notice arrives with no `finishReason` in the same event, or
        # spread over several, `feed` never saw it whole. Here the turn is complete and the
        # final `usageMetadata` has arrived. Fires at most once per response — if `feed`
        # already raised, this line is never reached.
        self._guard_retired()
        return chunks


# -- the shape LiteLLM expects -------------------------------------------------


def _delta_chunk(delta: Delta) -> ModelResponseStream:
    return ModelResponseStream(choices=[StreamingChoices(index=0, delta=delta, finish_reason=None)])


def _tool_open_chunk(index: int, call_id: str, name: str) -> ModelResponseStream:
    """Opens a tool call. ``role`` rides here because this may be the turn's first chunk."""
    return _delta_chunk(
        Delta(
            role="assistant",
            tool_calls=[
                {
                    "index": index,
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": ""},
                }
            ],
        )
    )


def _tool_delta_chunk(index: int, arguments: str) -> ModelResponseStream:
    return _delta_chunk(Delta(tool_calls=[{"index": index, "function": {"arguments": arguments}}]))


def _finish_chunk(reason: str) -> ModelResponseStream:
    return ModelResponseStream(
        choices=[StreamingChoices(index=0, delta=Delta(), finish_reason=reason)]
    )


def _usage_chunk(usage: Usage) -> ModelResponseStream:
    """Final chunk with the real usage; without it LiteLLM estimates by token counting.

    ``choices`` carries one empty entry instead of being ``[]``: the ``/v1/responses``
    route's iterator does ``chunk.choices[0].delta`` with no guard, and an empty list kills
    the stream before the terminal event — the client waits forever.
    """
    chunk = ModelResponseStream(
        choices=[StreamingChoices(index=0, delta=Delta(), finish_reason=None)]
    )
    chunk.usage = _litellm_usage(usage)
    return chunk


def _litellm_usage(usage: Usage) -> litellm.Usage:
    """``cached_tokens`` also has to go on the attribute the spend logging reads."""
    out = litellm.Usage(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        prompt_tokens_details={"cached_tokens": usage.cached_tokens},
    )
    out.cache_read_input_tokens = usage.cached_tokens
    return out


def _message(turn: _Turn) -> dict[str, Any]:
    """Assistant message in the OpenAI shape, with the reasoning in the standard field."""
    message: dict[str, Any] = {"role": "assistant"}
    if turn.tool_calls:
        message["tool_calls"] = turn.tool_calls
    else:
        message["content"] = "".join(turn.text)
    if turn.reasoning:
        message["reasoning_content"] = "".join(turn.reasoning)
    return message


def _model_response(model: str, turn: _Turn, *, finish_reason: str, usage: Usage) -> ModelResponse:
    return ModelResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[{"index": 0, "message": _message(turn), "finish_reason": finish_reason}],
        usage=_litellm_usage(usage),
    )


# -- dispatch --------------------------------------------------------------------


def _normalize(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Positional ``(model, messages)`` become kwargs.

    LiteLLM accepts both forms; without this, dispatch via ``kwargs["model"]`` did not see
    the model and every positional request fell through to the original.
    """
    if args and "model" not in kwargs:
        kwargs["model"] = args[0]
    if len(args) > 1 and "messages" not in kwargs:
        kwargs["messages"] = args[1]
    return kwargs


class _Reader(Protocol):
    def feed(self, event: dict[str, Any]) -> list[ModelResponseStream]: ...
    def close(self) -> list[ModelResponseStream]: ...


async def _drive(events: AsyncIterator[dict[str, Any]], reader: _Reader) -> None:
    """Non-streaming path: consumes everything through the same reader, discarding chunks.

    A single interpretation routine, shared with the streaming path — having two is what
    made the original's synchronous and asynchronous versions diverge.
    """
    async for event in events:
        reader.feed(event)
    reader.close()


async def _pump(
    events: AsyncIterator[dict[str, Any]], reader: _Reader
) -> AsyncIterator[ModelResponseStream]:
    """Streaming path: emits each event's chunks as they arrive."""
    async for event in events:
        for chunk in reader.feed(event):
            yield chunk
    for chunk in reader.close():
        yield chunk


async def _codex_turn(model: str, messages: list[Any], extra: dict[str, Any]) -> ModelResponse:
    spec = await _codex_spec(model, messages, extra)
    turn = _Turn()
    await _drive(_transport().stream(spec), _CodexReader(turn))
    return _model_response(
        model,
        turn,
        finish_reason=codex_finish_reason(turn.finish_raw, bool(turn.tool_calls)),
        usage=codex_usage(turn.usage_meta),
    )


def _responses_input(kwargs: dict[str, Any]) -> list[Any]:
    """``/v1/responses`` carries ``input``, not ``messages``.

    A plain string is the documented shorthand for a single user turn, and the item form is
    already what ``messages_to_input`` produces on the way out, so both are handed to the
    existing body builder unchanged rather than being converted twice.
    """
    value = kwargs.get("input")
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if isinstance(value, list):
        return list(value)
    return []


async def _codex_responses_turn(
    model: str, kwargs: dict[str, Any]
) -> ResponsesAPIResponse:
    """Serves ``/v1/responses`` from the subscription's own Responses payload.

    Codex **is** a Responses API endpoint, so the terminal event's `response` object is the
    shape this route has to return and is kept as the base — status, `previous_response_id`
    and the rest are the upstream's.

    What the terminal event does **not** carry is `output`. Measured against the real
    endpoint: the items arrive in the `response.output_item.done` events during the stream,
    and `response.completed` closes the turn without repeating them, so a straight
    passthrough returned `output: []` with a non-zero `output_tokens` — a completed turn
    whose text had vanished. They are rebuilt here from what the reader accumulated, which
    is the same source the chat path uses, so the two routes cannot disagree.

    Usage is rebuilt for a second reason: an absent or partial `usage` fails
    `ResponseAPIUsage` validation, which requires all three counters. Going through
    `codex_usage` is also what makes this route's spend log match the chat route's.
    """
    from litellm.types.llms.openai import ResponsesAPIResponse

    spec = await _codex_spec(model, _responses_input(kwargs), kwargs)
    turn = _Turn()
    await _drive(_transport().stream(spec), _CodexReader(turn))

    payload = dict(turn.response_payload)
    # `model` echoes the wire name; the caller asked for the public one and the spend log
    # reads this field.
    payload["model"] = model
    payload.setdefault("id", f"resp_{uuid.uuid4().hex[:24]}")
    payload.setdefault("created_at", int(time.time()))
    payload.setdefault("object", "response")
    if not payload.get("output"):
        payload["output"] = _responses_output(turn)
    payload["usage"] = _responses_usage(codex_usage(turn.usage_meta))
    return ResponsesAPIResponse(**payload)


def _responses_output(turn: _Turn) -> list[dict[str, Any]]:
    """``output`` items for a turn the terminal event did not carry them for.

    Reasoning comes first, then the message, then the tool calls — the order the Responses
    API documents and the order a client replays them in. Tool calls carry the composite
    id the chat path also emits, so a follow-up turn matches its output to the right call.
    """
    items: list[dict[str, Any]] = []
    reasoning = "".join(turn.reasoning)
    if reasoning:
        items.append(
            {
                "type": "reasoning",
                "id": f"rs_{uuid.uuid4().hex[:24]}",
                "summary": [{"type": "summary_text", "text": reasoning}],
            }
        )
    text = "".join(turn.text)
    if text:
        items.append(
            {
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        )
    for call in turn.tool_calls:
        function = call.get("function") or {}
        items.append(
            {
                "type": "function_call",
                "id": f"fc_{uuid.uuid4().hex[:24]}",
                "call_id": str(call.get("id") or ""),
                "name": str(function.get("name") or ""),
                "arguments": str(function.get("arguments") or ""),
                "status": "completed",
            }
        )
    return items


def _responses_usage(usage: Usage) -> ResponseAPIUsage:
    """``ResponseAPIUsage`` counters, which are named differently from the chat ones.

    All three are required by the model, so they are always supplied — a turn whose
    upstream reported nothing bills zero rather than failing to construct.
    """
    from litellm.types.llms.openai import ResponseAPIUsage

    return ResponseAPIUsage(
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
    )


async def _codex_responses_stream(
    model: str, kwargs: dict[str, Any]
) -> AsyncIterator[Any]:
    """``/v1/responses`` with ``stream: true``, as Responses API events.

    The Responses SSE protocol is not the chat chunk sequence: a client reads items, not
    deltas on a choice. The order below was **captured from this proxy's own native path**
    (`qwen-agent-coder`, `/v1/responses`, `stream: true`) rather than assumed, because the
    previous attempt at this route shipped a payload shape that the real endpoint never
    sends:

        response.created -> response.in_progress
        -> output_item.added -> content_part.added
        -> output_text.delta* -> output_text.done -> content_part.done
        -> output_item.done
        -> response.completed

    `item_id`/`output_index`/`content_index` tie the parts to their item — a client that
    tracks them needs all three.

    Events are emitted as LiteLLM's **typed** event models, not dicts. The proxy serialises
    a stream chunk with `_serialize_streaming_chunk`, which calls `.model_dump_json()`; a
    plain dict falls through to `str()` and reaches the client as a Python repr with single
    quotes, which no JSON parser accepts. Measured against a real proxy before this was
    written the second time.

    Text is emitted as it arrives; reasoning and tool calls are emitted as completed items
    once the turn closes. Reasoning deltas are deliberately not streamed: `_CodexReader`
    accumulates them for the chat path, and replaying them as `reasoning_text.delta` would
    mean a second interpretation of the same events — the divergence the module docstring
    exists to prevent.
    """
    spec = await _codex_spec(model, _responses_input(kwargs), kwargs)
    turn = _Turn()
    reader = _CodexReader(turn)

    state = _ResponsesStreamState(model=model)
    for out in state.created():
        yield out

    async for event in _transport().stream(spec):
        for chunk in reader.feed(event):
            text = _chunk_text(chunk)
            if not text:
                continue
            for out in state.open_message():
                yield out
            yield state.delta(text)
    reader.close()

    for out in state.close_message():
        yield out

    # Reasoning and tool calls are known only once the turn has closed, so they are
    # announced and completed back to back rather than streamed.
    for item in _responses_output(turn):
        if item.get("type") == "message":
            continue  # already streamed above
        for out in state.whole_item(item):
            yield out

    payload = dict(turn.response_payload)
    payload["model"] = model
    payload["id"] = state.response_id
    payload.setdefault("created_at", state.created_at)
    payload.setdefault("object", "response")
    payload["status"] = payload.get("status") or "completed"
    payload["output"] = state.items
    payload["usage"] = _responses_usage(codex_usage(turn.usage_meta)).model_dump()
    yield state.completed(payload)


def _chunk_text(chunk: ModelResponseStream) -> str:
    """Visible text on a chunk, ignoring reasoning and tool deltas."""
    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return ""
    delta = getattr(choices[0], "delta", None)
    return str(getattr(delta, "content", "") or "") if delta is not None else ""



class _ResponsesStreamState:
    """Sequence numbers, item ids and indices for one Responses stream.

    Kept in an object because every event carries `sequence_number`, and a client that
    reorders on it needs the numbering to be strictly increasing across the whole stream —
    including the items appended after the text has finished.
    """

    __slots__ = (
        "created_at",
        "items",
        "message_id",
        "message_open",
        "model",
        "output_index",
        "response_id",
        "text",
    )

    def __init__(self, model: str) -> None:
        self.model = model
        self.response_id = f"resp_{uuid.uuid4().hex[:24]}"
        self.created_at = int(time.time())
        self.output_index = 0
        self.message_id = f"msg_{uuid.uuid4().hex[:24]}"
        self.message_open = False
        self.text: list[str] = []
        self.items: list[dict[str, Any]] = []

    def _events(self) -> Any:
        from litellm.types.llms.openai import ResponsesAPIStreamEvents

        return ResponsesAPIStreamEvents

    def envelope(self, status: str) -> dict[str, Any]:
        return {
            "id": self.response_id,
            "created_at": self.created_at,
            "model": self.model,
            "object": "response",
            "status": status,
            "output": list(self.items),
        }

    def created(self) -> list[Any]:
        from litellm.types.llms.openai import (
            ResponseCreatedEvent,
            ResponseInProgressEvent,
            ResponsesAPIResponse,
        )

        kinds = self._events()
        envelope = ResponsesAPIResponse.model_validate(self.envelope("in_progress"))
        return [
            ResponseCreatedEvent(type=kinds.RESPONSE_CREATED, response=envelope),
            ResponseInProgressEvent(type=kinds.RESPONSE_IN_PROGRESS, response=envelope),
        ]

    def delta(self, text: str) -> Any:
        from litellm.types.llms.openai import OutputTextDeltaEvent

        self.text.append(text)
        return OutputTextDeltaEvent(
            type=self._events().OUTPUT_TEXT_DELTA,
            item_id=self.message_id,
            output_index=self.output_index,
            content_index=0,
            delta=text,
        )

    def open_message(self) -> list[Any]:
        """`output_item.added` + `content_part.added`, once, before the first delta."""
        if self.message_open:
            return []
        from litellm.types.llms.openai import ContentPartAddedEvent, OutputItemAddedEvent

        self.message_open = True
        kinds = self._events()
        return [
            OutputItemAddedEvent(
                type=kinds.OUTPUT_ITEM_ADDED,
                output_index=self.output_index,
                item=({
                    "id": self.message_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [],
                }),  # type: ignore[arg-type]  # pydantic coerces the dict; the annotation is narrower
            ),
            ContentPartAddedEvent(
                type=kinds.CONTENT_PART_ADDED,
                item_id=self.message_id,
                output_index=self.output_index,
                content_index=0,
                part=({"type": "output_text", "text": "", "annotations": [], "logprobs": []}),  # type: ignore[arg-type]  # pydantic coerces the dict; the annotation is narrower
            ),
        ]

    def close_message(self) -> list[Any]:
        """`output_text.done` + `content_part.done` + `output_item.done`."""
        if not self.message_open:
            return []
        from litellm.types.llms.openai import (
            ContentPartDoneEvent,
            OutputItemDoneEvent,
            OutputTextDoneEvent,
        )

        kinds = self._events()
        text = "".join(self.text)
        item: dict[str, Any] = {
            "id": self.message_id,
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }
        self.items.append(item)
        out = [
            OutputTextDoneEvent(
                type=kinds.OUTPUT_TEXT_DONE,
                item_id=self.message_id,
                output_index=self.output_index,
                content_index=0,
                text=text,
            ),
            ContentPartDoneEvent(
                type=kinds.CONTENT_PART_DONE,
                item_id=self.message_id,
                output_index=self.output_index,
                content_index=0,
                part=({"type": "output_text", "text": text, "annotations": [], "logprobs": []}),  # type: ignore[arg-type]  # pydantic coerces the dict; the annotation is narrower
            ),
            OutputItemDoneEvent(
                type=kinds.OUTPUT_ITEM_DONE, output_index=self.output_index, item=item  # type: ignore[arg-type]
            ),
        ]
        self.output_index += 1
        self.message_open = False
        return out

    def whole_item(self, item: dict[str, Any]) -> list[Any]:
        """An item known only at the end: announced and completed back to back."""
        from litellm.types.llms.openai import OutputItemAddedEvent, OutputItemDoneEvent

        kinds = self._events()
        self.items.append(item)
        out = [
            OutputItemAddedEvent(
                type=kinds.OUTPUT_ITEM_ADDED, output_index=self.output_index, item=item  # type: ignore[arg-type]
            ),
            OutputItemDoneEvent(
                type=kinds.OUTPUT_ITEM_DONE, output_index=self.output_index, item=item  # type: ignore[arg-type]
            ),
        ]
        self.output_index += 1
        return out

    def completed(self, payload: dict[str, Any]) -> Any:
        from litellm.types.llms.openai import ResponseCompletedEvent, ResponsesAPIResponse

        return ResponseCompletedEvent(
            type=self._events().RESPONSE_COMPLETED,
            response=ResponsesAPIResponse.model_validate(payload),
        )


async def _antigravity_turn(
    model: str, messages: list[Any], extra: dict[str, Any]
) -> ModelResponse:
    spec = await _antigravity_spec(model, messages, extra)
    turn = _Turn()
    reader = _AntigravityReader(turn, wire_model=str(spec.body.get("model") or model))
    await _drive(_transport().stream(spec), reader)
    return _model_response(
        model,
        turn,
        finish_reason=google_finish_reason(turn.finish_raw, bool(turn.tool_calls)),
        usage=google_usage(turn.usage_meta),
    )


async def _codex_stream(
    model: str, messages: list[Any], extra: dict[str, Any]
) -> AsyncIterator[ModelResponseStream]:
    spec = await _codex_spec(model, messages, extra)
    turn = _Turn()
    async for chunk in _pump(_transport().stream(spec), _CodexReader(turn)):
        yield chunk
    yield _finish_chunk(codex_finish_reason(turn.finish_raw, bool(turn.tool_calls)))
    yield _usage_chunk(codex_usage(turn.usage_meta))


async def _antigravity_stream(
    model: str, messages: list[Any], extra: dict[str, Any]
) -> AsyncIterator[ModelResponseStream]:
    spec = await _antigravity_spec(model, messages, extra)
    turn = _Turn()
    reader = _AntigravityReader(turn, wire_model=str(spec.body.get("model") or model))
    async for chunk in _pump(_transport().stream(spec), reader):
        yield chunk
    yield _finish_chunk(google_finish_reason(turn.finish_raw, bool(turn.tool_calls)))
    yield _usage_chunk(google_usage(turn.usage_meta))


async def dispatch_responses(*, provider: ProviderId | None = None, **kwargs: Any) -> Any:
    """``/v1/responses`` counterpart of `dispatch`; ``None`` means "not mine".

    Only Codex is served here. Anthropic has no branch for the same reason it has none in
    `dispatch` — LiteLLM's native path serves it with the token `_delegate_kwargs` injects,
    and that path already answers this route correctly. Antigravity is Gemini-shaped, so
    returning it through a Responses object would mean inventing item structure the
    upstream never sent; it stays on the route it works on.

    Streaming returns an async iterator of Responses API events, not chat chunks: see
    `_codex_responses_stream`, whose event order was captured from this proxy's own native
    path rather than assumed.
    """
    model = str(kwargs.get("model") or "")
    if provider == "google-antigravity" or (provider is None and is_gemini_model(model)):
        return None
    if provider == "openai-codex" or (provider is None and codex.is_codex_model(model)):
        _stamp_logging_identity(model, kwargs)
        if kwargs.get("stream"):
            return _codex_responses_stream(model, kwargs)
        return await _codex_responses_turn(model, kwargs)
    return None


async def dispatch(*, provider: ProviderId | None = None, **kwargs: Any) -> Any:
    """Serves the request if the model belongs to one of our subscriptions; ``None`` if not.

    ``None`` is the only way to say "not mine" without fabricating a response: the caller
    delegates to the original. One of our models that the upstream refuses propagates the
    error — `RemapRequired` and `RedeemRequired` included, for the reasons at the top of
    the module.

    ``provider`` comes from the deployment's `model_info.mysubs_provider` when the request
    goes through the Router, and **wins** over the name heuristic. It is what keeps a
    `claude-sonnet-4-6` served by Antigravity from being treated as Anthropic, or a
    `gpt-oss-120b-medium` from the same account from ending up at Codex — measured: seven
    of the 32 models in the real catalog dispatched to the wrong place.
    """
    model = str(kwargs.get("model") or "")
    messages = kwargs.get("messages") or []
    streaming = bool(kwargs.get("stream"))

    if provider == "google-antigravity" or (provider is None and is_gemini_model(model)):
        # Stamped before serving, not after: the spend log reads the identity off the
        # logging object, and the streaming path hands that object to the wrapper.
        _stamp_logging_identity(model, kwargs)
        if streaming:
            return _wrap_stream(_antigravity_stream(model, messages, kwargs), model, kwargs)
        return await _antigravity_turn(model, messages, kwargs)

    # After Gemini: `codex.is_codex_model` matches any name containing "gpt-", and a
    # hypothetical "gemini-gpt" belongs to Google.
    if provider == "openai-codex" or (provider is None and codex.is_codex_model(model)):
        _stamp_logging_identity(model, kwargs)
        if streaming:
            return _wrap_stream(_codex_stream(model, messages, kwargs), model, kwargs)
        return await _codex_turn(model, messages, kwargs)

    # `anthropic` has no branch of its own: it is served by LiteLLM's native path with the
    # prompt and the token that `_delegate_kwargs` injects. Returning `None` is what routes
    # it there.
    return None


def _cost_identity(model: str, kwargs: dict[str, Any]) -> tuple[str, str]:
    """The ``(model, provider)`` pair the cost calculation needs.

    ``model`` arrives here as the **public** name (``mysubs/codex/gpt-5.5``) — what the
    client asked for and what the Router resolved. No rate exists under that name, and
    ``custom_openai`` has no price table either, so every streamed call was logged at
    ``0.0``. Measured on ``litellm[proxy]`` 1.101.0, identical usage:

    ===================================== ============
    ``(model, custom_llm_provider)``       cost
    ===================================== ============
    ``("mysubs/codex/gpt-5.5", "custom_openai")``  ``0.0``
    ``("openai/gpt-5.5", "openai")``               ``0.0202325``
    ===================================== ============

    The wire name is already on the deployment, in ``litellm_params.model``, carrying the
    family prefix ``to_deployment`` picked from ``modelProvider``. It is the same pair the
    non-streaming path gets from the Router — which is why only streaming lost the cost,
    and why 83% of real calls were logged as free.

    Without a deployment there is nothing to look up: a direct ``litellm.acompletion``
    call has no Router. The old pair is kept for that case rather than guessing a family
    from the name, because a wrong guess prices the call against another model's rate.
    """
    wire = str(kwargs.get(_WIRE_MODEL_KEY) or "")
    if not wire:
        return model, "custom_openai"
    family = wire.split("/", 1)[0] if "/" in wire else "openai"
    return wire, family


def _stamp_logging_identity(model: str, kwargs: dict[str, Any]) -> None:
    """Record the **wire** ``(model, provider)`` pair on the caller's logging object.

    The spend log reads `custom_llm_provider` and `model` from
    `logging_obj.model_call_details`, not from the deployment. A request this plugin serves
    never reaches the provider client that would normally fill them in, so without this the
    row lands with the public name and **no provider at all** — measured on a live gateway:
    rows served natively read `anthropic/claude-opus-5` + `anthropic`, while rows served
    here read `mysubs/antigravity/...` with an empty provider.

    That empty provider is what leaves the Logs tab without an icon: the UI maps a provider
    id to a logo, and there is nothing to map. Declaring `custom_llm_provider` on the
    deployment fixes the Models tab only, because that one is built from the Router while
    this one is built per request.

    `model_group` keeps the public name, so the client still sees what it asked for.
    """
    logging_obj = kwargs.get("litellm_logging_obj")
    details = getattr(logging_obj, "model_call_details", None)
    if not isinstance(details, dict):
        return
    wire_model, provider = _cost_identity(model, kwargs)
    if not wire_model or wire_model == model:
        # No deployment to read the wire name from: leave what the proxy already set
        # rather than stamping a guessed family, which would bill against another rate.
        return
    details["model"] = wire_model
    details["custom_llm_provider"] = provider
    details.setdefault("model_group", model)
    with contextlib.suppress(Exception):
        logging_obj.model = wire_model  # type: ignore[union-attr]
        logging_obj.custom_llm_provider = provider  # type: ignore[union-attr]


def _wrap_stream(
    chunks: AsyncIterator[ModelResponseStream], model: str, kwargs: dict[str, Any]
) -> litellm.CustomStreamWrapper:
    """Wraps it in LiteLLM's iterator: that is what the proxy knows how to consume.

    Returning the raw generator gave the client objects without the protocol the
    ``/v1/chat/completions`` route expects — and no spend log callback would fire.

    The model and provider handed to the wrapper are the **wire** ones, not the public
    name: see ``_cost_identity``. The client still sees the name it asked for, because the
    spend log records ``model_group`` for that, but the cost calculation now gets a pair
    it can price.
    """
    cost_model, cost_provider = _cost_identity(model, kwargs)
    return litellm.CustomStreamWrapper(
        completion_stream=chunks,
        model=cost_model,
        custom_llm_provider=cost_provider,
        logging_obj=kwargs.get("litellm_logging_obj") or _logging_obj(cost_model, kwargs),
    )


def _logging_obj(model: str, kwargs: dict[str, Any]) -> Logging:
    """Logging object for when the caller does not bring its own.

    `CustomStreamWrapper` dereferences it in the constructor — passing ``None`` blows up
    before the first chunk. The proxy always injects its own; a direct library call does
    not.
    """
    return Logging(
        model=model,
        messages=kwargs.get("messages") or [],
        stream=True,
        call_type="acompletion",
        start_time=datetime.datetime.now(),
        litellm_call_id=str(kwargs.get("litellm_call_id") or uuid.uuid4()),
        function_id=str(kwargs.get("id") or uuid.uuid4()),
    )


async def _wrapped_acompletion(*args: Any, **kwargs: Any) -> Any:
    kwargs = _normalize(args, kwargs)
    served = await dispatch(**kwargs)
    if served is not None:
        return served
    original = _state.original_acompletion
    assert original is not None
    kwargs.pop(_WIRE_MODEL_KEY, None)
    return await original(**await _delegate_kwargs(kwargs))


def _wrapped_completion(*args: Any, **kwargs: Any) -> Any:
    """Synchronous path: runs the same `dispatch` in a private loop.

    Duplicating the logic in a synchronous version is what made the two drift apart in the
    original. The loop is private because `asyncio.run` refuses to run inside an already
    active loop, and the proxy calls this from threads with no loop at all.
    """
    kwargs = _normalize(args, kwargs)
    served = _run_sync(dispatch(**kwargs))
    if served is not None:
        return served
    original = _state.original_completion
    assert original is not None
    kwargs.pop(_WIRE_MODEL_KEY, None)
    return original(**_run_sync(_delegate_kwargs(kwargs)))


def _run_sync(coroutine: Coroutine[Any, Any, Any]) -> Any:
    """Runs a coroutine from synchronous code, whether or not a loop is active."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    # Synchronous call from inside a loop: running on a thread with its own loop is the
    # only way out that does not deadlock the caller's loop against itself.
    result: list[Any] = []
    error: list[BaseException] = []

    def run() -> None:
        try:
            result.append(asyncio.run(coroutine))
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result[0]


#: Places that refer to the entry functions. `litellm/__init__.py` does
#: `from .main import acompletion`, which **copies** the reference: rebinding only
#: `litellm.main` leaves `litellm.acompletion` pointing at the original function, and a
#: client calling `litellm.acompletion(...)` — the documented form — never goes through
#: dispatch. The original `sitecustomize.py` patches both (lines 2916-2917) and the port
#: started out patching only one: the request blew up with "LLM Provider NOT provided",
#: because it reached the native path with a name no provider knows.
_ASYNC_TARGETS: Final = ((litellm, "acompletion"), (litellm.main, "acompletion"))
_SYNC_TARGETS: Final = ((litellm, "completion"), (litellm.main, "completion"))


def _router_class() -> Any:
    """LiteLLM's `Router` class.

    Imported late: `litellm.router` drags in the whole proxy, and this module has to be
    importable without it.
    """
    from litellm.router import Router

    return Router


async def _delegate_kwargs(
    kwargs: dict[str, Any], *, provider: ProviderId | None = None
) -> dict[str, Any]:
    """Kwargs for the original, with the Claude prompt applied when it is a Claude model.

    This is what the original's ``_inject_claude_prompt`` does: the Anthropic subscription
    only validates the Claude Code identity as a system message, and without it the request
    is refused.

    A declared ``provider`` that is **not** `anthropic` blocks the injection. The
    Antigravity catalog serves `claude-sonnet-4-6` and `claude-opus-4-6-thinking`: without
    this guard, those requests carried the Anthropic subscription token to a Google
    endpoint — one account's credential sent to another.
    """
    model = str(kwargs.get("model") or "")
    if provider is not None and provider != "anthropic":
        return kwargs
    return anthropic.build_request(kwargs, model, await _access_token("anthropic"))


async def _wrapped_router_acompletion(
    self: Any, model: str, messages: list[Any], stream: bool = False, **kwargs: Any
) -> Any:
    """The path the **proxy** actually uses.

    The proxy does not call `litellm.acompletion`: it calls `Router.acompletion`, which
    resolves the deployment and builds the provider client **before** any module function is
    touched. Without this patch, a subscription model reached the native client and blew up
    with `Illegal header value b'Bearer '` — the deployment carries no `api_key` because the
    credential is OAuth and lives in the store, not in `config.yaml`.

    The original `sitecustomize.py` says so in the comment on line 2812: *"Proxy routes
    through Router.acompletion, not necessarily the module functions above."* The port
    patched only the module functions, and the symptom showed up only in the proxy — never
    on a library call.
    """
    # The provider comes from the deployment, not from the name: this is where the Router
    # has the information, and it is the only way to tell Anthropic's `claude-sonnet-4-6`
    # from the namesake served by Antigravity.
    declared = provider_of_deployment(self, model)
    # Read here because this is where the Router still has the deployment: the streaming
    # path needs the wire name to price the call, and by then the deployment is gone.
    wire = wire_model_of_deployment(self, model)
    served = await dispatch(
        provider=declared,
        model=model,
        messages=messages,
        stream=stream,
        **{_WIRE_MODEL_KEY: wire},
        **kwargs,
    )
    if served is not None:
        return served
    original = _state.original_router_acompletion
    assert original is not None
    delegated = await _delegate_kwargs(
        {"model": model, "messages": messages, **kwargs}, provider=declared
    )
    delegated.pop("model", None)
    delegated.pop("messages", None)
    delegated.pop(_WIRE_MODEL_KEY, None)
    return await original(self, model=model, messages=messages, stream=stream, **delegated)


def bind_responses_route(router: Any) -> bool:
    """Routes ``/v1/responses`` through the plugin for `router`. ``True`` if it bound.

    `Router.aresponses` is **not** a class method. `Router.__init__` builds it per instance
    with ``self.aresponses = self.factory_function(litellm.aresponses, ...)``, capturing
    `litellm.aresponses` by value at construction. Two consequences, both measured:

    - patching the class does nothing — there is no class attribute to override;
    - patching `litellm.aresponses` after the Router exists does nothing either, because
      the factory already holds the old reference.

    And the Router **does** already exist by the time this package loads: the proxy builds
    it before constructing the `CustomLogger` that brings us in (see `bootstrap`). So the
    only thing that works is replacing the bound attribute on the live instance, which is
    what this does.

    Idempotent per router: rebinding twice would save our own wrapper as the original and
    leave `unbind_responses_route` unable to restore anything.
    """
    if router is None:
        return False
    key = id(router)
    if key in _state.rebound_routers:
        return False
    original = getattr(router, "aresponses", None)
    if original is None:
        return False

    async def _wrapped_router_aresponses(**kwargs: Any) -> Any:
        model = str(kwargs.get("model") or "")
        declared = provider_of_deployment(router, model)
        # On the Router the mark is authoritative and the name heuristic is not consulted:
        # `codex.is_codex_model` matches any name containing "gpt-", so an operator's own
        # `gpt-4o` deployment would be answered from our subscription — measured, and the
        # reason this guard exists rather than deferring to `dispatch_responses` alone.
        if declared is None:
            return await original(**kwargs)
        served = await dispatch_responses(provider=declared, **kwargs)
        if served is not None:
            return served
        return await original(**kwargs)

    _state.rebound_routers[key] = (router, original)
    router.aresponses = _wrapped_router_aresponses
    return True


def unbind_responses_route() -> None:
    """Restores every `aresponses` this module replaced."""
    for router, original in _state.rebound_routers.values():
        try:
            router.aresponses = original
        except Exception:  # teardown is best-effort; a dead router is fine
            continue
    _state.rebound_routers.clear()


def install() -> None:
    """Applies the patch. Idempotent.

    The guard is not cosmetic: installing twice chained two wrappers, and the second saved
    the first as the "original" — `uninstall` then left the patch half applied and every
    request went through dispatch twice.
    """
    if _state.original_acompletion is not None:
        return
    _state.original_acompletion = litellm.main.acompletion
    _state.original_completion = litellm.main.completion
    router_class = _router_class()
    _state.original_router_acompletion = router_class.acompletion
    for module, name in _ASYNC_TARGETS:
        setattr(module, name, _wrapped_acompletion)
    for module, name in _SYNC_TARGETS:
        setattr(module, name, _wrapped_completion)
    router_class.acompletion = _wrapped_router_acompletion


def uninstall() -> None:
    """Restores the originals everywhere. With no patch applied, does nothing."""
    if _state.original_acompletion is None:
        return
    for module, name in _ASYNC_TARGETS:
        setattr(module, name, _state.original_acompletion)
    if _state.original_completion is not None:
        for module, name in _SYNC_TARGETS:
            setattr(module, name, _state.original_completion)
    if _state.original_router_acompletion is not None:
        _router_class().acompletion = _state.original_router_acompletion
        _state.original_router_acompletion = None
    unbind_responses_route()
    _state.original_acompletion = None
    _state.original_completion = None


__all__ = [
    "ANTIGRAVITY_USER_AGENT",
    "CODEX_URL",
    "StreamError",
    "bind_responses_route",
    "configure",
    "dispatch",
    "dispatch_responses",
    "install",
    "is_gemini_model",
    "unbind_responses_route",
    "uninstall",
]
