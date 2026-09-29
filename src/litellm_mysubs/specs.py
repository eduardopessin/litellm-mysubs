"""Building the per-provider request spec, and the process state it needs.

Extracted from `plugin.py`. What lives here is everything a request needs *before* the
transport opens a socket: the credential, the catalog, the transport itself, and the two
body builders that turn a canonical turn into each provider's wire.

The state object stays here rather than in `plugin.py` because this is what reads it on
every request; `plugin.py` only writes the patch bookkeeping into it. One object, so that
`uninstall` leaves no loose ends.
"""

from __future__ import annotations

import contextlib
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Final

import httpx
import litellm

from .credentials import refresher
from .credentials.store import CredentialStore, ProviderId
from .transport import hosts
from .transport.client import RequestSpec, Transport
from .turns import set_signature_sink
from .wire import antigravity, antigravity_models, codex

CODEX_URL: Final = "https://chatgpt.com/backend-api/codex/responses"

ANTIGRAVITY_USER_AGENT: Final = (
    "antigravity/hub/2.8.0 (aidev_client; os_type=darwin; arch=arm64; cl=963137146)"
)

#: Reasoning signatures from Gemini tool calls, to send back on the next turn. The cap
#: exists because a long session would accumulate one entry per call until the process
#: ends.
_SIGNATURE_LIMIT: Final = 512

#: This instance's transport identity (Antigravity's request ids). Codex's thread and
#: window ids are per conversation, not per process: see `codex.turn_metadata`.
_AGENT_ID: Final = uuid.uuid4().hex[:16]
_TRAJECTORY_ID: Final = uuid.uuid4().hex[:16]


class _State:
    """Module state, in a single object so that `uninstall` leaves no loose ends."""

    __slots__ = (
        "catalog",
        "original_acompletion",
        "original_completion",
        "original_router_acompletion",
        "rebound_messages_routers",
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
        #: Same, for the two Messages attributes; separate so unbinding one route never
        #: restores the other's.
        self.rebound_messages_routers: dict[int, tuple[Any, dict[str, Any]]] = {}
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
    """The token to retry with after a 401.

    `refresher.recover` decides, in omp's order: a token another worker already put in the
    store wins, a token this process minted moments ago is reused, and only otherwise is the
    refresh token spent — under the cross-process lock, and only by a store that owns it.

    A failure here returns `None`, which the transport translates into the upstream's real
    error. It does not raise: the original 401 is more informative than "I failed to
    renew" — and a store whose source cannot be re-read (a vault that does not answer, a
    file caught mid-write) must not replace the upstream's refusal with its own error.
    """
    store = _state.store
    if store is None:
        return None
    try:
        credential = await refresher.recover(
            store, _PROVIDER_IDS[provider], client_factory=httpx.AsyncClient
        )
    except Exception:
        return None
    return credential.access_token if credential is not None else None


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

    `refresher.fresh` renews a token within a minute of expiry (omp's
    `OAUTH_REFRESH_SKEW_MS`), while it still works, so there is no window between the check
    and the request; the renewal is shared with any other request of this worker that needs
    it, and locked against every other worker.

    No token at all is refused before anything is built, as omp refuses a request with no
    key: a Codex or Antigravity request went out with ``Authorization: Bearer `` — which
    httpx refuses on a real socket with ``Illegal header value b'Bearer '`` — and the client
    got a 500 naming neither the provider nor the missing step. Anthropic is the exception,
    and returns ``""``: `_delegate_kwargs` asks for its token on every call it hands to
    LiteLLM, Claude or not, and LiteLLM's own Anthropic client already refuses a Claude
    call without a key.
    """
    store = _state.store
    credential = (
        await refresher.fresh(store, _PROVIDER_IDS[provider], client_factory=httpx.AsyncClient)
        if store is not None
        else None
    )
    token = credential.access_token if credential is not None else ""
    if not token and provider != "anthropic":
        # omp: error/auth.ts :: MissingApiKeyError
        raise litellm.exceptions.AuthenticationError(
            message=f"No API key for provider: {_PROVIDER_IDS[provider]}",
            llm_provider=_PROVIDER_IDS[provider],
            model="",
        )
    return token


def _remember_signature(call_id: str, signature: str) -> None:
    signatures = _state.signatures
    signatures[call_id] = signature
    signatures.move_to_end(call_id)
    while len(signatures) > _SIGNATURE_LIMIT:
        signatures.popitem(last=False)


# `turns.py` writes thought signatures through this rather than importing the state back.
set_signature_sink(_remember_signature)


def _request_id() -> str:
    """``agent/<id>/<ts>/<traj>/<step>`` — the format the CCA expects."""
    _state.step += 1
    return f"agent/{_AGENT_ID}/{int(time.time() * 1000)}/{_TRAJECTORY_ID}/{_state.step}"


# omp: providers/openai-codex-responses.ts :: createCodexRequestContext
async def _codex_spec(model: str, messages: list[Any], extra: dict[str, Any]) -> RequestSpec:
    # No output caps to strip: `build_request_body` builds the body from scratch and does
    # not read `max_tokens`/`max_output_tokens`/`max_completion_tokens` from the kwargs. The
    # original had to delete them because it passed the kwargs on; here they never reach
    # the wire.
    token = await _access_token("codex")
    # One identity per request, shared by the body's `client_metadata` and the headers,
    # as omp builds it once and hands it to both.
    session_id = codex.session_key(model, messages, extra.get("tools"), extra)
    metadata = codex.turn_metadata(session_id, messages)
    body = codex.build_request_body(
        model,
        messages,
        tools=extra.get("tools"),
        extra=extra,
        session_id=session_id,
        metadata=metadata,
    )
    headers = codex.build_headers(
        token,
        session_id=session_id,
        metadata=metadata,
        model=str(body["model"]),
        service_tier=body.get("service_tier"),
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


# -- event interpretation, chunk shaping: see `turns.py` -----------------------


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

