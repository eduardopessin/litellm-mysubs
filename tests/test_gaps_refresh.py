"""Credential trouble on a real request: renewal that fails, a credential that vanishes, none.

The unit tests on `_refresh`/`_access_token` check what those functions return. What a
client sees depends on everything after that — the real `Transport` and its one retry on
401, the Router, the proxy's error mapping, the SDK — so here the request goes through the
real proxy app, a real `litellm.Router` and the real `Transport`, and only the hosts behind
it are fake: the subscription's inference endpoint and its OAuth token endpoint, both
answered by an `httpx.MockTransport`.

What is judged is what the client got and what the upstream received: never an empty
``Bearer``, never a loop of rejected retries, and the cause in the error the client reads.
"""

from __future__ import annotations

import asyncio
import json
import time
import types
from collections.abc import Callable, Iterable
from typing import Any, Final

import httpx
import litellm
import litellm.main
import litellm.proxy.proxy_server as proxy_server
import openai
import pytest

from litellm_mysubs import plugin, specs
from litellm_mysubs.credentials.store import Credential, ProviderId
from litellm_mysubs.transport.client import Transport
from litellm_mysubs.transport.hosts import HostRotation
from tests.test_plugin import FakeStore, codex_events, gemini_events

CODEX: Final = "mysubs/codex/gpt-5.5"
ANTIGRAVITY: Final = "mysubs/antigravity/gemini-3-pro"
CODEX_HOST: Final = "chatgpt.com"
TOKEN_HOSTS: Final = ("auth.openai.com", "oauth2.googleapis.com")
EXPIRED_BODY: Final = '{"detail": "Your authentication token has expired"}'

_LITELLM_ENTRY_POINTS: Final = {
    (module, name): getattr(module, name)
    for module in (litellm, litellm.main)
    for name in ("acompletion", "completion")
}


class OwningStore(FakeStore):
    """A store that owns the refresh token, as the proxy's file store does."""

    owns_refresh = True

    def __init__(self, credentials: dict[ProviderId, Credential] | None = None) -> None:
        super().__init__(credentials)
        #: What another worker writes to the shared source; `reload` picks it up.
        self.elsewhere: dict[ProviderId, Credential] = {}
        self.reload_error: Exception | None = None

    def reload(self) -> bool:
        super().reload()
        if self.reload_error is not None:
            raise self.reload_error
        self._credentials.update(self.elsewhere)
        return bool(self.elsewhere)


class Hosts:
    """Every host the plugin talks to, as one `MockTransport` handler.

    ``inference`` answers the subscription's endpoint, ``token`` the OAuth refresh; both
    see the request, so a test can act mid-flight (delete the credential, rotate it).
    """

    def __init__(
        self,
        inference: Callable[[httpx.Request], httpx.Response],
        token: Callable[[httpx.Request], httpx.Response] | None = None,
    ) -> None:
        self.inference = inference
        self.token = token or (lambda _: httpx.Response(400, json={"error": "invalid_grant"}))
        #: ``Authorization`` of each inference request, in order.
        self.bearers: list[str | None] = []
        self.token_calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.host in TOKEN_HOSTS:
            self.token_calls += 1
            return self.token(request)
        if request.url.path.endswith(":fetchAvailableModels"):
            return httpx.Response(200, json={"models": {}})
        self.bearers.append(request.headers.get("authorization"))
        return self.inference(request)

    def client(self, *_: Any, **__: Any) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self))


def sse(events: Iterable[dict[str, Any]]) -> httpx.Response:
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


def rejected(_: httpx.Request) -> httpx.Response:
    return httpx.Response(401, text=EXPIRED_BODY)


@pytest.fixture(autouse=True)
def proxy(monkeypatch: pytest.MonkeyPatch) -> Iterable[None]:
    plugin.uninstall()
    # A Router registers its deployments' models into the process-wide cost map; a name
    # left there reprices other tests' calls (measured: `gemini/gemini-3-flash` made the
    # `-agent` cost identity resolve to it).
    monkeypatch.setattr(litellm, "model_cost", dict(litellm.model_cost))
    for (module, name), function in _LITELLM_ENTRY_POINTS.items():
        monkeypatch.setattr(module, name, function)
    router = litellm.Router(
        model_list=[
            {
                "model_name": CODEX,
                "litellm_params": {"model": "openai/gpt-5.5"},
                "model_info": {"id": CODEX, "mysubs_provider": "openai-codex"},
            },
            {
                "model_name": ANTIGRAVITY,
                "litellm_params": {"model": "gemini/gemini-3-pro"},
                "model_info": {"id": ANTIGRAVITY, "mysubs_provider": "google-antigravity"},
            },
            # Unmarked and without credentials: served by name, i.e. handed to LiteLLM
            # through `_delegate_kwargs`, which asks for the Claude token on the way.
            {
                "model_name": "house-model",
                "litellm_params": {"model": "ollama/house", "mock_response": "from the house"},
            },
        ]
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "master_key", None)
    monkeypatch.setattr(specs._state, "catalog", specs.antigravity_models.ModelCatalog())
    plugin.install()
    yield
    plugin.uninstall()


def serve(hosts: Hosts, store: FakeStore | None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Wire the real `Transport` and the OAuth client to the fake hosts.

    The transport is built as `specs._transport` builds it, only with the fake hosts'
    client in place of the network.
    """
    monkeypatch.setattr(specs._state, "store", store)
    transport = Transport(client=hosts.client(), refresh=specs._refresh, rotation=HostRotation())
    monkeypatch.setattr(specs._state, "transport", transport)
    monkeypatch.setattr(specs, "httpx", types.SimpleNamespace(AsyncClient=hosts.client))


async def ask(model: str, *, stream: bool) -> str:
    """The answer text the OpenAI SDK assembles, or the SDK's exception."""
    client = openai.AsyncOpenAI(
        api_key="unused",
        base_url="http://proxy/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy_server.app), base_url="http://proxy"
        ),
    )
    messages: Any = [{"role": "user", "content": "hi"}]
    try:
        # A ceiling well above any legitimate path here: a refresh that loops or waits on
        # a dead token must fail the test, not hang the suite.
        async with asyncio.timeout(30):
            if not stream:
                answer = await client.chat.completions.create(model=model, messages=messages)
                return answer.choices[0].message.content or ""
            chunks = await client.chat.completions.create(
                model=model, messages=messages, stream=True
            )
            return "".join([c.choices[0].delta.content or "" async for c in chunks if c.choices])
    finally:
        await client.close()


def expired(token: str = "AT-old") -> Credential:
    return Credential(
        provider="openai-codex",
        access_token=token,
        refresh_token="RT",
        expires_at=time.time() - 10,
    )


def valid(token: str = "AT-old") -> Credential:
    return Credential(
        provider="openai-codex",
        access_token=token,
        refresh_token="RT",
        expires_at=time.time() + 3600,
    )


STREAMING = [pytest.param(False, id="whole"), pytest.param(True, id="stream")]


@pytest.mark.parametrize("stream", STREAMING)
class TestRenewalThatFails:
    async def test_the_upstream_401_reaches_the_client(
        self, stream: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Expired token, and the OAuth endpoint refuses the refresh token. The request
        still goes out once with the token there is — a clock skew is not proof it is dead
        — and the upstream's own refusal is what the client reads. One attempt: retrying a
        credential the server rejects is a loop of rejections at network speed.

        `invalid_grant` means the grant is dead, so the credential is dropped, as omp
        disables the row: the refresh token is presented exactly once, not again on the 401
        and not again by every sweep after it."""
        credential = expired()
        store = OwningStore({"openai-codex": credential})
        hosts = Hosts(rejected)
        serve(hosts, store, monkeypatch)

        with pytest.raises(openai.APIError, match="authentication token has expired"):
            await ask(CODEX, stream=stream)

        assert hosts.bearers == ["Bearer AT-old"]
        assert hosts.token_calls == 1
        assert store.get("openai-codex") is None

    async def test_a_source_that_cannot_be_reread_still_surfaces_the_401(
        self, stream: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The 401 sends `_refresh` back to the store, and the store's source — a vault, a
        file mid-write — can fail. That failure is not the client's answer: the request
        was refused by the upstream, and that refusal is what says what to fix."""
        store = OwningStore({"openai-codex": valid()})
        store.reload_error = RuntimeError("vault unreachable")
        hosts = Hosts(rejected)
        serve(hosts, store, monkeypatch)

        with pytest.raises(openai.APIError, match="authentication token has expired") as caught:
            await ask(CODEX, stream=stream)

        assert "vault unreachable" not in str(caught.value)
        assert hosts.bearers == ["Bearer AT-old"]

    async def test_a_store_that_does_not_own_the_token_never_renews(
        self, stream: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A reader store returns nothing new and must not spend the single-use refresh
        token; the 401 goes to the client after the one attempt."""
        store = FakeStore({"openai-codex": expired()})
        hosts = Hosts(rejected)
        serve(hosts, store, monkeypatch)

        with pytest.raises(openai.APIError, match="authentication token has expired"):
            await ask(CODEX, stream=stream)

        assert hosts.bearers == ["Bearer AT-old"]
        assert hosts.token_calls == 0


@pytest.mark.parametrize("stream", STREAMING)
class TestCredentialChangingMidFlight:
    async def test_deleted_mid_flight_is_not_retried_with_an_empty_token(
        self, stream: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Disconnect pressed while a request is in the air: the upstream refuses, the
        re-read finds nothing, and the refusal goes to the client. What must not happen is
        a retry carrying ``Bearer `` — or no refusal at all."""
        store = OwningStore({"openai-codex": valid()})

        def refuse_after_disconnect(request: httpx.Request) -> httpx.Response:
            store.delete("openai-codex")
            return rejected(request)

        hosts = Hosts(refuse_after_disconnect)
        serve(hosts, store, monkeypatch)

        with pytest.raises(openai.APIError, match="authentication token has expired"):
            await ask(CODEX, stream=stream)

        assert hosts.bearers == ["Bearer AT-old"]

    async def test_rotated_by_another_worker_the_request_is_answered(
        self, stream: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Another worker renewed while this one held the old token. The 401 re-reads the
        source, the retry carries the new token, and the client gets the answer — without
        this worker spending the refresh token."""
        store = OwningStore({"openai-codex": valid()})
        store.elsewhere = {"openai-codex": valid("AT-new")}

        def by_token(request: httpx.Request) -> httpx.Response:
            if request.headers.get("authorization") == "Bearer AT-new":
                return sse(codex_events(text="renewed"))
            return rejected(request)

        hosts = Hosts(by_token)
        serve(hosts, store, monkeypatch)

        assert await ask(CODEX, stream=stream) == "renewed"
        assert hosts.bearers == ["Bearer AT-old", "Bearer AT-new"]
        assert hosts.token_calls == 0


@pytest.mark.parametrize("stream", STREAMING)
@pytest.mark.parametrize(
    ("model", "provider"),
    [
        pytest.param(CODEX, "openai-codex", id="codex"),
        pytest.param(ANTIGRAVITY, "google-antigravity", id="antigravity"),
    ],
)
class TestNoCredential:
    """Before, a subscription nobody connected sent ``Authorization: Bearer `` upstream —
    httpx refuses that header on a real socket with ``Illegal header value`` — and the
    client got a 500 that named neither the provider nor the missing step. omp refuses a
    request without a key before building it (`MissingApiKeyError`)."""

    async def test_not_connected_is_a_401_that_names_the_provider(
        self, model: str, provider: str, stream: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        hosts = Hosts(lambda _: sse(codex_events()))
        serve(hosts, FakeStore(), monkeypatch)

        with pytest.raises(
            openai.AuthenticationError, match=f"No API key for provider: {provider}"
        ):
            await ask(model, stream=stream)

        assert hosts.bearers == []

    async def test_no_store_configured_is_the_same_401(
        self, model: str, provider: str, stream: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        hosts = Hosts(lambda _: sse(codex_events()))
        serve(hosts, None, monkeypatch)

        with pytest.raises(
            openai.AuthenticationError, match=f"No API key for provider: {provider}"
        ):
            await ask(model, stream=stream)

        assert hosts.bearers == []

    async def test_a_credential_with_an_empty_token_is_the_same_401(
        self, model: str, provider: str, stream: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = FakeStore(
            {
                provider: Credential(provider=provider, access_token="", project_id="p")  # type: ignore[arg-type]
            }
        )
        hosts = Hosts(lambda _: sse(gemini_events()))
        serve(hosts, store, monkeypatch)

        with pytest.raises(
            openai.AuthenticationError, match=f"No API key for provider: {provider}"
        ):
            await ask(model, stream=stream)

        assert hosts.bearers == []


async def test_models_that_are_not_ours_do_not_need_a_claude_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every call the plugin hands to LiteLLM asks for the Claude token on the way, Claude
    or not. A proxy with no Claude subscription must keep serving its other models."""
    hosts = Hosts(rejected)
    serve(hosts, FakeStore(), monkeypatch)

    assert await ask("house-model", stream=False) == "from the house"
    assert hosts.bearers == []
