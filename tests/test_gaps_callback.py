"""`MySubs`, driven the way the proxy drives it.

`config.yaml` loads the callback into ``litellm.callbacks`` and the proxy calls it from
`ProxyLogging`: ``async_pre_call_hook`` before the Router sees a request,
``async_post_call_success_hook`` once the response is built. Both are only reachable that
way — `ProxyLogging` skips a callback whose class does not override the hook, and reads the
quota headers off the ``_hidden_params`` LiteLLM's own provider client fills — so every test
here posts to the real proxy app with the callback registered, and judges what the client
received and what the page's usage view holds afterwards. Only the subscription hosts are
fake.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Final

import httpx
import litellm
import litellm.main
import litellm.proxy.proxy_server as proxy_server
import openai
import pytest

from litellm_mysubs import MySubs, plugin
from litellm_mysubs.bootstrap import ui_path
from litellm_mysubs.credentials.store import Credential
from litellm_mysubs.ui import install as ui_install
from tests.test_plugin import FakeStore, FakeTransport, codex_events

CLAUDE: Final = "mysubs/claudecode/claude-sonnet-4-5"
CODEX: Final = "mysubs/codex/gpt-5.5"

#: What Anthropic answers a Claude Max request with, measured names and scales: the
#: utilisation is a fraction, the reset an epoch.
QUOTA_HEADERS: Final = {
    "anthropic-ratelimit-unified-5h-utilization": "0.03",
    "anthropic-ratelimit-unified-5h-reset": "1790000000",
    "anthropic-ratelimit-unified-7d-utilization": "0.24",
    "anthropic-ratelimit-unified-7d-reset": "1790500000",
}

_LITELLM_ENTRY_POINTS: Final = {
    (module, name): getattr(module, name)
    for module in (litellm, litellm.main)
    for name in ("acompletion", "completion")
}


class AnthropicHost:
    """A local server standing in for ``api.anthropic.com``.

    LiteLLM's Anthropic client talks to it over a real socket, which is what puts the
    response headers into ``_hidden_params["additional_headers"]`` with the
    ``llm_provider-`` prefix — the only way those headers reach the callback.
    """

    def __init__(self, headers: dict[str, str]) -> None:
        self.tokens: list[str] = []
        tokens = self.tokens

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                self.rfile.read(int(self.headers["content-length"]))
                tokens.append(self.headers.get("authorization") or self.headers.get("x-api-key"))
                data = json.dumps(
                    {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-sonnet-4-5",
                        "content": [{"type": "text", "text": "from claude"}],
                        "stop_reason": "end_turn",
                        "stop_sequence": None,
                        "usage": {"input_tokens": 3, "output_tokens": 2},
                    }
                ).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                for name, value in headers.items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def anthropic_host() -> Iterable[AnthropicHost]:
    host = AnthropicHost(QUOTA_HEADERS)
    yield host
    host.close()


@pytest.fixture(autouse=True)
def proxy(monkeypatch: pytest.MonkeyPatch) -> Iterable[None]:
    """The proxy app as a normal start leaves it, minus whatever the test mounts on it.

    The callback mounts `/mysubs` on ``proxy_server.app`` and publishes its service; both
    are process globals, so the route list and the service are put back afterwards. The
    mount the package's import-time ``proxy_handler_instance`` already left there is taken
    out for the test, so the callback under test is the one that mounts — as on a proxy
    whose config names ``litellm_mysubs.MySubs``.
    """
    plugin.uninstall()
    # A Router registers its deployments' models into the process-wide cost map; a name
    # left there reprices other tests' calls (measured: `gemini/gemini-3-flash` made the
    # `-agent` cost identity resolve to it).
    monkeypatch.setattr(litellm, "model_cost", dict(litellm.model_cost))
    for (module, name), function in _LITELLM_ENTRY_POINTS.items():
        monkeypatch.setattr(module, name, function)
    monkeypatch.setattr(proxy_server, "master_key", None)
    monkeypatch.setattr(
        proxy_server.app.router,
        "routes",
        [r for r in proxy_server.app.router.routes if getattr(r, "path", None) != ui_path()],
    )
    monkeypatch.setattr(ui_install, "_SERVICE", None)
    monkeypatch.setattr(litellm, "callbacks", [])
    yield
    plugin.uninstall()
    plugin.unbind_responses_route()
    plugin.unbind_messages_route()


async def _stop_refresher() -> None:
    service = ui_install.shared_service()
    if service is not None:
        await service.stop_refresher()


def register(store: FakeStore) -> MySubs:
    """What the proxy does with ``callbacks: ["litellm_mysubs.MySubs"]``."""
    callback = MySubs(store=store)
    litellm.callbacks.append(callback)
    return callback


def openai_client() -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        api_key="unused",
        base_url="http://proxy/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy_server.app), base_url="http://proxy"
        ),
    )


def use_router(monkeypatch: pytest.MonkeyPatch, *deployments: dict[str, Any]) -> litellm.Router:
    router = litellm.Router(model_list=list(deployments))
    monkeypatch.setattr(proxy_server, "llm_router", router)
    return router


def codex_deployment() -> dict[str, Any]:
    return {
        "model_name": CODEX,
        "litellm_params": {"model": "openai/gpt-5.5"},
        "model_info": {"id": CODEX, "mysubs_provider": "openai-codex"},
    }


def claude_deployment(api_base: str) -> dict[str, Any]:
    return {
        "model_name": CLAUDE,
        "litellm_params": {"model": "anthropic/claude-sonnet-4-5", "api_base": api_base},
        "model_info": {"id": CLAUDE, "mysubs_provider": "anthropic"},
    }


class TestThePreCallHookIsTheStartupTrigger:
    async def test_the_first_request_is_already_served_by_the_subscription(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A proxy restarted with a subscription connected has nothing patched until the
        callback runs. The hook runs before the Router is called, so the very request that
        triggers `setup()` must reach the subscription — not LiteLLM's native OpenAI client,
        which has no key for this deployment."""
        use_router(monkeypatch, codex_deployment())
        transport = FakeTransport(codex_events(text="from codex"))
        plugin.configure(transport=transport)
        register(
            FakeStore({"openai-codex": Credential(provider="openai-codex", access_token="tok")})
        )
        client = openai_client()

        answer = await client.chat.completions.create(
            model=CODEX, messages=[{"role": "user", "content": "hi"}]
        )
        await client.close()
        await _stop_refresher()

        assert answer.choices[0].message.content == "from codex"
        assert len(transport.specs) == 1
        assert transport.specs[0].headers["Authorization"] == "Bearer tok"

    async def test_the_hook_leaves_the_request_as_it_arrived(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The hook's `None` is what keeps the plugin off the write path: the upstream gets
        the conversation the client sent, nothing added or dropped by the hook."""
        use_router(monkeypatch, codex_deployment())
        transport = FakeTransport(codex_events())
        plugin.configure(transport=transport)
        register(
            FakeStore({"openai-codex": Credential(provider="openai-codex", access_token="tok")})
        )
        client = openai_client()

        await client.chat.completions.create(
            model=CODEX,
            messages=[
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "first"},
            ],
        )
        await client.close()
        await _stop_refresher()

        body = transport.specs[0].body
        assert body["instructions"] == "be brief"
        sent = [
            part["text"]
            for item in body["input"]
            for part in item.get("content") or []
            if part.get("type") == "input_text"
        ]
        assert sent == ["first"]

    async def test_disabled_the_hook_patches_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`MYSUBS_DISABLE` is the operator's off switch: with it set, a connected
        subscription still must not take a request."""
        monkeypatch.setenv("MYSUBS_DISABLE", "1")
        use_router(monkeypatch, codex_deployment())
        transport = FakeTransport(codex_events())
        plugin.configure(transport=transport)
        register(
            FakeStore({"openai-codex": Credential(provider="openai-codex", access_token="tok")})
        )
        client = openai_client()

        with pytest.raises(openai.APIError):
            await client.chat.completions.create(
                model=CODEX, messages=[{"role": "user", "content": "hi"}]
            )
        await client.close()

        assert transport.specs == []


class TestThePostCallHookFeedsTheUsageView:
    async def test_claude_quota_headers_reach_the_card(
        self, monkeypatch: pytest.MonkeyPatch, anthropic_host: AnthropicHost
    ) -> None:
        """A subscription has no usage endpoint of its own on the request path: the only
        live reading is the headers on each answer. They arrive through LiteLLM's Anthropic
        client, prefixed, and the page must show them as percentages of each window."""
        use_router(monkeypatch, claude_deployment(anthropic_host.base))
        register(FakeStore({"anthropic": Credential(provider="anthropic", access_token="tok")}))
        client = openai_client()

        answer = await client.chat.completions.create(
            model=CLAUDE, messages=[{"role": "user", "content": "hi"}]
        )
        await client.close()
        await _stop_refresher()

        assert answer.choices[0].message.content == "from claude"
        service = ui_install.shared_service()
        assert service is not None
        card = next(c for c in service.cards() if c.provider == "anthropic")
        windows = {w.label: w for w in card.usage.windows}
        assert windows["5h"].used_percent == pytest.approx(3.0)
        assert windows["7d"].used_percent == pytest.approx(24.0)
        assert windows["7d"].resets_at == 1790500000

    async def test_an_answer_without_quota_headers_keeps_the_last_reading(
        self, monkeypatch: pytest.MonkeyPatch, anthropic_host: AnthropicHost
    ) -> None:
        """An answer that says nothing about the quota is no proof the quota changed: the
        card must not blink back to "no data" between requests."""
        use_router(monkeypatch, claude_deployment(anthropic_host.base))
        register(FakeStore({"anthropic": Credential(provider="anthropic", access_token="tok")}))
        client = openai_client()
        await client.chat.completions.create(
            model=CLAUDE, messages=[{"role": "user", "content": "hi"}]
        )
        silent = AnthropicHost({})
        try:
            proxy_server.llm_router.model_list[0]["litellm_params"]["api_base"] = silent.base
            await client.chat.completions.create(
                model=CLAUDE, messages=[{"role": "user", "content": "again"}]
            )
        finally:
            silent.close()
        await client.close()
        await _stop_refresher()

        service = ui_install.shared_service()
        assert service is not None
        card = next(c for c in service.cards() if c.provider == "anthropic")
        assert {w.label: w.used_percent for w in card.usage.windows} == pytest.approx(
            {"5h": 3.0, "7d": 24.0}
        )

    async def test_a_model_that_is_not_a_subscription_records_nothing(
        self, monkeypatch: pytest.MonkeyPatch, anthropic_host: AnthropicHost
    ) -> None:
        """An operator model answering with the same header names must not be read as the
        subscription's quota — the card would show someone else's usage."""
        use_router(
            monkeypatch,
            {
                "model_name": "house-model",
                "litellm_params": {
                    "model": "anthropic/house-model",
                    "api_base": anthropic_host.base,
                    "api_key": "sk-ant-api03-operator",
                },
            },
        )
        register(FakeStore({"anthropic": Credential(provider="anthropic", access_token="tok")}))
        client = openai_client()

        await client.chat.completions.create(
            model="house-model", messages=[{"role": "user", "content": "hi"}]
        )
        await client.close()
        await _stop_refresher()

        assert anthropic_host.tokens == ["sk-ant-api03-operator"]
        service = ui_install.shared_service()
        assert service is not None
        assert all(not card.usage.known for card in service.cards())
