"""Thinking, effort and tool choice as omp 18.4.4 sends them to Claude, read off the wire.

omp 18.4.4 (`providers/anthropic.ts :: buildParams`, pi-catalog `classes/anthropic.kdl`):

- A request that asks for no reasoning is omp's thinking-off request. Sonnet 5.5 answers
  ``thinking: {type: "disabled"}`` with 400 and gets ``between_tools``, its lowest setting;
  other adaptive-only models cannot switch adaptive thinking off, so the lowest effort is
  pinned; budget models get ``disabled``. mysubs sent none of it: a plain or
  ``reasoning_effort: "none"`` request to `claude-opus-5` went out with no field at all and
  thought at the default effort.
- Sonnet 5.5 refuses a ``tool_choice`` that forces a tool, like Opus 5.5 and Fable; omp
  downgrades it to ``auto`` rather than let the turn die with 400.
- A budget that does not fit under the model's ceiling shrinks, and below Anthropic's 1024
  minimum the turn goes out with thinking disabled.

What the client asked for — an effort, a ``thinking`` object, its own
``output_config.effort`` — is never replaced by the thinking-off request, nor by the
default effort. An explicit effort on an adaptive model used to reach the wire as `medium`:
on chat because LiteLLM's inner hop runs the wire a second time and that pass replaced the
first one's effort with the default; on Messages because a client's own
``output_config.effort`` beside ``thinking`` was replaced the same way.

Everything between the client and Anthropic is real: the proxy app, a `litellm.Router` with
the plugin installed and the Messages route bound, the `openai` and `anthropic` SDKs, and
LiteLLM's own Anthropic client talking over a socket to a local stand-in for
``api.anthropic.com`` that records what it received.
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

from litellm_mysubs import plugin
from litellm_mysubs.credentials.store import Credential
from tests.test_plugin import FakeStore

MESSAGES: Final = [{"role": "user", "content": "hi"}]
EFFORT_BETA: Final = "effort-2025-11-24"
#: An adaptive-only model, Sonnet 5.5, a budget model with its real ceiling, and one whose
#: declared ceiling leaves no room for a thinking budget.
MODELS: Final = {
    "claude-opus-5": 128000,
    "claude-sonnet-5-5": 128000,
    "claude-haiku-4-5": 64000,
    "claude-haiku-4-5-small": 4096,
}

_LITELLM_ENTRY_POINTS: Final = {
    (module, name): getattr(module, name)
    for module in (litellm, litellm.main)
    for name in ("acompletion", "completion")
}


class AnthropicHost:
    """A local server standing in for ``api.anthropic.com``: records body and headers."""

    def __init__(self) -> None:
        self.requests: list[tuple[dict[str, Any], dict[str, str]]] = []
        seen = self.requests

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                seen.append((body, {k.lower(): v for k, v in self.headers.items()}))
                data = json.dumps(
                    {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "model": body.get("model"),
                        "content": [{"type": "text", "text": "ok"}],
                        "stop_reason": "end_turn",
                        "stop_sequence": None,
                        "usage": {"input_tokens": 3, "output_tokens": 1},
                    }
                ).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def last(self) -> tuple[dict[str, Any], dict[str, str]]:
        assert len(self.requests) == 1, self.requests
        return self.requests[0]


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch) -> Iterable[AnthropicHost]:
    """The proxy over a real Router serving our Claude deployments from the fake host."""
    upstream = AnthropicHost()
    plugin.uninstall()
    monkeypatch.setattr(litellm, "model_cost", dict(litellm.model_cost))
    for (module, name), function in _LITELLM_ENTRY_POINTS.items():
        monkeypatch.setattr(module, name, function)
    router = litellm.Router(
        model_list=[
            {
                "model_name": name,
                "litellm_params": {
                    "model": f"anthropic/{name.removesuffix('-small')}",
                    "api_base": upstream.base,
                },
                "model_info": {
                    "id": name,
                    "mysubs_provider": "anthropic",
                    "max_output_tokens": ceiling,
                },
            }
            for name, ceiling in MODELS.items()
        ]
    )
    plugin.configure(
        store=FakeStore({"anthropic": Credential(provider="anthropic", access_token="tok")})
    )
    plugin.install()
    assert plugin.bind_messages_route(router)
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "master_key", None)
    yield upstream
    plugin.uninstall()
    plugin.unbind_messages_route()
    upstream.server.shutdown()
    upstream.server.server_close()


async def chat(model: str, **params: Any) -> None:
    client = openai.AsyncOpenAI(
        api_key="unused",
        base_url="http://proxy/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy_server.app), base_url="http://proxy"
        ),
    )
    await client.chat.completions.create(model=model, messages=MESSAGES, **params)


async def messages(model: str, **params: Any) -> None:
    anthropic = pytest.importorskip("anthropic")
    # Recent SDKs reject an `httpx` client and take their own fork of it.
    try:
        import httpx2 as sdk_http  # type: ignore[import-not-found]
    except ImportError:
        sdk_http = httpx
    client = anthropic.AsyncAnthropic(
        api_key="unused",
        base_url="http://proxy",
        max_retries=0,
        http_client=sdk_http.AsyncClient(
            transport=sdk_http.ASGITransport(app=proxy_server.app), base_url="http://proxy"
        ),
    )
    await client.messages.create(model=model, max_tokens=512, messages=MESSAGES, **params)


def effort_of(body: dict[str, Any]) -> object:
    return (body.get("output_config") or {}).get("effort")


class TestARequestThatDoesNotReason:
    @pytest.mark.parametrize("params", [{}, {"reasoning_effort": "none"}], ids=["plain", "none"])
    async def test_an_adaptive_only_model_is_pinned_to_the_lowest_effort(
        self, host: AnthropicHost, params: dict[str, Any]
    ) -> None:
        """Omitting `thinking` leaves adaptive thinking on at the default effort; omp pins
        `low` and sends the effort beta the field needs."""
        await chat("claude-opus-5", **params)

        body, headers = host.last()
        assert "thinking" not in body
        assert effort_of(body) == "low"
        assert EFFORT_BETA in headers["anthropic-beta"]

    @pytest.mark.parametrize("params", [{}, {"reasoning_effort": "none"}], ids=["plain", "none"])
    async def test_sonnet_5_5_gets_between_tools_and_no_effort(
        self, host: AnthropicHost, params: dict[str, Any]
    ) -> None:
        """`disabled` is a 400 on Sonnet 5.5; `between_tools` is its lowest setting, and
        pinning `low` there would cap the whole turn, not only thinking."""
        await chat("claude-sonnet-5-5", **params)

        body, _ = host.last()
        assert body["thinking"] == {"type": "between_tools"}
        assert "output_config" not in body

    async def test_between_tools_drops_the_sampling_parameters(self, host: AnthropicHost) -> None:
        """omp sends temperature/top_p only with thinking absent or disabled."""
        await chat("claude-sonnet-5-5", temperature=0.3, top_p=0.5)

        body, _ = host.last()
        assert body["thinking"] == {"type": "between_tools"}
        assert "temperature" not in body
        assert "top_p" not in body

    async def test_a_budget_model_gets_thinking_disabled_and_no_effort(
        self, host: AnthropicHost
    ) -> None:
        await chat("claude-haiku-4-5")

        body, headers = host.last()
        assert body["thinking"] == {"type": "disabled"}
        assert "output_config" not in body
        assert EFFORT_BETA not in headers["anthropic-beta"]

    async def test_the_messages_route_without_thinking_is_the_same_request(
        self, host: AnthropicHost
    ) -> None:
        await messages("claude-sonnet-5-5")

        body, _ = host.last()
        assert body["thinking"] == {"type": "between_tools"}


class TestWhatTheClientAskedForStands:
    async def test_an_explicit_effort_is_sent_as_asked(self, host: AnthropicHost) -> None:
        await chat("claude-opus-5", reasoning_effort="high")

        body, _ = host.last()
        assert body["thinking"]["type"] == "adaptive"
        assert effort_of(body) == "high"

    async def test_an_explicit_effort_on_sonnet_5_5_is_not_between_tools(
        self, host: AnthropicHost
    ) -> None:
        await chat("claude-sonnet-5-5", reasoning_effort="high")

        body, _ = host.last()
        assert body["thinking"]["type"] != "between_tools"

    async def test_the_clients_own_effort_is_not_pinned(self, host: AnthropicHost) -> None:
        """A Messages client that sets `output_config.effort` chose its effort; omp's
        Messages server reads it as a reasoning request, not as thinking off."""
        await messages("claude-opus-5", extra_body={"output_config": {"effort": "high"}})

        body, _ = host.last()
        assert "thinking" not in body
        assert effort_of(body) == "high"

    async def test_the_clients_own_effort_is_not_turned_into_between_tools(
        self, host: AnthropicHost
    ) -> None:
        await messages("claude-sonnet-5-5", extra_body={"output_config": {"effort": "high"}})

        body, _ = host.last()
        assert "thinking" not in body

    async def test_the_clients_effort_wins_over_the_adaptive_default(
        self, host: AnthropicHost
    ) -> None:
        await messages(
            "claude-opus-5",
            thinking={"type": "adaptive"},
            extra_body={"output_config": {"effort": "xhigh"}},
        )

        body, _ = host.last()
        assert body["thinking"]["type"] == "adaptive"
        assert effort_of(body) == "xhigh"

    async def test_between_tools_yields_to_an_effort_it_cannot_run_at(
        self, host: AnthropicHost
    ) -> None:
        """`between_tools` is a 400 at `xhigh`/`max`: with thinking turned off at that
        effort, omp sends no `thinking` and the model runs its default adaptive thinking."""
        await messages(
            "claude-sonnet-5-5",
            thinking={"type": "disabled"},
            extra_body={"output_config": {"effort": "max"}},
        )

        body, _ = host.last()
        assert "thinking" not in body


class TestForcedToolChoice:
    TOOL: Final = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}

    @pytest.mark.parametrize(
        "choice", ["required", {"type": "function", "function": {"name": "f"}}]
    )
    async def test_sonnet_5_5_gets_auto_instead_of_a_400(
        self, host: AnthropicHost, choice: object
    ) -> None:
        """400 `tool_choice: type "tool" and "any" are not supported for this model`."""
        await chat("claude-sonnet-5-5", tools=[self.TOOL], tool_choice=choice)

        body, _ = host.last()
        assert body["tool_choice"]["type"] == "auto"

    async def test_the_messages_route_is_downgraded_too(self, host: AnthropicHost) -> None:
        await messages(
            "claude-sonnet-5-5",
            tools=[{"name": "f", "input_schema": {"type": "object"}}],
            tool_choice={"type": "any", "disable_parallel_tool_use": True},
        )

        body, _ = host.last()
        assert body["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}

    async def test_a_model_that_accepts_forcing_keeps_it(self, host: AnthropicHost) -> None:
        await chat("claude-opus-5", tools=[self.TOOL], tool_choice="required")

        body, _ = host.last()
        assert body["tool_choice"]["type"] == "any"


class TestTheBudgetFitsTheCeiling:
    async def test_no_room_for_any_budget_sends_the_turn_without_thinking(
        self, host: AnthropicHost
    ) -> None:
        """A 4096 ceiling leaves 96 tokens after the output buffer, under Anthropic's 1024
        minimum. This used to go out with budget 8192 and `max_tokens` 4096, which Anthropic
        refuses: `max_tokens` must be greater than `thinking.budget_tokens`."""
        await chat("claude-haiku-4-5-small", reasoning_effort="high", max_tokens=100)

        body, _ = host.last()
        assert body["thinking"] == {"type": "disabled"}
        assert body["max_tokens"] == 100

    async def test_the_clients_max_tokens_reaches_the_wire(self, host: AnthropicHost) -> None:
        """LiteLLM's inner hop passes `max_completion_tokens=None` beside the client's
        `max_tokens`; reading the empty key sent the model's full ceiling instead of 100."""
        await chat("claude-opus-5", max_tokens=100)

        body, _ = host.last()
        assert body["max_tokens"] == 100
