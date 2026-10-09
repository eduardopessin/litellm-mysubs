"""What omp 18.8.6 changed for Claude, read off the wire.

omp 18.8.6 (pi-catalog `classes/anthropic.kdl`, `providers/anthropic.ts :: buildParams`):

- Opus 5.5 binds signed thinking to the exact preceding conversation, like Sonnet 5.5: every
  request asks for ``block_binding: {prefix_mismatch_behavior: "drop_block"}`` with its beta,
  and a turn that does not reason names ``thinking: {type: "adaptive"}`` to carry it.
- The sampling restriction moved into catalog rules: adaptive Claude (Opus 4.7+,
  Sonnet/Fable/Mythos 5+, Haiku 5.5+) never gets ``temperature``, ``top_p`` or ``top_k``.
  mysubs sent them to those models whenever thinking was off.

A chat request crosses the wire twice (LiteLLM's inner hop), so these go through the real
proxy, Router and LiteLLM client to a local stand-in for ``api.anthropic.com``; the harness
is `test_anthropic_thinking_omp1844`'s.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

import litellm
import litellm.proxy.proxy_server as proxy_server
import pytest

from litellm_mysubs import plugin
from litellm_mysubs.credentials.store import Credential
from tests.test_anthropic_thinking_omp1844 import (
    _LITELLM_ENTRY_POINTS,
    AnthropicHost,
    chat,
    effort_of,
)
from tests.test_plugin import FakeStore

BINDING_BETA: Final = "thinking-binding-controls-2026-08-01"
MODELS: Final = ("claude-opus-5-5", "claude-opus-5", "claude-sonnet-4-6")


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
                "litellm_params": {"model": f"anthropic/{name}", "api_base": upstream.base},
                "model_info": {
                    "id": name,
                    "mysubs_provider": "anthropic",
                    "max_output_tokens": 128000,
                },
            }
            for name in MODELS
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


class TestOpus55BindsItsThinking:
    async def test_an_off_turn_carries_the_binding(self, host: AnthropicHost) -> None:
        """What omp 18.8.6 sends for this request (its `onPayload`): adaptive named to carry
        the binding, no `display`, the lowest effort; the caller's `max_tokens` stands. The
        second pass used to read the block as reasoning and raise `max_tokens` to 12192."""
        await chat("claude-opus-5-5", max_tokens=100)

        body, headers = host.last()
        assert body["thinking"] == {
            "type": "adaptive",
            "block_binding": {"prefix_mismatch_behavior": "drop_block"},
        }
        assert effort_of(body) == "low"
        assert body["max_tokens"] == 100
        assert BINDING_BETA in headers["anthropic-beta"]


class TestSamplingParams:
    async def test_a_model_that_refuses_them_does_not_get_them(self, host: AnthropicHost) -> None:
        """Opus 5 answers them with 400 (omp: `supports-sampling-params #false`)."""
        await chat("claude-opus-5", temperature=0.3, top_p=0.5)

        body, _ = host.last()
        assert "temperature" not in body
        assert "top_p" not in body
        assert effort_of(body) == "low"

    async def test_the_rest_still_get_them(self, host: AnthropicHost) -> None:
        await chat("claude-sonnet-4-6", temperature=0.3)

        body, _ = host.last()
        assert body["temperature"] == 0.3
