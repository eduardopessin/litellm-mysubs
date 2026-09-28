"""The upstream stream is let go when a reader refuses the turn, on ``/v1/responses`` too.

`test_gaps_turn_brakes.py` shows it on the chat route. The streamed Responses route reads the
transport in a loop of its own, so the same failure — a brake trips, the upstream is left
suspended and keeps generating the turn the client was just told had failed — is checked
there as well, through the real proxy app and a real `litellm.Router` with the route bound.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from typing import Any, Final

import httpx
import litellm
import litellm.proxy.proxy_server as proxy_server
import openai
import pytest

from litellm_mysubs import plugin
from litellm_mysubs.transport.client import RequestSpec
from tests.test_plugin import FakeTransport, install_transport

CODEX: Final = "mysubs/codex/gpt-5.5"


class EndlessWhitespace(FakeTransport):
    """Codex stuck streaming blanks into a tool call's arguments, for as long as it is read."""

    def __init__(self) -> None:
        super().__init__()
        self.pulled = 0
        self.released = False

    async def stream(self, spec: RequestSpec) -> AsyncIterator[dict[str, Any]]:
        self.specs.append(spec)
        item = {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "write"}
        try:
            yield {"type": "response.output_item.added", "item": item}
            while self.pulled < 5000:
                self.pulled += 1
                yield {
                    "type": "response.function_call_arguments.delta",
                    "item_id": "fc_1",
                    "delta": " ",
                }
                await asyncio.sleep(0)
        finally:
            self.released = True


@pytest.fixture(autouse=True)
def proxy(monkeypatch: pytest.MonkeyPatch) -> Iterable[None]:
    plugin.uninstall()
    # A Router registers its deployments' models into the process-wide cost map.
    monkeypatch.setattr(litellm, "model_cost", dict(litellm.model_cost))
    router = litellm.Router(
        model_list=[
            {
                "model_name": CODEX,
                "litellm_params": {"model": "openai/gpt-5.5"},
                "model_info": {"id": CODEX, "mysubs_provider": "openai-codex"},
            }
        ]
    )
    assert plugin.bind_responses_route(router)
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "master_key", None)
    yield
    plugin.unbind_responses_route()
    plugin.uninstall()


async def test_a_braked_responses_stream_releases_the_upstream() -> None:
    transport = EndlessWhitespace()
    install_transport(transport)
    client = openai.AsyncOpenAI(
        api_key="unused",
        base_url="http://proxy/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy_server.app), base_url="http://proxy"
        ),
    )

    # LiteLLM 1.101 ends the SSE with an error frame, which the SDK raises; 1.103 sends a
    # `response.failed` event, which the SDK yields. Either way the client is told why.
    failure = ""
    try:
        async with asyncio.timeout(60):
            stream = await client.responses.create(model=CODEX, input="go", stream=True)
            async for event in stream:
                if event.type == "response.failed" and event.response.error is not None:
                    failure = event.response.error.message
    except openai.APIError as error:
        failure = str(error)
    await client.close()

    assert "whitespace loop" in failure
    assert transport.pulled < 300
    assert transport.released
