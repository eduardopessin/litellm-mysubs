"""Why a turn ended and what it cost, read off the real proxy the way a client reads it.

Two things omp settles and this plugin had hand-rolled:

- **Google finish reasons.** omp's `mapStopReasonString` knows two outcomes, ``STOP`` and
  ``MAX_TOKENS``; every other reason — SAFETY, RECITATION, MALFORMED_FUNCTION_CALL and
  whatever comes next — is an error its provider throws once the stream ends, and its
  gateway answers with the error shape of each dialect. Here the same turn reached the
  client as a finished answer: ``finish_reason: content_filter`` on chat, ``end_turn`` on
  Messages, a ``completed`` response on the Responses stream.
- **Usage.** omp's `mapGoogleUsage` falls back to ``total - candidates - thoughts`` when
  ``promptTokenCount`` is missing and clamps the cache to the prompt. Ours subtracted the
  cache from the prompt, and LiteLLM subtracts it again when it prices — a cached Gemini
  turn billed its cached tokens off the input.

Everything between the client and the subscription is real: the proxy app, a
`litellm.Router` with the plugin installed and both routes bound, the `openai` and
`anthropic` SDKs as clients, LiteLLM's own spend logging. Only the upstream HTTP is faked.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterable
from typing import Any

import httpx
import litellm
import litellm.proxy.proxy_server as proxy_server
import openai
import pytest
from litellm.integrations.custom_logger import CustomLogger

from litellm_mysubs import plugin
from tests.test_plugin import FakeTransport, codex_events, install_transport

CODEX = "mysubs/codex/gpt-5.5"
CODEX_WIRE = "openai/gpt-5.5"
ANTIGRAVITY = "mysubs/antigravity/gemini-2.5-pro"
ANTIGRAVITY_WIRE = "gemini/gemini-2.5-pro"

PARTS = ("Part ", "answer")
MESSAGES = [{"role": "user", "content": "hi"}]


def antigravity_events(finish: str, usage: dict[str, Any], *, tool: bool = False) -> list[Any]:
    """Cloud Code events: text over two events, the closing one with the finish and usage."""
    closing: list[dict[str, Any]] = [{"text": PARTS[1]}]
    if tool:
        closing.append({"functionCall": {"id": "call_1", "name": "lookup", "args": {"q": "x"}}})
    return [
        {"response": {"candidates": [{"content": {"parts": [{"text": PARTS[0]}]}}]}},
        {
            "response": {
                "candidates": [
                    {"content": {"role": "model", "parts": closing}, "finishReason": finish}
                ],
                "usageMetadata": usage,
            }
        },
    ]


class SpendRows(CustomLogger):
    """LiteLLM's own logging payload for each finished call, as a spend log reads it."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[dict[str, Any]] = []

    async def async_log_success_event(
        self, kwargs: Any, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        self.rows.append(kwargs.get("standard_logging_object") or {})

    async def one(self) -> dict[str, Any]:
        """The single row the call produced; success logging runs off the request path."""
        for _ in range(100):
            if self.rows:
                break
            await asyncio.sleep(0.02)
        assert len(self.rows) == 1, self.rows
        return self.rows[0]


@pytest.fixture(autouse=True)
def proxy(monkeypatch: pytest.MonkeyPatch) -> Iterable[SpendRows]:
    """The proxy app over a real Router serving both subscriptions on all three routes."""
    plugin.uninstall()
    router = litellm.Router(
        model_list=[
            {
                "model_name": CODEX,
                "litellm_params": {"model": CODEX_WIRE},
                "model_info": {"id": CODEX, "mysubs_provider": "openai-codex"},
            },
            {
                "model_name": ANTIGRAVITY,
                "litellm_params": {"model": ANTIGRAVITY_WIRE},
                "model_info": {"id": ANTIGRAVITY, "mysubs_provider": "google-antigravity"},
            },
        ]
    )
    plugin.install()
    assert plugin.bind_messages_route(router)
    assert plugin.bind_responses_route(router)
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "master_key", None)
    spend = SpendRows()
    # LiteLLM copies `callbacks` into its internal lists once and dedupes by class, so a
    # second test's recorder would never be registered: every list starts fresh here.
    monkeypatch.setattr(litellm, "callbacks", [spend])
    for name in ("_async_success_callback", "success_callback"):
        monkeypatch.setattr(litellm, name, [])
    yield spend
    plugin.uninstall()


def http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy_server.app), base_url="http://proxy"
    )


def openai_client() -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        api_key="unused", base_url="http://proxy/v1", http_client=http_client(), max_retries=0
    )


def anthropic_client() -> Any:
    anthropic = pytest.importorskip("anthropic")
    # Recent SDKs reject an `httpx` client and take their own fork of it.
    try:
        import httpx2 as sdk_http  # type: ignore[import-not-found]
    except ImportError:
        sdk_http = httpx
    return anthropic.AsyncAnthropic(
        api_key="unused",
        base_url="http://proxy",
        max_retries=0,
        http_client=sdk_http.AsyncClient(
            transport=sdk_http.ASGITransport(app=proxy_server.app), base_url="http://proxy"
        ),
    )


async def sse(response: AsyncIterator[bytes]) -> list[dict[str, Any]]:
    """The JSON payloads of a raw SSE body, ``[DONE]`` excluded."""
    body = b"".join([chunk async for chunk in response]).decode()
    payloads = []
    for frame in body.replace("\r\n", "\n").split("\n\n"):
        data = [line[5:].strip() for line in frame.split("\n") if line.startswith("data:")]
        if data and data[0] != "[DONE]":
            payloads.append(json.loads("\n".join(data)))
    return payloads


# -- a turn Google stopped ----------------------------------------------------------

SAFETY_USAGE = {"promptTokenCount": 120, "candidatesTokenCount": 45, "totalTokenCount": 165}
FAILED = "Generation failed with finish reason: SAFETY"


class TestAGoogleStopIsAnErrorOnEveryRoute:
    """omp: the provider throws ``Generation failed with finish reason: <reason>``; the
    gateway answers a non-streamed call with `formatError(502, "upstream_error", ...)`
    and ends a streamed one with its dialect's error event, after the text that did
    arrive. A client must never read the turn as finished."""

    @pytest.mark.parametrize(
        "reason",
        ["SAFETY", "RECITATION", "PROHIBITED_CONTENT", "MALFORMED_FUNCTION_CALL", "NEW_REASON"],
    )
    async def test_chat_answers_502_upstream_error(self, reason: str) -> None:
        """Before: HTTP 200 with ``finish_reason: content_filter`` and the text."""
        install_transport(FakeTransport(antigravity_events(reason, SAFETY_USAGE)))
        client = openai_client()

        with pytest.raises(openai.APIStatusError) as caught:
            await client.chat.completions.create(model=ANTIGRAVITY, messages=MESSAGES)
        await client.close()

        assert caught.value.status_code == 502
        body = caught.value.body
        assert isinstance(body, dict)
        assert body["type"] == "upstream_error"
        assert f"Generation failed with finish reason: {reason}" in body["message"]

    async def test_a_tool_call_does_not_mask_the_block(self) -> None:
        """omp upgrades only a benign finish to `toolUse`: SAFETY after a valid call is
        still an error. Before: HTTP 200 carrying the call as ``tool_calls``."""
        install_transport(FakeTransport(antigravity_events("SAFETY", SAFETY_USAGE, tool=True)))
        client = openai_client()

        with pytest.raises(openai.APIStatusError, match=FAILED) as caught:
            await client.chat.completions.create(model=ANTIGRAVITY, messages=MESSAGES)
        await client.close()

        assert caught.value.status_code == 502

    async def test_the_chat_stream_ends_in_an_error_after_the_text(self) -> None:
        """omp's `encodeStream`: the deltas that arrived, then ``{"error": ...}`` and no
        finish chunk. Before: a clean chunk with ``finish_reason: content_filter``."""
        install_transport(FakeTransport(antigravity_events("SAFETY", SAFETY_USAGE)))
        client = openai_client()
        received: list[str] = []
        finishes: list[str] = []

        with pytest.raises(openai.APIError, match=FAILED):
            stream = await client.chat.completions.create(
                model=ANTIGRAVITY, messages=MESSAGES, stream=True
            )
            async for chunk in stream:
                for choice in chunk.choices:
                    received.append(choice.delta.content or "")
                    if choice.finish_reason:
                        finishes.append(choice.finish_reason)
        await client.close()

        assert "".join(received) == "".join(PARTS)
        assert finishes == []

    async def test_messages_answers_502(self) -> None:
        """omp's `encodeResponse` throws on an `error` stop. Before: 200, ``end_turn``."""
        client = anthropic_client()
        install_transport(FakeTransport(antigravity_events("SAFETY", SAFETY_USAGE)))

        with pytest.raises(Exception, match=FAILED) as caught:
            await client.messages.create(model=ANTIGRAVITY, max_tokens=64, messages=MESSAGES)
        await client.close()

        assert getattr(caught.value, "status_code", None) == 502

    async def test_the_messages_stream_ends_in_an_error_event(self) -> None:
        """omp: the `error` event, which the SDK raises on. Before: `message_stop` with
        ``stop_reason: end_turn``, and the SDK returned the turn as finished."""
        client = anthropic_client()
        install_transport(FakeTransport(antigravity_events("SAFETY", SAFETY_USAGE)))
        texts: list[str] = []

        with pytest.raises(Exception, match=FAILED):
            async with client.messages.stream(
                model=ANTIGRAVITY, max_tokens=64, messages=MESSAGES
            ) as stream:
                async for text in stream.text_stream:
                    texts.append(text)
        await client.close()

        assert "".join(texts) == "".join(PARTS)

    async def test_responses_answers_502_upstream_error(self) -> None:
        """Before: HTTP 200 with a response marked ``incomplete``."""
        install_transport(FakeTransport(antigravity_events("SAFETY", SAFETY_USAGE)))
        client = openai_client()

        with pytest.raises(openai.APIStatusError, match=FAILED) as caught:
            await client.responses.create(model=ANTIGRAVITY, input="hi")
        await client.close()

        assert caught.value.status_code == 502

    async def test_the_responses_stream_does_not_complete(self) -> None:
        """omp's Responses server ends a failed stream with `response.failed` carrying the
        message. Before: `response.completed`, status ``completed``."""
        install_transport(FakeTransport(antigravity_events("SAFETY", SAFETY_USAGE)))

        async with (
            http_client() as client,
            client.stream(
                "POST", "/v1/responses", json={"model": ANTIGRAVITY, "input": "hi", "stream": True}
            ) as response,
        ):
            events = await sse(response.aiter_bytes())

        kinds = [event.get("type") for event in events]
        assert "response.completed" not in kinds
        assert any(FAILED in json.dumps(event) for event in events)


# -- what the turn cost -------------------------------------------------------------


def rates(wire: str) -> tuple[float, float, float]:
    info = litellm.get_model_info(wire)
    return (
        info["input_cost_per_token"],
        info["cache_read_input_token_cost"],
        info["output_cost_per_token"],
    )


def omp_cost(wire: str, *, uncached: int, cache_read: int, output: int) -> float:
    """omp's `calculateUsageCost` with LiteLLM's rates for the same model."""
    input_rate, cache_rate, output_rate = rates(wire)
    return uncached * input_rate + cache_read * cache_rate + output * output_rate


class Case:
    """An upstream turn and the omp ``input``/``cacheRead``/``output`` it maps to."""

    def __init__(
        self, model: str, wire: str, events: list[Any], uncached: int, cache_read: int, output: int
    ) -> None:
        self.model, self.wire, self.events = model, wire, events
        self.uncached, self.cache_read, self.output = uncached, cache_read, output

    @property
    def prompt(self) -> int:
        return self.uncached + self.cache_read

    @property
    def cost(self) -> float:
        return omp_cost(
            self.wire, uncached=self.uncached, cache_read=self.cache_read, output=self.output
        )


CASES = [
    pytest.param(
        # The under-billing: 1000 prompt tokens, 400 of them cached.
        Case(
            ANTIGRAVITY,
            ANTIGRAVITY_WIRE,
            antigravity_events(
                "STOP",
                {
                    "promptTokenCount": 1000,
                    "cachedContentTokenCount": 400,
                    "candidatesTokenCount": 50,
                    "totalTokenCount": 1050,
                },
            ),
            uncached=600,
            cache_read=400,
            output=50,
        ),
        id="antigravity-cached",
    ),
    pytest.param(
        # No promptTokenCount: prompt = 1080 - 50 - 30 = 1000; thoughts are output.
        Case(
            ANTIGRAVITY,
            ANTIGRAVITY_WIRE,
            antigravity_events(
                "STOP",
                {
                    "cachedContentTokenCount": 900,
                    "candidatesTokenCount": 50,
                    "thoughtsTokenCount": 30,
                    "totalTokenCount": 1080,
                },
            ),
            uncached=100,
            cache_read=900,
            output=80,
        ),
        id="antigravity-no-prompt-count",
    ),
    pytest.param(
        # Cache above the prompt: clamped to it, nothing uncached.
        Case(
            ANTIGRAVITY,
            ANTIGRAVITY_WIRE,
            antigravity_events(
                "STOP",
                {
                    "promptTokenCount": 300,
                    "cachedContentTokenCount": 500,
                    "candidatesTokenCount": 20,
                    "totalTokenCount": 320,
                },
            ),
            uncached=0,
            cache_read=300,
            output=20,
        ),
        id="antigravity-cache-above-prompt",
    ),
    pytest.param(
        Case(
            CODEX,
            CODEX_WIRE,
            codex_events(
                text="".join(PARTS),
                usage={
                    "input_tokens": 1000,
                    "input_tokens_details": {"cached_tokens": 400},
                    "output_tokens": 50,
                    "output_tokens_details": {"reasoning_tokens": 20},
                    "total_tokens": 1050,
                },
            ),
            uncached=600,
            cache_read=400,
            output=50,
        ),
        id="codex-cached",
    ),
]


class TestChatCountsAndBillsAsOmp:
    """omp's `openai-chat-server.ts :: buildUsage`: ``prompt_tokens = input + cacheRead``,
    the cache repeated in ``prompt_tokens_details``, total recomputed."""

    @pytest.mark.parametrize("case", CASES)
    async def test_non_streamed(self, case: Case, proxy: SpendRows) -> None:
        install_transport(FakeTransport(case.events))
        client = openai_client()

        response = await client.chat.completions.create(model=case.model, messages=MESSAGES)
        await client.close()

        usage = response.usage
        assert usage is not None
        assert usage.prompt_tokens == case.prompt
        assert usage.prompt_tokens_details is not None
        assert usage.prompt_tokens_details.cached_tokens == case.cache_read
        assert usage.completion_tokens == case.output
        assert usage.total_tokens == case.prompt + case.output
        row = await proxy.one()
        assert row["response_cost"] == pytest.approx(case.cost)

    @pytest.mark.parametrize("case", CASES)
    async def test_streamed(self, case: Case, proxy: SpendRows) -> None:
        install_transport(FakeTransport(case.events))
        client = openai_client()

        stream = await client.chat.completions.create(
            model=case.model,
            messages=MESSAGES,
            stream=True,
            stream_options={"include_usage": True},
        )
        usages = [chunk.usage async for chunk in stream if chunk.usage is not None]
        await client.close()

        assert usages, "the stream carried no usage"
        usage = usages[-1]
        assert usage.prompt_tokens == case.prompt
        assert usage.completion_tokens == case.output
        assert usage.total_tokens == case.prompt + case.output
        row = await proxy.one()
        assert row["response_cost"] == pytest.approx(case.cost)


class TestMessagesCountsAndBillsAsOmp:
    """omp's `anthropic-messages-server.ts :: encodeUsage`: ``input_tokens`` excludes the
    cache reads, which travel in ``cache_read_input_tokens``. Before, the Codex prompt
    went out whole in ``input_tokens`` and the streamed row billed the cache at the full
    input rate; the Antigravity one lost the cache from both."""

    @pytest.mark.parametrize("case", CASES)
    async def test_non_streamed(self, case: Case, proxy: SpendRows) -> None:
        client = anthropic_client()
        install_transport(FakeTransport(case.events))

        message = await client.messages.create(model=case.model, max_tokens=64, messages=MESSAGES)
        await client.close()

        assert message.usage.input_tokens == case.uncached
        assert message.usage.cache_read_input_tokens == case.cache_read
        assert message.usage.output_tokens == case.output
        row = await proxy.one()
        assert row["response_cost"] == pytest.approx(case.cost)

    @pytest.mark.parametrize("case", CASES)
    async def test_streamed(self, case: Case, proxy: SpendRows) -> None:
        client = anthropic_client()
        install_transport(FakeTransport(case.events))

        async with client.messages.stream(
            model=case.model, max_tokens=64, messages=MESSAGES
        ) as stream:
            message = await stream.get_final_message()
        await client.close()

        assert message.usage.input_tokens == case.uncached
        assert message.usage.cache_read_input_tokens == case.cache_read
        assert message.usage.output_tokens == case.output
        row = await proxy.one()
        assert row["response_cost"] == pytest.approx(case.cost)


class TestResponsesCountsAsOmp:
    """omp's `openai-responses-server.ts :: buildUsage`: ``input_tokens`` includes the
    cache reads, like the chat ``prompt_tokens``. The streamed envelope is encoded in
    `routes.py`, whose Responses port owns it."""

    @pytest.mark.parametrize("case", CASES)
    async def test_client_visible_usage(self, case: Case) -> None:
        install_transport(FakeTransport(case.events))
        client = openai_client()

        response = await client.responses.create(model=case.model, input="hi")
        await client.close()

        usage = response.usage
        assert usage is not None
        assert usage.input_tokens == case.prompt
        assert usage.output_tokens == case.output
        assert usage.total_tokens == case.prompt + case.output

    @pytest.mark.parametrize("case", CASES[:3])
    async def test_the_antigravity_row_bills_as_omp(self, case: Case, proxy: SpendRows) -> None:
        """Served through the chat turn, so priced from the chat usage."""
        install_transport(FakeTransport(case.events))
        client = openai_client()

        await client.responses.create(model=case.model, input="hi")
        await client.close()

        row = await proxy.one()
        assert row["response_cost"] == pytest.approx(case.cost)
