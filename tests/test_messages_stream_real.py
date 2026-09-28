"""`/v1/messages` streamed from Codex and Antigravity, read the way a client reads it.

Everything between the client and the upstream is real: the proxy app and its Anthropic
endpoint, `ProxyBaseLLMRequestProcessing`'s SSE writer, a real `litellm.Router` with the
Messages route bound on it. Only the subscription's HTTP is faked, with the events each
upstream actually sends.

Dict-level tests passed while this route was broken four ways at once, measured on the
proxy before the fix:

- text deltas went out as ``{"type": "text_delta", "text_delta": "ok"}`` — the key is
  ``text``;
- a streamed tool call ended the turn with ``stop_reason: tool_use`` and no ``tool_use``
  block, and reasoning never arrived as a ``thinking`` block;
- every frame was a bare ``data:`` line, which the Anthropic SDK drops unread: it
  dispatches on the SSE ``event`` field;
- a failure after the stream opened left only LiteLLM's own error frame, also a bare
  ``data:`` line, so the SDK saw the stream simply end.

So the stream is judged here as bytes off the socket, each event validated against
LiteLLM's own Anthropic types, and accumulated the way the SDK accumulates it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import httpx
import litellm
import litellm.proxy.proxy_server as proxy_server
import pytest
from litellm.types.llms.anthropic import (
    AnthropicFinishReason,
    AnthropicResponseContentBlockToolUse,
    ContentBlockDelta,
    ContentBlockStop,
    ContentJsonBlockDelta,
    ContentTextBlockDelta,
    ContentThinkingBlockDelta,
    ContentThinkingSignatureBlockDelta,
    MessageBlockDelta,
    MessageStartBlock,
    TextBlock,
)
from litellm.types.llms.openai import ChatCompletionThinkingBlock
from pydantic import TypeAdapter

from litellm_mysubs import plugin
from tests.test_plugin import FakeTransport, install_transport

CODEX = "mysubs/codex/gpt-5.5"
ANTIGRAVITY = "mysubs/antigravity/gemini-3-pro"

THOUGHT = "The user wants the weather; I should call the tool."
TEXT_PARTS = ("Let me ", "check ", "that.")
TEXT = "".join(TEXT_PARTS)
TOOL_ARGUMENT_PARTS = ('{"city": "Pa', 'ris", "units": "metric"}')
TOOL_INPUT = {"city": "Paris", "units": "metric"}


def codex_turn(*, tool: bool) -> list[dict[str, Any]]:
    """A Responses stream as Codex sends it: reasoning summary, message text, then a
    function call whose arguments arrive in fragments."""
    events: list[dict[str, Any]] = [
        {"type": "response.reasoning_summary_text.delta", "delta": THOUGHT},
        *({"type": "response.output_text.delta", "delta": part} for part in TEXT_PARTS),
    ]
    if tool:
        item = {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "get_weather"}
        events += [
            {"type": "response.output_item.added", "item": item},
            *(
                {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "delta": d}
                for d in TOOL_ARGUMENT_PARTS
            ),
            {
                "type": "response.output_item.done",
                "item": {**item, "arguments": "".join(TOOL_ARGUMENT_PARTS)},
            },
        ]
    events.append(
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": {"input_tokens": 120, "output_tokens": 45},
            },
        }
    )
    return events


def antigravity_turn(*, tool: bool, finish: str = "STOP") -> list[dict[str, Any]]:
    """Cloud Code events: a `thought` part, the text split over events, and a
    `functionCall` part on the closing event, the only one carrying `usageMetadata`."""
    closing: list[dict[str, Any]] = []
    if tool:
        closing.append(
            {
                "functionCall": {"id": "call_ag_1", "name": "get_weather", "args": TOOL_INPUT},
                "thoughtSignature": "sig-1",
            }
        )
    parts: list[list[dict[str, Any]]] = [
        [{"text": THOUGHT, "thought": True}],
        *([{"text": part}] for part in TEXT_PARTS),
        closing,
    ]
    return [
        {
            "response": {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": event_parts},
                        **({"finishReason": finish} if last else {}),
                    }
                ],
                "usageMetadata": (
                    {"promptTokenCount": 120, "candidatesTokenCount": 45, "totalTokenCount": 165}
                    if last
                    else {}
                ),
            }
        }
        for event_parts, last in ((p, i == len(parts) - 1) for i, p in enumerate(parts))
    ]


UPSTREAMS = [
    pytest.param(CODEX, codex_turn, id="codex"),
    pytest.param(ANTIGRAVITY, antigravity_turn, id="antigravity"),
]


@pytest.fixture(autouse=True)
def proxy(monkeypatch: pytest.MonkeyPatch) -> Iterable[litellm.Router]:
    """The proxy app answering with a real Router, the Messages route bound on it."""
    plugin.uninstall()
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
        ]
    )
    assert plugin.bind_messages_route(router)
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "master_key", None)
    yield router
    plugin.uninstall()


REQUEST: dict[str, Any] = {
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "weather in Paris?"}],
    "tools": [
        {
            "name": "get_weather",
            "description": "Current weather for a city.",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "units": {"type": "string"}},
            },
        }
    ],
}


async def post_messages(model: str, *, stream: bool) -> httpx.Response:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy_server.app), base_url="http://proxy"
    ) as client:
        response = await client.post(
            "/v1/messages", json={**REQUEST, "model": model, "stream": stream}
        )
    assert response.status_code == 200, response.text
    return response


def sse_events(body: str) -> list[tuple[str | None, dict[str, Any]]]:
    """``(event, data)`` per SSE frame, parsed the way the SDK's decoder does."""
    events: list[tuple[str | None, dict[str, Any]]] = []
    for raw in body.replace("\r\n", "\n").split("\n\n"):
        if not raw.strip():
            continue
        name: str | None = None
        data: list[str] = []
        for line in raw.split("\n"):
            field, _, value = line.partition(":")
            value = value.removeprefix(" ")
            if field == "event":
                name = value
            elif field == "data":
                data.append(value)
        events.append((name, json.loads("\n".join(data))))
    return events


_ADAPTERS: dict[str, TypeAdapter[Any]] = {
    "message_start": TypeAdapter(MessageStartBlock),
    "content_block_delta": TypeAdapter(ContentBlockDelta),
    "content_block_stop": TypeAdapter(ContentBlockStop),
    "message_delta": TypeAdapter(MessageBlockDelta),
}

#: The block each delta type belongs to, and the LiteLLM type its payload must satisfy.
#: `ContentBlockDelta` alone accepts a thinking delta shaped as a text one, because
#: `ContentTextBlockDelta.type` is a bare `str`.
_DELTAS: dict[str, tuple[str, TypeAdapter[Any]]] = {
    "text_delta": ("text", TypeAdapter(ContentTextBlockDelta)),
    "thinking_delta": ("thinking", TypeAdapter(ContentThinkingBlockDelta)),
    "signature_delta": ("thinking", TypeAdapter(ContentThinkingSignatureBlockDelta)),
    "input_json_delta": ("tool_use", TypeAdapter(ContentJsonBlockDelta)),
}

#: Block openers. Text and thinking use the TypedDicts LiteLLM's own stream adapter
#: emits (`ContentBlockContentBlockDict`); its `ToolUseBlock` requires a `caller` key that
#: neither Anthropic's SDK nor that adapter sends, so tool use is checked against the
#: response model instead.
_BLOCKS: dict[str, TypeAdapter[Any]] = {
    "text": TypeAdapter(TextBlock),
    "thinking": TypeAdapter(ChatCompletionThinkingBlock),
    "tool_use": TypeAdapter(AnthropicResponseContentBlockToolUse),
}


def accumulate(body: str) -> dict[str, Any]:
    """The finished message a streaming client ends up holding, or an assertion error.

    Enforces what the SDK's accumulator relies on: the event name equals the payload's
    `type`, one `message_start` first, blocks opened in index order and each delta
    landing on an open block of its own kind, everything closed before `message_delta`,
    `message_stop` last.
    """
    events = [(name, data) for name, data in sse_events(body) if data.get("type") != "ping"]
    assert events, "no events at all"
    for name, data in events:
        assert name == data.get("type"), (
            f"the SDK dispatches on the SSE event name and drops this frame: "
            f"event={name!r} type={data.get('type')!r}"
        )

    kinds = [data["type"] for _, data in events]
    assert kinds[0] == "message_start" and kinds.count("message_start") == 1
    assert kinds[-2:] == ["message_delta", "message_stop"], kinds

    message: dict[str, Any] = {}
    open_blocks: dict[int, dict[str, Any]] = {}
    fragments: dict[int, list[str]] = {}
    for _, event in events:
        kind = event["type"]
        if kind in _ADAPTERS:
            _ADAPTERS[kind].validate_python(event)
        if kind == "message_start":
            message = {**event["message"], "content": []}
        elif kind == "content_block_start":
            block = dict(event["content_block"])
            _BLOCKS[block["type"]].validate_python(block)
            assert event["index"] == len(message["content"]), "block indices must be contiguous"
            assert not open_blocks, "a block opened before the previous one stopped"
            message["content"].append(block)
            open_blocks[event["index"]] = block
            fragments[event["index"]] = []
        elif kind == "content_block_delta":
            delta = event["delta"]
            owner, adapter = _DELTAS[delta["type"]]
            adapter.validate_python(delta)
            block = open_blocks[event["index"]]
            assert block["type"] == owner, f"{delta['type']} on a {block['type']} block"
            if delta["type"] == "text_delta":
                block["text"] += delta["text"]
            elif delta["type"] == "thinking_delta":
                block["thinking"] += delta["thinking"]
            elif delta["type"] == "signature_delta":
                block["signature"] = delta["signature"]
            else:
                fragments[event["index"]].append(delta["partial_json"])
        elif kind == "content_block_stop":
            block = open_blocks.pop(event["index"])
            if block["type"] == "tool_use":
                joined = "".join(fragments[event["index"]])
                block["input"] = json.loads(joined) if joined else {}
        elif kind == "message_delta":
            assert not open_blocks, "message_delta while a block is still open"
            assert event["delta"]["stop_reason"] in AnthropicFinishReason.__args__
            message["stop_reason"] = event["delta"]["stop_reason"]
            message["usage"] = {**message["usage"], **event["usage"]}
    return message


def texts(message: dict[str, Any], kind: str) -> list[str]:
    field = "text" if kind == "text" else "thinking"
    return [block[field] for block in message["content"] if block["type"] == kind]


class TestAStreamedMessagesTurnIsWhatAClientReads:
    @pytest.mark.parametrize(("model", "upstream"), UPSTREAMS)
    async def test_thinking_text_and_tool_call_all_arrive(
        self, model: str, upstream: Any
    ) -> None:
        transport = install_transport(FakeTransport(upstream(tool=True)))

        response = await post_messages(model, stream=True)
        message = accumulate(response.text)

        assert transport.specs, "the request has to reach the subscription's wire"
        assert [block["type"] for block in message["content"]] == [
            "thinking",
            "text",
            "tool_use",
        ]
        assert texts(message, "thinking") == [THOUGHT]
        assert texts(message, "text") == [TEXT]
        tool = message["content"][2]
        assert tool["name"] == "get_weather"
        assert tool["input"] == TOOL_INPUT
        assert message["stop_reason"] == "tool_use"
        assert message["usage"]["input_tokens"] == 120
        assert message["usage"]["output_tokens"] == 45

    @pytest.mark.parametrize(("model", "upstream"), UPSTREAMS)
    async def test_a_turn_without_a_tool_ends_the_turn(self, model: str, upstream: Any) -> None:
        install_transport(FakeTransport(upstream(tool=False)))

        message = accumulate((await post_messages(model, stream=True)).text)

        assert [block["type"] for block in message["content"]] == ["thinking", "text"]
        assert texts(message, "text") == [TEXT]
        assert message["stop_reason"] == "end_turn"

    @pytest.mark.parametrize(("model", "upstream"), UPSTREAMS)
    async def test_the_stream_agrees_with_the_non_streamed_answer(
        self, model: str, upstream: Any
    ) -> None:
        """The tool call a client must answer, and why the turn stopped, cannot depend on
        whether it asked for a stream."""
        install_transport(FakeTransport(upstream(tool=True)))

        streamed = accumulate((await post_messages(model, stream=True)).text)
        whole = (await post_messages(model, stream=False)).json()

        def tools(message: dict[str, Any]) -> list[dict[str, Any]]:
            return [
                {key: block[key] for key in ("id", "name", "input")}
                for block in message["content"]
                if block["type"] == "tool_use"
            ]

        assert tools(streamed) == tools(whole)
        assert texts(streamed, "thinking") == texts(whole, "thinking")
        assert streamed["stop_reason"] == whole["stop_reason"] == "tool_use"


    async def test_a_blocked_answer_does_not_claim_a_stop_sequence(self) -> None:
        """Antigravity's server-side blocks reach the route as `content_filter`, which was
        sent as `stop_sequence` — telling the client one of its own stop sequences matched,
        when it sent none. omp's `mapStopReasonOut` has no such case and ends the turn."""
        install_transport(FakeTransport(antigravity_turn(tool=False, finish="SAFETY")))

        message = accumulate((await post_messages(ANTIGRAVITY, stream=True)).text)
        whole = (await post_messages(ANTIGRAVITY, stream=False)).json()

        assert message["stop_reason"] == whole["stop_reason"] == "end_turn"

    async def test_a_failure_mid_stream_reaches_the_client_as_an_error_event(self) -> None:
        """A Codex stream cut before `response.completed` is a failure, raised after
        `message_start` already left. LiteLLM writes its own error frame as a bare `data:`
        line; the client has to get omp's `error` event to know the turn failed."""
        install_transport(FakeTransport(codex_turn(tool=False)[:-1]))

        events = sse_events((await post_messages(CODEX, stream=True)).text)

        errors = [data for name, data in events if name == "error"]
        assert len(errors) == 1, events
        assert errors[0]["type"] == "error"
        assert errors[0]["error"]["type"] == "api_error"
        assert "response.completed" in errors[0]["error"]["message"]
        assert events[0][0] == "message_start"
        assert "message_stop" not in [name for name, _ in events], "a failed turn cannot stop"

    @pytest.mark.parametrize(
        ("choice", "codex_wire", "antigravity_wire"),
        [
            ({"type": "auto"}, "auto", {"mode": "VALIDATED"}),
            ({"type": "any"}, "required", {"mode": "ANY"}),
            ({"type": "none"}, "none", {"mode": "NONE"}),
            (
                {"type": "tool", "name": "get_weather"},
                {"type": "function", "name": "get_weather"},
                {"mode": "ANY", "allowedFunctionNames": ["get_weather"]},
            ),
        ],
    )
    async def test_the_tool_choice_reaches_the_upstream_in_its_own_shape(
        self, choice: dict[str, Any], codex_wire: Any, antigravity_wire: dict[str, Any]
    ) -> None:
        """Messages' ``tool_choice`` went through untranslated, and Codex answered every
        explicit one with 400 ``Invalid value: 'auto'`` (``'any'``, ``'tool'``) — measured on
        the live gateway on 0.1.14. omp's `mapToolChoice` maps it to the canonical form."""
        for model, upstream in ((CODEX, codex_turn), (ANTIGRAVITY, antigravity_turn)):
            transport = install_transport(FakeTransport(upstream(tool=True)))
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=proxy_server.app), base_url="http://proxy"
            ) as client:
                response = await client.post(
                    "/v1/messages",
                    json={**REQUEST, "model": model, "stream": True, "tool_choice": choice},
                )
            assert response.status_code == 200, response.text
            body = transport.specs[0].body
            if model == CODEX:
                assert body["tool_choice"] == codex_wire
            else:
                assert body["request"]["toolConfig"]["functionCallingConfig"] == antigravity_wire


def sdk_client() -> tuple[Any, Any]:
    """The Anthropic SDK pointed at the proxy app, or a skip where it is not installed."""
    anthropic = pytest.importorskip("anthropic")
    # Recent SDKs reject an `httpx` client and take their own fork of it.
    try:
        import httpx2 as sdk_http  # type: ignore[import-not-found]
    except ImportError:
        sdk_http = httpx
    http_client = sdk_http.AsyncClient(
        transport=sdk_http.ASGITransport(app=proxy_server.app), base_url="http://proxy"
    )
    client = anthropic.AsyncAnthropic(
        api_key="unused", base_url="http://proxy", http_client=http_client
    )
    return anthropic, client


@pytest.mark.parametrize(("model", "upstream"), UPSTREAMS)
async def test_the_anthropic_sdk_assembles_the_turn(model: str, upstream: Any) -> None:
    """The consumer Anthropic-native clients actually run. Before the frames carried an
    ``event:`` line, `get_final_message()` failed here with no snapshot: the SDK had
    received the whole stream and dispatched none of it."""
    _, client = sdk_client()
    install_transport(FakeTransport(upstream(tool=True)))

    async with client.messages.stream(model=model, **REQUEST) as stream:
        final = await stream.get_final_message()
    await client.close()

    assert [block.type for block in final.content] == ["thinking", "text", "tool_use"]
    assert final.content[0].thinking == THOUGHT
    assert final.content[1].text == TEXT
    assert final.content[2].name == "get_weather"
    assert final.content[2].input == TOOL_INPUT
    assert final.stop_reason == "tool_use"


async def test_the_anthropic_sdk_raises_on_a_failed_stream() -> None:
    """Without the `error` event the SDK returned the truncated turn as if it had
    finished: `message_start` had arrived and nothing it recognised said otherwise."""
    anthropic, client = sdk_client()
    install_transport(FakeTransport(codex_turn(tool=False)[:-1]))

    with pytest.raises(anthropic.APIStatusError, match=r"response\.completed"):
        async with client.messages.stream(model=CODEX, **REQUEST) as stream:
            await stream.get_final_message()
    await client.close()
