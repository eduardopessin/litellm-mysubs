"""`/v1/responses` served from Codex and Antigravity, read by the real OpenAI SDK.

Everything between the client and the upstream is real: the proxy app and its Responses
endpoint, `async_data_generator`'s SSE writer, a real `litellm.Router` with the Responses
route bound on it, and `openai.AsyncOpenAI` as the client. Only the subscription's HTTP is
faked, with the events each upstream actually sends.

The encoding is omp's (`openai-responses-server.ts`), so what is pinned here is what an
OpenAI client needs from it, measured on the proxy before the port:

- no event carried `sequence_number`, which every event type in the SDK requires;
- reasoning and tool calls reached a streaming client only as whole items after the text,
  with no `reasoning_summary_*` or `function_call_arguments.*` events at all;
- a streamed `function_call` went out with the composite ``call_id|item_id`` id, which
  fails the ``^[a-zA-Z0-9_-]+$`` charset clients validate `call_id` against;
- a turn cut by the output limit ended in `response.completed`, not `response.incomplete`;
- a failure after the stream opened left no `response.failed` for the client to read;
- on the way in, `instructions`, `reasoning` and every `function_call`/`function_call_output`
  item in `input` were dropped before reaching the upstream.
"""

from __future__ import annotations

import contextlib
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
from tests.test_plugin import FakeTransport, install_transport

CODEX = "mysubs/codex/gpt-5.5"
ANTIGRAVITY = "mysubs/antigravity/gemini-3-pro"

THOUGHT_PARTS = ("The user wants the weather; ", "I should call the tool.")
THOUGHT = "".join(THOUGHT_PARTS)
TEXT_PARTS = ("Let me ", "check ", "that.")
TEXT = "".join(TEXT_PARTS)
TOOL_ARGUMENT_PARTS = ('{"city": "Pa', 'ris", "units"', ': "metric"}')
TOOL_ARGUMENTS = "".join(TOOL_ARGUMENT_PARTS)
TOOL_INPUT = {"city": "Paris", "units": "metric"}


def codex_turn(*, tool: bool = True, terminal: str = "response.completed") -> list[dict[str, Any]]:
    """A Codex Responses stream as the backend sends it, envelope events included.

    The reasoning summary and the text arrive split, the function call's arguments in
    fragments that are not JSON on their own, and `response.completed` does not repeat
    `output` (measured on the live endpoint).
    """
    events: list[dict[str, Any]] = [
        {"type": "response.created", "response": {"id": "resp_up", "status": "in_progress"}},
        {"type": "response.in_progress", "response": {"id": "resp_up", "status": "in_progress"}},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "reasoning", "id": "rs_up", "summary": []},
        },
        {
            "type": "response.reasoning_summary_part.added",
            "item_id": "rs_up",
            "output_index": 0,
            "summary_index": 0,
            "part": {"type": "summary_text", "text": ""},
        },
        *(
            {
                "type": "response.reasoning_summary_text.delta",
                "item_id": "rs_up",
                "output_index": 0,
                "summary_index": 0,
                "delta": part,
            }
            for part in THOUGHT_PARTS
        ),
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "reasoning",
                "id": "rs_up",
                "summary": [{"type": "summary_text", "text": THOUGHT}],
                "encrypted_content": "gAAAA-opaque",
            },
        },
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {"type": "message", "id": "msg_up", "role": "assistant", "content": []},
        },
        *(
            {
                "type": "response.output_text.delta",
                "item_id": "msg_up",
                "output_index": 1,
                "content_index": 0,
                "delta": part,
            }
            for part in TEXT_PARTS
        ),
        {
            "type": "response.output_item.done",
            "output_index": 1,
            "item": {
                "type": "message",
                "id": "msg_up",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": TEXT, "annotations": []}],
            },
        },
    ]
    if tool:
        item = {
            "type": "function_call",
            "id": "fc_up",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": "",
            "status": "in_progress",
        }
        events += [
            {"type": "response.output_item.added", "output_index": 2, "item": item},
            *(
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": "fc_up",
                    "output_index": 2,
                    "delta": part,
                }
                for part in TOOL_ARGUMENT_PARTS
            ),
            {
                "type": "response.output_item.done",
                "output_index": 2,
                "item": {**item, "arguments": TOOL_ARGUMENTS, "status": "completed"},
            },
        ]
    events.append(
        {
            "type": terminal,
            "response": {
                "id": "resp_up",
                "status": "incomplete" if terminal == "response.incomplete" else "completed",
                "usage": {
                    "input_tokens": 120,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 45,
                    "output_tokens_details": {"reasoning_tokens": 12},
                    "total_tokens": 165,
                },
            },
        }
    )
    return events


def antigravity_turn(*, tool: bool = True) -> list[dict[str, Any]]:
    """Cloud Code events: `thought` parts, the text split over events, and a
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
        *([{"text": part, "thought": True}] for part in THOUGHT_PARTS),
        *([{"text": part}] for part in TEXT_PARTS),
        closing,
    ]
    return [
        {
            "response": {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": event_parts},
                        **({"finishReason": "STOP"} if last else {}),
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
    pytest.param(CODEX, codex_turn, "call_1", id="codex"),
    pytest.param(ANTIGRAVITY, antigravity_turn, "call_ag_1", id="antigravity"),
]

TOOLS: list[Any] = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "units": {"type": "string"}},
        },
    }
]


@pytest.fixture(autouse=True)
def proxy(monkeypatch: pytest.MonkeyPatch) -> Iterable[litellm.Router]:
    """The proxy app answering with a real Router, the Responses route bound on it."""
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
    assert plugin.bind_responses_route(router)
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "master_key", None)
    yield router
    plugin.uninstall()


@pytest.fixture
async def client() -> AsyncIterator[openai.AsyncOpenAI]:
    """The OpenAI SDK pointed at the proxy app."""
    http_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy_server.app), base_url="http://proxy"
    )
    sdk = openai.AsyncOpenAI(
        api_key="unused", base_url="http://proxy/v1", http_client=http_client, max_retries=0
    )
    yield sdk
    await sdk.close()


async def stream_events(client: openai.AsyncOpenAI, model: str, **extra: Any) -> list[Any]:
    """Every event the SDK parses off `responses.create(stream=True)`."""
    stream = await client.responses.create(
        model=model, input="weather in Paris?", tools=TOOLS, stream=True, **extra
    )
    return [event async for event in stream]


class TestTheSdkReadsAStreamedTurn:
    @pytest.mark.parametrize(("model", "upstream", "call_id"), UPSTREAMS)
    async def test_the_final_response_carries_reasoning_text_and_the_call(
        self, client: openai.AsyncOpenAI, model: str, upstream: Any, call_id: str
    ) -> None:
        """What `client.responses.stream(...)` hands back once the turn has finished."""
        install_transport(FakeTransport(upstream()))

        async with client.responses.stream(
            model=model, input="weather in Paris?", tools=TOOLS
        ) as stream:
            final = await stream.get_final_response()

        assert final.status == "completed"
        assert final.model == model
        assert [item.type for item in final.output] == ["reasoning", "message", "function_call"]
        reasoning, _, call = final.output
        assert [part.text for part in reasoning.summary] == [THOUGHT]
        assert final.output_text == TEXT
        assert call.name == "get_weather"
        assert json.loads(call.arguments) == TOOL_INPUT
        # The call id travels without the `|item_id` half: clients validate its charset.
        assert call.call_id == call_id
        assert final.usage is not None
        assert (final.usage.input_tokens, final.usage.output_tokens) == (120, 45)
        assert final.usage.total_tokens == 165

    @pytest.mark.parametrize(("model", "upstream", "call_id"), UPSTREAMS)
    async def test_every_event_is_numbered_and_each_item_streams_its_own_deltas(
        self, client: openai.AsyncOpenAI, model: str, upstream: Any, call_id: str
    ) -> None:
        install_transport(FakeTransport(upstream()))

        events = await stream_events(client, model)

        # Every event type in the SDK declares `sequence_number` as required.
        assert [event.sequence_number for event in events] == list(range(len(events)))
        kinds = [event.type for event in events]
        assert kinds[:2] == ["response.created", "response.in_progress"]
        assert kinds[-1] == "response.completed"

        reasoning = "".join(
            e.delta for e in events if e.type == "response.reasoning_summary_text.delta"
        )
        assert reasoning == THOUGHT
        assert "response.reasoning_summary_part.done" in kinds
        text = "".join(e.delta for e in events if e.type == "response.output_text.delta")
        assert text == TEXT
        arguments = [e.delta for e in events if e.type == "response.function_call_arguments.delta"]
        assert json.loads("".join(arguments)) == TOOL_INPUT
        done = next(e for e in events if e.type == "response.function_call_arguments.done")
        assert json.loads(done.arguments) == TOOL_INPUT

        # Each item is announced, then finished, under one output index; the terminal
        # response lists the same items under the same ids.
        added = [e for e in events if e.type == "response.output_item.added"]
        finished = [e for e in events if e.type == "response.output_item.done"]
        assert [e.output_index for e in added] == [0, 1, 2]
        assert [e.item.id for e in added] == [e.item.id for e in finished]
        assert [item.id for item in events[-1].response.output] == [e.item.id for e in added]

    async def test_codex_fragments_arrive_as_the_upstream_sent_them(
        self, client: openai.AsyncOpenAI
    ) -> None:
        """One delta per upstream fragment: the stream is relayed, not replayed."""
        install_transport(FakeTransport(codex_turn()))

        events = await stream_events(client, CODEX)

        def deltas(kind: str) -> list[str]:
            return [e.delta for e in events if e.type == kind]

        assert deltas("response.reasoning_summary_text.delta") == list(THOUGHT_PARTS)
        assert deltas("response.output_text.delta") == list(TEXT_PARTS)
        assert deltas("response.function_call_arguments.delta") == list(TOOL_ARGUMENT_PARTS)

    async def test_a_turn_cut_by_the_output_limit_ends_incomplete(
        self, client: openai.AsyncOpenAI
    ) -> None:
        """`response.completed` with `status: incomplete` is a contradiction a client reading
        the event type takes at its word."""
        install_transport(FakeTransport(codex_turn(tool=False, terminal="response.incomplete")))

        events = await stream_events(client, CODEX)

        terminal = events[-1]
        assert terminal.type == "response.incomplete"
        assert terminal.response.status == "incomplete"
        assert terminal.response.incomplete_details.reason == "max_output_tokens"
        assert terminal.response.output_text == TEXT


class SpendRows(CustomLogger):
    """The rows LiteLLM's success callbacks receive, as a spend logger reads them."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[dict[str, Any]] = []

    async def async_log_success_event(
        self, kwargs: Any, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        self.rows.append(kwargs.get("standard_logging_object") or {})


async def test_the_streamed_usage_reaches_the_client_and_the_spend_row(
    client: openai.AsyncOpenAI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LiteLLM's success handler rewrites the terminal event's `usage` in place into chat
    counters, and the plugin handed it the event it then returned to the client: the SDK
    read `input_tokens: None` off a stream whose row was priced. The row is kept whole."""
    spend = SpendRows()
    monkeypatch.setattr(litellm, "callbacks", [spend])
    install_transport(FakeTransport(codex_turn(tool=False)))

    async with client.responses.stream(model=CODEX, input="hi") as stream:
        final = await stream.get_final_response()

    assert final.usage is not None
    assert (final.usage.input_tokens, final.usage.output_tokens) == (120, 45)
    assert final.usage.output_tokens_details.reasoning_tokens == 12
    assert len(spend.rows) == 1
    row = spend.rows[0]
    assert (row["prompt_tokens"], row["completion_tokens"]) == (120, 45)
    assert row["response_cost"], "the row still has to price the turn"


class TestAFailedStreamIsAFailure:
    @pytest.mark.parametrize(
        ("events", "streamed"),
        [
            pytest.param(codex_turn()[:-1], TEXT, id="cut-before-completed"),
            pytest.param(
                [
                    *codex_turn()[:9],
                    {
                        "type": "response.failed",
                        "response": {"error": {"code": "server_error", "message": "overloaded"}},
                    },
                ],
                TEXT_PARTS[0],
                id="upstream-response-failed",
            ),
        ],
    )
    async def test_the_sdk_sees_the_turn_end_in_response_failed(
        self, client: openai.AsyncOpenAI, events: list[dict[str, Any]], streamed: str
    ) -> None:
        """omp answers a stream that failed after it opened with `response.failed`.

        Without it, LiteLLM 1.101's own error frame — a bare ``data: {"error": ...}`` —
        was the only sign: the SDK raises on it, but an event-driven consumer never saw
        the turn end. 1.103 emits a `response.failed` of its own, unless the stream already
        carried a terminal event, and no error frame; so whether the raw iterator also
        raises depends on the version, and the terminal event is what is pinned.
        """
        install_transport(FakeTransport(events))

        seen: list[Any] = []
        stream = await client.responses.create(model=CODEX, input="hi", stream=True)
        with contextlib.suppress(openai.APIError):
            async for event in stream:
                seen.append(event)

        kinds = [event.type for event in seen]
        assert "response.completed" not in kinds, "a failed turn cannot complete"
        assert kinds.count("response.failed") == 1, kinds
        assert kinds[-1] == "response.failed", kinds
        assert [event.sequence_number for event in seen] == list(range(len(seen)))
        failed = seen[-1].response
        assert failed.status == "failed"
        assert failed.error is not None and failed.error.message
        # What had streamed before the failure is closed and kept, as omp does.
        assert failed.output_text == streamed

    async def test_the_stream_helper_raises_instead_of_returning_a_response(
        self, client: openai.AsyncOpenAI
    ) -> None:
        install_transport(FakeTransport(codex_turn()[:-1]))

        with pytest.raises((openai.APIError, RuntimeError)):
            async with client.responses.stream(model=CODEX, input="hi") as stream:
                await stream.get_final_response()


class TestTheSdkReadsANonStreamedTurn:
    @pytest.mark.parametrize(("model", "upstream", "call_id"), UPSTREAMS)
    async def test_the_response_carries_reasoning_text_and_the_call(
        self, client: openai.AsyncOpenAI, model: str, upstream: Any, call_id: str
    ) -> None:
        install_transport(FakeTransport(upstream()))

        response = await client.responses.create(
            model=model, input="weather in Paris?", tools=TOOLS
        )

        assert response.status == "completed"
        assert response.object == "response"
        assert response.id.startswith("resp_")
        assert [item.type for item in response.output] == ["reasoning", "message", "function_call"]
        assert [part.text for part in response.output[0].summary] == [THOUGHT]
        assert response.output_text == TEXT
        call = response.output[2]
        assert call.name == "get_weather"
        assert json.loads(call.arguments) == TOOL_INPUT
        assert call.call_id == call_id
        assert response.usage is not None
        assert (response.usage.input_tokens, response.usage.output_tokens) == (120, 45)
        reasoning_tokens = response.usage.output_tokens_details.reasoning_tokens
        assert reasoning_tokens == (12 if model == CODEX else 0)

    async def test_a_truncated_turn_is_incomplete(self, client: openai.AsyncOpenAI) -> None:
        install_transport(FakeTransport(codex_turn(tool=False, terminal="response.incomplete")))

        response = await client.responses.create(model=CODEX, input="hi")

        assert response.status == "incomplete"
        assert response.incomplete_details is not None
        assert response.incomplete_details.reason == "max_output_tokens"


FOLLOW_UP: list[Any] = [
    {"role": "user", "content": [{"type": "input_text", "text": "weather in Paris?"}]},
    {
        "type": "reasoning",
        "id": "rs_prev",
        "summary": [{"type": "summary_text", "text": THOUGHT}],
    },
    {
        "type": "message",
        "role": "assistant",
        "id": "msg_prev",
        "content": [{"type": "output_text", "text": TEXT, "annotations": []}],
    },
    {
        "type": "function_call",
        "id": "fc_prev",
        "call_id": "call_1",
        "name": "get_weather",
        "arguments": TOOL_ARGUMENTS,
    },
    {"type": "function_call_output", "call_id": "call_1", "output": '{"temp": 21}'},
]


class TestTheRequestReachesTheUpstream:
    """omp's `parseRequest` bridges every field of a Responses request a follow-up turn
    depends on; before the port they were dropped on the way to the subscription."""

    async def test_codex_receives_instructions_effort_and_the_tool_exchange(
        self, client: openai.AsyncOpenAI
    ) -> None:
        transport = install_transport(FakeTransport(codex_turn(tool=False)))

        await client.responses.create(
            model=CODEX,
            instructions="Answer briefly.",
            input=FOLLOW_UP,
            tools=TOOLS,
            reasoning={"effort": "high", "summary": "detailed"},
        )

        body = transport.specs[0].body
        assert body["instructions"] == "Answer briefly."
        assert body["reasoning"] == {"effort": "high", "summary": "detailed"}
        items = [
            {key: item.get(key) for key in ("type", "role", "call_id", "name", "arguments")}
            for item in body["input"]
        ]
        assert items == [
            {"type": "message", "role": "user", "call_id": None, "name": None, "arguments": None},
            {
                "type": "message",
                "role": "assistant",
                "call_id": None,
                "name": None,
                "arguments": None,
            },
            {
                "type": "function_call",
                "role": None,
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": TOOL_ARGUMENTS,
            },
            {
                "type": "function_call_output",
                "role": None,
                "call_id": "call_1",
                "name": None,
                "arguments": None,
            },
        ]
        assert body["input"][3]["output"] == '{"temp": 21}'

    async def test_antigravity_receives_the_system_prompt_and_the_tool_exchange(
        self, client: openai.AsyncOpenAI
    ) -> None:
        transport = install_transport(FakeTransport(antigravity_turn(tool=False)))

        await client.responses.create(
            model=ANTIGRAVITY, instructions="Answer briefly.", input=FOLLOW_UP, tools=TOOLS
        )

        request = transport.specs[0].body["request"]
        system = json.dumps(request.get("systemInstruction"))
        assert "Answer briefly." in system
        parts = [part for content in request["contents"] for part in content["parts"]]
        call = next(part["functionCall"] for part in parts if "functionCall" in part)
        assert call["name"] == "get_weather"
        assert call["args"] == TOOL_INPUT
        result = next(part["functionResponse"] for part in parts if "functionResponse" in part)
        assert result["name"] == "get_weather"
        assert "21" in json.dumps(result["response"])
