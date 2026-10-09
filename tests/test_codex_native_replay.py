"""A chat client's Codex history goes back as the responses it came from.

omp replays each Codex response's own output items on the next request
(`providers/openai-codex-responses.ts :: convertMessages`, the `providerPayload` branch):
the encrypted reasoning and each message's `phase`. A chat client sends back only text and
tool calls, so before this every step of a tool loop reached the model with its earlier
reasoning gone and its progress notes looking like final answers.

Same harness as `test_codex_omp1844.py`: the real proxy app, the `openai` SDK as client, a
real Router and `Transport`; only the Codex backend is fake, and it answers each request
with the next scripted response.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any, Final

import httpx
import litellm.proxy.proxy_server as proxy_server
import openai
import pytest

from litellm_mysubs import specs
from litellm_mysubs.wire import codex
from tests.test_codex_omp1844 import CODEX, STREAMING, WEATHER_TOOL, router, serve  # noqa: F401

USER: Final = {"role": "user", "content": "weather in Paris?"}


def _summary(text: str) -> list[dict[str, str]]:
    return [{"type": "summary_text", "text": text}]


def _response(*items: dict[str, Any]) -> list[dict[str, Any]]:
    """A Codex stream: each item added, then finished as given, then completed."""
    events: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        opened = {**item, "content": []} if item["type"] == "message" else dict(item)
        if item["type"] == "function_call":
            opened["arguments"] = ""
        events.append({"type": "response.output_item.added", "output_index": index, "item": opened})
        if item["type"] == "message":
            for part in item["content"]:
                events.append(
                    {
                        "type": "response.output_text.delta",
                        "item_id": item["id"],
                        "output_index": index,
                        "delta": part["text"],
                    }
                )
        if item["type"] == "function_call":
            events.append(
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": item["id"],
                    "output_index": index,
                    "delta": item["arguments"],
                }
            )
        events.append({"type": "response.output_item.done", "output_index": index, "item": item})
    events.append({"type": "response.completed", "response": {"status": "completed", "usage": {}}})
    return events


def _message(item_id: str, text: str, phase: str) -> dict[str, Any]:
    return {
        "type": "message",
        "id": item_id,
        "role": "assistant",
        "status": "completed",
        "phase": phase,
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


TOOL_STEP: Final = _response(
    {
        "type": "reasoning",
        "id": "rs_1",
        "summary": _summary("look it up"),
        "encrypted_content": "enc-1",
    },
    _message("msg_1", "Checking the weather.", "commentary"),
    {
        "type": "function_call",
        "id": "fc_1",
        "status": "completed",
        "call_id": "call_1",
        "name": "get_weather",
        "arguments": '{"city":"Paris"}',
    },
)
ANSWER: Final = _response(
    {
        "type": "reasoning",
        "id": "rs_2",
        "summary": _summary("sum up"),
        "encrypted_content": "enc-2",
    },
    _message("msg_2", "Sunny.", "final_answer"),
)


class Backend:
    """Answers each request with the next scripted response, recording the bodies."""

    def __init__(self, *responses: list[dict[str, Any]]) -> None:
        self.script = list(responses)
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        events = (
            self.script.pop(0) if self.script else _response(_message("m", "ok", "final_answer"))
        )
        text = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})


async def ask(messages: list[dict[str, Any]], *, stream: bool) -> dict[str, Any]:
    """One chat turn through the proxy, returned as the assistant message a client keeps."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy_server.app), base_url="http://proxy"
    ) as http:
        client = openai.AsyncOpenAI(
            api_key="sk-anything", base_url="http://proxy/v1", http_client=http, max_retries=0
        )
        if not stream:
            answer = await client.chat.completions.create(
                model=CODEX,
                messages=messages,
                tools=[WEATHER_TOOL],  # type: ignore[arg-type, list-item]
            )
            return answer.choices[0].message.model_dump(exclude_none=True)
        text = ""
        calls: dict[int, dict[str, Any]] = {}
        chunks = await client.chat.completions.create(
            model=CODEX,
            messages=messages,
            tools=[WEATHER_TOOL],
            stream=True,  # type: ignore[arg-type, list-item]
        )
        async for chunk in chunks:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            text += delta.content or ""
            for call in delta.tool_calls or []:
                kept = calls.setdefault(
                    call.index,
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                )
                kept["id"] += call.id or ""
                if call.function is not None:
                    kept["function"]["name"] += call.function.name or ""
                    kept["function"]["arguments"] += call.function.arguments or ""
        message: dict[str, Any] = {"role": "assistant", "content": text or None}
        if calls:
            message["tool_calls"] = [calls[index] for index in sorted(calls)]
        return message


def _result(message: dict[str, Any], text: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": message["tool_calls"][0]["id"], "content": text}


def _assistant_input(body: dict[str, Any]) -> list[dict[str, Any]]:
    """The replayed assistant side of an input: everything that is not client input."""
    return [
        item
        for item in body["input"]
        if item.get("role") not in ("user", "developer")
        and item.get("type") != "function_call_output"
    ]


REPLAYED_TOOL_STEP: Final = [
    {"type": "reasoning", "summary": _summary("look it up"), "encrypted_content": "enc-1"},
    {
        "type": "message",
        "role": "assistant",
        "phase": "commentary",
        "content": [{"type": "output_text", "text": "Checking the weather.", "annotations": []}],
    },
    {
        "type": "function_call",
        "call_id": "call_1",
        "name": "get_weather",
        "arguments": '{"city":"Paris"}',
    },
]


@pytest.mark.usefixtures("router")
class TestNativeReplay:
    @pytest.mark.parametrize("stream", STREAMING)
    async def test_a_tool_step_goes_back_as_the_response_it_was(self, stream: bool) -> None:
        """The follow-up carries the step's encrypted reasoning and its `commentary`
        phase, ids and output-only statuses stripped, and the call keeps pairing with
        its result."""
        backend = Backend(TOOL_STEP)
        serve(backend)

        step = await ask([USER], stream=stream)
        await ask([USER, step, _result(step, "sun")], stream=stream)

        follow_up = backend.bodies[1]
        assert _assistant_input(follow_up) == REPLAYED_TOOL_STEP
        assert follow_up["input"][-1] == {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "sun",
        }

    @pytest.mark.parametrize("stream", STREAMING)
    async def test_a_final_answer_goes_back_with_its_reasoning_and_phase(
        self, stream: bool
    ) -> None:
        backend = Backend(TOOL_STEP, ANSWER)
        serve(backend)

        step = await ask([USER], stream=stream)
        history = [USER, step, _result(step, "sun")]
        answer = await ask(history, stream=stream)
        await ask([*history, answer, {"role": "user", "content": "and tomorrow?"}], stream=stream)

        assert _assistant_input(backend.bodies[2]) == [
            *REPLAYED_TOOL_STEP,
            {"type": "reasoning", "summary": _summary("sum up"), "encrypted_content": "enc-2"},
            {
                "type": "message",
                "role": "assistant",
                "phase": "final_answer",
                "content": [{"type": "output_text", "text": "Sunny.", "annotations": []}],
            },
        ]

    async def test_an_edited_history_is_re_encoded(self) -> None:
        """A client that changed the call's arguments gets its own turn sent, not the
        kept response: the reasoning described a call the history no longer holds."""
        backend = Backend(TOOL_STEP)
        serve(backend)

        step = await ask([USER], stream=False)
        step["tool_calls"][0]["function"]["arguments"] = '{"city": "Lyon"}'
        await ask([USER, step, _result(step, "rain")], stream=False)

        assert _assistant_input(backend.bodies[1]) == [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Checking the weather.", "annotations": []}
                ],
                "status": "completed",
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"city":"Lyon"}',
            },
        ]

    async def test_an_empty_final_answer_is_not_replayed(self) -> None:
        """gpt-5.6 can end a turn whose text all went to `commentary` with an empty
        `final_answer`; that empty slot is dropped, as omp drops it."""
        backend = Backend(
            _response(
                {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "enc-1"},
                _message("msg_1", "Done: it is sunny.", "commentary"),
                {"type": "reasoning", "id": "rs_2", "summary": [], "encrypted_content": "enc-2"},
                _message("msg_2", "", "final_answer"),
            )
        )
        serve(backend)

        answer = await ask([USER], stream=False)
        await ask([USER, answer, {"role": "user", "content": "thanks"}], stream=False)

        assert _assistant_input(backend.bodies[1]) == [
            {"type": "reasoning", "summary": [], "encrypted_content": "enc-1"},
            {
                "type": "message",
                "role": "assistant",
                "phase": "commentary",
                "content": [
                    {"type": "output_text", "text": "Done: it is sunny.", "annotations": []}
                ],
            },
        ]


class TestScope:
    """Native items are model-bound: the reasoning is encrypted for the model and account
    that produced it (omp replays them only for the same model)."""

    def _kept(self) -> OrderedDict[str, codex.NativeTurn]:
        kept = codex.native_turn(
            codex.replay_scope("acct-1", "gpt-6-luna"),
            [
                {"type": "reasoning", "id": "rs_1", "encrypted_content": "enc-1"},
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "f",
                    "arguments": "{}",
                },
            ],
            "",
            [("call_1|fc_1", "{}")],
        )
        assert kept is not None
        return OrderedDict([kept])

    def _history(self) -> list[dict[str, Any]]:
        call = {
            "id": "call_1|fc_1",
            "type": "function",
            "function": {"name": "f", "arguments": "{}"},
        }
        return [
            USER,
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "call_1|fc_1", "content": "x"},
        ]

    @pytest.mark.parametrize(
        ("model", "account", "replayed"),
        [
            pytest.param("gpt-6-luna", "acct-1", True, id="same"),
            pytest.param("gpt-6-astra", "acct-1", False, id="other-model"),
            pytest.param("gpt-6-luna", "acct-2", False, id="other-account"),
        ],
    )
    def test_only_the_producing_model_and_account_get_the_reasoning(
        self, model: str, account: str, replayed: bool
    ) -> None:
        body = codex.build_request_body(
            model, self._history(), native_turns=self._kept(), account=account
        )
        has_reasoning = any(item.get("type") == "reasoning" for item in body["input"])
        assert has_reasoning is replayed


def test_the_process_cache_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A long-lived proxy keeps the most recent turns only."""
    monkeypatch.setattr(specs, "_NATIVE_TURN_LIMIT", 2)
    turn = codex.NativeTurn(scope="s", text="t", calls=(), items=())
    for key in ("a", "b", "c"):
        specs._remember_native_turn(key, turn)
    assert list(specs._state.native_turns) == ["b", "c"]
