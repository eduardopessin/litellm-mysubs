"""The brakes in the stream readers, judged from the client's side of the real proxy.

Each brake exists for an upstream that misbehaves while answering 200: Codex streaming
whitespace into a tool call's arguments with no end, Gemini reasoning in a loop the
subscription bills by the token, a flash model spilling its planning object into the
visible text. The units say when each one trips; here the upstream actually misbehaves —
endlessly, where that is the failure — behind the real proxy app and a real
`litellm.Router`, and what is judged is what the OpenAI SDK hands its caller and how much
of the upstream stream was consumed before the brake let go.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterable
from typing import Any, Final

import httpx
import litellm
import litellm.main
import litellm.proxy.proxy_server as proxy_server
import openai
import pytest

from litellm_mysubs import plugin, routes, specs
from litellm_mysubs.transport.client import RequestSpec
from litellm_mysubs.wire import antigravity_models
from tests.test_plugin import FakeTransport, codex_events, gemini_events, install_transport

CODEX: Final = "mysubs/codex/gpt-5.5"
GEMINI_PRO: Final = "mysubs/antigravity/gemini-3-pro"
GEMINI_FLASH: Final = "mysubs/antigravity/gemini-3-flash"

#: omp's limits (``providers/openai-codex-responses.ts``): 256 consecutive whitespace-only
#: argument deltas, or 16 KiB of them.
WHITESPACE_EVENT_LIMIT: Final = 256
WHITESPACE_CHAR_LIMIT: Final = 16 * 1024

#: Upstream events a brake may let through before it must have tripped. Well under
#: anything a runaway stream would reach, and the test would otherwise hang, not fail.
CEILING: Final = 5000

_LITELLM_ENTRY_POINTS: Final = {
    (module, name): getattr(module, name)
    for module in (litellm, litellm.main)
    for name in ("acompletion", "completion")
}


class EndlessTransport(FakeTransport):
    """An upstream that sends ``head`` and then repeats ``loop`` for as long as it is read.

    ``pulled`` is how far the plugin read before letting go, and ``released`` whether it
    closed the stream — an upstream left open keeps the subscription's request running.
    """

    def __init__(self, head: list[dict[str, Any]], loop: dict[str, Any]) -> None:
        super().__init__()
        self.head = head
        self.loop = loop
        self.pulled = 0
        self.released = False

    async def stream(self, spec: RequestSpec) -> AsyncIterator[dict[str, Any]]:
        self.specs.append(spec)
        try:
            for event in self.head:
                self.pulled += 1
                yield event
            while self.pulled < CEILING:
                self.pulled += 1
                yield self.loop
                await asyncio.sleep(0)
        finally:
            self.released = True


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
            *(
                {
                    "model_name": model,
                    "litellm_params": {"model": f"gemini/{model.rsplit('/', 1)[-1]}"},
                    "model_info": {"id": model, "mysubs_provider": "google-antigravity"},
                }
                for model in (GEMINI_PRO, GEMINI_FLASH)
            ),
        ]
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "master_key", None)
    # omp's back-off between re-samples of a looping non-streamed turn; the count of
    # attempts is what is judged here, not the wait.
    monkeypatch.setattr(routes, "THINKING_LOOP_RETRY_BASE_DELAY", 0.0)
    monkeypatch.setattr(routes, "WHITESPACE_LOOP_RETRY_DELAY", 0.0)
    catalog = antigravity_models.ModelCatalog()
    catalog.update({"models": {"gemini-3-pro-high": {}, "gemini-3-flash": {}}})
    monkeypatch.setattr(specs._state, "catalog", catalog)
    plugin.install()
    yield
    plugin.uninstall()


def sdk() -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        api_key="unused",
        base_url="http://proxy/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy_server.app), base_url="http://proxy"
        ),
    )


async def answer(model: str, *, stream: bool) -> tuple[str, str]:
    """``(content, reasoning)`` as the SDK caller ends up with them."""
    client = sdk()
    messages: Any = [{"role": "user", "content": "go"}]
    try:
        async with asyncio.timeout(60):
            if not stream:
                whole = await client.chat.completions.create(model=model, messages=messages)
                message = whole.choices[0].message
                return message.content or "", getattr(message, "reasoning_content", "") or ""
            content, reasoning = [], []
            chunks = await client.chat.completions.create(
                model=model, messages=messages, stream=True
            )
            async for chunk in chunks:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    content.append(delta.content or "")
                    reasoning.append(getattr(delta, "reasoning_content", "") or "")
            return "".join(content), "".join(reasoning)
    finally:
        await client.close()


STREAMING = [pytest.param(False, id="whole"), pytest.param(True, id="stream")]


def tool_call_opening() -> list[dict[str, Any]]:
    item = {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "write"}
    return [
        {"type": "response.output_item.added", "item": item},
        {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "delta": '{"a'},
    ]


@pytest.mark.parametrize("stream", STREAMING)
class TestCodexWhitespaceLoop:
    async def test_an_endless_run_of_whitespace_deltas_is_cut(self, stream: bool) -> None:
        """Whitespace deltas keep the connection busy, so no idle timeout ever fires: the
        brake is the only thing that ends the turn. The client gets an error that says
        why, and the upstream is read no further than the limit."""
        transport = EndlessTransport(
            tool_call_opening(),
            {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "delta": " "},
        )
        install_transport(transport)

        with pytest.raises(openai.APIError, match="whitespace-only tool-call argument"):
            await answer(CODEX, stream=stream)

        # A whole turn is replayed twice first, as omp does; a streamed one is not.
        attempts = len(transport.specs)
        assert attempts == (1 if stream else 1 + routes.WHITESPACE_LOOP_RETRY_LIMIT)
        assert transport.pulled <= attempts * (len(tool_call_opening()) + WHITESPACE_EVENT_LIMIT)
        assert transport.released

    async def test_a_few_huge_whitespace_deltas_are_cut_by_size(self, stream: bool) -> None:
        """The byte limit catches what the event count would let through: a handful of
        deltas, each kilobytes of blanks."""
        transport = EndlessTransport(
            tool_call_opening(),
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "delta": "\n" * 4096,
            },
        )
        install_transport(transport)

        with pytest.raises(openai.APIError, match="whitespace-only tool-call argument"):
            await answer(CODEX, stream=stream)

        per_attempt = len(tool_call_opening()) + WHITESPACE_CHAR_LIMIT // 4096
        assert transport.pulled <= len(transport.specs) * per_attempt

    async def test_pretty_printed_arguments_are_not_a_loop(self, stream: bool) -> None:
        """Whitespace-only deltas are normal in indented JSON; a tool call with a few of
        them completes and reaches the client whole. The non-streamed answer carries the
        arguments as omp re-serializes them (`JSON.stringify` of the parsed object)."""
        events = [
            *tool_call_opening()[:1],
            *(
                {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "delta": d}
                for d in ["{", "\n", "  ", '"path"', ": ", '"a.txt"', "\n", "}"]
            ),
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "write",
                    "arguments": '{\n  "path": "a.txt"\n}',
                },
            },
            *codex_events(chunks=[])[-1:],
        ]
        install_transport(FakeTransport(events))
        client = sdk()
        messages: Any = [{"role": "user", "content": "go"}]
        if stream:
            chunks = await client.chat.completions.create(
                model=CODEX, messages=messages, stream=True
            )
            arguments = "".join(
                [
                    call.function.arguments or ""
                    async for chunk in chunks
                    if chunk.choices
                    for call in chunk.choices[0].delta.tool_calls or []
                    if call.function
                ]
            )
        else:
            whole = await client.chat.completions.create(model=CODEX, messages=messages)
            calls = whole.choices[0].message.tool_calls or []
            arguments = calls[0].function.arguments
        await client.close()

        assert json.loads(arguments) == {"path": "a.txt"}


def thought(text: str) -> dict[str, Any]:
    return {"response": {"candidates": [{"content": {"parts": [{"text": text, "thought": True}]}}]}}


@pytest.mark.parametrize("stream", STREAMING)
class TestAntigravityReasoningLoop:
    async def test_a_reasoning_loop_is_aborted_instead_of_billed(self, stream: bool) -> None:
        """Gemini repeating one sentence in its reasoning never reaches an answer, and every
        repetition is billed. The detector watches the thought text as it streams: the
        client gets an error naming the loop, and the upstream is let go long before the
        repetition would have ended on its own."""
        transport = EndlessTransport(
            [thought("The user wants the config. ")],
            thought("Let me check the file one more time. "),
        )
        install_transport(transport)

        with pytest.raises(openai.APIError, match="Thinking loop detected"):
            await answer(GEMINI_PRO, stream=stream)

        # A streamed turn is not asked again: its reasoning already went out. A whole one
        # is, as omp's `completeSimple` re-samples, up to three attempts in all.
        assert len(transport.specs) == (1 if stream else routes.THINKING_LOOP_MAX_ATTEMPTS)
        assert transport.pulled < 100 * len(transport.specs)
        assert transport.released

    async def test_reasoning_that_moves_on_is_answered(self, stream: bool) -> None:
        """The same number of thought events, each saying something new, is a turn like any
        other."""
        events = [
            thought(f"Step {i}: inspect module {i * 7} and record its exports. ") for i in range(40)
        ]
        install_transport(FakeTransport([*events, *gemini_events(text="done")]))

        content, reasoning = await answer(GEMINI_PRO, stream=stream)

        assert content == "done"
        assert reasoning.startswith("Step 0:") and "Step 39:" in reasoning


def visible(text: str, *, finish: bool = False) -> dict[str, Any]:
    candidate: dict[str, Any] = {"content": {"parts": [{"text": text}]}}
    if finish:
        candidate["finishReason"] = "STOP"
    return {"response": {"candidates": [candidate], "usageMetadata": {}}}


@pytest.mark.parametrize("stream", STREAMING)
class TestPlanningLeakFilterAtClose:
    async def test_a_held_tail_that_was_not_planning_is_delivered(self, stream: bool) -> None:
        """Text that opens with a brace is held in case it is the planning object. When the
        turn ends with the object never closed and no planning key in it, the held text is
        the model's answer — dropping it at close lost the end of the reply."""
        install_transport(
            FakeTransport(
                [
                    visible("Result: "),
                    visible("{"),
                    visible('"answer": 42'),
                    visible("", finish=True),
                ]
            )
        )

        content, _ = await answer(GEMINI_FLASH, stream=stream)

        assert content == 'Result: {"answer": 42'

    async def test_a_held_planning_prefix_is_dropped_at_close(self, stream: bool) -> None:
        """The reverse: a planning object cut off by the end of the turn is still planning,
        and half of it must not reach the client."""
        install_transport(
            FakeTransport(
                [
                    visible("Done."),
                    visible('{"thought": "next I will'),
                    visible(" call the tool", finish=True),
                ]
            )
        )

        content, _ = await answer(GEMINI_FLASH, stream=stream)

        assert content == "Done."

    async def test_a_closed_planning_object_is_cut_and_the_answer_kept(self, stream: bool) -> None:
        install_transport(
            FakeTransport(
                [
                    visible('{"thought": "plan", "ca'),
                    visible('ll": "read"}The file has 3 lines.'),
                    visible("", finish=True),
                ]
            )
        )

        content, _ = await answer(GEMINI_FLASH, stream=stream)

        assert content == "The file has 3 lines."
