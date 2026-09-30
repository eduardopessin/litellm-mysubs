"""The stream readers as omp 18.4.4's stream handlers read the same upstream events.

`turns._CodexReader` ports ``providers/openai-codex-responses.ts :: CodexStreamProcessor``
and `turns._AntigravityReader` ports ``providers/google-gemini-cli.ts ::
streamGoogleGeminiCli``; the chat deltas are what omp's ``openai-chat-server.ts`` writes
for them. Each test is one upstream behaviour the earlier readers got wrong, judged from
the OpenAI SDK's side of the real proxy app over a real `litellm.Router`; only the
upstream HTTP is faked.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import Any, Final

import httpx
import litellm
import litellm.proxy.proxy_server as proxy_server
import openai
import pytest

from litellm_mysubs import plugin, routes, specs
from litellm_mysubs.transport.client import RequestSpec
from litellm_mysubs.turns import WHITESPACE_DELTA_EVENT_LIMIT
from litellm_mysubs.wire import antigravity_models
from tests.test_plugin import FakeTransport, codex_events, gemini_events, install_transport

CODEX: Final = "mysubs/codex/gpt-5.5"
GEMINI_PRO: Final = "mysubs/antigravity/gemini-3-pro"
GEMINI_FLASH: Final = "mysubs/antigravity/gemini-3-flash"
STREAMING = [pytest.param(False, id="whole"), pytest.param(True, id="stream")]


@pytest.fixture(autouse=True)
def proxy(monkeypatch: pytest.MonkeyPatch) -> Iterable[None]:
    plugin.uninstall()
    monkeypatch.setattr(litellm, "model_cost", dict(litellm.model_cost))
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
    monkeypatch.setattr(routes, "THINKING_LOOP_RETRY_BASE_DELAY", 0.0)
    monkeypatch.setattr(routes, "WHITESPACE_LOOP_RETRY_DELAY", 0.0)
    catalog = antigravity_models.ModelCatalog()
    catalog.update({"models": {"gemini-3-pro-high": {}, "gemini-3-flash": {}}})
    monkeypatch.setattr(specs._state, "catalog", catalog)
    plugin.install()
    yield
    plugin.uninstall()


class Attempts(FakeTransport):
    """An upstream that answers each request with the next stream in ``attempts``."""

    def __init__(self, attempts: list[list[dict[str, Any]]]) -> None:
        super().__init__()
        self.attempts = attempts

    async def stream(self, spec: RequestSpec) -> AsyncIterator[dict[str, Any]]:
        self.specs.append(spec)
        for event in self.attempts[min(len(self.specs), len(self.attempts)) - 1]:
            yield event


@dataclass
class Turn:
    """A chat answer as the SDK caller assembles it, streamed or not."""

    content: str = ""
    reasoning: str = ""
    finish: str | None = None
    calls: dict[int, dict[str, str]] = field(default_factory=dict)


async def answer(model: str, *, stream: bool) -> Turn:
    client = openai.AsyncOpenAI(
        api_key="unused",
        base_url="http://proxy/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy_server.app), base_url="http://proxy"
        ),
    )
    turn = Turn()
    messages: Any = [{"role": "user", "content": "go"}]
    try:
        async with asyncio.timeout(60):
            if not stream:
                whole = await client.chat.completions.create(model=model, messages=messages)
                choice = whole.choices[0]
                turn.content = choice.message.content or ""
                turn.reasoning = getattr(choice.message, "reasoning_content", None) or ""
                turn.finish = choice.finish_reason
                for index, call in enumerate(choice.message.tool_calls or []):
                    turn.calls[index] = {
                        "id": call.id,
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    }
                return turn
            chunks = await client.chat.completions.create(
                model=model, messages=messages, stream=True
            )
            async for chunk in chunks:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                turn.content += choice.delta.content or ""
                turn.reasoning += getattr(choice.delta, "reasoning_content", None) or ""
                turn.finish = choice.finish_reason or turn.finish
                for call in choice.delta.tool_calls or []:
                    slot = turn.calls.setdefault(
                        call.index, {"id": "", "name": "", "arguments": ""}
                    )
                    slot["id"] += call.id or ""
                    if call.function is not None:
                        slot["name"] += call.function.name or ""
                        slot["arguments"] += call.function.arguments or ""
            return turn
    finally:
        await client.close()


# -- Codex -----------------------------------------------------------------------


def function_call(
    item_id: str, *, deltas: Iterable[str] = (), arguments: str, done: bool = True
) -> list[dict[str, Any]]:
    item = {"type": "function_call", "id": item_id, "call_id": f"call_{item_id}", "name": "read"}
    events: list[dict[str, Any]] = [{"type": "response.output_item.added", "item": item}]
    events += [
        {"type": "response.function_call_arguments.delta", "item_id": item_id, "delta": delta}
        for delta in deltas
    ]
    if done:
        events.append(
            {"type": "response.output_item.done", "item": {**item, "arguments": arguments}}
        )
    return events


def completed(
    status: str = "completed", *, kind: str = "response.completed", reason: str | None = None
) -> dict[str, Any]:
    response: dict[str, Any] = {"status": status, "usage": {}}
    if reason:
        response["incomplete_details"] = {"reason": reason}
    return {"type": kind, "response": response}


def reasoning(item_id: str, parts: list[str]) -> list[dict[str, Any]]:
    """A reasoning item whose summary streams part by part, as Codex sends it."""
    item = {"type": "reasoning", "id": item_id, "summary": []}
    events: list[dict[str, Any]] = [{"type": "response.output_item.added", "item": item}]
    for index, text in enumerate(parts):
        key = {"item_id": item_id, "summary_index": index}
        events += [
            {
                "type": "response.reasoning_summary_part.added",
                **key,
                "part": {"type": "summary_text", "text": ""},
            },
            {"type": "response.reasoning_summary_text.delta", **key, "delta": text},
            {
                "type": "response.reasoning_summary_part.done",
                **key,
                "part": {"type": "summary_text", "text": text},
            },
        ]
    summary = [{"type": "summary_text", "text": text} for text in parts]
    events.append({"type": "response.output_item.done", "item": {**item, "summary": summary}})
    return events


@pytest.mark.parametrize("stream", STREAMING)
class TestCodex:
    async def test_arguments_sent_only_when_the_call_closes_reach_the_client(
        self, stream: bool
    ) -> None:
        """Codex can send a call's arguments whole in ``output_item.done``. omp's chat
        encoder then writes them as the call's only argument delta; before, the streamed
        call reached the client with no arguments at all."""
        install_transport(
            FakeTransport([*function_call("fc_1", arguments='{"path":"a"}'), completed()])
        )

        turn = await answer(CODEX, stream=stream)

        assert json.loads(turn.calls[0]["arguments"]) == {"path": "a"}
        assert turn.finish == "tool_calls"

    async def test_text_before_a_call_is_part_of_the_answer(self, stream: bool) -> None:
        """omp's `encodeResponse` keeps the text beside the calls. Before, the whole answer
        dropped it whenever the turn had a call, while the stream carried it."""
        events = [
            *codex_events(text="Let me read it.")[:-1],
            *function_call("fc_1", deltas=['{"path":"a"}'], arguments='{"path":"a"}'),
            completed(),
        ]
        install_transport(FakeTransport(events))

        turn = await answer(CODEX, stream=stream)

        assert turn.content == "Let me read it."
        assert turn.calls[0]["name"] == "read"

    async def test_summary_parts_are_separate_paragraphs(self, stream: bool) -> None:
        """omp ends every summary part with a paragraph break; before, two parts ran into
        one sentence."""
        install_transport(
            FakeTransport([*reasoning("rs_1", ["First idea.", "Second idea."]), *codex_events()])
        )

        turn = await answer(CODEX, stream=stream)

        assert turn.reasoning.startswith("First idea.\n\nSecond idea.")
        assert turn.content == "hello"

    async def test_response_done_is_a_terminal_event(self, stream: bool) -> None:
        """omp ends the turn on ``response.done`` as on ``response.completed``. Before,
        the turn failed as truncated."""
        install_transport(FakeTransport([*codex_events()[:-1], completed(kind="response.done")]))

        turn = await answer(CODEX, stream=stream)

        assert (turn.content, turn.finish) == ("hello", "stop")

    async def test_a_delta_for_a_closed_call_is_dropped(self, stream: bool) -> None:
        """A late fragment addressed to a call that already closed lands nowhere, rather
        than being appended to the client's copy of the arguments."""
        events = [
            *function_call("fc_1", deltas=['{"path":"a"}'], arguments='{"path":"a"}'),
            {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "delta": "}}"},
            completed(),
        ]
        install_transport(FakeTransport(events))

        turn = await answer(CODEX, stream=stream)

        assert json.loads(turn.calls[0]["arguments"]) == {"path": "a"}

    async def test_the_whitespace_brake_counts_consecutive_deltas_only(self, stream: bool) -> None:
        """omp's brake is a run of 256 whitespace-only deltas on one call; real argument
        bytes reset it. Before, blanks were summed over the whole turn and a long, indented
        argument tripped it."""
        deltas = ['{"a":', *[" "] * 200, '"x",', *[" "] * 200, '"b":1}']
        install_transport(
            FakeTransport(
                [*function_call("fc_1", deltas=deltas, arguments='{"a":"x","b":1}'), completed()]
            )
        )

        turn = await answer(CODEX, stream=stream)

        assert json.loads(turn.calls[0]["arguments"]) == {"a": "x", "b": 1}

    async def test_a_call_cut_by_the_output_limit_does_not_finish_as_a_call(
        self, stream: bool
    ) -> None:
        """omp promotes a truncated turn to its calls only when the calls are provably
        whole; a closed item is not proof. Before, any call made it ``tool_calls``."""
        events = [
            *function_call("fc_1", deltas=['{"path":"a"}'], arguments='{"path":"a"}'),
            completed("incomplete", kind="response.incomplete", reason="max_output_tokens"),
        ]
        install_transport(FakeTransport(events))

        turn = await answer(CODEX, stream=stream)

        assert turn.finish == "length"

    async def test_an_open_call_with_whole_arguments_is_handed_back(self, stream: bool) -> None:
        events = [
            *function_call("fc_1", deltas=['{"path":"a"}'], arguments="", done=False),
            completed("incomplete", kind="response.incomplete", reason="max_output_tokens"),
        ]
        install_transport(FakeTransport(events))

        turn = await answer(CODEX, stream=stream)

        assert turn.finish == "tool_calls"
        assert json.loads(turn.calls[0]["arguments"]) == {"path": "a"}


# -- the loop guard ----------------------------------------------------------------


def codex_looping_reasoning() -> list[dict[str, Any]]:
    item = {"type": "reasoning", "id": "rs_1", "summary": []}
    return [
        {"type": "response.output_item.added", "item": item},
        *(
            {"type": "response.reasoning_text.delta", "item_id": "rs_1", "delta": delta}
            for delta in ["Let me check the file one more time. "] * 12
        ),
        {"type": "response.output_item.done", "item": item},
        *codex_events(),
    ]


class TestLoopGuard:
    @pytest.mark.parametrize("stream", STREAMING)
    async def test_codex_reasoning_is_guarded_too(self, stream: bool) -> None:
        """omp guards every model's stream against exact cycles, not only Gemini's.
        Before, a Codex turn repeating itself ran to the end and was billed."""
        transport = install_transport(FakeTransport(codex_looping_reasoning()))

        with pytest.raises(openai.APIError, match="Thinking loop detected"):
            await answer(CODEX, stream=stream)

        assert len(transport.specs) == (1 if stream else routes.THINKING_LOOP_MAX_ATTEMPTS)

    async def test_a_whole_turn_that_loops_is_asked_again(self) -> None:
        """omp's `completeSimple` re-samples a turn its guard stopped; the client, which
        has seen nothing yet, gets the second attempt's answer. Before: the error."""
        transport = install_transport(
            Attempts([codex_looping_reasoning(), codex_events(text="fresh answer")])
        )

        turn = await answer(CODEX, stream=False)

        assert turn.content == "fresh answer"
        assert len(transport.specs) == 2


def whitespace_loop(*, text: str | None = None) -> list[dict[str, Any]]:
    """A call whose arguments turn into an endless run of blanks, after optional text."""
    head = codex_events(text=text)[:-1] if text else []
    blanks = [" "] * WHITESPACE_DELTA_EVENT_LIMIT
    return [*head, *function_call("fc_1", deltas=['{"a":', *blanks], arguments="", done=False)]


class TestWhitespaceBrake:
    async def test_a_whole_turn_is_replayed_past_the_loop(self) -> None:
        """omp replays a turn whose only product was the looping call; nothing of a
        non-streamed turn has reached the client. Before: the error."""
        transport = install_transport(Attempts([whitespace_loop(), codex_events(text="done")]))

        turn = await answer(CODEX, stream=False)

        assert turn.content == "done"
        assert len(transport.specs) == 2

    async def test_a_turn_that_already_said_something_is_not_replayed(self) -> None:
        """Visible text before the loop makes the replay unsafe for omp too."""
        transport = install_transport(
            Attempts([whitespace_loop(text="Writing it now."), codex_events(text="done")])
        )

        with pytest.raises(openai.APIError, match="whitespace-only tool-call argument"):
            await answer(CODEX, stream=False)

        assert len(transport.specs) == 1


# -- Antigravity -------------------------------------------------------------------


def thought(text: str, *, finish: str | None = None) -> dict[str, Any]:
    candidate: dict[str, Any] = {"content": {"parts": [{"text": text, "thought": True}]}}
    if finish:
        candidate["finishReason"] = finish
    return {"response": {"candidates": [candidate], "usageMetadata": {"totalTokenCount": 9}}}


def calls(*ids: str | None) -> list[dict[str, Any]]:
    parts = [
        {"functionCall": {"name": "read", "args": {"n": n}, **({"id": i} if i else {})}}
        for n, i in enumerate(ids)
    ]
    return [
        {
            "response": {
                "candidates": [{"content": {"parts": parts}, "finishReason": "STOP"}],
                "usageMetadata": {"totalTokenCount": 9},
            }
        }
    ]


@pytest.mark.parametrize("stream", STREAMING)
class TestAntigravity:
    async def test_an_empty_answer_fails(self, stream: bool) -> None:
        """omp: "Cloud Code Assist API returned an empty response". Before: an empty turn
        delivered as a finished one, which leaves an agent with nothing to act on."""
        install_transport(FakeTransport(gemini_events(text=" ", usage={"totalTokenCount": 3})))

        with pytest.raises(openai.APIError, match="returned an empty response"):
            await answer(GEMINI_PRO, stream=stream)

    async def test_reasoning_with_no_answer_fails(self, stream: bool) -> None:
        install_transport(FakeTransport([thought("Working it out.", finish="STOP")]))

        with pytest.raises(openai.APIError, match="thought-only response without final output"):
            await answer(GEMINI_PRO, stream=stream)

    async def test_repeated_and_missing_call_ids_are_made_distinct(self, stream: bool) -> None:
        """omp mints a fresh id for a call without one or with one already used. Before,
        two calls with the same id reached the client, which cannot answer both."""
        install_transport(FakeTransport(calls("dup", "dup", None)))

        turn = await answer(GEMINI_PRO, stream=stream)

        ids = [turn.calls[index]["id"] for index in range(3)]
        assert ids[0] == "dup"
        assert len(set(ids)) == 3 and all(ids)
        assert [json.loads(turn.calls[i]["arguments"]) for i in range(3)] == [
            {"n": 0},
            {"n": 1},
            {"n": 2},
        ]

    async def test_leaked_thinking_markup_is_reasoning(self, stream: bool) -> None:
        """omp heals reasoning a model leaks into the visible text. Before, the client
        showed the ``<thinking>`` section as part of the answer."""
        install_transport(
            FakeTransport(gemini_events(chunks=["<think", "ing>plan it</thinking>", "The answer."]))
        )

        turn = await answer(GEMINI_PRO, stream=stream)

        assert turn.content == "The answer."
        assert "plan it" in turn.reasoning

    async def test_an_in_band_error_without_a_code_fails(self, stream: bool) -> None:
        """omp throws on any in-band ``error``. Before, one without a numeric code was
        ignored and the turn went on as if nothing happened."""
        install_transport(
            FakeTransport([{"error": {"message": "backend exploded"}}, *gemini_events(text="late")])
        )

        with pytest.raises(
            openai.APIError, match="Cloud Code Assist stream error: backend exploded"
        ):
            await answer(GEMINI_PRO, stream=stream)

    async def test_json_that_is_not_a_planning_object_is_the_answer(self, stream: bool) -> None:
        """omp re-judges the whole held text on every delta: an object that opened like
        ``{"th`` and turned out to be ``{"thing": ..}`` is an answer. Before, the hold was
        only checked on its first delta, and the ``command`` key erased the answer."""
        text = '{"thing": 1, "command": "ls"}'
        install_transport(FakeTransport(gemini_events(chunks=[text[:4], text[4:]])))

        turn = await answer(GEMINI_FLASH, stream=stream)

        assert turn.content == text
