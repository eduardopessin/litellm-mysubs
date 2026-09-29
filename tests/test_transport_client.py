"""HTTP layer: what the transport does with each response, without opening a socket.

What matters here is not that a GET returns 200 — it is the opposite: *how many* times a
rejected request is repeated, *how long* it waits in between, *when* repeating it stops
being legal, and what crosses the boundary when none of that helps. The policy is omp's
(`fetchWithRetry`, the Codex and Cloud Code replays, the watchdogs); these tests pin what
it does on the wire.

All of it with `httpx.MockTransport`: no network, and no clock but the watchdogs' — the
backoff waits are recorded instead of slept.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import httpx
import pytest

from litellm_mysubs.transport.client import (
    RedeemRequired,
    RemapRequired,
    RequestSpec,
    Response,
    StreamTimeout,
    Transport,
    UpstreamError,
)
from litellm_mysubs.transport.hosts import HOSTS, STREAM_PATH, HostRotation

CODEX_URL = "https://chatgpt.com/backend-api/codex/responses"
PRIMARY = HOSTS[0] + STREAM_PATH
SANDBOX = HOSTS[1] + STREAM_PATH

Reply = httpx.Response | Exception | Callable[[httpx.Request], Awaitable[httpx.Response]]


def spec(
    provider: str = "codex",
    *,
    url: str | None = None,
    token: str = "old",
    model: str = "gpt-5.5",
    on_response: Callable[[int, Any], None] | None = None,
) -> RequestSpec:
    return RequestSpec(
        url=url or (CODEX_URL if provider == "codex" else PRIMARY),
        headers={"Authorization": f"Bearer {token}", "X-Fixo": "1"},
        body={"model": model, "stream": True},
        provider="codex" if provider == "codex" else "antigravity",
        model=model,
        on_response=on_response,
    )


def gemini_spec(**kwargs: Any) -> RequestSpec:
    return spec("antigravity", model="gemini-3-pro", **kwargs)


class Recorder:
    """`MockTransport` handler that stores the requests and serves scripted replies.

    A reply is a response, an exception to raise as the network failure, or an async
    callable for the cases that need to wait.
    """

    def __init__(self, *replies: Reply) -> None:
        self._replies = list(replies)
        self.requests: list[httpx.Request] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self._replies:
            raise AssertionError(f"one request too many for {request.url}")
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, httpx.Response):
            return reply
        return await reply(request)

    @property
    def tokens(self) -> list[str]:
        return [r.headers.get("Authorization", "") for r in self.requests]

    @property
    def urls(self) -> list[str]:
        return [str(r.url) for r in self.requests]


class Waits(list[float]):
    """Stands in for `asyncio.sleep`: records each backoff instead of sleeping it."""

    async def __call__(self, delay: float) -> None:
        self.append(delay)


def transport(handler: Recorder, **kwargs: Any) -> Transport:
    kwargs.setdefault("sleep", Waits())
    return Transport(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), **kwargs)


def sse(*events: dict[str, Any] | str) -> str:
    """A conformant body: one ``data:`` line and one blank line per event."""
    return "".join(f"data: {e if isinstance(e, str) else json.dumps(e)}\n\n" for e in events)


def ok(text: str = "", **kwargs: Any) -> httpx.Response:
    return httpx.Response(200, text=text, **kwargs)


def status(code: int, text: str = "", **kwargs: Any) -> httpx.Response:
    return httpx.Response(code, text=text, **kwargs)


def delta(text: str) -> dict[str, Any]:
    return {"type": "response.output_text.delta", "delta": text}


CREATED = {"type": "response.created", "response": {"id": "resp_1"}}
COMPLETED = {"type": "response.completed", "response": {"status": "completed"}}


def codex_ok(text: str = "hi") -> httpx.Response:
    return ok(sse(CREATED, delta(text), COMPLETED))


def part(text: str, *, finish: str | None = None, thought: bool = False) -> dict[str, Any]:
    candidate: dict[str, Any] = {
        "content": {"parts": [{"text": text, **({"thought": True} if thought else {})}]}
    }
    if finish:
        candidate["finishReason"] = finish
    return {"response": {"candidates": [candidate]}}


EMPTY_STOP = {"response": {"candidates": [{"finishReason": "STOP"}], "usageMetadata": {}}}


def gemini_ok(text: str = "hi") -> httpx.Response:
    return ok(sse(part(text, finish="STOP")))


def body_of(chunks: list[bytes], after: Callable[[], Awaitable[None]]) -> AsyncIterator[bytes]:
    """A body that sends ``chunks`` and then waits on ``after`` — to stall, or to fail."""

    async def generate() -> AsyncIterator[bytes]:
        for chunk in chunks:
            yield chunk
        await after()

    return generate()


async def stall() -> None:
    await asyncio.sleep(3600)


async def drain(transport_: Transport, request: RequestSpec) -> list[dict[str, Any]]:
    return [event async for event in transport_.stream(request)]


class TestNonStreaming:
    async def test_delivers_status_headers_and_body(self) -> None:
        recorder = Recorder(ok('{"ok":true}', headers={"X-Request-Id": "abc"}))
        async with transport(recorder) as client:
            response = await client.request(spec())

        assert isinstance(response, Response)
        assert response.status == 200
        assert response.text == '{"ok":true}'
        assert response.headers["x-request-id"] == "abc"

    async def test_sends_the_spec_verbatim(self) -> None:
        """Headers and body come from `wire/`; the transport does not edit them."""
        recorder = Recorder(ok("{}"))
        async with transport(recorder) as client:
            await client.request(spec())

        sent = recorder.requests[0]
        assert sent.method == "POST"
        assert str(sent.url) == CODEX_URL
        assert sent.headers["X-Fixo"] == "1"
        assert json.loads(sent.read()) == {"model": "gpt-5.5", "stream": True}


class TestTokenRefresh:
    async def test_401_refreshes_once_and_retries_with_the_new_token(self) -> None:
        recorder = Recorder(status(401, "expired"), ok("{}"))

        async def refresh(provider: str) -> str:
            assert provider == "codex"
            return "new"

        async with transport(recorder, refresh=refresh) as client:
            response = await client.request(spec())

        assert response.status == 200
        assert recorder.tokens == ["Bearer old", "Bearer new"]

    async def test_second_401_raises_instead_of_looping(self) -> None:
        """Refreshing in a cycle against a rejected credential is a loop at network speed."""
        recorder = Recorder(status(401, "one"), status(401, "two"))
        calls: list[str] = []

        async def refresh(provider: str) -> str:
            calls.append(provider)
            return "new"

        async with transport(recorder, refresh=refresh) as client:
            with pytest.raises(UpstreamError) as raised:
                await client.request(spec())

        assert len(recorder.requests) == 2
        assert calls == ["codex"]
        assert (raised.value.status, raised.value.body) == (401, "two")

    async def test_without_callback_raises_the_real_401(self) -> None:
        recorder = Recorder(status(401, "no callback"))
        async with transport(recorder) as client:
            with pytest.raises(UpstreamError) as raised:
                await client.request(spec())

        assert len(recorder.requests) == 1
        assert (raised.value.status, raised.value.body) == (401, "no callback")

    async def test_callback_returning_none_raises_without_retrying(self) -> None:
        """Non-refreshable token: repeating with the same one gave the same 401."""
        recorder = Recorder(status(401, "not refreshable"))

        async def refresh(provider: str) -> None:
            return None

        async with transport(recorder, refresh=refresh) as client:
            with pytest.raises(UpstreamError):
                await client.request(spec())

        assert len(recorder.requests) == 1

    async def test_an_empty_token_is_not_a_token(self) -> None:
        """A refresh that returns `""` failed; using it sent `Bearer ` and spent the only
        repetition available on a request guaranteed to be rejected."""
        recorder = Recorder(status(401, "empty"))

        async def refresh(provider: str) -> str:
            return ""

        async with transport(recorder, refresh=refresh) as client:
            with pytest.raises(UpstreamError):
                await client.request(spec())

        assert len(recorder.requests) == 1

    async def test_a_persistent_401_does_not_move_to_the_other_host(self) -> None:
        """omp: a 401 is the credential's, not the host's — the other one refuses it too."""
        recorder = Recorder(status(401, "one"), status(401, "two"))

        async def refresh(provider: str) -> str:
            return "new"

        async with transport(recorder, refresh=refresh, rotation=HostRotation()) as client:
            with pytest.raises(UpstreamError):
                await drain(client, gemini_spec())

        assert recorder.urls == [PRIMARY, PRIMARY]


class TestDecisionsAreNotReimplemented:
    async def test_unsupported_model_raises_remap_without_retrying(self) -> None:
        """Which model to use belongs to `plugin.py`; the transport only flags the option."""
        body = "The 'gpt-6' model is not supported when using Codex with a ChatGPT account"
        recorder = Recorder(status(400, body))

        async with transport(recorder) as client:
            with pytest.raises(RemapRequired) as raised:
                await client.request(spec())

        assert len(recorder.requests) == 1
        assert (raised.value.status, raised.value.body) == (400, body)

    async def test_quota_is_retried_in_place_then_raises_redeem(self) -> None:
        """omp re-sends a Codex 429 five times, 0.5 s longer each time, before giving up."""
        waits = Waits()
        recorder = Recorder(*(status(429, f"quota {n}") for n in range(6)))
        async with transport(recorder, sleep=waits) as client:
            with pytest.raises(RedeemRequired) as raised:
                await client.request(spec())

        assert len(recorder.requests) == 6
        assert waits == [0.5, 1.0, 1.5, 2.0, 2.5]
        assert (raised.value.status, raised.value.body) == (429, "quota 5")

    async def test_other_400_is_not_a_model_problem(self) -> None:
        """An invalid payload does not become `RemapRequired` — switching models does not
        fix it — and is not repeated either."""
        recorder = Recorder(status(400, "Invalid value at 'input'"))
        async with transport(recorder) as client:
            with pytest.raises(UpstreamError) as raised:
                await client.request(spec())

        assert not isinstance(raised.value, RemapRequired | RedeemRequired)
        assert len(recorder.requests) == 1

    async def test_error_never_invents_a_status(self) -> None:
        recorder = Recorder(*(status(503, "upstream down") for _ in range(6)))
        async with transport(recorder) as client:
            with pytest.raises(UpstreamError) as raised:
                await client.request(spec())

        assert raised.value.status == 503
        assert "upstream down" in str(raised.value)

    async def test_antigravity_429_is_not_redeemable(self) -> None:
        """The Antigravity table has no reset credit; using the Codex one invented it."""
        recorder = Recorder(*(status(429, "rate") for _ in range(4)))
        async with transport(recorder) as client:
            with pytest.raises(UpstreamError) as raised:
                await client.request(gemini_spec())

        assert not isinstance(raised.value, RedeemRequired)

    async def test_antigravity_429_does_not_try_the_other_host(self) -> None:
        """A 429 is the account's verdict, not the endpoint's.

        Both hosts front the same account and the same quota, so asking the second one
        repeats a refusal that is already known — it only doubles the latency of the
        failure. Measured against the real backend: ~22 s through the rotation against
        ~11 s when the error propagates at once. The host that answered gets the in-place
        retries omp gives its last host instead.
        """
        waits = Waits()
        recorder = Recorder(*(status(429, "RESOURCE_EXHAUSTED") for _ in range(4)))

        async with transport(recorder, rotation=HostRotation(), sleep=waits) as client:
            with pytest.raises(UpstreamError) as raised:
                await drain(client, gemini_spec())

        assert recorder.urls == [PRIMARY] * 4
        assert waits == [1.0, 2.0, 4.0]
        assert raised.value.status == 429


class TestStatusRetry:
    """omp's `fetchWithRetry`: 408, 429 and 5xx are re-sent in place, and so is a failed
    send; the server's own hint sets the wait, the provider's backoff otherwise."""

    async def test_a_transient_status_is_sent_again(self) -> None:
        waits = Waits()
        recorder = Recorder(status(503, "busy"), codex_ok("answer"))
        async with transport(recorder, sleep=waits) as client:
            events = await drain(client, spec())

        assert delta("answer") in events
        assert waits == [0.5]

    async def test_the_retry_after_header_sets_the_wait(self) -> None:
        waits = Waits()
        recorder = Recorder(status(429, "slow down", headers={"retry-after": "7"}), codex_ok())
        async with transport(recorder, sleep=waits) as client:
            await drain(client, spec())

        assert waits == [7.0]

    async def test_a_body_hint_sets_the_wait(self) -> None:
        """Cloud Code puts it in the error details: ``"retryDelay": "34.5s"``."""
        waits = Waits()
        body = '{"error": {"details": [{"retryDelay": "34.5s"}]}}'
        recorder = Recorder(status(429, body), gemini_ok())
        async with transport(recorder, sleep=waits) as client:
            await drain(client, gemini_spec())

        assert waits == [34.5]

    async def test_a_hint_beyond_the_budget_is_not_waited_out(self) -> None:
        """An hour-long reset is a final answer: waiting it out holds the client for
        nothing. omp caps every wait at five minutes and returns the refusal instead."""
        waits = Waits()
        recorder = Recorder(status(429, "quota", headers={"retry-after": "3600"}))
        async with transport(recorder, sleep=waits) as client:
            with pytest.raises(RedeemRequired):
                await drain(client, spec())

        assert len(recorder.requests) == 1
        assert waits == []

    async def test_a_failed_send_is_sent_again(self) -> None:
        waits = Waits()
        recorder = Recorder(httpx.ConnectError("refused"), codex_ok("answer"))
        async with transport(recorder, sleep=waits) as client:
            events = await drain(client, spec())

        assert delta("answer") in events
        assert waits == [0.5]

    async def test_a_send_that_keeps_failing_raises_the_network_error(self) -> None:
        recorder = Recorder(*(httpx.ConnectError("refused") for _ in range(6)))
        async with transport(recorder) as client:
            with pytest.raises(httpx.ConnectError):
                await drain(client, spec())

        assert len(recorder.requests) == 6

    async def test_the_last_host_backs_off_exponentially(self) -> None:
        """One send on the first host, then omp's four on the last: 1 s, 2 s, 4 s."""
        waits = Waits()
        recorder = Recorder(*(status(503, f"busy {n}") for n in range(5)))
        async with transport(recorder, rotation=HostRotation(), sleep=waits) as client:
            with pytest.raises(UpstreamError) as raised:
                await drain(client, gemini_spec())

        assert recorder.urls == [PRIMARY, SANDBOX, SANDBOX, SANDBOX, SANDBOX]
        assert waits == [1.0, 2.0, 4.0]
        assert raised.value.body == "busy 4"

    @pytest.mark.parametrize("code", [403, 404])
    async def test_a_refusal_is_not_repeated(self, code: int) -> None:
        recorder = Recorder(status(code, "no"))
        async with transport(recorder) as client:
            with pytest.raises(UpstreamError):
                await drain(client, spec())

        assert len(recorder.requests) == 1


class TestStreaming:
    async def test_events_arrive_in_order_and_stop_at_done(self) -> None:
        body = sse(CREATED, delta("a"), delta("b"), "[DONE]", delta("c"))
        async with transport(Recorder(ok(body))) as client:
            events = await drain(client, spec())

        assert events == [CREATED, delta("a"), delta("b")]

    async def test_sse_framing_follows_the_spec(self) -> None:
        """A ``data:`` with no space, JSON across two ``data:`` lines, CRLF endings and
        comments are all valid SSE; reading line by line dropped the first two."""
        body = (
            ": keep-alive\r\n"
            'data:{"type": "response.output_text.delta", "delta": "a"}\r\n\r\n'
            'data: {"type": "response.output_text.delta",\r\n'
            'data:  "delta": "b"}\r\n\r\n'
            f"data: {json.dumps(COMPLETED)}\r\n\r\n"
        )
        async with transport(Recorder(ok(body))) as client:
            events = await drain(client, spec())

        assert events == [delta("a"), delta("b"), COMPLETED]

    async def test_a_malformed_event_raises(self) -> None:
        """omp's `readSseJson` throws on a frame that is not JSON; skipping it delivered a
        response with a hole in it as if it were whole."""
        body = sse(delta("a"), "<html>502 Bad Gateway</html>", COMPLETED)
        async with transport(Recorder(ok(body))) as client:
            with pytest.raises(json.JSONDecodeError):
                await drain(client, spec())

    async def test_a_body_cut_mid_event_invents_nothing(self) -> None:
        """Without `[DONE]` and with the last event truncated: the complete events are
        delivered, the cut-off one ends the stream quietly."""
        body = sse(delta("a")) + 'data: {"type": "response.output_text.delta", "del'
        async with transport(Recorder(ok(body))) as client:
            events = await drain(client, spec())

        assert events == [delta("a")]

    async def test_retry_happens_before_the_first_event(self) -> None:
        recorder = Recorder(status(401, "x"), codex_ok("a"))

        async def refresh(provider: str) -> str:
            return "new"

        async with transport(recorder, refresh=refresh) as client:
            events = await drain(client, spec())

        assert delta("a") in events
        assert recorder.tokens == ["Bearer old", "Bearer new"]

    async def test_no_reopen_after_an_event_was_delivered(self) -> None:
        """After the first event is delivered, reopening duplicated the prefix already
        consumed — even when the stream then breaks off."""
        recorder = Recorder(ok(sse(part("partial"))), gemini_ok())

        async with transport(recorder, rotation=HostRotation()) as client:
            events = await drain(client, gemini_spec())

        assert events == [part("partial")]
        assert len(recorder.requests) == 1


class TestCodexReplay:
    """omp's `#tryRetryProviderError`: a Codex attempt that fails before any content is
    sent again, up to five times, 0.5 s longer each time."""

    async def test_a_stream_cut_before_content_is_replayed(self) -> None:
        waits = Waits()
        recorder = Recorder(ok(sse(CREATED)), codex_ok("answer"))
        async with transport(recorder, sleep=waits) as client:
            events = await drain(client, spec())

        assert events == [CREATED, delta("answer"), COMPLETED]
        assert waits == [0.5]

    async def test_a_transient_failure_event_is_replayed(self) -> None:
        failed = {"type": "response.failed", "response": {"error": {"code": "server_error"}}}
        recorder = Recorder(ok(sse(CREATED, failed)), codex_ok("answer"))
        async with transport(recorder) as client:
            events = await drain(client, spec())

        assert failed not in events
        assert delta("answer") in events

    async def test_a_final_failure_event_is_delivered_as_is(self) -> None:
        failed = {
            "type": "response.failed",
            "response": {"error": {"code": "context_length_exceeded", "message": "too long"}},
        }
        recorder = Recorder(ok(sse(CREATED, failed)))
        async with transport(recorder) as client:
            events = await drain(client, spec())

        assert events == [CREATED, failed]

    async def test_a_connection_dropped_before_content_is_replayed(self) -> None:
        async def drop() -> None:
            raise httpx.RemoteProtocolError("peer closed connection")

        dropped = httpx.Response(200, content=body_of([sse(CREATED).encode()], drop))
        recorder = Recorder(dropped, codex_ok("answer"))
        async with transport(recorder) as client:
            events = await drain(client, spec())

        assert events == [CREATED, delta("answer"), COMPLETED]

    async def test_whitespace_already_delivered_is_never_replayed(self) -> None:
        """omp counts any delta, whitespace included: it already reached the consumer."""
        recorder = Recorder(ok(sse(CREATED, delta(" "))), codex_ok())
        async with transport(recorder) as client:
            events = await drain(client, spec())

        assert events == [CREATED, delta(" ")]
        assert len(recorder.requests) == 1

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param(
                {"type": "response.output_item.added", "item": {"type": "function_call"}},
                id="tool-call-announced",
            ),
            pytest.param(
                {
                    "type": "response.output_item.done",
                    "item": {"type": "message", "content": [{"text": "whole answer"}]},
                },
                id="text-carried-by-the-item",
            ),
            pytest.param(
                {"type": "response.reasoning_summary_text.done", "text": "thought"},
                id="summary-delivered-whole",
            ),
        ],
    )
    async def test_content_without_a_delta_still_commits(self, content: dict[str, Any]) -> None:
        """A tool call is visible the moment it is announced, and an item can carry its
        text whole: either way the client has it, and a replay would repeat it."""
        recorder = Recorder(ok(sse(CREATED, content)), codex_ok())
        async with transport(recorder) as client:
            events = await drain(client, spec())

        assert events == [CREATED, content]
        assert len(recorder.requests) == 1

    async def test_an_announced_message_is_not_content_yet(self) -> None:
        added = {"type": "response.output_item.added", "item": {"type": "message"}}
        recorder = Recorder(ok(sse(CREATED, added)), codex_ok("answer"))
        async with transport(recorder) as client:
            events = await drain(client, spec())

        assert added not in events
        assert delta("answer") in events

    async def test_the_replay_budget_is_five(self) -> None:
        waits = Waits()
        recorder = Recorder(*(ok(sse(CREATED)) for _ in range(6)))
        async with transport(recorder, sleep=waits) as client:
            events = await drain(client, spec())

        assert len(recorder.requests) == 6
        assert waits == [0.5, 1.0, 1.5, 2.0, 2.5]
        # The last attempt is delivered as it came; the reader rejects it for having no
        # terminal event.
        assert events == [CREATED]


class TestAntigravityReplay:
    """omp's `streamGoogleGeminiCli`: an empty ``STOP`` is sent again on the same host —
    500 ms, then 1 s — before the other host is asked."""

    async def test_an_empty_stop_is_sent_again(self) -> None:
        waits = Waits()
        recorder = Recorder(ok(sse(EMPTY_STOP)), ok(sse(part(" "), EMPTY_STOP)), gemini_ok("a"))
        async with transport(recorder, sleep=waits) as client:
            events = await drain(client, gemini_spec())

        assert events == [part("a", finish="STOP")]
        assert waits == [0.5, 1.0]

    async def test_an_empty_host_hands_over_to_the_other(self) -> None:
        recorder = Recorder(*(ok(sse(EMPTY_STOP)) for _ in range(3)), gemini_ok("a"))
        async with transport(recorder, rotation=HostRotation()) as client:
            events = await drain(client, gemini_spec())

        assert events == [part("a", finish="STOP")]
        assert recorder.urls == [PRIMARY] * 3 + [SANDBOX]

    async def test_an_empty_answer_from_the_last_host_reaches_the_reader(self) -> None:
        """Nothing left to try: the reader gets what the last attempt sent, and decides."""
        recorder = Recorder(*(ok(sse(EMPTY_STOP)) for _ in range(3)))
        async with transport(recorder) as client:
            events = await drain(client, gemini_spec())

        assert events == [EMPTY_STOP]
        assert len(recorder.requests) == 3

    async def test_a_thought_only_stop_is_not_sent_again(self) -> None:
        """omp: a complete answer that only thought; replaying burns another reasoning pass."""
        thought = part("thinking…", thought=True, finish="STOP")
        recorder = Recorder(ok(sse(thought)))
        async with transport(recorder) as client:
            events = await drain(client, gemini_spec())

        assert events == [thought]
        assert len(recorder.requests) == 1

    async def test_a_transient_error_in_band_moves_to_the_other_host(self) -> None:
        overloaded = {"error": {"code": 503, "message": "overloaded"}}
        recorder = Recorder(ok(sse(overloaded)), gemini_ok("a"))
        async with transport(recorder, rotation=HostRotation()) as client:
            events = await drain(client, gemini_spec())

        assert events == [part("a", finish="STOP")]
        assert recorder.urls == [PRIMARY, SANDBOX]

    async def test_a_quota_error_in_band_stays_with_its_host(self) -> None:
        exhausted = {"error": {"code": 429, "message": "out of quota"}}
        recorder = Recorder(ok(sse(exhausted)))
        async with transport(recorder, rotation=HostRotation()) as client:
            events = await drain(client, gemini_spec())

        assert events == [exhausted]
        assert recorder.urls == [PRIMARY]


class TestWatchdogs:
    async def test_codex_first_event_timeout_is_replayed_then_raised(self) -> None:
        silent = [httpx.Response(200, content=body_of([], stall)) for _ in range(6)]
        recorder = Recorder(*silent)
        async with transport(recorder, first_event_timeout=0.02) as client:
            with pytest.raises(StreamTimeout, match="waiting for the first event"):
                await drain(client, spec())

        assert len(recorder.requests) == 6

    @pytest.mark.parametrize(
        ("filler", "stalls"),
        [
            pytest.param({"type": "codex.rate_limits"}, True, id="notices"),
            pytest.param(delta("."), False, id="deltas"),
        ],
    )
    async def test_only_progress_keeps_a_codex_stream_alive(
        self, filler: dict[str, Any], stalls: bool
    ) -> None:
        """Rate-limit notices every so often kept a stalled Codex stream open forever
        under a read timeout; omp's idle watchdog counts progress events only."""

        async def generate() -> AsyncIterator[bytes]:
            yield sse(CREATED, delta("a")).encode()
            for _ in range(10):
                await asyncio.sleep(0.02)
                yield sse(filler).encode()
            yield sse(COMPLETED).encode()

        body = httpx.Response(200, content=generate())
        async with transport(Recorder(body), idle_timeout=0.05) as client:
            if stalls:
                with pytest.raises(StreamTimeout, match="stalled while waiting for the next"):
                    await drain(client, spec())
            else:
                assert (await drain(client, spec()))[-1] == COMPLETED

    async def test_antigravity_has_no_idle_watchdog(self) -> None:
        """omp arms only the first-event watchdog on the Cloud Code stream."""

        async def generate() -> AsyncIterator[bytes]:
            yield sse(part("a")).encode()
            await asyncio.sleep(0.1)
            yield sse(part("b", finish="STOP")).encode()

        body = httpx.Response(200, content=generate())
        async with transport(Recorder(body), idle_timeout=0.02) as client:
            events = await drain(client, gemini_spec())

        assert events == [part("a"), part("b", finish="STOP")]

    async def test_a_silent_first_host_hands_over_to_the_other(self) -> None:
        silent = httpx.Response(200, content=body_of([], stall))
        recorder = Recorder(silent, gemini_ok("a"))
        async with transport(
            recorder, rotation=HostRotation(), first_event_timeout=0.02
        ) as client:
            events = await drain(client, gemini_spec())

        assert events == [part("a", finish="STOP")]
        assert recorder.urls == [PRIMARY, SANDBOX]

    async def test_headers_that_never_come_are_a_failed_send(self) -> None:
        async def late(request: httpx.Request) -> httpx.Response:
            await stall()
            raise AssertionError("unreachable")

        recorder = Recorder(late, codex_ok("a"))
        async with transport(recorder, first_event_timeout=0.02) as client:
            events = await drain(client, spec())

        assert delta("a") in events
        assert len(recorder.requests) == 2

    async def test_flash_gets_the_short_first_event_watchdog(self) -> None:
        """Flash does not inherit the five minutes omp keeps for a cold Pro start."""
        from litellm_mysubs.transport.hosts import first_event_timeout

        assert first_event_timeout("gemini-3-flash") < first_event_timeout("gemini-3-pro")
        assert first_event_timeout("gemini-3.1-flash-lite") == first_event_timeout("gemini-3-pro")


class TestOnResponse:
    """The Codex wire reads turn state and quota from the headers of the response it is
    served from — exactly once, and never from an attempt that was thrown away."""

    async def test_a_stream_reports_before_its_first_event(self) -> None:
        seen: list[tuple[int, str]] = []
        order: list[str] = []

        def on_response(code: int, headers: Any) -> None:
            seen.append((code, headers["x-codex-turn-state"]))
            order.append("headers")

        recorder = Recorder(
            ok(sse(CREATED, delta("a"), COMPLETED), headers={"x-codex-turn-state": "ts-1"})
        )
        async with transport(recorder) as client:
            async for _ in client.stream(spec(on_response=on_response)):
                order.append("event")

        assert seen == [(200, "ts-1")]
        assert order[0] == "headers"

    async def test_a_request_reports_before_reading_the_body(self) -> None:
        seen: list[int] = []
        recorder = Recorder(ok("{}"))
        async with transport(recorder) as client:
            await client.request(spec(on_response=lambda code, _: seen.append(code)))

        assert seen == [200]

    async def test_refused_and_replayed_attempts_are_not_reported(self) -> None:
        seen: list[str] = []
        recorder = Recorder(
            status(503, "busy", headers={"x-attempt": "1"}),
            ok(sse(CREATED), headers={"x-attempt": "2"}),
            ok(sse(CREATED, delta("a"), COMPLETED), headers={"x-attempt": "3"}),
        )
        async with transport(recorder) as client:
            await drain(client, spec(on_response=lambda _, h: seen.append(h["x-attempt"])))

        assert seen == ["3"]

    async def test_what_the_callback_raises_propagates(self) -> None:
        def broken(code: int, headers: Any) -> None:
            raise RuntimeError("consumer bug")

        async with transport(Recorder(codex_ok())) as client:
            with pytest.raises(RuntimeError, match="consumer bug"):
                await drain(client, spec(on_response=broken))


class TestHostFailover:
    async def test_failover_tries_the_next_host(self) -> None:
        recorder = Recorder(status(503, "capacity"), gemini_ok("a"))

        async with transport(recorder, rotation=HostRotation()) as client:
            events = await drain(client, gemini_spec())

        assert events == [part("a", finish="STOP")]
        assert recorder.urls == [PRIMARY, SANDBOX]

    async def test_a_failed_send_tries_the_next_host(self) -> None:
        recorder = Recorder(httpx.ConnectError("unreachable"), gemini_ok("a"))

        async with transport(recorder, rotation=HostRotation()) as client:
            events = await drain(client, gemini_spec())

        assert events == [part("a", finish="STOP")]
        assert recorder.urls == [PRIMARY, SANDBOX]

    async def test_a_refused_request_does_not_try_the_next_host(self) -> None:
        """omp moves on only for a transient failure: the other host would refuse a 404
        the same way, at twice the latency."""
        recorder = Recorder(status(404, "model not found"))

        async with transport(recorder, rotation=HostRotation()) as client:
            with pytest.raises(UpstreamError) as raised:
                await drain(client, gemini_spec())

        assert (raised.value.status, raised.value.body) == (404, "model not found")
        assert recorder.urls == [PRIMARY]

    async def test_both_hosts_failing_propagates_the_last_error(self) -> None:
        recorder = Recorder(status(503, "first"), *(status(502, "last") for _ in range(4)))

        async with transport(recorder, rotation=HostRotation()) as client:
            with pytest.raises(UpstreamError) as raised:
                await drain(client, gemini_spec())

        assert (raised.value.status, raised.value.body) == (502, "last")

    async def test_the_good_host_is_committed_only_after_a_full_stream(self) -> None:
        rotation = HostRotation()
        recorder = Recorder(status(503, "x"), gemini_ok("a"))

        async with transport(recorder, rotation=rotation) as client:
            stream = client.stream(gemini_spec())
            await anext(stream)
            # One delivered event is not a complete stream: the host does not count yet.
            assert rotation.index == 0
            await stream.aclose()

        assert rotation.index == 0

    async def test_a_completed_stream_commits_the_host(self) -> None:
        rotation = HostRotation()
        recorder = Recorder(status(503, "x"), gemini_ok("a"))

        async with transport(recorder, rotation=rotation) as client:
            await drain(client, gemini_spec())

        assert rotation.current == HOSTS[1]

    async def test_a_stream_without_a_finish_reason_does_not_commit(self) -> None:
        """omp commits the host after content **and** a finish reason: a stream cut short
        is not the host answering."""
        rotation = HostRotation()
        recorder = Recorder(status(503, "x"), ok(sse(part("cut"))))

        async with transport(recorder, rotation=rotation) as client:
            await drain(client, gemini_spec())

        assert rotation.current == HOSTS[0]

    async def test_a_reused_rotation_still_fails_over_on_the_next_request(self) -> None:
        """The "already emitted" mark is per request, not per rotation.

        The rotation lives in the session and outlives the stream. If the first request left
        it marked, the second lost failover — and the failure is silent: it looks like an
        upstream error, not like a host left untried.
        """
        recorder = Recorder(gemini_ok("1"), status(503, "x"), gemini_ok("2"))

        async with transport(recorder, rotation=HostRotation()) as client:
            assert await drain(client, gemini_spec()) == [part("1", finish="STOP")]
            assert await drain(client, gemini_spec()) == [part("2", finish="STOP")]

        assert len(recorder.requests) == 3

    async def test_a_stream_in_flight_does_not_cost_another_request_its_failover(
        self,
    ) -> None:
        """The rotation is shared by every request in flight. With the "emitted" mark on
        it, a stream that had started forbade failover to a request still waiting on its
        first host, whose 503 then reached the client untried elsewhere."""
        started = asyncio.Event()

        async def refused_after_the_other_starts(request: httpx.Request) -> httpx.Response:
            await started.wait()
            return status(503, "capacity")

        def route(request: httpx.Request) -> Reply:
            if request.headers["Authorization"] == "Bearer streaming":
                return gemini_ok("streaming")
            if str(request.url) == PRIMARY:
                return refused_after_the_other_starts
            return gemini_ok("waiting")

        class Router(Recorder):
            async def __call__(self, request: httpx.Request) -> httpx.Response:
                self.requests.append(request)
                reply = route(request)
                return reply if isinstance(reply, httpx.Response) else await reply(request)

        async with transport(Router(), rotation=HostRotation()) as client:
            waiting = asyncio.create_task(drain(client, gemini_spec(token="waiting")))
            await asyncio.sleep(0)
            streaming = client.stream(gemini_spec(token="streaming"))
            await anext(streaming)
            started.set()
            assert await waiting == [part("waiting", finish="STOP")]
            await streaming.aclose()

    async def test_without_rotation_only_the_spec_url_is_tried(self) -> None:
        recorder = Recorder(status(403, "x"))
        async with transport(recorder) as client:
            with pytest.raises(UpstreamError):
                await client.request(spec())

        assert recorder.urls == [CODEX_URL]

    async def test_a_url_outside_the_rotation_is_not_rewritten(self) -> None:
        """Codex has no alternative hosts; a rotation being present must not divert it."""
        recorder = Recorder(*(status(503, "x") for _ in range(6)))

        async with transport(recorder, rotation=HostRotation()) as client:
            with pytest.raises(UpstreamError):
                await client.request(spec())

        assert set(recorder.urls) == {CODEX_URL}


class TestClientOwnership:
    async def test_an_injected_client_survives_the_transport(self) -> None:
        """Closing it broke the owner, which may have requests in flight."""
        injected = httpx.AsyncClient(transport=httpx.MockTransport(Recorder(ok("{}"))))

        async with Transport(client=injected) as client:
            await client.request(spec())

        assert injected.is_closed is False
        await injected.aclose()

    async def test_an_owned_client_is_closed(self) -> None:
        client = Transport()
        internal = client._client
        await client.aclose()

        assert internal.is_closed is True
