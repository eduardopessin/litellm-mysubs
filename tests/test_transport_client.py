"""HTTP layer: what the transport does with each response, without opening a socket.

What matters here is not that a GET returns 200 — it is the opposite: *how many* times a
rejected request is repeated, *when* repeating it stops being legal, and what crosses the
boundary when none of that helps. These are exactly the points where the original version
diverged between the synchronous and the asynchronous path.

All of it with `httpx.MockTransport`: no network, no clock, deterministic.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from litellm_mysubs.transport.client import (
    RedeemRequired,
    RemapRequired,
    RequestSpec,
    Response,
    Transport,
    UpstreamError,
)
from litellm_mysubs.transport.hosts import HOSTS, STREAM_PATH, HostRotation

CODEX_URL = "https://chatgpt.com/backend-api/codex/responses"

Handler = Callable[[httpx.Request], httpx.Response]


def spec(
    provider: str = "codex",
    *,
    url: str | None = None,
    token: str = "old",
) -> RequestSpec:
    return RequestSpec(
        url=url or CODEX_URL,
        headers={"Authorization": f"Bearer {token}", "X-Fixo": "1"},
        body={"model": "gpt-5.5", "stream": True},
        provider="codex" if provider == "codex" else "antigravity",
        model="gpt-5.5",
    )


class Recorder:
    """`MockTransport` handler that stores the requests and serves scripted responses."""

    def __init__(self, *responses: httpx.Response) -> None:
        self._responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError(f"one request too many for {request.url}")
        return self._responses.pop(0)

    @property
    def tokens(self) -> list[str]:
        return [r.headers.get("Authorization", "") for r in self.requests]

    @property
    def urls(self) -> list[str]:
        return [str(r.url) for r in self.requests]


def transport(handler: Handler, **kwargs: Any) -> Transport:
    return Transport(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), **kwargs)


def sse_body(*lines: str) -> str:
    return "".join(f"{line}\n" for line in lines)


def ok(text: str = "", **kwargs: Any) -> httpx.Response:
    return httpx.Response(200, text=text, **kwargs)


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
        recorder = Recorder(httpx.Response(401, text="expired"), ok("{}"))

        async def refresh(provider: str) -> str:
            assert provider == "codex"
            return "new"

        async with transport(recorder, refresh=refresh) as client:
            response = await client.request(spec())

        assert response.status == 200
        assert recorder.tokens == ["Bearer old", "Bearer new"]

    async def test_second_401_raises_instead_of_looping(self) -> None:
        """Refreshing in a cycle against a rejected credential is a loop at network speed."""
        recorder = Recorder(httpx.Response(401, text="one"), httpx.Response(401, text="two"))
        calls: list[str] = []

        async def refresh(provider: str) -> str:
            calls.append(provider)
            return "new"

        async with transport(recorder, refresh=refresh) as client:
            with pytest.raises(UpstreamError) as raised:
                await client.request(spec())

        assert len(recorder.requests) == 2
        assert calls == ["codex"]
        assert raised.value.status == 401
        assert raised.value.body == "two"

    async def test_without_callback_raises_the_real_401(self) -> None:
        recorder = Recorder(httpx.Response(401, text="no callback"))
        async with transport(recorder) as client:
            with pytest.raises(UpstreamError) as raised:
                await client.request(spec())

        assert len(recorder.requests) == 1
        assert (raised.value.status, raised.value.body) == (401, "no callback")

    async def test_callback_returning_none_raises_without_retrying(self) -> None:
        """Non-refreshable token: repeating with the same one gave the same 401."""
        recorder = Recorder(httpx.Response(401, text="not refreshable"))

        async def refresh(provider: str) -> None:
            return None

        async with transport(recorder, refresh=refresh) as client:
            with pytest.raises(UpstreamError):
                await client.request(spec())

        assert len(recorder.requests) == 1

    async def test_an_empty_token_is_not_a_token(self) -> None:
        """A refresh that returns `""` failed; using it sent `Bearer ` and spent the only
        repetition available on a request guaranteed to be rejected."""
        recorder = Recorder(httpx.Response(401, text="empty"))

        async def refresh(provider: str) -> str:
            return ""

        async with transport(recorder, refresh=refresh) as client:
            with pytest.raises(UpstreamError):
                await client.request(spec())

        assert len(recorder.requests) == 1


class TestDecisionsAreNotReimplemented:
    async def test_unsupported_model_raises_remap_without_retrying(self) -> None:
        """Which model to use belongs to `plugin.py`; the transport only flags the option."""
        body = "The 'gpt-6' model is not supported when using Codex with a ChatGPT account"
        recorder = Recorder(httpx.Response(400, text=body))

        async with transport(recorder) as client:
            with pytest.raises(RemapRequired) as raised:
                await client.request(spec())

        assert len(recorder.requests) == 1
        assert raised.value.status == 400
        assert raised.value.body == body

    async def test_quota_raises_redeem_without_retrying(self) -> None:
        recorder = Recorder(httpx.Response(429, text="quota"))
        async with transport(recorder) as client:
            with pytest.raises(RedeemRequired) as raised:
                await client.request(spec())

        assert len(recorder.requests) == 1
        assert raised.value.status == 429

    async def test_remap_and_redeem_still_carry_the_upstream_error(self) -> None:
        """Whoever does not handle them propagates the real error, not one the transport
        invented."""
        assert issubclass(RemapRequired, UpstreamError)
        assert issubclass(RedeemRequired, UpstreamError)

    async def test_other_400_is_not_a_model_problem(self) -> None:
        """An invalid payload does not become `RemapRequired` — switching models does not
        fix it."""
        recorder = Recorder(httpx.Response(400, text="Invalid value at 'input'"))
        async with transport(recorder) as client:
            with pytest.raises(UpstreamError) as raised:
                await client.request(spec())

        assert not isinstance(raised.value, RemapRequired | RedeemRequired)

    async def test_error_never_invents_a_status(self) -> None:
        recorder = Recorder(httpx.Response(503, text="upstream down"))
        async with transport(recorder) as client:
            with pytest.raises(UpstreamError) as raised:
                await client.request(spec())

        assert raised.value.status == 503
        assert "upstream down" in str(raised.value)

    async def test_antigravity_429_is_not_redeemable(self) -> None:
        """The Antigravity table has no reset credit; using the Codex one invented it."""
        recorder = Recorder(httpx.Response(429, text="rate"))
        async with transport(recorder) as client:
            with pytest.raises(UpstreamError) as raised:
                await client.request(spec("antigravity"))

        assert not isinstance(raised.value, RedeemRequired)

    async def test_antigravity_429_does_not_try_the_other_host(self) -> None:
        """A 429 is the account's verdict, not the endpoint's.

        Both hosts front the same account and the same quota, so asking the second one
        repeats a refusal that is already known — it only doubles the latency of the
        failure. Measured against the real backend: ~22 s through the rotation against
        ~11 s when the error propagates at once.
        """
        rotation = HostRotation()
        recorder = Recorder(httpx.Response(429, text="RESOURCE_EXHAUSTED"))

        async with transport(recorder, rotation=rotation) as client:
            with pytest.raises(UpstreamError) as raised:
                await client.request(spec("antigravity", url=HOSTS[0] + STREAM_PATH))

        assert recorder.urls == [HOSTS[0] + STREAM_PATH]
        assert raised.value.status == 429


class TestStreaming:
    async def test_events_arrive_in_order_and_stop_at_done(self) -> None:
        body = sse_body(
            ": keep-alive",
            'data: {"n": 1}',
            "",
            'data: {"n": 2}',
            "data: [DONE]",
            'data: {"n": 3}',
        )
        async with transport(Recorder(ok(body))) as client:
            events = await drain(client, spec())

        assert events == [{"n": 1}, {"n": 2}]

    async def test_a_body_cut_mid_event_invents_nothing(self) -> None:
        """Without `[DONE]` and with the last line truncated: the complete events are
        delivered."""
        body = 'data: {"n": 1}\ndata: {"n": 2, "part'
        async with transport(Recorder(ok(body))) as client:
            events = await drain(client, spec())

        assert events == [{"n": 1}]

    async def test_retry_happens_before_the_first_event(self) -> None:
        recorder = Recorder(httpx.Response(401, text="x"), ok(sse_body('data: {"n": 1}')))

        async def refresh(provider: str) -> str:
            return "new"

        async with transport(recorder, refresh=refresh) as client:
            events = await drain(client, spec())

        assert events == [{"n": 1}]
        assert recorder.tokens == ["Bearer old", "Bearer new"]

    async def test_no_reopen_after_an_event_was_delivered(self) -> None:
        """After the first event is delivered, reopening duplicated the prefix already
        consumed."""
        rotation = HostRotation()
        first = ok(sse_body('data: {"n": 1}'))
        recorder = Recorder(first, ok(sse_body('data: {"n": 1}')))

        async with transport(recorder, rotation=rotation) as client:
            events = await drain(client, spec("antigravity", url=HOSTS[0] + STREAM_PATH))

        assert events == [{"n": 1}]
        assert len(recorder.requests) == 1
        assert rotation.started is True


class TestHostFailover:
    def _spec(self) -> RequestSpec:
        return spec("antigravity", url=HOSTS[0] + STREAM_PATH)

    async def test_failover_tries_the_next_host(self) -> None:
        rotation = HostRotation()
        recorder = Recorder(httpx.Response(503, text="capacity"), ok(sse_body('data: {"n":1}')))

        async with transport(recorder, rotation=rotation) as client:
            events = await drain(client, self._spec())

        assert events == [{"n": 1}]
        assert recorder.urls == [HOSTS[0] + STREAM_PATH, HOSTS[1] + STREAM_PATH]

    async def test_both_hosts_failing_propagates_the_last_error(self) -> None:
        rotation = HostRotation()
        recorder = Recorder(httpx.Response(503, text="first"), httpx.Response(404, text="last"))

        async with transport(recorder, rotation=rotation) as client:
            with pytest.raises(UpstreamError) as raised:
                await drain(client, self._spec())

        assert (raised.value.status, raised.value.body) == (404, "last")

    async def test_the_good_host_is_committed_only_after_a_full_stream(self) -> None:
        rotation = HostRotation()
        recorder = Recorder(httpx.Response(503, text="x"), ok(sse_body('data: {"n":1}')))

        async with transport(recorder, rotation=rotation) as client:
            stream = client.stream(self._spec())
            await anext(stream)
            # One delivered event is not a complete stream: the host does not count yet.
            assert rotation.index == 0
            await stream.aclose()

        assert rotation.index == 0

    async def test_a_completed_stream_commits_the_host(self) -> None:
        rotation = HostRotation()
        recorder = Recorder(
            httpx.Response(503, text="x"), ok(sse_body('data: {"n":1}', "data: [DONE]"))
        )

        async with transport(recorder, rotation=rotation) as client:
            await drain(client, self._spec())

        assert rotation.index == 1
        assert rotation.current == HOSTS[1]

    async def test_a_reused_rotation_still_fails_over_on_the_next_request(self) -> None:
        """The "already emitted" mark is per request, not per rotation.

        The rotation lives in the session and outlives the stream. If the first request left
        it marked, the second lost failover — and the failure is silent: it looks like an
        upstream error, not like a host left untried.
        """
        rotation = HostRotation()
        recorder = Recorder(
            ok(sse_body('data: {"n":1}', "data: [DONE]")),
            httpx.Response(503, text="x"),
            ok(sse_body('data: {"n":2}', "data: [DONE]")),
        )

        async with transport(recorder, rotation=rotation) as client:
            assert await drain(client, self._spec()) == [{"n": 1}]
            assert await drain(client, self._spec()) == [{"n": 2}]

        assert len(recorder.requests) == 3

    async def test_without_rotation_only_the_spec_url_is_tried(self) -> None:
        recorder = Recorder(httpx.Response(503, text="x"))
        async with transport(recorder) as client:
            with pytest.raises(UpstreamError):
                await client.request(spec())

        assert recorder.urls == [CODEX_URL]

    async def test_a_url_outside_the_rotation_is_not_rewritten(self) -> None:
        """Codex has no alternative hosts; a rotation being present must not divert it."""
        rotation = HostRotation()
        recorder = Recorder(httpx.Response(503, text="x"))

        async with transport(recorder, rotation=rotation) as client:
            with pytest.raises(UpstreamError):
                await client.request(spec())

        assert recorder.urls == [CODEX_URL]


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
