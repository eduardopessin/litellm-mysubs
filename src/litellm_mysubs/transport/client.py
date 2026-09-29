"""Asynchronous HTTP layer: the only part of the package that opens sockets.

The original opened the stream inside the translator itself, and the "async" version was
the synchronous one wrapped in a ``run_in_executor``: each request occupied a pool worker
for the whole response — which on a subscription stream is minutes — and request N+1
queued with no symptom visible from the outside. Here the transport is a native
`httpx.AsyncClient`.

What it executes is omp's, decided elsewhere:

* `retry.py` — what a refusal means and whether to re-send it in place (`fetchWithRetry`);
* `hosts.py` — the Antigravity hosts, the last good one first;
* `replay.py` — which events commit an attempt, and which failures omp sends again while
  nothing has been delivered;
* `sse.py` — how the body becomes events.

The watchdogs are omp's too, and they replace every `httpx` deadline: a first-event
watchdog (response headers, then the first event) on both providers, and on Codex an idle
watchdog that only *progress* events reset. Antigravity has no idle watchdog once its
first event arrived — omp arms none there.

Boundary: a `RequestSpec` goes in (URL, headers and body already built by `wire/`), a
`Response` or a sequence of raw SSE events comes out. The transport does not know what a
`ModelResponse` is, nor which model to substitute for which.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    Mapping,
)
from dataclasses import dataclass, field
from enum import Enum
from types import TracebackType
from typing import Any, Final, Literal

import httpx

from . import hosts, replay, retry, sse
from .hosts import HostRotation
from .replay import Verdict
from .retry import Action, Decision

#: No `httpx` deadline: the watchdogs own every wait, as omp's do (it passes `timeout:
#: false` so the runtime's own ceiling cannot pre-empt them). A read timeout would cut an
#: Antigravity stream omp lets idle, and a Codex one on keep-alives omp does not count.
TIMEOUT: Final = httpx.Timeout(None)

# omp: utils/idle-iterator.ts :: DEFAULT_STREAM_IDLE_TIMEOUT_MS
#: Codex: the longest silence between two progress events.
CODEX_IDLE_TIMEOUT_S: Final = 300.0
# omp: utils/idle-iterator.ts :: DEFAULT_STREAM_FIRST_EVENT_TIMEOUT_MS
# omp: utils/idle-iterator.ts :: getOpenAIStreamFirstEventTimeoutMs
#: Codex: the wait for the response headers, and again for the first progress event;
#: never shorter than the idle timeout.
CODEX_FIRST_EVENT_TIMEOUT_S: Final = 300.0
# omp: providers/openai-codex-responses.ts :: wrapCodexSseStream
CODEX_FIRST_EVENT_ERROR: Final = (
    "OpenAI Codex SSE stream timed out while waiting for the first event"
)
CODEX_IDLE_ERROR: Final = "OpenAI Codex SSE stream stalled while waiting for the next event"

#: Takes the provider name, returns a fresh access token or ``None`` when it is not
#: refreshable. Asynchronous because the refresh is itself an HTTP request.
RefreshCallback = Callable[[str], Awaitable[str | None]]

#: Status and headers of the response a request is served from.
ResponseCallback = Callable[[int, Mapping[str, str]], None]

Provider = Literal["codex", "antigravity"]


@dataclass(frozen=True, slots=True)
class RequestSpec:
    """A request already translated to the provider wire."""

    url: str
    headers: Mapping[str, str]
    body: Mapping[str, Any]
    provider: Provider
    model: str
    #: Called exactly once per request, with the status and headers of the 2xx response
    #: the caller is served from — before its first event, and for `Transport.request`
    #: before the body is read. Never for an attempt that failed or was replayed: its
    #: headers describe a response nobody received. Whatever it raises propagates.
    on_response: ResponseCallback | None = None


@dataclass(frozen=True, slots=True)
class Response:
    """Non-streaming response, still uninterpreted."""

    status: int
    headers: Mapping[str, str]
    text: str


class UpstreamError(Exception):
    """Error propagated from the upstream, with the **real** status and body.

    A number is never invented: a 500 fabricated on top of a 429 erases the only piece of
    information that tells the user the quota has run out.
    """

    __slots__ = ("body", "status")

    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


class RemapRequired(UpstreamError):  # noqa: N818 — name fixed by the boundary contract
    """The account does not serve this model name, and the name may be a resolvable alias.

    It derives from `UpstreamError` on purpose: whoever does not handle it propagates the
    real upstream error instead of one invented by the transport. Which model to use — if
    any — is `plugin.py`'s decision.
    """

    __slots__ = ()


class RedeemRequired(UpstreamError):  # noqa: N818 — name fixed by the boundary contract
    """Quota exhausted; there may be reset credit left to redeem.

    Same rule: redeeming credit is an effect that costs money, not a transport decision.
    """

    __slots__ = ()


class StreamTimeout(httpx.ReadTimeout):
    """A watchdog fired — omp's `StreamTimeoutError`, with omp's message.

    An `httpx` timeout, so it is what a read timeout used to be for everyone downstream.
    """


class _Failover(Exception):  # noqa: N818 — control flow, never leaves the transport
    """This endpoint gave up before anything was delivered: ask the next one."""


class _Stop(Enum):
    """Why an attempt ended before anything of it was delivered."""

    RETRY = "retry"
    """A failure omp sends again: Codex re-sends it, Antigravity asks the next host."""

    EMPTY = "empty"
    """Antigravity: the stream ended with nothing in it — omp's empty-response retry."""

    FATAL = "fatal"
    """Nothing to replay: deliver what was held and the failure itself."""


@dataclass(slots=True)
class _Call:
    """What one request carries across its attempts, hosts and replays."""

    spec: RequestSpec
    headers: dict[str, str]
    refreshed: bool = False
    #: omp's `providerRetryAttempt`: Codex replays spent, shared by the whole request.
    replays: int = 0
    #: An event reached the caller: from here on, no replay and no other host.
    delivered: bool = False
    notified: bool = False

    def notify(self, response: httpx.Response) -> None:
        if not self.notified:
            self.notified = True
            if self.spec.on_response is not None:
                self.spec.on_response(response.status_code, response.headers)


@dataclass(slots=True)
class _Attempt:
    """One opened response, as far as the hold-back is concerned."""

    held: list[dict[str, Any]] = field(default_factory=list)
    stop: _Stop | None = None
    #: The failure behind ``stop``: an in-band event, or what reading raised.
    event: dict[str, Any] | None = None
    error: Exception | None = None
    #: Antigravity: an answer arrived, and the last finish reason seen.
    meaningful: bool = False
    finish: str | None = None


class Transport:
    """Opens connections and runs omp's retries, replays, failover and watchdogs."""

    __slots__ = (
        "_client",
        "_first_event_timeout",
        "_idle_timeout",
        "_owns_client",
        "_refresh",
        "_rotation",
        "_sleep",
    )

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        refresh: RefreshCallback | None = None,
        rotation: HostRotation | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        first_event_timeout: float | None = None,
        idle_timeout: float | None = None,
    ) -> None:
        """``first_event_timeout``/``idle_timeout`` are omp's `streamFirstEventTimeoutMs`
        and `streamIdleTimeoutMs`, in seconds: ``None`` keeps omp's defaults, ``0``
        disables the watchdog."""
        #: An injected client belongs to whoever injected it — closing it would break
        #: the owner, who may still have requests in flight. Only what was created here
        #: gets closed.
        self._owns_client = client is None
        self._client = httpx.AsyncClient(timeout=TIMEOUT) if client is None else client
        self._refresh = refresh
        self._rotation = rotation
        self._sleep = sleep
        self._first_event_timeout = first_event_timeout
        self._idle_timeout = idle_timeout

    async def __aenter__(self) -> Transport:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def request(self, spec: RequestSpec) -> Response:
        """Non-streaming request: open, read the whole body, close."""
        call = _Call(spec, dict(spec.headers))
        urls = self._candidate_urls(spec)
        for position, url in enumerate(urls):
            try:
                response = await self._open(call, url, is_last=position == len(urls) - 1)
            except _Failover:
                continue
            try:
                call.notify(response)
                async with asyncio.timeout(self._first_event(spec)):
                    text = (await response.aread()).decode("utf-8", "replace")
            finally:
                await response.aclose()
            return Response(status=response.status_code, headers=dict(response.headers), text=text)
        raise AssertionError("the last endpoint never fails over")  # pragma: no cover

    async def stream(self, spec: RequestSpec) -> AsyncIterator[dict[str, Any]]:
        """Raw SSE events, in the order they arrive.

        Every retry — status, token refresh, replay, another host — happens while nothing
        of the response has reached the caller. Once an event has, there is no way back:
        reopening would resend the prefix it has already consumed. A body cut in half ends
        the iterator with the complete events that did arrive.
        """
        call = _Call(spec, dict(spec.headers))
        urls = self._candidate_urls(spec)
        for position, url in enumerate(urls):
            try:
                async for event in self._endpoint(call, url, is_last=position == len(urls) - 1):
                    yield event
            except _Failover:
                continue
            return

    # omp: providers/google-gemini-cli.ts :: streamGoogleGeminiCli
    # omp: providers/openai-codex-responses.ts :: recoverStreamError
    async def _endpoint(
        self, call: _Call, url: str, *, is_last: bool
    ) -> AsyncIterator[dict[str, Any]]:
        """One URL: open it, stream it, and replay on it what omp replays there."""
        empty_retries = 0
        while True:
            # omp re-sends an empty Cloud Code response with a plain fetch: one send.
            response = await self._open(call, url, is_last=is_last, single=empty_retries > 0)
            attempt = _Attempt()
            try:
                async for event in self._attempt(call, response, attempt):
                    yield event
            finally:
                await response.aclose()

            if attempt.stop is None:
                self._commit(call.spec, url, attempt)
                call.notify(response)
                return
            if attempt.stop is _Stop.RETRY and call.spec.provider == "codex":
                if call.replays < retry.CODEX_MAX_RETRIES:
                    call.replays += 1
                    await self._sleep(retry.CODEX_RETRY_DELAY_S * call.replays)
                    continue
            elif attempt.stop is _Stop.EMPTY and empty_retries < hosts.MAX_EMPTY_RETRIES:
                empty_retries += 1
                await self._sleep(hosts.empty_retry_delay(empty_retries))
                continue
            elif attempt.stop is not _Stop.FATAL and not is_last:
                raise _Failover

            for event in self._surface(call, response, attempt):
                yield event
            return

    async def _attempt(
        self, call: _Call, response: httpx.Response, attempt: _Attempt
    ) -> AsyncIterator[dict[str, Any]]:
        """The events of one response, held back until one of them commits the attempt."""
        spec = call.spec
        verdict_of = (
            replay.codex_verdict if spec.provider == "codex" else replay.antigravity_verdict
        )
        events = self._events(spec, response)
        live = False
        try:
            while True:
                try:
                    event = await anext(events)
                except StopAsyncIteration:
                    break
                except httpx.TransportError as error:
                    if live:
                        raise
                    attempt.stop, attempt.error = _Stop.RETRY, error
                    return
                except json.JSONDecodeError as error:
                    if live:
                        raise
                    attempt.stop, attempt.error = _Stop.FATAL, error
                    return
                if spec.provider == "antigravity":
                    attempt.meaningful = attempt.meaningful or replay.google_meaningful(event)
                    attempt.finish = replay.google_finish(event) or attempt.finish
                if not live:
                    verdict = verdict_of(event)
                    if verdict is Verdict.HOLD:
                        attempt.held.append(event)
                        continue
                    if verdict is Verdict.RETRY:
                        attempt.stop, attempt.event = _Stop.RETRY, event
                        return
                    live = True
                    for held in self._release(call, response, attempt.held):
                        yield held
                call.delivered = True
                yield event
        finally:
            await events.aclose()
        if not live:
            # Codex: ended before its terminal event — omp's retryable
            # `CodexProviderStreamError`. Antigravity: the empty response.
            attempt.stop = _Stop.RETRY if spec.provider == "codex" else _Stop.EMPTY

    def _release(
        self, call: _Call, response: httpx.Response, held: list[dict[str, Any]]
    ) -> Iterator[dict[str, Any]]:
        call.notify(response)
        for event in held:
            call.delivered = True
            yield event
        held.clear()

    def _surface(
        self, call: _Call, response: httpx.Response, attempt: _Attempt
    ) -> Iterator[dict[str, Any]]:
        """An attempt nobody replays: what it held, then its failure — as it arrived."""
        failure = [attempt.event] if attempt.event is not None else []
        if attempt.error is None:
            call.notify(response)
        if attempt.held or failure:
            yield from self._release(call, response, [*attempt.held, *failure])
        if attempt.error is not None:
            raise attempt.error

    # omp: utils/idle-iterator.ts :: iterateWithIdleTimeout
    async def _events(
        self, spec: RequestSpec, response: httpx.Response
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Decoded events under the first-event and idle watchdogs.

        The first-event deadline runs until the first *progress* event; the idle one, from
        the last progress event. On Codex a keep-alive or a rate-limit notice is not
        progress — counting it would keep a stalled response open forever.
        """
        first = self._first_event(spec)
        idle = self._idle(spec)
        is_progress = replay.is_codex_progress if spec.provider == "codex" else _always
        loop = asyncio.get_running_loop()
        deadline = loop.time() + first if first else None
        awaiting_first = True
        decoder = sse.SseDecoder()
        lines = aiter(response.aiter_lines())
        while not decoder.done:
            try:
                async with asyncio.timeout_at(deadline):
                    line = await anext(lines)
            except StopAsyncIteration:
                break
            except TimeoutError:
                message = self._timeout_message(spec, first=awaiting_first)
                raise StreamTimeout(message, request=response.request) from None
            for frame in decoder.feed(line):
                if isinstance(frame, sse.Malformed):
                    raise frame.error
                if not isinstance(frame, dict):
                    continue
                if is_progress(frame):
                    awaiting_first = False
                    deadline = loop.time() + idle if idle else None
                yield frame
        for frame in decoder.close():
            if isinstance(frame, sse.Malformed):
                raise frame.error
            if isinstance(frame, dict):
                yield frame

    # omp: fetch-retry.ts :: fetchWithRetry
    # omp: utils/idle-iterator.ts :: armPreResponseTimeout
    async def _open(
        self, call: _Call, url: str, *, is_last: bool, single: bool = False
    ) -> httpx.Response:
        """A 2xx response still unread; closing it is the caller's job.

        omp's `fetchWithRetry`: a retryable status or a failed send is re-sent in place,
        after the server's hint or the provider's backoff — on Codex's single URL, and on
        the last Antigravity host; any other host gets one send and hands over to the
        next. A 401 refreshes the token and repeats **once** — repeating without a limit
        using a credential the server rejects is a loop of rejections at network speed.

        The first-event watchdog already runs here, before the headers: Codex arms it per
        send, Antigravity once over the host's whole sequence, waits included.
        """
        spec = call.spec
        budget = retry.CODEX_BUDGET if spec.provider == "codex" else retry.ANTIGRAVITY_BUDGET
        loop = asyncio.get_running_loop()
        first = self._first_event(spec)
        host_deadline = (
            loop.time() + first if first and spec.provider == "antigravity" else None
        )
        attempt = 0
        while True:
            deadline = host_deadline or (loop.time() + first if first else None)
            request = self._client.build_request(
                "POST", url, json=dict(spec.body), headers=call.headers
            )
            try:
                async with asyncio.timeout_at(deadline):
                    response = await self._client.send(request, stream=True)
                    if response.is_success:
                        return response
                    try:
                        body = (await response.aread()).decode("utf-8", "replace")
                    finally:
                        await response.aclose()
            except (TimeoutError, httpx.TransportError) as error:
                failure = (
                    StreamTimeout(self._timeout_message(spec, first=True), request=request)
                    if isinstance(error, TimeoutError)
                    else error
                )
                if not is_last and not call.delivered:
                    raise _Failover from failure
                delay = None if single else budget.after_network_error(attempt)
                if delay is None:
                    raise failure from None
                await self._wait(delay, host_deadline, failure)
                attempt += 1
                continue

            status = response.status_code
            decision = self._decide(spec, status, body)
            if decision.action is Action.REFRESH_TOKEN and not call.refreshed:
                token = await self._refresh(spec.provider) if self._refresh else None
                if token:
                    call.headers["Authorization"] = f"Bearer {token}"
                    call.refreshed = True
                    continue
            # In place where the answer will not move to another host: the last one, or
            # a status that never fails over.
            limit = 1 if single else budget.max_attempts
            if not is_last and decision.action is Action.FAIL:
                limit = 1
            if attempt + 1 < limit:
                delay = budget.after_status(attempt, status, response.headers, body)
                if delay is not None:
                    await self._wait(delay, host_deadline, UpstreamError(status, body))
                    attempt += 1
                    continue
            if decision.action is Action.REMAP_MODEL:
                raise RemapRequired(status, body)
            if decision.action is Action.REDEEM_CREDIT:
                raise RedeemRequired(status, body)
            if decision.action is Action.FAIL and not is_last and not call.delivered:
                raise _Failover
            raise UpstreamError(status, body)

    async def _wait(self, delay: float, deadline: float | None, failure: Exception) -> None:
        """Back off; a host deadline that expires meanwhile ends in the failure waited on."""
        try:
            async with asyncio.timeout_at(deadline):
                await self._sleep(delay)
        except TimeoutError:
            raise failure from None

    def _decide(self, spec: RequestSpec, status: int, body: str) -> Decision:
        """Classification is delegated; nothing about the error content is decided here.

        ``can_remap``/``can_redeem`` go in as ``True``: they are capabilities of the
        caller, and the transport knows neither the aliases nor has authority to spend
        credit. Passing them affirmatively makes the possibility reach `plugin.py` as a
        dedicated exception — which, by deriving from `UpstreamError`, still propagates the
        real status and body if nobody handles it.
        """
        if spec.provider == "codex":
            return retry.decide_codex(status, body, can_remap=True, can_redeem=True)
        return retry.decide_antigravity(status)

    def _first_event(self, spec: RequestSpec) -> float | None:
        if self._first_event_timeout is not None:
            return self._first_event_timeout or None
        if spec.provider == "codex":
            return max(CODEX_FIRST_EVENT_TIMEOUT_S, self._idle(spec) or 0.0)
        return hosts.first_event_timeout(str(spec.body.get("model") or spec.model))

    def _idle(self, spec: RequestSpec) -> float | None:
        if spec.provider != "codex":
            return None
        if self._idle_timeout is not None:
            return self._idle_timeout or None
        return CODEX_IDLE_TIMEOUT_S

    @staticmethod
    def _timeout_message(spec: RequestSpec, *, first: bool) -> str:
        if spec.provider == "antigravity":
            return hosts.FIRST_EVENT_TIMEOUT_ERROR
        return CODEX_FIRST_EVENT_ERROR if first else CODEX_IDLE_ERROR

    def _candidate_urls(self, spec: RequestSpec) -> list[str]:
        """URLs to try, in order. With no rotation, only the one that came in the request."""
        if self._rotation is None:
            return [spec.url]
        for host in self._rotation.hosts:
            if spec.url.startswith(host):
                return self._rotation.urls(spec.url[len(host) :])
        return [spec.url]

    def _commit(self, spec: RequestSpec, url: str, attempt: _Attempt) -> None:
        """Remember the host only after omp's success: an answer and a normal finish."""
        if (
            self._rotation is not None
            and spec.provider == "antigravity"
            and attempt.meaningful
            and attempt.finish in replay.GOOGLE_OK_FINISHES
        ):
            self._rotation.commit(url)


def _always(_: dict[str, Any]) -> bool:
    return True
