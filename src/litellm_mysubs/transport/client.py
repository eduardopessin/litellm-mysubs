"""Asynchronous HTTP layer: the only part of the package that opens sockets.

The original opened the stream inside the translator itself, and the "async" version was
the synchronous one wrapped in a ``run_in_executor``: each request occupied a pool worker
for the whole response — which on a subscription stream is minutes — and request N+1
queued with no symptom visible from the outside. Here the transport is a native
`httpx.AsyncClient`.

There is no OMP anchor in this layer: OMP talks to the runtime `fetch` and the shape of the
loop has no direct counterpart in ``providers/*.ts``. What **is** ported — error
classification and endpoint rotation — lives in ``retry.py`` and ``hosts.py``, with anchors
there. This module executes decisions, it does not take them.

Boundary: a `RequestSpec` goes in (URL, headers and body already built by `wire/`), a
`Response` or a sequence of raw SSE events comes out. The transport does not know what a
`ModelResponse` is, nor which model to substitute for which.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Final, Literal

import httpx

from . import sse
from .hosts import HostRotation
from .retry import Action, Decision, decide_antigravity, decide_codex

#: omp watchdogs: 300 s for the first event and 300 s of inactivity between events. The
#: httpx `read` timeout is exactly the second one; the total has to be ``None`` — a
#: legitimate long-reasoning stream exceeds the bounds of any global timeout.
TIMEOUT: Final = httpx.Timeout(None, connect=30.0, read=300.0, write=60.0)

#: Takes the provider name, returns a fresh access token or ``None`` when it is not
#: refreshable. Asynchronous because the refresh is itself an HTTP request.
RefreshCallback = Callable[[str], Awaitable[str | None]]

Provider = Literal["codex", "antigravity"]


@dataclass(frozen=True, slots=True)
class RequestSpec:
    """A request already translated to the provider wire."""

    url: str
    headers: Mapping[str, str]
    body: Mapping[str, Any]
    provider: Provider
    model: str


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


def _drain(pending: list[str]) -> Iterator[str]:
    """Iterable that empties as it is consumed: whatever is left over is observable."""
    while pending:
        yield pending.pop(0)


def _decode(line: str) -> tuple[list[dict[str, Any]], bool]:
    """The events of one line, and whether the stream has ended.

    `sse.iter_events` signals ``[DONE]`` by **stopping**, not by returning any marker —
    from the outside that is indistinguishable from a line with no data. So a blank line is
    appended after the real one: if it is left unconsumed, `iter_events` stopped early,
    which only happens at ``[DONE]``.

    One line yields at most one event, so materialising them buffers nothing.
    """
    pending = [line, ""]
    events = list(sse.iter_events(_drain(pending)))
    return events, bool(pending)


class Transport:
    """Opens connections, applies the `retry.py` decision and rotates hosts via `hosts.py`."""

    __slots__ = ("_client", "_owns_client", "_refresh", "_rotation")

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        refresh: RefreshCallback | None = None,
        rotation: HostRotation | None = None,
    ) -> None:
        #: An injected client belongs to whoever injected it — closing it would break
        #: the owner, who may still have requests in flight. Only what was created here
        #: gets closed.
        self._owns_client = client is None
        self._client = httpx.AsyncClient(timeout=TIMEOUT) if client is None else client
        self._refresh = refresh
        self._rotation = rotation

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
        response = await self._open(spec)
        try:
            text = (await response.aread()).decode("utf-8", "replace")
        finally:
            await response.aclose()
        self._commit(response)
        return Response(
            status=response.status_code,
            headers=dict(response.headers),
            text=text,
        )

    async def stream(self, spec: RequestSpec) -> AsyncIterator[dict[str, Any]]:
        """Raw SSE events, in the order they arrive.

        The opening loop — with token refresh and failover — runs **before** the first
        `yield`, and only once. Once the caller has seen an event there is no way back:
        reopening on another host or with another token would resend the prefix it has
        already consumed. A body cut in half ends the iterator with the complete events
        that did arrive; the truncated line produces no event at all.
        """
        response = await self._open(spec)
        try:
            async for line in response.aiter_lines():
                events, done = _decode(line)
                for event in events:
                    self._mark_started()
                    yield event
                if done:
                    break
            self._commit(response)
        finally:
            await response.aclose()

    async def _open(self, spec: RequestSpec) -> httpx.Response:
        """A 200 response still unread; closing it is the caller's job.

        A 401 refreshes the token and repeats **once** on the same endpoint — repeating
        without a limit using a credential the server rejects is a loop of rejections at
        network speed. Once the endpoint is exhausted, the next one is tried for as long as
        `hosts.py` allows it.
        """
        headers = dict(spec.headers)
        urls = self._candidate_urls(spec)
        # This request has emitted nothing yet, and the rotation outlives the previous
        # request: without resetting the flag, one complete stream would leave
        # `can_failover` at `False` forever and the next request would silently lose
        # failover.
        self._clear_started()
        last = 0, ""
        for position, url in enumerate(urls):
            refreshed = False
            while True:
                response = await self._client.send(
                    self._client.build_request("POST", url, json=dict(spec.body), headers=headers),
                    stream=True,
                )
                if response.status_code == 200:
                    return response

                status = response.status_code
                body = (await response.aread()).decode("utf-8", "replace")
                await response.aclose()

                decision = self._decide(spec, status, body)
                if decision.action is Action.REMAP_MODEL:
                    raise RemapRequired(status, body)
                if decision.action is Action.REDEEM_CREDIT:
                    raise RedeemRequired(status, body)
                if decision.action is Action.ABORT:
                    raise UpstreamError(status, body)
                if decision.action is Action.REFRESH_TOKEN and not refreshed:
                    token = await self._refresh(spec.provider) if self._refresh else None
                    if token:
                        headers["Authorization"] = f"Bearer {token}"
                        refreshed = True
                        continue
                last = status, body
                break

            # Guard for the `hosts.py` invariant: as long as `_open` runs before the
            # first `yield`, `started` is always false here and `is_last` is already
            # enforced by the `for` — the condition is redundant both ways today. It stays
            # because it is the only thing stopping the invariant from being lost silently
            # if someone ever calls `_open` mid-stream: there the right answer is to stop,
            # not to reopen on another host and duplicate what the client has already
            # seen.
            if not self._can_failover(is_last=position == len(urls) - 1):
                break
        raise UpstreamError(*last)

    def _decide(self, spec: RequestSpec, status: int, body: str) -> Decision:
        """Classification is delegated; nothing about the error content is decided here.

        ``can_remap``/``can_redeem`` go in as ``True``: they are capabilities of the
        caller, and the transport knows neither the aliases nor has authority to spend
        credit. Passing them affirmatively makes the possibility reach `plugin.py` as a
        dedicated exception — which, by deriving from `UpstreamError`, still propagates the
        real status and body if nobody handles it.
        """
        if spec.provider == "codex":
            return decide_codex(status, body, can_remap=True, can_redeem=True)
        return decide_antigravity(status)

    def _candidate_urls(self, spec: RequestSpec) -> list[str]:
        """URLs to try, in order. With no rotation, only the one that came in the request."""
        if self._rotation is None:
            return [spec.url]
        for host in self._rotation.hosts:
            if spec.url.startswith(host):
                return self._rotation.urls(spec.url[len(host) :])
        return [spec.url]

    def _can_failover(self, *, is_last: bool) -> bool:
        return self._rotation is not None and self._rotation.can_failover(is_last=is_last)

    def _mark_started(self) -> None:
        if self._rotation is not None:
            self._rotation.mark_started()

    def _clear_started(self) -> None:
        """Reset the emission flag at the start of each request.

        `hosts.py` exposes no reset — the flag there is the `started` field, and the
        rotation exists to be reused across requests. The field is written directly instead
        of adding a method to an already verified file.
        """
        if self._rotation is not None:
            self._rotation.started = False

    def _commit(self, response: httpx.Response) -> None:
        """Remember the endpoint only after the response has been consumed in full."""
        if self._rotation is not None:
            self._rotation.commit(str(response.request.url))
