"""Automatic path: the proxy opens the OAuth callback port itself.

It only works when LiteLLM runs **on the same machine as the browser**. The `redirect_uri`
registered by the three providers points at `localhost`/`127.0.0.1`, and that `localhost`
is the machine that opens the browser, not the proxy's. On a remote proxy the redirect
hits a port nobody is listening on and the user is left with an error page — so nothing
here raises: `start()` returns `started=False` with a reason and the route falls back to
showing the paste, which is the way out that always works.

Two measurements ground the design:

* **The page cannot read the window that failed.** Trying to read the URL of the provider
  window gives `SecurityError` (different origin) and there is no trick around it. The
  loop only closes with a real server listening on the browser's side — which is exactly
  what this module launches.
* **The port has to be free at the instant of `start()`.** A port held by the previous
  login gives no visible error: the browser redirects to a dead server and the user sees a
  blank page. So the probe happens before promising `started=True`, and `cancel()`/`stop()`
  exist to guarantee the second login finds the port empty.

The port and the path come out of the `redirect_uri` carried inside `request.url`, and not
from a parallel table: the `redirect_uri` is compared byte-for-byte by the provider, and
two sources of truth diverge silently — the divergence only shows up as an opaque refusal,
already after the login.
"""

from __future__ import annotations

import asyncio
import urllib.parse
from dataclasses import dataclass
from typing import Any, Final

from ..credentials.callback_server import (
    CallbackError,
    CallbackResult,
    PortInUseError,
    _close_all,
    _open_listeners,
)
from ..credentials.callback_server import serve_once as _serve_once
from ..credentials.store import ProviderId

__all__ = [
    "DEFAULT_WAIT_S",
    "LocalAttempt",
    "LocalCallbackFlow",
]

#: How long the port stays listening. Same as the server's `DEFAULT_TIMEOUT_S`: it covers a
#: login with MFA and an account switch without leaving the port held if the user closes
#: the browser.
DEFAULT_WAIT_S: float = 300.0

#: Implicit ports per scheme. None of the three providers omits the port in the registered
#: `redirect_uri`, but deriving from the URL without covering the omitted case would send a
#: `None` travelling down to the `bind`, where the error no longer names the cause.
_DEFAULT_PORTS: Final[dict[str, int]] = {"http": 80, "https": 443}

#: Bind attempts inside the task. Only the first is immediate; the rest exist for the case
#: where the previous server is still handing the port back to the kernel when the user
#: presses «Connect» twice in a row.
_BIND_ATTEMPTS: Final = 3

#: Pause between bind attempts. A `close()` already issued resolves in microseconds; this is
#: slack, not useful waiting.
_RETRY_S: Final = 0.05

#: Ceiling for waiting until a cancelled task releases the port. Past this we give up
#: waiting — staying blocked would be worse than a probe that fails and shows the paste.
_DRAIN_GRACE_S: Final = 2.0


@dataclass(frozen=True, slots=True)
class LocalAttempt:
    """The outcome of a `start()`, as the HTTP route sees it."""

    provider: ProviderId
    started: bool
    """Whether the callback port became (or is about to become) ours."""

    reason: str
    """Why not, when `started` is false. Empty when it is true."""


@dataclass(frozen=True, slots=True)
class _Endpoint:
    """Where to listen, derived from the `redirect_uri` the provider will receive."""

    host: str
    port: int
    path: str


def _redirect_uri(url: str) -> str:
    """The `redirect_uri` carried in the authorization URL.

    It is read from the already-assembled URL instead of repeating the provider's constant:
    this is the value the provider will compare against, and listening anywhere other than
    there is listening nowhere.
    """
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    values = query.get("redirect_uri")
    if not values or not values[0]:
        raise ValueError("the authorization URL carries no `redirect_uri`")
    return values[0]


def _endpoint(url: str) -> _Endpoint:
    """`host`, port and path where the callback will land."""
    parts = urllib.parse.urlsplit(_redirect_uri(url))
    host = parts.hostname
    if not host:
        raise ValueError(f"`redirect_uri` without host: {parts.geturl()!r}")
    port = parts.port if parts.port is not None else _DEFAULT_PORTS.get(parts.scheme)
    if port is None:
        raise ValueError(f"`redirect_uri` without port and unknown scheme: {parts.scheme!r}")
    # A `redirect_uri` with no path lands on the root; `urlsplit` returns "" in that case and
    # the server compares paths literally, so "" would never match the "/" the browser sends.
    return _Endpoint(host=host, port=port, path=parts.path or "/")


def _probe(endpoint: _Endpoint) -> None:
    """Raises if the port is not free right now.

    It uses the server's own binder, not a raw `bind`, so that the verdict is the same one
    `serve_once` will get — including dual-stack, where half the loopback being occupied is
    already an impediment. The sockets are closed immediately: a window remains between the
    probe and the real bind, covered by the `_BIND_ATTEMPTS` retries.
    """
    _close_all(_open_listeners(endpoint.host, endpoint.port))


async def _sleep(seconds: float) -> None:
    """Deliberate indirection: it is the only real waiting point, and the tests replace it."""
    await asyncio.sleep(seconds)


async def _drain(task: asyncio.Task[CallbackResult], *, timeout: float) -> None:
    """Cancels the task and waits for it to release the port.

    `asyncio.wait` and not `await task`: the cancelled task will end in an exception, and a
    direct `await` would turn cleaning up an aborted login into a failure of the cleaner.
    """
    task.cancel()
    await asyncio.wait({task}, timeout=timeout)


def _absorb(task: asyncio.Task[CallbackResult]) -> None:
    """Consumes the task's exception.

    Without this, asyncio dumps "Task exception was never retrieved" into the proxy logs
    when the user simply gives up on the login — which is the most common outcome of all.
    """
    if not task.cancelled():
        task.exception()


class LocalCallbackFlow:
    """Opens the callback port inside the proxy process itself.

    One live task per provider, at most. Everything is best-effort: whatever fails here is
    returned as `started=False` and the user goes on through the paste.
    """

    def __init__(self, *, wait_s: float = DEFAULT_WAIT_S) -> None:
        self._wait_s = wait_s
        self._tasks: dict[ProviderId, asyncio.Task[CallbackResult]] = {}

    def start(self, provider: ProviderId, request: Any) -> LocalAttempt:
        """Puts the `redirect_uri` port on listen and returns without waiting for anyone.

        Never blocks: it is called from inside an HTTP route, and the callback only arrives
        minutes later — the time it takes the user to authenticate with the provider.
        """
        try:
            endpoint = _endpoint(str(request.url))
        except ValueError as exc:
            return LocalAttempt(provider=provider, started=False, reason=str(exc))

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Called from synchronous code (a CLI, a test without a loop). There is nowhere
            # to put the task, and creating our own loop here would leave the port held by a
            # thread nobody supervises.
            return LocalAttempt(
                provider=provider,
                started=False,
                reason="no active event loop to listen for the callback",
            )

        # This provider's previous `start()` holds the port and a `state` that is no longer
        # the one the browser will bring: a user who presses «Connect» twice generates a new
        # `state`, and the old server would wait forever for one that never arrives.
        previous = self._tasks.pop(provider, None)

        if previous is None:
            # We only probe when the port is not ours. With a live `previous` the probe would
            # fail against our own server and would refuse a legitimate retry.
            try:
                _probe(endpoint)
            except PortInUseError as exc:
                return LocalAttempt(provider=provider, started=False, reason=str(exc))
            except (CallbackError, OSError) as exc:
                return LocalAttempt(provider=provider, started=False, reason=str(exc))

        task = loop.create_task(
            self._listen(endpoint, provider=provider, state=str(request.state), previous=previous)
        )
        task.add_done_callback(_absorb)
        self._tasks[provider] = task
        return LocalAttempt(provider=provider, started=True, reason="")

    async def _listen(
        self,
        endpoint: _Endpoint,
        *,
        provider: ProviderId,
        state: str,
        previous: asyncio.Task[CallbackResult] | None,
    ) -> CallbackResult:
        """Waits for the callback, after the port's previous owner has left."""
        if previous is not None:
            await _drain(previous, timeout=_DRAIN_GRACE_S)

        last: PortInUseError | None = None
        for attempt in range(_BIND_ATTEMPTS):
            if attempt:
                await _sleep(_RETRY_S)
            try:
                return await _serve_once(
                    port=endpoint.port,
                    path=endpoint.path,
                    expected_state=state,
                    host=endpoint.host,
                    timeout_s=self._wait_s,
                    # The server's `GET /mysubs-status` publishes this, and it is where the
                    # page opened in the browser gets the name of the connecting provider.
                    provider=provider,
                )
            except PortInUseError as exc:
                last = exc
        raise last if last is not None else CallbackError("bind with no attempts")

    def pending(self, provider: ProviderId) -> bool:
        """Whether a port is still listening for this provider."""
        task = self._tasks.get(provider)
        return task is not None and not task.done()

    async def result(self, provider: ProviderId) -> CallbackResult | None:
        """The callback already received, or `None` if still waiting or if the listen failed.

        It neither blocks nor consumes: the page polls this in a loop and has to see the same
        result both times it asks — the second probe is usually the one that catches the
        response in flight.
        """
        task = self._tasks.get(provider)
        if task is None or not task.done() or task.cancelled():
            return None
        return None if task.exception() is not None else task.result()

    def cancel(self, provider: ProviderId) -> None:
        """Releases this provider's port, without waiting for the close.

        Synchronous on purpose, so it can be called from a route that waits for nothing. The
        listeners' `close()` happens in the server's `finally`, on the next loop iteration.
        """
        task = self._tasks.pop(provider, None)
        if task is not None:
            task.cancel()

    async def stop(self) -> None:
        """Releases every port and waits for the close.

        This is what gets called on proxy shutdown. A port that survives the process that
        opened it does not exist, but a port that survives a *reload* does — and then it is
        the second login that stops working, with nothing on screen to explain why.
        """
        tasks = list(self._tasks.values())
        self._tasks.clear()
        if not tasks:
            return
        for task in tasks:
            task.cancel()
        # `asyncio.wait` with `timeout` returns the pending sets instead of raising: giving up
        # on a stubborn task is better than holding up the shutdown.
        await asyncio.wait(set(tasks), timeout=_DRAIN_GRACE_S)
