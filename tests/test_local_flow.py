"""Callback port opened by the proxy itself.

The tests measure the two properties that make this path worth having: `start()` returns
immediately (it is called from inside an HTTP route) and the port really is free after a
cancel (a stuck port ruins the second login, which is precisely what the user does right
after a failure).

The port is tested with the server's real binder, not with a double: the failure mode that
matters is a `bind` the kernel refuses, and a double never refuses anything.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest

from litellm_mysubs.credentials import oauth
from litellm_mysubs.credentials.callback_server import (
    CallbackResult,
    PortInUseError,
    _close_all,
    _open_listeners,
)
from litellm_mysubs.ui import local_flow
from litellm_mysubs.ui.local_flow import LocalCallbackFlow

PATH = "/callback"

#: High ephemeral ports, one per test, so that a race between tests does not disguise itself
#: as `PortInUseError`.
_PORTS = iter(range(45500, 45800))


@pytest.fixture
def port() -> Iterator[int]:
    yield next(_PORTS)


@dataclass(frozen=True, slots=True)
class FakeRequest:
    """The same as an `oauth.AuthRequest` exposes to this module: `url` and `state`."""

    url: str
    state: str
    verifier: str = ""


def auth_request(port: int, *, state: str = "state-1", path: str = PATH) -> FakeRequest:
    return FakeRequest(
        url=(
            "https://claude.ai/oauth/authorize?client_id=x&state=" + state + "&redirect_uri="
            f"http%3A%2F%2Flocalhost%3A{port}{path.replace('/', '%2F')}"
        ),
        state=state,
    )


def port_is_free(port: int) -> bool:
    """Whether the server's binder can claim the port again.

    It is the exact path the user's retry walks; a raw `bind` without `SO_REUSEADDR` would
    measure the `TIME_WAIT` of the test's own socket instead of the server's.
    """
    try:
        listeners = _open_listeners("localhost", port)
    except PortInUseError:
        return False
    _close_all(listeners)
    return True


async def until_free(port: int, *, limit: float = 2.0) -> bool:
    """Wait for the port to be released, with a ceiling. `close()` propagates on the next
    loop cycle.

    The alternative — a fixed `sleep` — is either slow or unstable depending on machine load.
    """
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        if port_is_free(port):
            return True
        await asyncio.sleep(0.01)
    return False


async def deliver(port: int, query: str, *, host: str = "127.0.0.1") -> None:
    """Raw GET against the port, as the browser would do when redirected."""
    reader, writer = await asyncio.open_connection(host, port)
    writer.write(
        f"GET {PATH}?{query} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n".encode()
    )
    await writer.drain()
    await reader.read()
    writer.close()
    with contextlib.suppress(OSError):
        await writer.wait_closed()


async def settle() -> None:
    """Give the loop the chance to run the just-created task up to its first await."""
    for _ in range(5):
        await asyncio.sleep(0)


class TestStartupDoesNotBlock:
    async def test_start_returns_before_the_callback_arrives(
        self, port: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        waiting: asyncio.Event = asyncio.Event()

        async def never(**kwargs: Any) -> CallbackResult:
            waiting.set()
            await asyncio.sleep(3600)
            raise AssertionError("unreachable")

        monkeypatch.setattr(local_flow, "_serve_once", never)
        flow = LocalCallbackFlow()

        before = time.monotonic()
        attempt = flow.start("anthropic", auth_request(port))
        elapsed = time.monotonic() - before

        assert attempt.started
        assert attempt.reason == ""
        # The HTTP route that calls this answers the user right away; the callback only
        # arrives minutes later, once they have finished authenticating.
        assert elapsed < 0.1

        await waiting.wait()
        assert flow.pending("anthropic")
        await flow.stop()

    async def test_the_port_and_path_come_from_the_real_redirect_uri(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}

        async def capture(**kwargs: Any) -> CallbackResult:
            seen.update(kwargs)
            await asyncio.sleep(3600)
            raise AssertionError("unreachable")

        monkeypatch.setattr(local_flow, "_serve_once", capture)
        monkeypatch.setattr(local_flow, "_probe", lambda endpoint: None)

        # The provider's real URL, not one built by the test: it is the derivation against
        # the true value that has to be proven. A parallel table would diverge silently and
        # the provider would refuse the `redirect_uri` only after the user logged in.
        request = oauth.begin("anthropic")
        flow = LocalCallbackFlow()
        assert flow.start("anthropic", request).started
        await settle()

        assert seen["host"] == "localhost"
        assert seen["port"] == 54545
        assert seen["path"] == "/callback"
        assert seen["expected_state"] == request.state

        await flow.stop()


class TestFailuresDoNotRaise:
    async def test_a_busy_port_returns_started_false_with_a_reason(self, port: int) -> None:
        holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # `SO_REUSEADDR` as the server sets it: without it the test's own `bind` failed
        # against the `TIME_WAIT` left by an earlier run on this port, and the suite
        # started depending on how long ago it had run.
        holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        holder.bind(("127.0.0.1", port))
        holder.listen(1)
        flow = LocalCallbackFlow()
        try:
            attempt = flow.start("anthropic", auth_request(port))
        finally:
            holder.close()

        # No exception: the caller has to be able to show the paste right after this.
        assert not attempt.started
        assert str(port) in attempt.reason
        assert not flow.pending("anthropic")

    def test_without_an_event_loop_it_does_not_blow_up(self, port: int) -> None:
        # A synchronous CLI may call this; creating a loop of our own here would leave the
        # port pinned to a thread nobody supervises.
        attempt = LocalCallbackFlow().start("anthropic", auth_request(port))

        assert not attempt.started
        assert "loop" in attempt.reason

    async def test_a_url_without_redirect_uri_is_refused_without_touching_the_network(
        self,
    ) -> None:
        flow = LocalCallbackFlow()
        attempt = flow.start("anthropic", FakeRequest(url="https://claude.ai/x?a=1", state="s"))

        assert not attempt.started
        assert "redirect_uri" in attempt.reason

    async def test_a_failure_inside_the_task_does_not_reach_the_caller_of_start(
        self, port: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def explode(**kwargs: Any) -> CallbackResult:
            raise RuntimeError("upstream down")

        monkeypatch.setattr(local_flow, "_serve_once", explode)
        flow = LocalCallbackFlow()

        assert flow.start("anthropic", auth_request(port)).started
        await settle()

        # The task died, but `start()` had already returned and `result()` merely says
        # there is nothing — the page falls back to the paste instead of showing a traceback.
        assert not flow.pending("anthropic")
        assert await flow.result("anthropic") is None


class TestResult:
    async def test_result_is_none_while_waiting_and_the_callback_once_it_arrives(
        self, port: int
    ) -> None:
        flow = LocalCallbackFlow(wait_s=5.0)
        request = auth_request(port, state="true-state")
        assert flow.start("anthropic", request).started
        await settle()

        assert await flow.result("anthropic") is None

        await deliver(port, f"code=real-code&state={request.state}")
        await asyncio.wait_for(asyncio.shield(flow._tasks["anthropic"]), 2.0)

        received = await flow.result("anthropic")
        assert received is not None
        assert received.code == "real-code"
        assert received.state == request.state
        # Probing again has to give the same: the page asks in a loop.
        assert await flow.result("anthropic") == received

    async def test_a_swapped_state_produces_no_result(self, port: int) -> None:
        flow = LocalCallbackFlow(wait_s=5.0)
        assert flow.start("anthropic", auth_request(port, state="ours")).started
        await settle()

        await deliver(port, "code=stolen&state=some-other")
        with contextlib.suppress(Exception):
            await asyncio.wait_for(asyncio.shield(flow._tasks["anthropic"]), 2.0)

        assert await flow.result("anthropic") is None


class TestPortReleased:
    async def test_a_second_start_cancels_the_first_and_takes_the_port(
        self, port: int
    ) -> None:
        flow = LocalCallbackFlow(wait_s=5.0)
        first = flow.start("anthropic", auth_request(port, state="first"))
        assert first.started
        await settle()
        first_task = flow._tasks["anthropic"]

        # The user pressed "Connect" again: new `state`, and the old server would sit
        # waiting for a `state` that never arrives, holding the port.
        request = auth_request(port, state="second")
        assert flow.start("anthropic", request).started
        await asyncio.sleep(0.1)

        assert first_task.cancelled()

        # The proof that the port really became usable: the second server answers on it.
        await deliver(port, f"code=from-the-second&state={request.state}")
        await asyncio.wait_for(asyncio.shield(flow._tasks["anthropic"]), 2.0)

        received = await flow.result("anthropic")
        assert received is not None
        assert received.code == "from-the-second"

    async def test_cancel_releases_the_port(self, port: int) -> None:
        flow = LocalCallbackFlow(wait_s=5.0)
        assert flow.start("anthropic", auth_request(port)).started
        await settle()
        assert not port_is_free(port)

        flow.cancel("anthropic")

        assert await until_free(port)
        assert not flow.pending("anthropic")

    async def test_stop_releases_the_ports_of_every_provider(self) -> None:
        anthropic_port = next(_PORTS)
        codex_port = next(_PORTS)
        flow = LocalCallbackFlow(wait_s=5.0)
        assert flow.start("anthropic", auth_request(anthropic_port)).started
        assert flow.start("openai-codex", auth_request(codex_port)).started
        await settle()
        assert not port_is_free(anthropic_port)
        assert not port_is_free(codex_port)

        await flow.stop()

        # A proxy `reload` re-enters through here; a port that survives `stop()` makes the
        # next login fail with nothing on screen to explain why.
        assert await until_free(anthropic_port)
        assert await until_free(codex_port)
        assert not flow.pending("anthropic")
        assert not flow.pending("openai-codex")
