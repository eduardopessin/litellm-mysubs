"""Loopback server that catches the OAuth redirect.

The tests speak HTTP by hand against the real port. A mock of the parser would prove that
the parser calls the parser; what matters is that a browser connecting to the port gets a
response and that the port is free afterwards.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
from collections.abc import Iterator

import pytest

from litellm_mysubs.credentials.callback_server import (
    CallbackError,
    CallbackResult,
    CallbackTimeoutError,
    PortInUseError,
    _open_listeners,
    serve_once,
)

PATH = "/callback"
STATE = "state-of-this-session"

#: High ephemeral ports, one per test, so that a race between tests does not disguise
#: itself as a `PortInUseError`.
_PORTS = iter(range(45100, 45400))


@pytest.fixture
def port() -> Iterator[int]:
    yield next(_PORTS)


async def request(port: int, target: str, *, host: str = "127.0.0.1") -> str:
    """Make a raw GET and return the whole response."""
    reader, writer = await asyncio.open_connection(host, port)
    writer.write(f"GET {target} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n".encode())
    await writer.drain()
    body = await reader.read()
    writer.close()
    with contextlib.suppress(OSError):
        await writer.wait_closed()
    return body.decode("utf-8", "replace")


async def callback(port: int, query: str) -> str:
    return await request(port, f"{PATH}?{query}")


def port_is_free(port: int) -> bool:
    """Whether the real binder can claim the port again.

    It uses `_open_listeners` and not a raw `bind` because that is the path the user's retry
    goes through — and a raw `bind` without `SO_REUSEADDR` would refuse the port because of
    the `TIME_WAIT` of the connection already served, measuring the test's socket instead of
    the server.
    """
    try:
        listeners = _open_listeners("localhost", port)
    except PortInUseError:
        return False
    for sock in listeners:
        sock.close()
    return True


class TestSuccessfulCallback:
    async def test_the_code_and_the_state_reach_the_waiter(self, port: int) -> None:
        waiting = asyncio.ensure_future(
            serve_once(port=port, path=PATH, expected_state=STATE, timeout_s=5.0)
        )
        await asyncio.sleep(0.05)
        response = await callback(port, f"code=abc123&state={STATE}")

        assert await waiting == CallbackResult(code="abc123", state=STATE)
        assert "200 OK" in response
        assert port_is_free(port)

    async def test_the_page_tells_the_user_the_window_can_be_closed(self, port: int) -> None:
        waiting = asyncio.ensure_future(
            serve_once(port=port, path=PATH, expected_state=STATE, timeout_s=5.0)
        )
        await asyncio.sleep(0.05)
        response = await callback(port, f"code=x&state={STATE}")
        await waiting

        head, _, body = response.partition("\r\n\r\n")
        assert "close this window" in body
        # Without a correct `Content-Length` the browser keeps loading forever.
        assert f"Content-Length: {len(body.encode())}" in head

    async def test_the_callback_arrives_over_ipv6_loopback(self, port: int) -> None:
        if not socket.has_ipv6:
            pytest.skip("kernel without IPv6")
        waiting = asyncio.ensure_future(
            serve_once(port=port, path=PATH, expected_state=STATE, timeout_s=5.0)
        )
        await asyncio.sleep(0.05)
        # Windows tries `::1` first. If only IPv4 were bound, the user's code would go to
        # whoever held this port on IPv6.
        await request(port, f"{PATH}?code=via-v6&state={STATE}", host="::1")

        assert (await waiting).code == "via-v6"


class TestCodeVariants:
    async def test_authcode_from_native_clients_is_accepted(self, port: int) -> None:
        waiting = asyncio.ensure_future(
            serve_once(port=port, path=PATH, expected_state=STATE, timeout_s=5.0)
        )
        await asyncio.sleep(0.05)
        await callback(port, f"authCode=native&state={STATE}")

        assert (await waiting).code == "native"

    async def test_the_fragment_anthropic_appends_to_the_code_is_cut(self, port: int) -> None:
        waiting = asyncio.ensure_future(
            serve_once(port=port, path=PATH, expected_state=STATE, timeout_s=5.0)
        )
        await asyncio.sleep(0.05)
        await callback(port, f"code=abc%23xyz&state={STATE}")

        # `abc#xyz` exchanged whole for the token returns `invalid_grant`.
        assert (await waiting).code == "abc"


class TestRefusals:
    async def test_a_swapped_state_does_not_let_the_code_out(self, port: int) -> None:
        waiting = asyncio.ensure_future(
            serve_once(port=port, path=PATH, expected_state=STATE, timeout_s=5.0)
        )
        await asyncio.sleep(0.05)
        response = await callback(port, "code=someone-elses-code&state=attacker-state")

        with pytest.raises(CallbackError) as caught:
            await waiting
        assert "someone-elses-code" not in str(caught.value)
        assert "403" in response.split("\r\n", 1)[0]
        assert port_is_free(port)

    async def test_the_provider_description_travels_in_the_exception(self, port: int) -> None:
        waiting = asyncio.ensure_future(
            serve_once(port=port, path=PATH, expected_state=STATE, timeout_s=5.0)
        )
        await asyncio.sleep(0.05)
        await callback(
            port,
            f"error=access_denied&error_description=O+utilizador+recusou+o+acesso&state={STATE}",
        )

        with pytest.raises(CallbackError) as caught:
            await waiting
        assert "access_denied" in str(caught.value)
        # It is the only sentence that tells the user what to do next.
        assert "O utilizador recusou o acesso" in str(caught.value)

    async def test_the_error_wins_even_without_a_valid_state(self, port: int) -> None:
        waiting = asyncio.ensure_future(
            serve_once(port=port, path=PATH, expected_state=STATE, timeout_s=5.0)
        )
        await asyncio.sleep(0.05)
        await callback(port, "error=server_error&error_description=Upstream+em+baixo")

        with pytest.raises(CallbackError) as caught:
            await waiting
        assert "Upstream em baixo" in str(caught.value)


class TestBrowserNoise:
    async def test_a_request_to_another_path_answers_404_and_the_wait_continues(
        self, port: int
    ) -> None:
        waiting = asyncio.ensure_future(
            serve_once(port=port, path=PATH, expected_state=STATE, timeout_s=5.0)
        )
        await asyncio.sleep(0.05)
        # The browser asks for this on its own; letting it cancel the login makes the flow
        # hostage to the browser in use.
        noise = await request(port, "/favicon.ico")
        assert "404" in noise.split("\r\n", 1)[0]
        assert not waiting.done()

        await callback(port, f"code=after-the-noise&state={STATE}")
        assert (await waiting).code == "after-the-noise"


class TestPortAndWait:
    async def test_a_busy_port_raises_an_actionable_error_instead_of_changing_port(
        self, port: int
    ) -> None:
        holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        holder.bind(("127.0.0.1", port))
        holder.listen(1)
        try:
            with pytest.raises(PortInUseError) as caught:
                await serve_once(port=port, path=PATH, expected_state=STATE, timeout_s=5.0)
        finally:
            holder.close()

        # The provider validates the registered redirect URI: a port fallback would trade
        # this clear error for an opaque refusal after the login had already happened.
        assert str(port) in str(caught.value)

    async def test_the_exhausted_wait_raises_timeout_and_releases_the_port(
        self, port: int
    ) -> None:
        with pytest.raises(CallbackTimeoutError):
            await serve_once(port=port, path=PATH, expected_state=STATE, timeout_s=0.2)

        # Retrying after a timeout is the normal case; a stuck port made the failure
        # permanent.
        assert port_is_free(port)

    async def test_the_port_is_reusable_right_after_a_failed_login(self, port: int) -> None:
        first = asyncio.ensure_future(
            serve_once(port=port, path=PATH, expected_state=STATE, timeout_s=5.0)
        )
        await asyncio.sleep(0.05)
        await callback(port, "code=x&state=wrong")
        with pytest.raises(CallbackError):
            await first

        second = asyncio.ensure_future(
            serve_once(port=port, path=PATH, expected_state=STATE, timeout_s=5.0)
        )
        await asyncio.sleep(0.05)
        await callback(port, f"code=second-attempt&state={STATE}")

        assert (await second).code == "second-attempt"
