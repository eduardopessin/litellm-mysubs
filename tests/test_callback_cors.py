"""CORS and status endpoint of the callback server.

What is proved here is what the remote page needs: that it can **read** the response
(without `Access-Control-Allow-Origin` the body is opaque to it, measured), that it knows
whether an interceptor is running, and that none of this steals the callback from the wait
in progress. HTTP is spoken by hand against the real port, like the rest of the suite: a
mock would prove that the mock returns headers.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Iterator

import pytest

from litellm_mysubs.credentials.callback_server import (
    CallbackError,
    CallbackResult,
    PortInUseError,
    _open_listeners,
    serve_once,
)

PATH = "/callback"
STATE = "state-of-this-session"
STATUS = "/mysubs-status"
PROVIDER = "anthropic"

#: Own range, disjoint from the one in `test_callback_server.py` (45100-45399), so that the
#: two files running in the same session do not accuse each other of `PortInUseError`.
_PORTS = iter(range(45500, 45800))


@pytest.fixture
def port() -> Iterator[int]:
    yield next(_PORTS)


async def raw(port: int, method: str, target: str, *, host: str = "127.0.0.1") -> str:
    """A raw request with the given method; returns the whole response."""
    reader, writer = await asyncio.open_connection(host, port)
    writer.write(
        f"{method} {target} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n".encode()
    )
    await writer.drain()
    body = await reader.read()
    writer.close()
    with contextlib.suppress(OSError):
        await writer.wait_closed()
    return body.decode("utf-8", "replace")


async def get(port: int, target: str) -> str:
    return await raw(port, "GET", target)


def headers_of(response: str) -> str:
    return response.partition("\r\n\r\n")[0]


def status_json(response: str) -> dict[str, object]:
    parsed = json.loads(response.partition("\r\n\r\n")[2])
    assert isinstance(parsed, dict)
    return parsed


def port_is_free(port: int) -> bool:
    try:
        listeners = _open_listeners("localhost", port)
    except PortInUseError:
        return False
    for sock in listeners:
        sock.close()
    return True


def start(port: int) -> asyncio.Task[CallbackResult]:
    """The wait running in the background, with `provider` already published in the state."""
    return asyncio.ensure_future(
        serve_once(
            port=port,
            path=PATH,
            expected_state=STATE,
            timeout_s=5.0,
            provider=PROVIDER,
        )
    )


class TestCorsOnEveryResponse:
    """Without these headers the page's `fetch` goes out, but the response is unreadable
    for it."""

    async def test_the_success_200_is_readable_by_the_page(self, port: int) -> None:
        waiting = start(port)
        await asyncio.sleep(0.05)
        response = await get(port, f"{PATH}?code=abc&state={STATE}")
        await waiting

        assert "200 OK" in response
        assert "Access-Control-Allow-Origin: *" in headers_of(response)

    async def test_the_403_of_the_swapped_state_also_carries_cors(self, port: int) -> None:
        waiting = start(port)
        await asyncio.sleep(0.05)
        response = await get(port, f"{PATH}?code=stolen&state=other")
        with pytest.raises(CallbackError):
            await waiting

        assert "403 Forbidden" in response
        assert "Access-Control-Allow-Origin: *" in headers_of(response)
        # The refusal stays a refusal: the code does not leave in the body.
        assert "stolen" not in response

    async def test_the_404_of_another_path_also_carries_cors(self, port: int) -> None:
        waiting = start(port)
        await asyncio.sleep(0.05)
        response = await get(port, "/favicon.ico")

        assert "404 Not Found" in response
        assert "Access-Control-Allow-Origin: *" in headers_of(response)

        await get(port, f"{PATH}?code=after&state={STATE}")
        await waiting

    async def test_the_400_of_the_provider_refusal_also_carries_cors(self, port: int) -> None:
        waiting = start(port)
        await asyncio.sleep(0.05)
        response = await get(port, f"{PATH}?error=access_denied&state={STATE}")
        with pytest.raises(CallbackError):
            await waiting

        assert "400 Bad Request" in response
        assert "Access-Control-Allow-Origin: *" in headers_of(response)


class TestStatusEndpoint:
    async def test_tells_the_page_an_interceptor_exists_without_ending_the_wait(
        self, port: int
    ) -> None:
        waiting = start(port)
        await asyncio.sleep(0.05)
        response = await get(port, STATUS)

        assert "200 OK" in response
        assert "Access-Control-Allow-Origin: *" in headers_of(response)
        assert status_json(response) == {
            "mysubs": True,
            "provider": PROVIDER,
            "pending": True,
            "done": False,
        }

        # The probe is not the callback: the flow must still resolve afterwards.
        await get(port, f"{PATH}?code=after-the-probe&state={STATE}")
        await get(port, STATUS)
        assert (await waiting).code == "after-the-probe"

    async def test_turns_to_done_after_the_callback_has_arrived(self, port: int) -> None:
        waiting = start(port)
        await asyncio.sleep(0.05)
        # The order is the real one: the page is already probing when the callback arrives,
        # and it is the next probe that tells it that it can stop. If the server closed at
        # the instant of the callback, this connection was refused and the page was left
        # without knowing the outcome.
        await get(port, STATUS)
        await get(port, f"{PATH}?code=ok&state={STATE}")

        assert status_json(await get(port, STATUS)) == {
            "mysubs": True,
            "provider": PROVIDER,
            "pending": False,
            "done": True,
        }
        assert (await waiting).code == "ok"


class TestPreflight:
    async def test_options_on_any_path_answers_204_without_ending_the_wait(
        self, port: int
    ) -> None:
        waiting = start(port)
        await asyncio.sleep(0.05)
        for target in (STATUS, PATH, "/whatever-it-is"):
            response = await raw(port, "OPTIONS", target)
            assert "204 No Content" in response, target
            assert "Access-Control-Allow-Origin: *" in headers_of(response), target
            assert "Access-Control-Allow-Methods: GET, OPTIONS" in headers_of(response), target

        await get(port, f"{PATH}?code=after-the-preflight&state={STATE}")
        assert (await waiting).code == "after-the-preflight"

    async def test_the_port_is_free_after_everything(self, port: int) -> None:
        waiting = start(port)
        await asyncio.sleep(0.05)
        await raw(port, "OPTIONS", STATUS)
        await get(port, STATUS)
        await get(port, "/favicon.ico")
        await get(port, f"{PATH}?code=end&state={STATE}")
        # The final probe closes the page's cycle; without it the server would wait for the
        # linger.
        await get(port, STATUS)
        await waiting

        # A stuck port would make the first failure permanent: the retry is the next step.
        assert port_is_free(port)
