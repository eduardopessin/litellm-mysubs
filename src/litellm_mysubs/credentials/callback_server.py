"""Single-use loopback server that catches the OAuth redirect.

This is the opposite of what ``oauth.py`` does. There, LiteLLM is far from the browser and
the code arrives by paste. Here the process runs on the user's workstation — their browser
reaches our port — and so it is worth opening the exact port the provider has registered and
catching the callback with no manual intervention.

The design facts come from the OMP source (``@oh-my-pi/pi-ai@18.2.6``,
``src/registry/oauth/callback-server.ts``), which is the client that actually made these
flows work on Windows, macOS and Linux:

* **Dual-stack bind.** ``localhost`` resolves to ``127.0.0.1`` *and* ``::1``, and the
  Windows resolver orders ``::1`` first. A server binding only the IPv4 literal leaves the
  IPv6 loopback on the same port free for another process — which then receives the user's
  authorization code. So with ``host="localhost"`` both families are bound on the same port.
  With a literal (``127.0.0.1``) only that one is bound.
* **A missing IPv6 is not fatal.** A kernel booted with ``ipv6.disable=1`` made the whole
  flow blow up on the ``::1`` bind (OMP issue #8814). Whether IPv6 exists is decided by a
  real bind attempt, not by assumption.
* **Exact port, never a fallback.** The provider compares the ``redirect_uri`` with the
  registered one; falling back to a random port trades a clear local error for an opaque
  refusal on the provider's side, already after the user has logged in. An occupied port
  raises ``PortInUseError`` naming the port.
* **Guaranteed close.** Every listener closes in ``finally``, errors included. A port left
  stuck prevents the retry — and the retry is precisely what the user attempts after a
  failure, so the failure would become permanent.

Implementation: ``asyncio.start_server`` with a minimal request line parser. It is a
single-use server talking to a browser, always ``GET``; ``http.server`` would force a thread
and a cooperative shutdown far more fragile to close within a bounded time.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import json
import socket
import urllib.parse
from dataclasses import dataclass
from typing import Final

__all__ = [
    "DEFAULT_TIMEOUT_S",
    "CallbackError",
    "CallbackResult",
    "CallbackTimeoutError",
    "PortInUseError",
    "serve_once",
]

#: How long the callback is waited for. Five minutes cover a login with MFA and an account
#: switch without leaving the port stuck forever if the user closes the browser.
DEFAULT_TIMEOUT_S: float = 300.0

#: Request line ceiling. A browser sends ~1 KiB; above that it is junk or an attack, and
#: reading without a limit would leave the process at the mercy of whoever reaches the
#: loopback.
_MAX_REQUEST_LINE: Final = 8192

#: Ceiling on headers drained before the blank line. They are drained so the browser does
#: not see an RST mid-send and show an error instead of the success page.
_MAX_HEADERS: Final = 64

#: Maximum time waiting for the listeners to close gracefully. The port has already been
#: released by ``Server.close()`` at that point; this is only courtesy to in-flight
#: handlers.
_CLOSE_GRACE_S: Final = 2.0

#: How long responses keep being served after the outcome is decided, **and only** if some
#: page has already probed ``/mysubs-status``. Without this the server closed at the very
#: instant the callback arrived and the next probe hit a refused connection: the page never
#: got to see ``done`` and could not tell "it worked" from "it died". The purely local flow
#: never probes, so it never pays this time.
_DONE_LINGER_S: Final = 2.0


@dataclass(frozen=True, slots=True)
class CallbackResult:
    """What the redirect brought: the authorization code and the ``state`` with it."""

    code: str
    state: str


class CallbackError(RuntimeError):
    """A callback flow failure (port, CSRF, provider refusal)."""


class PortInUseError(CallbackError):
    """The port required by the registered ``redirect_uri`` is already occupied."""


class CallbackTimeoutError(CallbackError):
    """The wait ran out with no valid callback."""


def _bind(family: int, address: str, port: int) -> socket.socket:
    """A socket already bound on ``address:port``, not yet listening.

    ``IPV6_V6ONLY`` is mandatory: without it the IPv6 socket would also claim the IPv4
    mapping of the same port and this function's second bind would fail with ``EADDRINUSE``
    against ourselves.
    """
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if family == socket.AF_INET6:
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        sock.bind((address, port))
        sock.setblocking(False)
    except OSError:
        sock.close()
        raise
    return sock


def _targets(host: str) -> list[tuple[int, str]]:
    """The families to bind for a given ``host``.

    Only the name ``localhost`` is ambiguous between families; a literal is exactly one.
    """
    if host == "localhost":
        return [(socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")]
    return [(socket.AF_INET6 if ":" in host else socket.AF_INET, host)]


def _open_listeners(host: str, port: int) -> list[socket.socket]:
    """Bound sockets, one per available family.

    A family failing because it does not exist in the kernel is ignored as long as another
    remains; an occupied family aborts everything, because handing the code to half the
    loopback is the same as handing it to whoever holds the other half.
    """
    opened: list[socket.socket] = []
    failures: list[str] = []
    for family, address in _targets(host):
        if family == socket.AF_INET6 and not socket.has_ipv6:
            failures.append("::1: no IPv6 support in the interpreter")
            continue
        try:
            opened.append(_bind(family, address, port))
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                _close_all(opened)
                raise PortInUseError(
                    f"Port {port} ({address}) is already in use. The provider only accepts "
                    f"the redirect URI registered on this port, so there is no alternative: "
                    f"terminate the process holding it and repeat the login."
                ) from exc
            failures.append(f"{address}: {exc.strerror or exc}")
    if not opened:
        detail = "; ".join(failures) or "no eligible address family"
        raise CallbackError(f"Could not listen on {host}:{port} — {detail}.")
    return opened


def _close_all(socks: list[socket.socket]) -> None:
    for sock in socks:
        with contextlib.suppress(OSError):
            sock.close()


def _page(title: str, message: str) -> bytes:
    return (
        "<!doctype html><html lang=en><meta charset=utf-8>"
        f"<title>{title}</title>"
        "<body style='font:16px system-ui;margin:4rem auto;max-width:34rem'>"
        f"<h1>{title}</h1><p>{message}</p>"
        "</body></html>"
    ).encode()


#: CORS headers present on **every** response.
#:
#: Measured: a page served by the proxy (``http://192.168.1.28:4141``) can ``fetch``
#: ``http://localhost:54545/…`` and the request does arrive here — but without these headers
#: the response is opaque to it and neither the status nor the body can be read, so the page
#: had no way of knowing the callback arrived.
#:
#: ``*`` and not a fixed origin because the user reaches the proxy by IP, by host name or
#: through a tunnel, and pinning one of them would break the others. What that exposes is
#: what this server has: it is **single-use**, it lives for the seconds of a login, and it
#: serves nothing beyond the state of the flow in progress — there are no files, no API and
#: no credential behind it. The authorization code still only comes out against the correct
#: ``state``.
_CORS: Final = (
    "Access-Control-Allow-Origin: *\r\n"
    "Access-Control-Allow-Headers: *\r\n"
    "Access-Control-Allow-Methods: GET, OPTIONS\r\n"
)


def _http(status: str, content_type: str, body: bytes) -> bytes:
    """A complete HTTP response, CORS included.

    A correct ``Content-Length`` and ``Connection: close`` are not cosmetic: without them
    the browser waits for more body and the user sees a page loading forever instead of the
    confirmation.
    """
    head = (
        f"HTTP/1.1 {status}\r\n"
        f"Content-Type: {content_type}\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"{_CORS}"
        "Connection: close\r\n"
        "\r\n"
    ).encode()
    return head + body


def _response(status: str, title: str, message: str) -> bytes:
    """The page the browser shows the user."""
    return _http(status, "text/html; charset=utf-8", _page(title, message))


_OK = _response(
    "200 OK",
    "Signed in",
    "The credential was captured. You can close this window and return to the terminal.",
)
_NOT_FOUND = _response("404 Not Found", "Not found", "This address is not part of the login.")
_FORBIDDEN = _response(
    "403 Forbidden",
    "Request refused",
    "The <code>state</code> does not match this session's. The login was ignored; "
    "start again from the terminal.",
)
_BAD_REQUEST = _response(
    "400 Bad Request",
    "Login failed",
    "The provider did not return an authorization code. Go back to the terminal for details.",
)

#: The endpoint the remote page probes. Without it the page would only learn by timeout
#: whether an interceptor is running and whether the callback already arrived — and a
#: timeout does not distinguish "still authenticating" from "nobody is listening".
_STATUS_PATH: Final = "/mysubs-status"

#: Preflight: 204 with no body, on any path. The browser sends it before the ``fetch`` when
#: the page adds headers; a 404 answer here would kill the real request before it left.
_NO_CONTENT: Final = (f"HTTP/1.1 204 No Content\r\n{_CORS}Connection: close\r\n\r\n").encode()


def _status(provider: str, *, done: bool) -> bytes:
    """The state of the flow in progress, as JSON.

    ``mysubs`` is the mark that distinguishes this server from anything else answering on
    the same port: the page has to know it is talking to us before trusting the rest.
    """
    body = json.dumps(
        {"mysubs": True, "provider": provider, "pending": not done, "done": done}
    ).encode()
    return _http("200 OK", "application/json; charset=utf-8", body)


def _parse_query(target: str) -> dict[str, list[str]]:
    """Parameters from the request line.

    ``urlsplit`` is not used: it discards the fragment, and Anthropic does return
    ``code=abc#state`` — the ``#`` is part of the value exactly as it arrives on the wire.
    Cutting it is the responsibility of whoever reads the ``code``.
    """
    _, _, query = target.partition("?")
    return urllib.parse.parse_qs(query, keep_blank_values=True)


def _first(params: dict[str, list[str]], key: str) -> str:
    values = params.get(key)
    return values[0] if values else ""


def _extract(
    params: dict[str, list[str]], expected_state: str
) -> tuple[bytes, CallbackResult | CallbackError]:
    """The response to give the browser and the outcome of the wait.

    It returns the error instead of raising it because the HTTP status and the outcome are
    two distinct decisions: a mismatched ``state`` deserves 403 and a provider refusal
    deserves 400, and reconstructing that from the exception text in the caller would couple
    the logic to the wording of the messages.

    The order is deliberate. The provider's ``error`` is read before anything else because
    in that response there is no code to protect and the ``error_description`` is the only
    sentence that tells the user what to do. The ``state`` comes before the ``code`` because
    a mismatched ``state`` is the CSRF signal: the code alongside it belongs to someone else
    and cannot leave here.
    """
    error = _first(params, "error")
    if error:
        description = _first(params, "error_description")
        message = f"The provider refused the login: {error}. {description}".rstrip()
        return _BAD_REQUEST, CallbackError(message)

    state = _first(params, "state")
    if state != expected_state:
        return _FORBIDDEN, CallbackError(
            "The callback `state` does not match this session's — the request was "
            "discarded without even reading the code. Start the login again."
        )

    # `authCode` is the variant some native clients use on the same redirect, and Anthropic
    # does return `code=abc#state`: what counts is what comes before the `#`.
    code = (_first(params, "code") or _first(params, "authCode")).partition("#")[0]
    if not code:
        return _BAD_REQUEST, CallbackError("The callback arrived without an authorization code.")
    return _OK, CallbackResult(code=code, state=state)


async def _reply(writer: asyncio.StreamWriter, payload: bytes) -> None:
    """Writes the response and closes before the outcome is published.

    Publishing the result first would let ``serve_once`` close the listeners with the body
    still undrained, and the browser would show a connection error on top of a login that
    actually worked.
    """
    with contextlib.suppress(OSError, asyncio.IncompleteReadError):
        writer.write(payload)
        await writer.drain()
    writer.close()
    with contextlib.suppress(OSError, asyncio.CancelledError):
        await writer.wait_closed()


async def _read_request(reader: asyncio.StreamReader) -> tuple[str, str] | None:
    """The method and target from the request line, or ``None`` if it is not readable.

    The method stopped being assumed to be ``GET`` once the remote page gained the ability
    to probe the server: the browser sends ``OPTIONS`` before the ``fetch`` and treating it
    as junk made the real request never leave.
    """
    try:
        line = await reader.readline()
    except (OSError, ValueError):
        return None
    if not line or len(line) > _MAX_REQUEST_LINE:
        return None
    parts = line.decode("latin-1").split()
    if len(parts) < 2 or parts[0].upper() not in ("GET", "OPTIONS"):
        return None
    for _ in range(_MAX_HEADERS):
        try:
            header = await reader.readline()
        except (OSError, ValueError):
            break
        if header in (b"", b"\r\n", b"\n"):
            break
    return parts[0].upper(), parts[1]


async def serve_once(
    *,
    port: int,
    path: str,
    expected_state: str,
    host: str = "localhost",
    timeout_s: float = DEFAULT_TIMEOUT_S,
    provider: str = "",
) -> CallbackResult:
    """Listens on ``host:port`` until a valid callback reaches ``path``.

    Requests to other paths answer 404 and do **not** end the wait: the browser asks for
    ``/favicon.ico`` on its own initiative, and letting that cancel the login made the flow
    depend on which browser was used.

    ``provider`` is only published on ``/mysubs-status``, so the remote page knows which
    login this is; empty is acceptable and nothing else in the flow reads it.
    """
    loop = asyncio.get_running_loop()
    outcome: asyncio.Future[CallbackResult] = loop.create_future()
    # There is only a remote page following this if somebody probes the status; while
    # nobody probes, the server behaves exactly as before.
    probed = False
    seen_done = asyncio.Event()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        nonlocal probed
        request = await _read_request(reader)
        if request is None:
            await _reply(writer, _BAD_REQUEST)
            return
        method, target = request
        if method == "OPTIONS":
            # Preflight on any path, and the wait stays intact: it is a question about
            # headers, not the callback.
            await _reply(writer, _NO_CONTENT)
            return
        route = target.partition("?")[0]
        if route == _STATUS_PATH:
            # The page's probe is not the callback: answer and the wait carries on.
            probed = True
            done = outcome.done()
            await _reply(writer, _status(provider, done=done))
            if done:
                seen_done.set()
            return
        if route != path:
            # 404 and nothing else: the `outcome` stays intact and the wait carries on.
            await _reply(writer, _NOT_FOUND)
            return
        payload, outcome_value = _extract(_parse_query(target), expected_state)
        await _reply(writer, payload)
        if outcome.done():
            return
        if isinstance(outcome_value, CallbackError):
            outcome.set_exception(outcome_value)
        else:
            outcome.set_result(outcome_value)

    socks = _open_listeners(host, port)
    servers: list[asyncio.Server] = []
    try:
        for sock in socks:
            servers.append(await asyncio.start_server(handle, sock=sock))
        socks.clear()  # from here on the sockets belong to the servers
        try:
            return await asyncio.wait_for(outcome, timeout_s)
        except TimeoutError as exc:
            raise CallbackTimeoutError(
                f"{timeout_s:.0f}s passed with no callback on {host}:{port}{path}. "
                f"The login never completed in the browser."
            ) from exc
        finally:
            # Closing at the very instant the outcome is decided makes the page's next
            # probe hit a refused connection — it would be left not knowing whether the
            # login worked. We wait only until it reads `done`, and at most the linger.
            if probed and outcome.done() and not seen_done.is_set():
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(seen_done.wait(), _DONE_LINGER_S)
    finally:
        _close_all(socks)
        for server in servers:
            server.close()
        if servers:
            with contextlib.suppress(TimeoutError, OSError):
                await asyncio.wait_for(
                    asyncio.gather(*(s.wait_closed() for s in servers)), _CLOSE_GRACE_S
                )
