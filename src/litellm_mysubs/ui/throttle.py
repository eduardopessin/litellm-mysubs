"""Failed-attempt limit per origin, for the one route with no administrator guard.

The pairing code has 60 bits and lasts ten minutes: against someone guessing from the
Internet, the arithmetic is already won. What is missing is the other attacker — the one
sitting on the same LAN as the proxy, hammering `/mysubs/api/deposit` at no cost, and
leaving no trace: today one failed attempt and a million failed attempts produce exactly
the same record, which is none.

This module settles both halves: block the origin that insists, and write down what
happened. Never the what — neither the attempted code nor any fragment of the credential
goes into the log. A security log that leaks the secret it protects is worse than no log.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Mapping
from typing import Protocol

#: Ten failures before blocking. The legitimate user mistypes — the `pairing` alphabet
#: already drops the characters that get confused, but copying half of it remains — and
#: ten attempts cover a botched paste with room to spare. For someone guessing, ten
#: attempts in a space of 2^60 are worth nothing.
DEFAULT_MAX_FAILURES: int = 10

#: The window over which failures add up. Sliding on purpose: a raw counter would
#: accumulate months of mistakes and block someone who never attacked anyone.
DEFAULT_WINDOW_S: float = 60.0

#: How long the origin stays out. Five minutes reduce continuous hammering to a few dozen
#: attempts per hour, and are a short annoyance for whoever landed here by mistake: the
#: pairing code lives ten minutes, so there is still room to try again with the same code
#: before it expires.
DEFAULT_BLOCK_S: float = 300.0

_LOG = logging.getLogger("litellm_mysubs")


class _Address(Protocol):
    @property
    def host(self) -> str: ...


class _RequestLike(Protocol):
    """What is read from the request. A protocol instead of `fastapi.Request` so the test
    can exercise the "no `client`" case, which `TestClient` never produces."""

    @property
    def headers(self) -> Mapping[str, str]: ...

    @property
    def client(self) -> _Address | None: ...


def client_origin(request: _RequestLike) -> str:
    """The origin the attempt is charged to.

    `X-Forwarded-For` is a `client, proxy1, proxy2` list; the first element is the real
    client. Trusting it is only correct behind a proxy that **rewrites** it — if a header
    set by the attacker himself arrives here, he picks the origin and escapes the limit by
    inventing a new value on every request.

    Ignoring it has the symmetric flaw, and a worse one in the normal case: behind an
    ingress every request arrives with the ingress IP, every client collapses into a
    single origin, and ten mistakes by one person lock out the whole installation. Between
    over-blocking people who did nothing and failing to block someone who forges headers in
    a misconfigured deployment, the header is the lesser evil — and the log writes the
    origin used, so the confusion is diagnosable.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded.strip():
        return forwarded.split(",")[0].strip()
    client = request.client
    if client is not None and client.host:
        return client.host
    # ASGI does not guarantee `client`: it is missing in test transports and on unix
    # sockets. A constant origin is preferable to not counting at all — at least it adds
    # up.
    return "unknown"


class Throttle:
    """Recent failures per origin, and who is locked out.

    In memory, like `PairingRegistry`: a block that survives a restart is not worth the
    file it would require, and restarting a proxy is not a primitive a LAN attacker
    controls.

    **Only failures count.** A valid deposit erases the origin's history, so whoever holds
    a legitimate code never gets here, however many subscriptions they connect afterwards.

    **One limiter per worker, and that is accepted.** Measured with `--num_workers 4`:
    gunicorn spreads the requests and each process counts up to its own limit, so 22
    attempts went through before the first 429 instead of 10. The effective ceiling is
    `max_failures * workers`.

    Sharing the state in a file — like the refresher's `flock` — was considered and
    refused: it added I/O on every request path and a new failure mode, for a gain the
    arithmetic says does not exist. With 60 bits of entropy and a 10 min TTL, even 16
    workers give 1600 attempts per window against a space of 2^60 — a 1 in 720 trillion
    chance of hitting. The limiter exists to stop hammering and leave a trace, not to be
    the only defence; the entropy is the defence.
    """

    def __init__(
        self,
        *,
        max_failures: int = DEFAULT_MAX_FAILURES,
        window_s: float = DEFAULT_WINDOW_S,
        block_s: float = DEFAULT_BLOCK_S,
    ) -> None:
        self._max_failures = max_failures
        self._window_s = window_s
        self._block_s = block_s
        self._failures: dict[str, list[float]] = {}
        self._blocked_until: dict[str, float] = {}
        # The proxy serves the UI from a thread pool: two attempts from the same origin can
        # land on different threads, and both mutate the two dictionaries.
        self._lock = threading.Lock()

    def blocked(self, origin: str, *, now: float | None = None) -> float:
        """Seconds left until this origin is served again. `0.0` = free."""
        moment = time.time() if now is None else now
        with self._lock:
            until = self._blocked_until.get(origin)
            if until is None or moment >= until:
                return 0.0
            return until - moment

    def record_failure(self, origin: str, *, now: float | None = None) -> None:
        """An attempt that redeemed nothing. It may be the one that shuts the door."""
        moment = time.time() if now is None else now
        with self._lock:
            # Sweeping here is what keeps the registry bounded. Without it, a long-lived
            # proxy accumulates one entry per origin that ever failed, and nothing clears
            # them: `record_success` only reaches whoever gets it right.
            self._purge_locked(moment)
            recent = [t for t in self._failures.get(origin, ()) if moment - t < self._window_s]
            recent.append(moment)
            count = len(recent)
            if count >= self._max_failures:
                self._blocked_until[origin] = moment + self._block_s
                # The list goes: the failures were already paid for with this block, and
                # keeping them would get the origin blocked again on the first mistake
                # after it comes back.
                self._failures.pop(origin, None)
            else:
                self._failures[origin] = recent

        if count >= self._max_failures:
            _LOG.error(
                "mysubs: origin %s blocked for %.0fs after %d failed deposits",
                origin,
                self._block_s,
                count,
            )
        else:
            _LOG.warning(
                "mysubs: failed deposit from origin %s (%d/%d in window)",
                origin,
                count,
                self._max_failures,
            )

    def record_success(self, origin: str) -> None:
        """Valid redemption: this origin's history ceases to exist."""
        with self._lock:
            self._failures.pop(origin, None)
            self._blocked_until.pop(origin, None)

    def purge(self, *, now: float | None = None) -> int:
        """Discards history outside the window and expired blocks. Returns how many origins
        left the registry."""
        moment = time.time() if now is None else now
        with self._lock:
            return self._purge_locked(moment)

    def _purge_locked(self, now: float) -> int:
        before = len(self._failures) + len(self._blocked_until)
        self._failures = {
            origin: alive
            for origin, failures in self._failures.items()
            if (alive := [t for t in failures if now - t < self._window_s])
        }
        self._blocked_until = {
            origin: until for origin, until in self._blocked_until.items() if now < until
        }
        return before - len(self._failures) - len(self._blocked_until)


def retry_after(seconds: float) -> str:
    """The `Retry-After`, in whole seconds. Rounds up and never yields `0`: a client that
    reads `Retry-After: 0` resends immediately, and gets another 429."""
    return str(max(1, math.ceil(seconds)))
