"""Periodic credential renewal, independent of traffic.

Until now the only place a token was renewed was the request path
(``plugin.py :: _access_token``). That covers a busy proxy and fails in exactly the case
where the user looks at the UI: an installation left idle overnight wakes up with the three
cards saying "token expired", and the first thing the user does is reconnect the
subscription by hand — a re-login that was never needed, because the refresh token was
still valid. Nobody had run to exchange it.

This module is the missing loop: it sweeps the credentials every ``interval_s`` and
exchanges those within ``skew_s`` of expiry.

Three precautions, all inherited from incidents already measured elsewhere in the package:

* **Single owner, in two layers.** The first is the rule at the top of ``store.py``: a store
  with ``owns_refresh=False`` reads and never exchanges, so the sweep does not even start.
  The second is ``file_lock``: being the owner is not enough when LiteLLM runs with
  ``--num_workers > 1``, because each worker is a distinct process with its own refresher
  waking up at the same time. Without the lock there are N refreshers exchanging the same
  single-use refresh token.
* **Re-read inside the lock.** Between deciding "it is expiring" and acquiring the lock
  there is room for another process to complete a whole renewal. Renewing on top of what it
  just wrote spends an already-rotated refresh token, and the result is the
  ``invalid_grant`` that all the rest of the package exists to avoid.
* **One failure does not bring the loop down.** Provider down, network cut, broken
  credential: it lands in the ``RefreshReport`` and the sweep carries on. A refresher that
  dies on the first failure is worse than having no refresher at all, because it keeps
  looking like it is working.

``sweep()`` is the unit with behaviour; ``start()``/``stop()`` are just the loop around it.
It is also what gets called by hand from the UI without waiting for the next interval.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import httpx

from . import oauth
from .file_store import DEFAULT_PATH
from .lock import LockBusyError, file_lock
from .store import PROVIDER_IDS, CredentialStore, ProviderId

__all__ = [
    "DEFAULT_INTERVAL_S",
    "DEFAULT_SKEW_S",
    "BackgroundRefresher",
    "RefreshReport",
]

_LOG = logging.getLogger(__name__)

#: How often the loop wakes up. One minute is generous against the five-minute skew: even
#: losing a whole sweep, four opportunities remain before the token dies.
DEFAULT_INTERVAL_S: float = 60.0

#: How long before expiry a renewal happens. The five minutes are the same the request path
#: already uses: they cover a clock out of sync with the provider's and a slow network round
#: trip, without being so early that a token is burned on every sweep.
DEFAULT_SKEW_S: float = 300.0

#: Stable reasons, so whoever reads the report does not have to compare sentences. Only a
#: real failure escapes this list: that is the upstream message, the only one that says what
#: to do next.
REASON_RENEWED: Final = "renewed"
REASON_STILL_VALID: Final = "still valid"
REASON_UNKNOWN_EXPIRY: Final = "unknown validity"
REASON_NOT_CONNECTED: Final = "not connected"
REASON_LOCKED: Final = "another process is renewing"
REASON_ALREADY_RENEWED: Final = "already renewed by another"
REASON_NOT_OWNER: Final = "store does not own the refresh"


@dataclass(frozen=True, slots=True)
class RefreshReport:
    """What happened to one provider during a sweep.

    One is returned per provider, even for those left untouched: the UI wants to know the
    credential was looked at and considered valid, and "did not appear in the report" does
    not distinguish that from "the sweep blew up before getting there".
    """

    provider: ProviderId
    renewed: bool
    reason: str


async def _sleep(seconds: float) -> None:
    """Deliberate indirection: it is the only real wait point, and tests replace it."""
    await asyncio.sleep(seconds)


def _describe(exc: BaseException) -> str:
    """The failure as text, preferring the message over the class name.

    An ``OAuthError`` carries the upstream body; swapping it for the type name would erase
    the only actionable part. With no message (a bare ``TimeoutError``, for instance) the
    name remains, which still distinguishes a network failure from a broken credential.
    """
    text = str(exc).strip()
    return text or type(exc).__name__


class BackgroundRefresher:
    """Renews expiring credentials without depending on there being requests."""

    def __init__(
        self,
        store: CredentialStore,
        *,
        interval_s: float = DEFAULT_INTERVAL_S,
        skew_s: float = DEFAULT_SKEW_S,
        client_factory: Any = None,
    ) -> None:
        self.store = store
        self.interval_s = interval_s
        self.skew_s = skew_s
        #: Injectable so tests open no sockets; in production it is the `httpx` client.
        self.client_factory: Any = httpx.AsyncClient if client_factory is None else client_factory
        self._task: asyncio.Task[None] | None = None

    # -- sweep -----------------------------------------------------------------

    async def sweep(self, *, now: float | None = None) -> list[RefreshReport]:
        """One pass over every provider. Never raises.

        ``now`` is injectable so tests can put a credential on the edge of expiry without
        touching the machine's clock.
        """
        moment = time.time() if now is None else now

        if not self.store.owns_refresh:
            # Rule 1 from the top of `store.py`, applied before even looking at the
            # credentials: this process reads from a source another one owns, and exchanging
            # a token here would invalidate the owner's copy.
            return [RefreshReport(p, False, REASON_NOT_OWNER) for p in PROVIDER_IDS]

        reports: list[RefreshReport] = []
        for provider in PROVIDER_IDS:
            try:
                reports.append(await self._sweep_one(provider, moment))
            except LockBusyError:
                # Not a failure: it is the other worker doing the work. Recording this as
                # an error would fill the report with red on a proxy with several healthy
                # workers.
                reports.append(RefreshReport(provider, False, REASON_LOCKED))
            except Exception as exc:  # see "one failure does not bring the loop down", above
                reports.append(RefreshReport(provider, False, _describe(exc)))
        return reports

    async def _sweep_one(self, provider: ProviderId, now: float) -> RefreshReport:
        credential = self.store.get(provider)
        if credential is None:
            return RefreshReport(provider, False, REASON_NOT_CONNECTED)

        due = self._due(credential.expires_at, now)
        if due is not None:
            return RefreshReport(provider, False, due)

        with file_lock(self._lock_target(provider), timeout_s=0.0):
            return await self._renew(provider, now)

    def _due(self, expires_at: float, now: float) -> str | None:
        """``None`` if it is time to renew; the reason not to renew otherwise.

        ``expires_at`` at zero is "unknown", not "expired" — it is what `Credential` assumes
        and what happens to a token pasted by hand. Treating it as lapsed would put this
        loop burning one refresh token per sweep, minute after minute, over the one
        credential we know nothing about. What discovers that such a token died is the 401
        on the request path.
        """
        if expires_at <= 0:
            return REASON_UNKNOWN_EXPIRY
        if expires_at - now >= self.skew_s:
            return REASON_STILL_VALID
        return None

    async def _renew(self, provider: ProviderId, now: float) -> RefreshReport:
        """The exchange itself, already holding the lock."""
        # The re-read is the whole point of the lock: while waiting for it, another process
        # may have renewed and written. Renewing on top would spend an already-rotated
        # single-use refresh token, which is how one gets to `invalid_grant` in a loop.
        self.store.reload()
        current = self.store.get(provider)
        if current is None:
            return RefreshReport(provider, False, REASON_NOT_CONNECTED)
        if self._due(current.expires_at, now) is not None:
            return RefreshReport(provider, False, REASON_ALREADY_RENEWED)

        async with self.client_factory() as client:
            renewed = await oauth.refresh(current, client=client, store=self.store)
        self.store.set(provider, renewed)
        return RefreshReport(provider, True, REASON_RENEWED)

    def _lock_target(self, provider: ProviderId) -> Path:
        """The lock target: one per provider, next to the credentials file.

        Per provider and not global because the three subscriptions renew at independent
        moments, and a shared lock would put a slow Antigravity delaying Anthropic's renewal
        until the next sweep.

        Stores with no file (the Secret Manager one) fall back to the default path: the lock
        does not guard the file, it guards the *right to exchange the token*, and for that
        it only needs to be the same path across every worker on the same machine.
        """
        raw = getattr(self.store, "path", None)
        base = Path(raw) if isinstance(raw, str | Path) else DEFAULT_PATH
        return base.with_name(f"{base.name}.{provider}")

    # -- loop ------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        """Starts the loop. Calling it twice does not create two refreshers.

        With no loop running it does not raise: this is usually called from plugin startup,
        and blowing up there would bring the whole proxy down over a convenience.
        """
        if self.running:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            _LOG.warning(
                "background refresher did not start: no event loop running; "
                "credentials will only be renewed on the request path"
            )
            return
        self._task = loop.create_task(self._loop())

    async def stop(self) -> None:
        """Cancels and **waits** for the task.

        Waiting is not courtesy: a sweep halfway through an exchange holds the lock, and
        giving up without seeing it finish would leave the lock file stuck until the process
        died.
        """
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    async def _loop(self) -> None:
        while True:
            try:
                await self.sweep()
            except asyncio.CancelledError:
                raise
            except Exception:  # the loop survives everything but cancellation
                # `sweep()` already absorbs what is its own; anything reaching here is
                # unforeseen, and letting it kill the task would give a silently dead
                # refresher — `running` would keep saying yes until someone noticed the
                # expired cards.
                _LOG.exception("background refresher sweep failed; the loop continues")
            await _sleep(self.interval_s)
