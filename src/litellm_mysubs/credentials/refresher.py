"""Credential renewal: the one place a refresh token is spent.

Every renewal — the one before a request, the one after a 401, the button in the UI and
the periodic sweep — goes through ``_refresh_stored`` here, the port of omp's
``OAuthRefresher``. omp makes ``AuthStorage`` the sole refresh authority for the same
reason this module exists: Anthropic's and OpenAI's refresh tokens are single-use and
rotating, and two renewals of the same credential leave one of them holding a token the
provider has already rotated away. The measured symptom is ``invalid_grant`` in a loop,
across every worker, until somebody logs in by hand.

The guarantees, each with its omp counterpart:

* **One exchange at a time, per credential.** In the process, concurrent callers join the
  renewal already in flight (omp's per-credential single-flight). Across processes — each
  LiteLLM worker is one, and the credentials file may sit on a volume several pods share —
  the exchange happens under ``file_lock`` on ``credentials.json.<provider>.lock`` (omp's
  refresh lease). Same lock file as 0.1.16, so a mixed rollout still excludes itself.
* **Re-read inside the lock.** Between deciding "it is expiring" and holding the lock
  another process may have renewed and written. What it wrote is adopted instead of
  renewing on top of it.
* **Compare-and-set on the way out.** The result is written only while the store still
  holds the credential that was exchanged; a login or a peer that wrote meanwhile wins.
* **A dead grant is dropped.** ``invalid_grant``, ``revoked``, a bare 401: the credential is
  removed (omp disables the row) — again only if it is still the one that failed, so a peer
  that rotated it in the meantime keeps its fresh copy. Keeping a dead credential only had
  the sweep present it again every minute.
* **Bounded.** The exchange has ``REFRESH_TIMEOUT_S``; waiting for a peer's lock has
  ``LOCK_WAIT_S``.

Where it diverges from omp, on purpose: omp's broker refresher is the only refresher of its
store and forces every renewal it starts. Here each worker runs its own sweep, so the sweep
re-checks freshness inside the lock (``REASON_ALREADY_RENEWED``) and does not wait for a
busy lock (``REASON_LOCKED``): the worker holding it is doing the same job.

The sweep exists because renewing only on the request path fails exactly when the user
looks at the UI: an installation idle overnight wakes up with every card expired, and the
user reconnects by hand a subscription whose refresh token was still valid.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from collections.abc import Callable, Coroutine
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import httpx

from . import oauth
from .file_store import DEFAULT_PATH
from .lock import LockBusyError, async_file_lock
from .store import OAUTH_REFRESH_SKEW_S, PROVIDER_IDS, Credential, CredentialStore, ProviderId

__all__ = [
    "DEFAULT_INTERVAL_S",
    "DEFAULT_SKEW_S",
    "BackgroundRefresher",
    "RefreshReport",
    "force_refresh",
    "fresh",
    "lock_target",
    "recover",
]

_LOG = logging.getLogger(__name__)

# omp: auth-broker/types.ts :: DEFAULT_REFRESH_INTERVAL_MS
#: How often the sweep wakes up. Against the five-minute skew, even a lost sweep leaves four
#: more chances before the token dies.
DEFAULT_INTERVAL_S: float = 60.0

# omp: auth-broker/types.ts :: DEFAULT_REFRESH_SKEW_MS
#: How long before expiry the sweep renews. Wider than the request path's minute so that
#: in steady state a request never has to wait for a renewal.
DEFAULT_SKEW_S: float = 300.0

# omp: auth/refresh.ts :: OAUTH_REFRESH_OPERATION_TIMEOUT_MS
#: Ceiling of one exchange, client creation included. A token endpoint that hangs must not
#: pin the lock — and with it every worker that needs this credential.
REFRESH_TIMEOUT_S: Final = 10.0

# omp: auth/refresh.ts :: OAUTH_REFRESH_LEASE_TTL_MS
#: How long the request path waits for another process's renewal. omp trusts a silent lease
#: holder this long; a live holder is done well within it (``REFRESH_TIMEOUT_S``), and a
#: dead one releases the ``flock`` with its process.
LOCK_WAIT_S: Final = 15.0

# omp: auth/refresh.ts :: OAUTH_REFRESH_LEASE_POLL_MS
LOCK_POLL_S: Final = 0.05

# omp: auth/refresh.ts :: OAUTH_REMINT_COOLDOWN_MS
#: How long a 401 may be answered with a token this process minted itself instead of a new
#: one. Re-minting a token the provider rejected moments ago only rotates the refresh token
#: again — during a provider-wide outage, once per rejected request.
REMINT_COOLDOWN_S: Final = 300.0

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

ClientFactory = Callable[[], Any]


@dataclass(frozen=True, slots=True)
class _Outcome:
    """What a renewal attempt left in the store."""

    credential: Credential | None
    #: Whether *this* attempt exchanged the refresh token (as opposed to adopting a peer's).
    refreshed: bool


#: Renewals in flight, per lock target and wait mode. The mode is part of the key because a
#: sweep gives up on a busy lock and a request waits for it: a request joining the sweep's
#: attempt would inherit a `LockBusyError` instead of the peer's fresh token.
_IN_FLIGHT: dict[tuple[str, bool], asyncio.Task[_Outcome]] = {}

#: Access token this process last minted, per lock target, and when.
_RECENT_MINTS: dict[str, tuple[str, float]] = {}

#: Access token last handed to a request, per lock target. It is what tells a 401 on the
#: token the store still holds from a 401 on one a peer has already replaced.
_SERVED: dict[str, str] = {}


def lock_target(store: Any, provider: ProviderId) -> Path:
    """The lock target: one per provider, next to the credentials file.

    Per provider and not global because the three subscriptions renew at independent
    moments, and a shared lock would put a slow Antigravity in front of Anthropic's renewal.

    Stores with no file (the Secret Manager one) fall back to the default path: the lock
    does not guard the file, it guards the *right to exchange the token*, and for that it
    only needs to be the same path across every worker on the same machine.
    """
    raw = getattr(store, "path", None)
    base = Path(raw) if isinstance(raw, str | Path) else DEFAULT_PATH
    return base.with_name(f"{base.name}.{provider}")


def _fresh_at(credential: Credential, now: float, skew_s: float) -> bool:
    """omp's ``Date.now() + skew < expires``, with our zero: unknown expiry is never due.

    A hand-pasted token carries no validity; treating it as lapsed would burn a refresh
    token on every pass over the one credential we know nothing about.
    """
    return credential.expires_at <= 0 or now + skew_s < credential.expires_at


def _update_if_matches(
    store: Any, provider: ProviderId, expected: Credential, credential: Credential
) -> bool:
    """The store's own compare-and-set when it has one; re-read and compare otherwise."""
    method = getattr(store, "update_if_matches", None)
    if method is not None:
        return bool(method(provider, expected, credential))
    store.reload()
    if store.get(provider) != expected:
        return False
    store.set(provider, credential)
    return True


def _delete_if_matches(store: Any, provider: ProviderId, expected: Credential) -> bool:
    method = getattr(store, "delete_if_matches", None)
    if method is not None:
        return bool(method(provider, expected))
    store.reload()
    if store.get(provider) != expected:
        return False
    store.delete(provider)
    return True


# omp: auth/refresh.ts :: OAuthRefresher.refreshStored
async def _refresh_stored(
    store: Any,
    provider: ProviderId,
    *,
    observed: Credential | None,
    force: bool,
    skew_s: float,
    wait_s: float,
    client_factory: ClientFactory,
    now: float | None,
) -> _Outcome:
    """Renews under the lock, deciding again with what the store holds by then.

    ``observed`` is what the caller decided on. A different, still fresh credential in the
    store means a peer renewed while this one waited — it is adopted, even when ``force``
    asks for a new token. ``force`` only skips the "still fresh" check (a 401, the button).
    """
    async with async_file_lock(
        lock_target(store, provider), timeout_s=wait_s, poll_s=LOCK_POLL_S
    ):
        store.reload()
        current = store.get(provider)
        if current is None:
            return _Outcome(None, False)
        moment = time.time() if now is None else now
        fresh_now = _fresh_at(current, moment, skew_s)
        if observed is not None and current != observed and fresh_now:
            return _Outcome(current, False)
        if not force and fresh_now:
            return _Outcome(current, False)

        try:
            async with asyncio.timeout(REFRESH_TIMEOUT_S):
                async with client_factory() as client:
                    renewed = await oauth.refresh(current, client=client, store=store)
        except TimeoutError:
            raise oauth.OAuthError(
                provider,
                f"token refresh timed out after {REFRESH_TIMEOUT_S:g}s for provider: {provider}",
            ) from None
        except Exception as exc:
            if not oauth.is_definitive_failure(exc):
                raise
            if _delete_if_matches(store, provider, current):
                _LOG.warning(
                    "%s: the provider rejected the refresh token for good (%s); "
                    "credential removed — connect the subscription again",
                    provider,
                    exc,
                )
                raise
            # Someone replaced the credential while the exchange failed: theirs is live.
            store.reload()
            return _Outcome(store.get(provider), False)

        if not _update_if_matches(store, provider, current, renewed):
            store.reload()
            return _Outcome(store.get(provider), False)
        _RECENT_MINTS[str(lock_target(store, provider))] = (renewed.access_token, time.time())
        return _Outcome(renewed, True)


def _land(key: tuple[str, bool], task: asyncio.Task[_Outcome]) -> None:
    if _IN_FLIGHT.get(key) is task:
        del _IN_FLIGHT[key]
    if not task.cancelled():
        # Retrieved here so a renewal whose waiters all left does not log "exception was
        # never retrieved"; each waiter still gets it through its own `await`.
        task.exception()


# omp: auth/refresh.ts :: OAuthRefresher.refreshSingleFlight
async def _single_flight(
    target: str, wait: bool, start: Callable[[], Coroutine[Any, Any, _Outcome]]
) -> _Outcome:
    """Joins the renewal of ``target`` already in flight, or starts it.

    Shielded: a caller that goes away — a client that disconnects mid-request — must not
    cancel an exchange the provider may already have answered with a rotated token.
    """
    loop = asyncio.get_running_loop()
    key = (target, wait)
    task = _IN_FLIGHT.get(key)
    if task is None or task.done() or task.get_loop() is not loop:
        task = loop.create_task(start())
        _IN_FLIGHT[key] = task
        task.add_done_callback(functools.partial(_land, key))
    return await asyncio.shield(task)


def _factory(client_factory: ClientFactory | None) -> ClientFactory:
    return httpx.AsyncClient if client_factory is None else client_factory


async def fresh(
    store: Any, provider: ProviderId, *, client_factory: ClientFactory | None = None
) -> Credential | None:
    """The credential for a request, renewed first when it is within a minute of expiry.

    A failed renewal is not raised: the request goes out with the token there is, and the
    upstream's own answer is what the client reads. A still valid token touches neither the
    lock nor the store's source — this runs once per request.

    A store that does not own the refresh only re-reads its source: the owner may have
    renewed already, and that saves the round trip of a certain 401.
    """
    credential: Credential | None = store.get(provider)
    if credential is None:
        return None
    target = str(lock_target(store, provider))
    if credential.is_expired(leeway_s=OAUTH_REFRESH_SKEW_S):
        credential = await _renewed_or(store, provider, credential, client_factory)
    _SERVED[target] = credential.access_token
    return credential


async def _renewed_or(
    store: Any,
    provider: ProviderId,
    credential: Credential,
    client_factory: ClientFactory | None,
) -> Credential:
    """What `fresh` hands out for a due credential: the renewed one, else ``credential``."""
    if not getattr(store, "owns_refresh", False) or not credential.refresh_token:
        try:
            store.reload()
            return store.get(provider) or credential
        except Exception:
            return credential
    try:
        outcome = await _single_flight(
            str(lock_target(store, provider)),
            True,
            lambda: _refresh_stored(
                store,
                provider,
                observed=credential,
                force=False,
                skew_s=OAUTH_REFRESH_SKEW_S,
                wait_s=LOCK_WAIT_S,
                client_factory=_factory(client_factory),
                now=None,
            ),
        )
    except Exception as exc:
        _LOG.debug("%s: renewal before the request failed: %s", provider, exc)
        return credential
    return outcome.credential or credential


# omp: auth/refresh.ts :: OAuthRefresher.refresh
# omp: auth/refresh.ts :: OAuthRefresher.recentMint
async def recover(
    store: Any, provider: ProviderId, *, client_factory: ClientFactory | None = None
) -> Credential | None:
    """The credential to retry with after the upstream answered 401.

    In omp's order: a peer that replaced the token since it was handed out wins; a token
    this process minted within ``REMINT_COOLDOWN_S`` is reused; otherwise a new one is
    minted even though the clock says the old one is valid — the provider has just said it
    is not. A store that cannot renew (not the owner, no refresh token) only re-reads, and
    offers what it read only while the clock still calls it valid: retrying with a token
    both the clock and the provider call dead is a second certain 401. Raises when the
    re-read or the renewal fails.
    """
    store.reload()
    current: Credential | None = store.get(provider)
    if current is None:
        return None
    if not getattr(store, "owns_refresh", False) or not current.refresh_token:
        return None if current.is_expired(leeway_s=OAUTH_REFRESH_SKEW_S) else current
    target = str(lock_target(store, provider))
    now = time.time()
    served = _SERVED.get(target)
    if served is not None and served != current.access_token and _fresh_at(
        current, now, OAUTH_REFRESH_SKEW_S
    ):
        return current
    if (target, True) not in _IN_FLIGHT and _recent_mint(target, current, now):
        return current
    outcome = await _single_flight(
        target,
        True,
        lambda: _refresh_stored(
            store,
            provider,
            observed=current,
            force=True,
            skew_s=OAUTH_REFRESH_SKEW_S,
            wait_s=LOCK_WAIT_S,
            client_factory=_factory(client_factory),
            now=None,
        ),
    )
    return outcome.credential


def _recent_mint(target: str, current: Credential, now: float) -> bool:
    mint = _RECENT_MINTS.get(target)
    if mint is None:
        return False
    access, at = mint
    if (
        now - at >= REMINT_COOLDOWN_S
        or current.access_token != access
        or not _fresh_at(current, now, OAUTH_REFRESH_SKEW_S)
    ):
        del _RECENT_MINTS[target]
        return False
    return True


# omp: auth/refresh.ts :: OAuthRefresher.refreshById
async def force_refresh(
    store: Any, provider: ProviderId, *, client_factory: ClientFactory | None = None
) -> Credential:
    """Mints a new token now, whatever the clock says — the UI's renew button.

    Still single-flight, locked and compare-and-set: pressing the button while a worker
    renews joins or follows that renewal instead of racing it.
    """
    outcome = await _single_flight(
        str(lock_target(store, provider)),
        True,
        lambda: _refresh_stored(
            store,
            provider,
            observed=None,
            force=True,
            skew_s=OAUTH_REFRESH_SKEW_S,
            wait_s=LOCK_WAIT_S,
            client_factory=_factory(client_factory),
            now=None,
        ),
    )
    if outcome.credential is None:
        raise LookupError(f"{provider} is not connected")
    return outcome.credential


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


# omp: auth-broker/refresher.ts :: AuthBrokerRefresher
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

    # omp: auth-broker/refresher.ts :: AuthBrokerRefresher.tick
    async def sweep(self, *, now: float | None = None) -> list[RefreshReport]:
        """One pass over every provider, all of them at once. Never raises.

        ``now`` is injectable so tests can put a credential on the edge of expiry without
        touching the machine's clock.
        """
        moment = time.time() if now is None else now

        if not self.store.owns_refresh:
            # Rule 1 from the top of `store.py`, applied before even looking at the
            # credentials: this process reads from a source another one owns, and exchanging
            # a token here would invalidate the owner's copy.
            return [RefreshReport(p, False, REASON_NOT_OWNER) for p in PROVIDER_IDS]

        return list(await asyncio.gather(*(self._report(p, moment) for p in PROVIDER_IDS)))

    async def _report(self, provider: ProviderId, now: float) -> RefreshReport:
        try:
            return await self._sweep_one(provider, now)
        except LockBusyError:
            # Not a failure: it is the other worker doing the work. Recording this as an
            # error would fill the report with red on a proxy with several healthy workers.
            return RefreshReport(provider, False, REASON_LOCKED)
        except Exception as exc:  # one failure does not bring the sweep down
            return RefreshReport(provider, False, _describe(exc))

    async def _sweep_one(self, provider: ProviderId, now: float) -> RefreshReport:
        credential = self.store.get(provider)
        if credential is None:
            return RefreshReport(provider, False, REASON_NOT_CONNECTED)

        due = self._due(credential.expires_at, now)
        if due is not None:
            return RefreshReport(provider, False, due)

        outcome = await _single_flight(
            str(lock_target(self.store, provider)),
            False,
            lambda: _refresh_stored(
                self.store,
                provider,
                observed=None,
                force=False,
                skew_s=self.skew_s,
                # No wait: the worker holding the lock is doing this very renewal, and the
                # next sweep finds its result. Waiting would pile every worker onto one
                # lock for a task that is not urgent.
                wait_s=0.0,
                client_factory=self.client_factory,
                now=now,
            ),
        )
        if outcome.refreshed:
            return RefreshReport(provider, True, REASON_RENEWED)
        if outcome.credential is None:
            return RefreshReport(provider, False, REASON_NOT_CONNECTED)
        return RefreshReport(provider, False, REASON_ALREADY_RENEWED)

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

    # -- loop ------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        """Starts the loop, sweeping at once. Calling it twice does not create two loops.

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
        """Cancels and **waits** for the loop.

        A renewal already in flight is not cancelled with it: it is shielded, finishes on
        its own and releases its lock — abandoning it after the provider rotated would lose
        the new refresh token.
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
