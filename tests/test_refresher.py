"""Background refresher: the loop that renews without depending on traffic."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from litellm_mysubs.credentials import refresher as module
from litellm_mysubs.credentials.lock import LockBusyError
from litellm_mysubs.credentials.refresher import BackgroundRefresher
from litellm_mysubs.credentials.store import Credential, CredentialStore, ProviderId

NOW = 1_700_000_000.0

#: Inside the five-minute margin: time to refresh.
SOON = NOW + 60.0

#: Well beyond the margin: touching this would burn a refresh token for nothing.
LATER = NOW + 3600.0


class FakeStore(CredentialStore):
    """In-memory store that counts what was done to it."""

    def __init__(
        self,
        creds: dict[ProviderId, Credential],
        *,
        owns: bool = True,
        path: Path | None = None,
    ) -> None:
        self.owns_refresh = owns
        #: The refresher derives the lock target from here. Pointed at a `tmp_path` so the
        #: lock files do not end up in the `~/.litellm` of whoever runs the tests.
        self.path = path
        self.creds = dict(creds)
        self.reloads = 0
        #: What a re-read will see, simulating another process writing while we were
        #: waiting for the lock.
        self.on_reload: dict[ProviderId, Credential] = {}

    def get(self, provider: ProviderId) -> Credential | None:
        return self.creds.get(provider)

    def set(self, provider: ProviderId, credential: Credential) -> None:
        self.creds[provider] = credential

    def delete(self, provider: ProviderId) -> None:
        self.creds.pop(provider, None)

    def reload(self) -> bool:
        self.reloads += 1
        self.creds.update(self.on_reload)
        return True


class FakeClient:
    """Enough for the refresher's `async with`; never talks to anyone."""

    async def __aenter__(self) -> FakeClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None


def credential(provider: ProviderId, expires_at: float) -> Credential:
    return Credential(
        provider=provider,
        access_token=f"old-{provider}",
        refresh_token=f"rt-{provider}",
        expires_at=expires_at,
        project_id="proj",
    )


class SpyRefresh:
    """Replaces `oauth.refresh`, recording who was refreshed."""

    def __init__(self, *, fail: set[ProviderId] | None = None) -> None:
        self.calls: list[ProviderId] = []
        self.fail = fail or set()

    async def __call__(
        self, cred: Credential, *, client: Any, store: CredentialStore | None = None
    ) -> Credential:
        self.calls.append(cred.provider)
        if cred.provider in self.fail:
            raise RuntimeError(f"upstream is down for {cred.provider}")
        return cred.with_access_token(f"new-{cred.provider}", LATER)


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> SpyRefresh:
    fake = SpyRefresh()
    monkeypatch.setattr(module.oauth, "refresh", fake)
    return fake


def build(store: FakeStore, tmp_path: Path) -> BackgroundRefresher:
    store.path = tmp_path / "credentials.json"
    return BackgroundRefresher(store, client_factory=FakeClient)


def by_provider(reports: list[module.RefreshReport]) -> dict[ProviderId, module.RefreshReport]:
    return {r.provider: r for r in reports}


class TestWhenItRefreshes:
    async def test_credential_close_to_expiry_is_refreshed_and_stored(
        self, spy: SpyRefresh, tmp_path: Path
    ) -> None:
        store = FakeStore({"anthropic": credential("anthropic", SOON)})
        reports = await build(store, tmp_path).sweep(now=NOW)

        assert spy.calls == ["anthropic"]
        assert by_provider(reports)["anthropic"] == module.RefreshReport(
            "anthropic", True, module.REASON_RENEWED
        )
        # Refreshing without storing is the worst of both worlds: it spends the refresh
        # token and the next sweep again finds the credential about to expire.
        assert store.creds["anthropic"].access_token == "new-anthropic"

    async def test_credential_far_from_expiry_is_not_touched(
        self, spy: SpyRefresh, tmp_path: Path
    ) -> None:
        store = FakeStore({"anthropic": credential("anthropic", LATER)})
        reports = await build(store, tmp_path).sweep(now=NOW)

        assert spy.calls == []
        assert by_provider(reports)["anthropic"].reason == module.REASON_STILL_VALID
        assert store.creds["anthropic"].access_token == "old-anthropic"

    async def test_unknown_expiry_does_not_burn_the_token_every_minute(
        self, spy: SpyRefresh, tmp_path: Path
    ) -> None:
        # `expires_at=0` is a hand-pasted token: "we do not know", not "it expired".
        # Treating it as expired would make the loop rotate the refresh token every minute.
        store = FakeStore({"anthropic": credential("anthropic", 0.0)})
        reports = await build(store, tmp_path).sweep(now=NOW)

        assert spy.calls == []
        assert by_provider(reports)["anthropic"].renewed is False

    async def test_provider_not_connected_is_not_an_error(
        self, spy: SpyRefresh, tmp_path: Path
    ) -> None:
        reports = await build(FakeStore({}), tmp_path).sweep(now=NOW)

        assert spy.calls == []
        assert {r.reason for r in reports} == {module.REASON_NOT_CONNECTED}
        assert all(r.renewed is False for r in reports)


class TestSingleOwner:
    async def test_store_that_is_not_the_owner_refreshes_nothing(
        self, spy: SpyRefresh, tmp_path: Path
    ) -> None:
        # Everything expiring and still nothing is touched: the owner's copy would be
        # invalidated.
        store = FakeStore(
            {p: credential(p, SOON) for p in module.PROVIDER_IDS},
            owns=False,
        )
        reports = await build(store, tmp_path).sweep(now=NOW)

        assert spy.calls == []
        assert len(reports) == len(module.PROVIDER_IDS)
        assert all(r.renewed is False for r in reports)

    async def test_busy_lock_does_not_count_as_an_error(
        self, spy: SpyRefresh, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def busy(path: Path, *, timeout_s: float = 0.0) -> Any:
            raise LockBusyError(f"busy: {path}")

        monkeypatch.setattr(module, "file_lock", busy)
        store = FakeStore({"anthropic": credential("anthropic", SOON)})
        report = by_provider(await build(store, tmp_path).sweep(now=NOW))["anthropic"]

        assert spy.calls == []
        assert report.renewed is False
        # Its own reason, distinct from a failure: with several workers this is normal.
        assert report.reason == module.REASON_LOCKED

    async def test_re_read_inside_the_lock_avoids_refreshing_on_top(
        self, spy: SpyRefresh, tmp_path: Path
    ) -> None:
        store = FakeStore({"anthropic": credential("anthropic", SOON)})
        # Another process refreshed between the decision and the lock acquisition.
        store.on_reload = {"anthropic": credential("anthropic", LATER)}

        report = by_provider(await build(store, tmp_path).sweep(now=NOW))["anthropic"]

        assert store.reloads == 1
        # Spending the refresh token on top of what it wrote is the `invalid_grant`.
        assert spy.calls == []
        assert report.reason == module.REASON_ALREADY_RENEWED


class TestResilience:
    async def test_failure_in_one_provider_does_not_block_the_next(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spy = SpyRefresh(fail={"anthropic"})
        monkeypatch.setattr(module.oauth, "refresh", spy)
        store = FakeStore({p: credential(p, SOON) for p in module.PROVIDER_IDS})

        reports = by_provider(await build(store, tmp_path).sweep(now=NOW))

        assert reports["anthropic"].renewed is False
        assert "is down" in reports["anthropic"].reason
        assert reports["openai-codex"].renewed is True
        assert reports["google-antigravity"].renewed is True
        assert store.creds["openai-codex"].access_token == "new-openai-codex"


class TestLoop:
    async def test_start_twice_does_not_create_two_tasks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sweeps = 0
        parked = asyncio.Event()

        async def park(_: float) -> None:
            # Each task does exactly one sweep and stays here. Without this brake the
            # count would measure the scheduler's speed, not the number of tasks.
            await parked.wait()

        monkeypatch.setattr(module, "_sleep", park)
        refresher = build(FakeStore({}), tmp_path)

        async def counted(**_: Any) -> list[module.RefreshReport]:
            nonlocal sweeps
            sweeps += 1
            return []

        refresher.sweep = counted  # type: ignore[method-assign]

        refresher.start()
        first = refresher._task
        refresher.start()
        assert refresher._task is first
        assert refresher.running is True

        for _ in range(6):
            await asyncio.sleep(0)

        # Two tasks would give two sweeps — and each sweep exchanges single-use refresh
        # tokens.
        assert sweeps == 1

        await refresher.stop()

        assert refresher.running is False
        # `stop()` has to **wait** for the task: cancelling and moving on would leave a
        # sweep halfway through an exchange with the lock file still in hand.
        assert first is not None and first.done()

    async def test_stop_really_stops_and_is_safe_without_start(self, tmp_path: Path) -> None:
        refresher = build(FakeStore({}), tmp_path)
        await refresher.stop()  # with no task at all it does not blow up
        assert refresher.running is False

    async def test_a_sweep_that_blows_up_does_not_kill_the_loop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rounds = 0

        async def exploding(**_: Any) -> list[module.RefreshReport]:
            nonlocal rounds
            rounds += 1
            raise RuntimeError("unexpected")

        async def instant(_: float) -> None:
            await asyncio.sleep(0)

        monkeypatch.setattr(module, "_sleep", instant)
        refresher = build(FakeStore({}), tmp_path)
        refresher.sweep = exploding  # type: ignore[method-assign]

        refresher.start()
        for _ in range(6):
            await asyncio.sleep(0)

        # A refresher that died silently would still report `running=True`.
        assert rounds > 1
        assert refresher.running is True
        await refresher.stop()
        assert refresher.running is False

    def test_without_an_event_loop_start_gives_up_without_blowing_up(
        self, tmp_path: Path
    ) -> None:
        # This is called from the plugin startup; blowing up here would bring down the
        # whole proxy over a convenience.
        refresher = build(FakeStore({}), tmp_path)
        refresher.start()
        assert refresher.running is False
