"""Active quota probe: what it may assert, and what it must refuse to assert.

The payloads here are the ones **measured** against the real accounts, HTTP 200, copied
without touch-ups. They cover the two ways this probe can lie:

* scaling a number wrong — `utilization: 15.0` is 15%, but the same name in the headers
  comes as a fraction, and passing this one through the same `* 100` showed 1500%;
* drawing a bar where there is no data — a 401, a 500 or a broken body must yield an empty
  `UsageSnapshot()`, so the UI writes "no data" instead of "0% used".

All with `httpx.MockTransport`: no network, no clock.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from litellm_mysubs.catalog.usage import UsageSnapshot
from litellm_mysubs.catalog.usage_probe import (
    ANTHROPIC_USAGE_URL,
    CODEX_USAGE_URL,
    probe,
)
from litellm_mysubs.credentials.store import Credential

Handler = Callable[[httpx.Request], httpx.Response]

ANTHROPIC = Credential(provider="anthropic", access_token="tok-a")
CODEX = Credential(provider="openai-codex", access_token="tok-c")
GOOGLE = Credential(provider="google-antigravity", access_token="tok-g")

#: Measured response of `GET /api/oauth/usage` on a Max account. The per-model windows came
#: back `null`: they are windows this plan does not have, not windows at zero.
ANTHROPIC_PAYLOAD: dict[str, object] = {
    "five_hour": {
        "utilization": 15.0,
        "resets_at": "2026-09-19T12:30:00.221085+00:00",
        "locked_reason": None,
    },
    "seven_day": {"utilization": 31.0, "resets_at": "2026-09-23T23:00:00.221105+00:00"},
    "seven_day_opus": None,
    "seven_day_sonnet": None,
}

#: Measured response of `GET /backend-api/wham/usage` on a Plus account.
CODEX_PAYLOAD: dict[str, object] = {
    "user_id": "user-1",
    "email": "someone@example.com",
    "plan_type": "plus",
    "rate_limit": {
        "allowed": True,
        "limit_reached": False,
        "primary_window": {
            "used_percent": 0,
            "limit_window_seconds": 18000,
            "reset_after_seconds": 18000,
            "reset_at": 1789830279,
        },
        "secondary_window": {
            "used_percent": 19,
            "limit_window_seconds": 604800,
            "reset_after_seconds": 400000,
            "reset_at": 1790230279,
        },
    },
}


def client(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def json_handler(payload: object, *, status: int = 200) -> tuple[Handler, list[httpx.Request]]:
    """Handler that records the requests, so one can assert what was (or was not) sent."""
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json=payload)

    return handle, seen


def labels(snapshot: UsageSnapshot) -> dict[str, float]:
    return {window.label: window.used_percent for window in snapshot.windows}


class TestAnthropic:
    @pytest.mark.asyncio
    async def test_utilization_is_already_a_percentage(self) -> None:
        """15.0 is 15%, not 1500%: this path does not multiply, unlike the headers."""
        handle, seen = json_handler(ANTHROPIC_PAYLOAD)
        async with client(handle) as http:
            snapshot = await probe(ANTHROPIC, client=http)

        assert labels(snapshot) == {"5h": 15.0, "7d": 31.0}
        assert str(seen[0].url) == ANTHROPIC_USAGE_URL
        assert seen[0].method == "GET"

    @pytest.mark.asyncio
    async def test_iso_reset_becomes_epoch(self) -> None:
        handle, _ = json_handler(ANTHROPIC_PAYLOAD)
        async with client(handle) as http:
            snapshot = await probe(ANTHROPIC, client=http)

        five_hour = next(w for w in snapshot.windows if w.label == "5h")
        # `2026-09-19T12:30:00.221085+00:00` as a UTC epoch.
        assert five_hour.resets_at == pytest.approx(1789821000.221085)

    @pytest.mark.asyncio
    async def test_scoped_window_adds_a_row_when_the_plan_has_it(self) -> None:
        """A filled `seven_day_opus` is one extra window, labelled from the key."""
        payload = {**ANTHROPIC_PAYLOAD, "seven_day_opus": {"utilization": 7.5}}
        handle, _ = json_handler(payload)
        async with client(handle) as http:
            snapshot = await probe(ANTHROPIC, client=http)

        assert labels(snapshot) == {"5h": 15.0, "7d": 31.0, "7d opus": 7.5}

    @pytest.mark.asyncio
    async def test_all_null_windows_are_not_known(self) -> None:
        """An account with no reported window is "unknown", not "0% used"."""
        payload = {"five_hour": None, "seven_day": None, "seven_day_opus": None}
        handle, _ = json_handler(payload)
        async with client(handle) as http:
            snapshot = await probe(ANTHROPIC, client=http)

        assert snapshot == UsageSnapshot()
        assert snapshot.known is False

    @pytest.mark.asyncio
    async def test_sends_the_measured_oauth_identity(self) -> None:
        """The route belongs to the CLI: without the OAuth beta and the Claude Code UA it is
        different traffic."""
        handle, seen = json_handler(ANTHROPIC_PAYLOAD)
        async with client(handle) as http:
            await probe(ANTHROPIC, client=http)

        headers = seen[0].headers
        assert headers["authorization"] == "Bearer tok-a"
        assert headers["anthropic-beta"] == "oauth-2025-04-20"
        assert headers["user-agent"] == "claude-cli/2.1.257 (external, cli)"
        assert headers["accept"] == "application/json"


class TestCodex:
    @pytest.mark.asyncio
    async def test_windows_plan_and_epoch_reset(self) -> None:
        handle, seen = json_handler(CODEX_PAYLOAD)
        async with client(handle) as http:
            snapshot = await probe(CODEX, client=http)

        assert labels(snapshot) == {"5h": 0.0, "7d": 19.0}
        assert snapshot.plan == "plus"
        # `reset_at` is an epoch in seconds, not milliseconds.
        assert {w.label: w.resets_at for w in snapshot.windows} == {
            "5h": 1789830279.0,
            "7d": 1790230279.0,
        }
        assert str(seen[0].url) == CODEX_USAGE_URL
        # The path does not carry `/codex/`: with it, measured, the backend returns 403.
        assert "/codex/" not in str(seen[0].url)

    @pytest.mark.asyncio
    async def test_labels_come_from_the_window_length(self) -> None:
        """The label comes from `limit_window_seconds`, not from the field position."""
        payload = {
            "plan_type": "pro",
            "rate_limit": {
                "primary_window": {"used_percent": 4, "limit_window_seconds": 3600},
                "secondary_window": {"used_percent": 8, "limit_window_seconds": 2592000},
            },
        }
        handle, _ = json_handler(payload)
        async with client(handle) as http:
            snapshot = await probe(CODEX, client=http)

        assert labels(snapshot) == {"1h": 4.0, "30d": 8.0}

    @pytest.mark.asyncio
    async def test_payload_without_rate_limit_keeps_the_plan(self) -> None:
        """The plan alone is already a fact about the account; `known` stays true."""
        handle, _ = json_handler({"plan_type": "plus"})
        async with client(handle) as http:
            snapshot = await probe(CODEX, client=http)

        assert snapshot.windows == ()
        assert snapshot.plan == "plus"
        assert snapshot.known is True


class TestFailuresAreEmptyNotZero:
    @pytest.mark.parametrize("status", [401, 403, 429, 500])
    @pytest.mark.asyncio
    async def test_non_200_is_empty(self, status: int) -> None:
        handle, _ = json_handler(ANTHROPIC_PAYLOAD, status=status)
        async with client(handle) as http:
            assert await probe(ANTHROPIC, client=http) == UsageSnapshot()

    @pytest.mark.asyncio
    async def test_broken_json_is_empty(self) -> None:
        def handle(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"{not json")

        async with client(handle) as http:
            assert await probe(CODEX, client=http) == UsageSnapshot()

    @pytest.mark.asyncio
    async def test_json_that_is_not_an_object_is_empty(self) -> None:
        handle, _ = json_handler([1, 2, 3])
        async with client(handle) as http:
            assert await probe(ANTHROPIC, client=http) == UsageSnapshot()

    @pytest.mark.asyncio
    async def test_timeout_does_not_propagate(self) -> None:
        """A quota probe that brings the page down is worse than no probe at all."""

        def handle(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("took too long", request=request)

        async with client(handle) as http:
            assert await probe(ANTHROPIC, client=http) == UsageSnapshot()

    @pytest.mark.asyncio
    async def test_connect_error_does_not_propagate(self) -> None:
        def handle(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no network", request=request)

        async with client(handle) as http:
            assert await probe(CODEX, client=http) == UsageSnapshot()


class TestAntigravity:
    @pytest.mark.asyncio
    async def test_no_request_is_made(self) -> None:
        """Antigravity has its own path (POST with `project`) in `ui/service.py`.

        The proof that it is not probed here is that the transport was never called: a GET
        to this endpoint would spend a request just to receive a 404.
        """
        handle, seen = json_handler(ANTHROPIC_PAYLOAD)
        async with client(handle) as http:
            snapshot = await probe(GOOGLE, client=http)

        assert snapshot == UsageSnapshot()
        assert seen == []
