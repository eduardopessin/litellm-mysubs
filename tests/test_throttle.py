"""The attempt limit on the only route without an administrator guard.

What these tests protect: that the limiter counts failures only — whoever gets it right can
never be blocked, or the product stops working for the legitimate user —, that the window
slides rather than being a raw counter, and that the route returns **before** touching the
code registry when the origin is blocked. The last one is what gives the 429 its value: if
the redemption happened anyway, a blocked attacker would keep burning valid codes and
telling from the responses which ones existed.
"""

from __future__ import annotations

from typing import Any

from litellm_mysubs.ui.throttle import Throttle, client_origin

from .test_ui_interceptor import build, issue, payload


class FakeClient:
    def __init__(self, host: str) -> None:
        self.host = host


class FakeRequest:
    """Only what `client_origin` reads. It exists because `TestClient` never produces a
    request without `client`, and that is precisely one of the cases to cover."""

    def __init__(self, *, headers: dict[str, str] | None = None, host: str | None = None) -> None:
        self.headers = headers or {}
        self.client = None if host is None else FakeClient(host)


class TestLimiter:
    def test_below_the_limit_nothing_is_blocked(self) -> None:
        """A slip while pasting the code must not cost five minutes of waiting."""
        throttle = Throttle(max_failures=3, window_s=60.0, block_s=300.0)
        for i in range(2):
            throttle.record_failure("10.0.0.1", now=1000.0 + i)
        assert throttle.blocked("10.0.0.1", now=1002.0) == 0.0

    def test_reaching_the_limit_blocks(self) -> None:
        throttle = Throttle(max_failures=3, window_s=60.0, block_s=300.0)
        for i in range(3):
            throttle.record_failure("10.0.0.1", now=1000.0 + i)
        assert throttle.blocked("10.0.0.1", now=1002.0) > 0.0

    def test_blocked_counts_down_and_reaches_zero(self) -> None:
        """The block is temporary by design: an origin locked out forever over ten slips
        would leave the user with no way out short of restarting the proxy."""
        throttle = Throttle(max_failures=2, window_s=60.0, block_s=300.0)
        for i in range(2):
            throttle.record_failure("10.0.0.1", now=1000.0 + i)
        assert throttle.blocked("10.0.0.1", now=1001.0) == 300.0
        assert throttle.blocked("10.0.0.1", now=1151.0) == 150.0
        assert throttle.blocked("10.0.0.1", now=1301.0) == 0.0

    def test_failures_outside_the_window_do_not_add_up(self) -> None:
        """The proof that the window slides. With a raw counter, two failures a month would
        after half a dozen months lock out someone who never attacked anyone."""
        throttle = Throttle(max_failures=3, window_s=60.0, block_s=300.0)
        throttle.record_failure("10.0.0.1", now=1000.0)
        throttle.record_failure("10.0.0.1", now=1030.0)
        # This one falls outside the window of the first two: only it and the 1030 one count.
        throttle.record_failure("10.0.0.1", now=1070.0)
        assert throttle.blocked("10.0.0.1", now=1070.0) == 0.0
        throttle.record_failure("10.0.0.1", now=1075.0)
        assert throttle.blocked("10.0.0.1", now=1075.0) > 0.0

    def test_a_success_clears_the_history(self) -> None:
        """Whoever has a legitimate code is never blocked, however many subscriptions they
        connect."""
        throttle = Throttle(max_failures=3, window_s=60.0, block_s=300.0)
        for i in range(2):
            throttle.record_failure("10.0.0.1", now=1000.0 + i)
        throttle.record_success("10.0.0.1")
        for i in range(2):
            throttle.record_failure("10.0.0.1", now=1010.0 + i)
        assert throttle.blocked("10.0.0.1", now=1012.0) == 0.0

    def test_origins_do_not_affect_each_other(self) -> None:
        """A global limiter turned an attacker into a denial of service for everybody — which
        is cheaper for them than guessing the code."""
        throttle = Throttle(max_failures=2, window_s=60.0, block_s=300.0)
        for i in range(2):
            throttle.record_failure("10.0.0.1", now=1000.0 + i)
        assert throttle.blocked("10.0.0.1", now=1002.0) > 0.0
        assert throttle.blocked("10.0.0.2", now=1002.0) == 0.0

    def test_purge_drops_old_state_and_reports_it(self) -> None:
        """Without a sweep, a long-lived proxy keeps one entry per origin that ever failed:
        only `record_success` deletes, and that never reaches whoever keeps failing."""
        throttle = Throttle(max_failures=5, window_s=60.0, block_s=300.0)
        throttle.record_failure("10.0.0.1", now=1000.0)
        throttle.record_failure("10.0.0.2", now=1000.0)
        assert throttle.purge(now=1030.0) == 0
        assert throttle.purge(now=1100.0) == 2
        assert throttle.purge(now=1100.0) == 0

    def test_purge_also_drops_expired_blocks(self) -> None:
        throttle = Throttle(max_failures=1, window_s=60.0, block_s=300.0)
        throttle.record_failure("10.0.0.1", now=1000.0)
        assert throttle.purge(now=1100.0) == 0
        assert throttle.purge(now=1400.0) == 1

    def test_a_blocked_origin_is_not_re_blocked_by_one_slip(self) -> None:
        """Once the block lapses the count restarts: dragging along failures already paid for
        shut the door again on the first slip."""
        throttle = Throttle(max_failures=2, window_s=60.0, block_s=300.0)
        for i in range(2):
            throttle.record_failure("10.0.0.1", now=1000.0 + i)
        throttle.record_failure("10.0.0.1", now=1400.0)
        assert throttle.blocked("10.0.0.1", now=1400.0) == 0.0


class TestClientOrigin:
    def test_a_single_forwarded_element_is_the_client(self) -> None:
        request = FakeRequest(headers={"x-forwarded-for": "203.0.113.7"}, host="10.0.0.1")
        assert client_origin(request) == "203.0.113.7"

    def test_the_first_forwarded_element_wins(self) -> None:
        """The list is `client, proxy1, proxy2`. Using the last one merged every client behind
        an ingress into a single origin."""
        request = FakeRequest(
            headers={"x-forwarded-for": "203.0.113.7, 10.0.0.9, 10.0.0.10"}, host="10.0.0.1"
        )
        assert client_origin(request) == "203.0.113.7"

    def test_without_the_header_the_socket_peer_is_used(self) -> None:
        request = FakeRequest(host="10.0.0.1")
        assert client_origin(request) == "10.0.0.1"

    def test_an_empty_header_falls_back_to_the_peer(self) -> None:
        request = FakeRequest(headers={"x-forwarded-for": "   "}, host="10.0.0.1")
        assert client_origin(request) == "10.0.0.1"

    def test_without_a_client_there_is_still_an_origin(self) -> None:
        """ASGI does not guarantee `client`. Counting everything under a constant origin is
        worse than having IPs, but much better than not counting at all."""
        assert client_origin(FakeRequest()) == "unknown"


class TestDepositRoute:
    """The route, from the outside. This is where the 429 is proven to come before the
    redemption."""

    def flood(self, client: Any, times: int) -> Any:
        response = None
        for i in range(times):
            response = client.post("/mysubs/api/deposit", json=payload(f"AAAA-BBBB-{i:04d}"))
        return response

    def test_the_first_ten_invented_codes_are_refused_with_403(self) -> None:
        client, _, _ = build()
        assert self.flood(client, 10).status_code == 403

    def test_the_eleventh_attempt_is_throttled(self) -> None:
        client, _, _ = build()
        self.flood(client, 10)
        response = client.post("/mysubs/api/deposit", json=payload("AAAA-BBBB-CCCC"))
        assert response.status_code == 429
        assert int(response.headers["Retry-After"]) > 0

    def test_a_valid_code_is_also_refused_while_blocked(self) -> None:
        """The proof that the route returns before the redemption: the code is good, and even
        so it is neither spent nor the credential stored. Without this, whoever is blocked
        kept burning codes and telling which ones existed by the side effects."""
        client, _, store = build()
        code = issue(client)
        self.flood(client, 10)
        response = client.post("/mysubs/api/deposit", json=payload(code))
        assert response.status_code == 429
        assert store.creds == {}

    def test_a_success_clears_the_counter_on_the_route(self) -> None:
        """Nine slips followed by a good deposit must not leave the origin one step from the
        block: the tenth slip still answers 403."""
        client, _, _ = build()
        code = issue(client)
        self.flood(client, 9)
        assert client.post("/mysubs/api/deposit", json=payload(code)).status_code == 200
        assert self.flood(client, 1).status_code == 403

    def test_the_throttled_body_says_nothing_about_the_code(self) -> None:
        """A 429 that told a good code from a bad one would hand back the oracle the single
        403 refuses to give."""
        client, _, _ = build()
        code = issue(client)
        self.flood(client, 10)
        good = client.post("/mysubs/api/deposit", json=payload(code))
        bad = client.post("/mysubs/api/deposit", json=payload("ZZZZ-ZZZZ-ZZZZ"))
        assert good.status_code == bad.status_code == 429
        assert good.json() == bad.json()
        assert code not in good.text

    def test_two_origins_are_counted_apart_on_the_route(self) -> None:
        """Behind an ingress, blocking one client must not shut the door on the others."""
        client, _, _ = build()
        for i in range(10):
            client.post(
                "/mysubs/api/deposit",
                json=payload(f"AAAA-BBBB-{i:04d}"),
                headers={"X-Forwarded-For": "203.0.113.7"},
            )
        blocked = client.post(
            "/mysubs/api/deposit",
            json=payload("AAAA-BBBB-CCCC"),
            headers={"X-Forwarded-For": "203.0.113.7"},
        )
        free = client.post(
            "/mysubs/api/deposit",
            json=payload("AAAA-BBBB-CCCC"),
            headers={"X-Forwarded-For": "203.0.113.8"},
        )
        assert blocked.status_code == 429
        assert free.status_code == 403

    def test_a_failed_deposit_is_logged_without_the_code(self) -> None:
        """Today there is no trace at all, and that is half the problem. The other half would
        be a log carrying the very secret it protects."""
        import logging

        client, _, _ = build()
        records: list[logging.LogRecord] = []

        class Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        log = logging.getLogger("litellm_mysubs")
        handler = Capture()
        log.addHandler(handler)
        try:
            client.post("/mysubs/api/deposit", json=payload("SECR-ETXX-XXXX", "tok-secret"))
        finally:
            log.removeHandler(handler)

        messages = [r.getMessage() for r in records]
        assert any(r.levelno == logging.WARNING for r in records)
        assert not any("SECR-ETXX-XXXX" in m or "tok-secret" in m for m in messages)

    def test_crossing_the_limit_is_logged_at_error(self) -> None:
        """The per-failure warning is background noise; an origin turning blocked is the event
        somebody has to see in `journalctl`."""
        import logging

        client, _, _ = build()
        records: list[logging.LogRecord] = []

        class Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        log = logging.getLogger("litellm_mysubs")
        handler = Capture()
        log.addHandler(handler)
        try:
            self.flood(client, 10)
        finally:
            log.removeHandler(handler)

        assert sum(1 for r in records if r.levelno == logging.ERROR) == 1
