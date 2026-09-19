"""Usage state read from the response headers.

The values in these tests were measured against the real proxy, not invented.
"""

from __future__ import annotations

from litellm_mysubs.catalog.usage import from_anthropic_headers, from_codex_headers, from_headers

#: Real response of `gpt-5.5` measured on 2026-09-18.
CODEX_HEADERS = {
    "x-codex-active-limit": "premium",
    "x-codex-credits-balance": "0",
    "x-codex-plan-type": "plus",
    "x-codex-primary-reset-at": "1789790331",
    "x-codex-primary-used-percent": "0",
    "x-codex-primary-window-minutes": "300",
    "x-codex-secondary-reset-at": "1789999782",
    "x-codex-secondary-used-percent": "19",
    "x-codex-secondary-window-minutes": "10080",
}

#: Real response of `claude-opus-5`, as the proxy re-exposes it.
ANTHROPIC_HEADERS = {
    "llm_provider-anthropic-ratelimit-unified-5h-reset": "1789789200",
    "llm_provider-anthropic-ratelimit-unified-5h-utilization": "0.03",
    "llm_provider-anthropic-ratelimit-unified-7d-reset": "1790204400",
    "llm_provider-anthropic-ratelimit-unified-7d-utilization": "0.24",
}


class TestScales:
    def test_anthropic_fraction_becomes_a_percentage(self) -> None:
        """Anthropic gives 0.24 for 24%; Codex gives 19 for 19%.

        Treating both scales as equal showed 0.24% where it is 24% — a card saying "almost
        unused" on an account at a quarter of the weekly limit.
        """
        snapshot = from_anthropic_headers(ANTHROPIC_HEADERS)
        assert [round(w.used_percent) for w in snapshot.windows] == [3, 24]

    def test_codex_percentage_is_taken_as_is(self) -> None:
        snapshot = from_codex_headers(CODEX_HEADERS)
        assert [round(w.used_percent) for w in snapshot.windows] == [0, 19]


class TestWindows:
    def test_window_minutes_become_readable_labels(self) -> None:
        """300 minutes is "5h" and 10080 is "7d" — that is how the user thinks about the limit."""
        snapshot = from_codex_headers(CODEX_HEADERS)
        assert [w.label for w in snapshot.windows] == ["5h", "7d"]

    def test_reset_is_a_countdown_not_a_timestamp(self) -> None:
        snapshot = from_codex_headers(CODEX_HEADERS, now=1789790331 - 600)
        assert snapshot.windows[0].resets_in_s(now=1789790331 - 600) == 600

    def test_a_reset_in_the_past_never_goes_negative(self) -> None:
        """A negative counter on screen reads as an error; zero reads as "already reset"."""
        snapshot = from_codex_headers(CODEX_HEADERS)
        assert snapshot.windows[0].resets_in_s(now=1789790331 + 5000) == 0


class TestUnknown:
    def test_a_provider_without_quota_headers_is_reported_as_unknown(self) -> None:
        """Measured: Antigravity returns no quota headers. A zeroed bar would read as
        "unused" — the opposite of what is known, which is nothing."""
        snapshot = from_headers("google-antigravity", {"content-type": "application/json"})
        assert not snapshot.known
        assert snapshot.windows == ()

    def test_garbage_values_do_not_become_zero(self) -> None:
        """`used_percent=0` because of an unreadable value would be inventing a fact."""
        snapshot = from_codex_headers({"x-codex-primary-used-percent": "a lot"})
        assert snapshot.windows == ()

    def test_an_empty_snapshot_carries_no_timestamp(self) -> None:
        """With no data there is no snapshot, and with no snapshot there is no age to show."""
        assert from_headers("anthropic", {}).taken_at == 0.0


class TestProxyPrefix:
    def test_headers_are_read_with_or_without_the_proxy_prefix(self) -> None:
        """The same header arrives under different names depending on whether one talks
        directly to upstream or through the proxy."""
        direct = {k.replace("llm_provider-", ""): v for k, v in ANTHROPIC_HEADERS.items()}
        assert (
            from_anthropic_headers(direct).windows
            == from_anthropic_headers(ANTHROPIC_HEADERS).windows
        )
