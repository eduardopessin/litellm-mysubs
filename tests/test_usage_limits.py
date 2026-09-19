"""Scoped windows from Anthropic's `/api/oauth/usage`.

The payload used here was measured against the real account on 2026-09-19: the `limits`
array carried Fable's `weekly_scoped` at 13% while the top-level keys
`seven_day_opus`/`seven_day_sonnet` all came back `null`. It is that difference these tests
guard.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from litellm_mysubs.catalog.usage import from_anthropic_usage

#: Real response of `GET /api/oauth/usage`, reduced to the keys the parser reads.
USAGE_WITH_LIMITS: dict[str, Any] = {
    "five_hour": {"utilization": 16.0, "resets_at": "2026-09-19T12:30:00.908758+00:00"},
    "seven_day": {"utilization": 32.0, "resets_at": "2026-09-23T23:00:00.908779+00:00"},
    "seven_day_opus": None,
    "seven_day_sonnet": None,
    "limits": [
        {
            "kind": "session",
            "group": "session",
            "percent": 17,
            "severity": "normal",
            "resets_at": "2026-09-19T12:30:00.908758+00:00",
            "scope": None,
            "is_active": False,
        },
        {
            "kind": "weekly_all",
            "group": "weekly",
            "percent": 32,
            "severity": "normal",
            "resets_at": "2026-09-23T23:00:00.908779+00:00",
            "scope": None,
            "is_active": True,
        },
        {
            "kind": "weekly_scoped",
            "group": "weekly",
            "percent": 13,
            "severity": "normal",
            "resets_at": "2026-09-23T22:59:59.908951+00:00",
            "scope": {"model": {"id": None, "display_name": "Fable"}, "surface": None},
            "is_active": False,
        },
    ],
}

#: The old format: top-level keys only, no array at all.
USAGE_WITHOUT_LIMITS: dict[str, Any] = {
    "five_hour": {"utilization": 16.0, "resets_at": "2026-09-19T12:30:00.908758+00:00"},
    "seven_day": {"utilization": 32.0, "resets_at": "2026-09-23T23:00:00.908779+00:00"},
    "seven_day_opus": None,
    "seven_day_sonnet": None,
}


def _epoch(raw: str) -> float:
    return datetime.fromisoformat(raw).timestamp()


class TestLimitsArray:
    def test_scoped_window_appears_with_the_model_name(self) -> None:
        """Fable only exists in `limits`; reading the top-level keys lost it entirely."""
        snapshot = from_anthropic_usage(USAGE_WITH_LIMITS)
        assert [(w.label, w.used_percent) for w in snapshot.windows] == [
            ("5h", 17.0),
            ("7d", 32.0),
            ("7d Fable", 13.0),
        ]

    def test_reset_times_come_from_the_limits_entries(self) -> None:
        """`weekly_scoped` resets one second before `weekly_all` — measured, not rounded."""
        windows = {w.label: w.resets_at for w in from_anthropic_usage(USAGE_WITH_LIMITS).windows}
        assert windows["5h"] == _epoch("2026-09-19T12:30:00.908758+00:00")
        assert windows["7d"] == _epoch("2026-09-23T23:00:00.908779+00:00")
        assert windows["7d Fable"] == _epoch("2026-09-23T22:59:59.908951+00:00")

    def test_inactive_scoped_window_still_counts(self) -> None:
        """Fable came back `is_active:false` and the dashboard still shows it at 13%."""
        labels = [w.label for w in from_anthropic_usage(USAGE_WITH_LIMITS).windows]
        assert "7d Fable" in labels

    def test_scoped_without_display_name_is_skipped(self) -> None:
        """Without a name the window would be indistinguishable from the global `7d`; the others
        remain."""
        payload = dict(USAGE_WITH_LIMITS)
        payload["limits"] = [
            USAGE_WITH_LIMITS["limits"][0],
            USAGE_WITH_LIMITS["limits"][1],
            {**USAGE_WITH_LIMITS["limits"][2], "scope": {"model": {"id": None}}},
        ]
        assert [w.label for w in from_anthropic_usage(payload).windows] == ["5h", "7d"]

    def test_unknown_kind_is_skipped(self) -> None:
        payload = dict(USAGE_WITH_LIMITS)
        payload["limits"] = [
            USAGE_WITH_LIMITS["limits"][0],
            {"kind": "monthly_experiment", "percent": 99, "resets_at": None, "scope": None},
        ]
        assert [w.label for w in from_anthropic_usage(payload).windows] == ["5h"]


class TestFallback:
    def test_payload_without_limits_uses_the_top_level_keys(self) -> None:
        """Accounts that do not return the array yet must not lose the two base windows."""
        snapshot = from_anthropic_usage(USAGE_WITHOUT_LIMITS)
        assert [(w.label, w.used_percent) for w in snapshot.windows] == [
            ("5h", 16.0),
            ("7d", 32.0),
        ]

    def test_empty_limits_array_falls_back_too(self) -> None:
        payload = {**USAGE_WITHOUT_LIMITS, "limits": []}
        assert [w.label for w in from_anthropic_usage(payload).windows] == ["5h", "7d"]

    def test_nothing_usable_is_unknown(self) -> None:
        """With no numbers no zeroed bar is drawn: the card says it does not know."""
        assert not from_anthropic_usage({"limits": [], "seven_day_opus": None}).known
