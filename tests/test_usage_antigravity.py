"""The three Antigravity families, from `:fetchAvailableModels`.

The numbers come from a real measurement against the connected account: Anthropic 0.00%
used, Google 0.14% and OpenAI 0.00%. The payloads are recorded here — no test touches the
network.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from litellm_mysubs.catalog.usage import UsageSnapshot, from_antigravity_models
from litellm_mysubs.credentials.store import Credential, ProviderId
from litellm_mysubs.transport.hosts import MODELS_PATH
from litellm_mysubs.ui.service import MySubsService

#: Slice of the measured payload: one model per family, with the fractions that came back.
MEASURED: dict[str, Any] = {
    "models": {
        "claude-sonnet-4-5": {
            "modelProvider": "MODEL_PROVIDER_ANTHROPIC",
            "apiProvider": "API_PROVIDER_ANTHROPIC",
            "quotaInfo": {"remainingFraction": 1.0, "resetTime": "2026-09-19T15:15:55Z"},
        },
        "gemini-3-pro": {
            "modelProvider": "MODEL_PROVIDER_GOOGLE",
            "apiProvider": "API_PROVIDER_GOOGLE_GEMINI",
            "quotaInfo": {"remainingFraction": 0.99855, "resetTime": "2026-09-19T15:10:06Z"},
        },
        "gpt-5": {
            "modelProvider": "MODEL_PROVIDER_OPENAI",
            "apiProvider": "API_PROVIDER_OPENAI",
            "quotaInfo": {"remainingFraction": 1.0, "resetTime": "2026-09-19T15:15:55Z"},
        },
    }
}


def used(snapshot: UsageSnapshot) -> dict[str, float]:
    return {window.label: window.used_percent for window in snapshot.windows}


class FakeStore:
    owns_refresh = True

    def __init__(self, credential: Credential) -> None:
        self.credential = credential

    def get(self, provider: ProviderId) -> Credential | None:
        return self.credential if provider == self.credential.provider else None

    def set(self, credential: Credential) -> None: ...

    def delete(self, provider: ProviderId) -> None: ...

    def reload(self) -> None: ...

    def connected(self) -> list[ProviderId]:
        return [self.credential.provider]


class TestFamilies:
    def test_three_families_match_the_measured_dashboard(self) -> None:
        snapshot = from_antigravity_models(MEASURED, now=1.0)
        assert used(snapshot) == pytest.approx(
            {"Anthropic": 0.0, "Google": 0.145, "OpenAI": 0.0}, abs=1e-6
        )

    def test_remaining_fraction_is_not_read_as_used(self) -> None:
        """`0.99855` remaining is 0.14% used; reading it as used would give 99.855%."""
        google = next(w for w in from_antigravity_models(MEASURED).windows if w.label == "Google")
        assert google.used_percent < 1.0
        assert google.used_percent != pytest.approx(99.855, abs=1e-3)

    def test_labels_come_from_the_provider_enum(self) -> None:
        assert sorted(used(from_antigravity_models(MEASURED))) == ["Anthropic", "Google", "OpenAI"]

    def test_unknown_enum_keeps_the_family_instead_of_dropping_it(self) -> None:
        payload = {
            "models": {
                "grok-9": {
                    "modelProvider": "MODEL_PROVIDER_XAI",
                    "quotaInfo": {"remainingFraction": 0.5},
                }
            }
        }
        assert used(from_antigravity_models(payload)) == pytest.approx({"Xai": 50.0}, abs=1e-6)

    def test_most_consumed_model_wins_within_a_family(self) -> None:
        """Two Google models: the bar must show the one that has consumed the most."""
        payload = {
            "models": {
                "gemini-flash": {
                    "modelProvider": "MODEL_PROVIDER_GOOGLE",
                    "quotaInfo": {"remainingFraction": 0.9, "resetTime": "2026-09-19T15:10:06Z"},
                },
                "gemini-pro": {
                    "modelProvider": "MODEL_PROVIDER_GOOGLE",
                    "quotaInfo": {"remainingFraction": 0.25, "resetTime": "2026-09-19T16:00:00Z"},
                },
            }
        }
        snapshot = from_antigravity_models(payload)
        assert used(snapshot) == pytest.approx({"Google": 75.0}, abs=1e-6)
        assert len(snapshot.windows) == 1

    def test_quota_without_reset_time_still_counts(self) -> None:
        """Measured: some models have `quotaInfo` and no `resetTime`. They count towards the
        fraction."""
        payload = {
            "models": {
                "no-reset": {
                    "modelProvider": "MODEL_PROVIDER_GOOGLE",
                    "quotaInfo": {"remainingFraction": 0.4},
                },
                "with-reset": {
                    "modelProvider": "MODEL_PROVIDER_GOOGLE",
                    "quotaInfo": {"remainingFraction": 0.8, "resetTime": "2026-09-19T15:10:06Z"},
                },
            }
        }
        window = from_antigravity_models(payload).windows[0]
        assert window.used_percent == pytest.approx(60.0, abs=1e-6)
        # The most consumed one carries no reset: the only one the family published wins.
        assert window.resets_at > 0.0


class TestEmptyAndMalformed:
    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"models": {}},
            {"models": {"a": {"modelProvider": "MODEL_PROVIDER_GOOGLE"}}},
            {"models": {"a": {"quotaInfo": {}}}},
        ],
        ids=["no-models", "empty-models", "no-quotaInfo", "quota-without-fraction"],
    )
    def test_nothing_measured_means_an_empty_snapshot(self, payload: dict[str, Any]) -> None:
        """With no data no zeroed bar is fabricated: the card stays without numbers."""
        assert not from_antigravity_models(payload).known

    def test_models_as_a_list_does_not_raise(self) -> None:
        """Guard against a shape change: today it is a dict, but a list must not blow up."""
        payload = {
            "models": [
                {
                    "modelProvider": "MODEL_PROVIDER_GOOGLE",
                    "quotaInfo": {"remainingFraction": 0.99855},
                }
            ]
        }
        assert used(from_antigravity_models(payload)) == pytest.approx({"Google": 0.145}, abs=1e-6)


class TestFetchUsageFallback:
    async def test_falls_back_to_models_when_quota_summary_is_empty(self) -> None:
        """`:quotaSummary` answers 200 with no groups; the families come from the catalog."""
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.url.path)
            if request.url.path == MODELS_PATH:
                return httpx.Response(200, json=MEASURED)
            return httpx.Response(200, json={"groups": []})

        credential = Credential(
            provider="google-antigravity", access_token="tok", project_id="proj-1"
        )
        service = MySubsService(
            store=FakeStore(credential),  # type: ignore[arg-type]
            router_source=lambda: None,
            client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

        snapshot = await service.fetch_usage("google-antigravity")

        assert MODELS_PATH in seen, seen
        assert used(snapshot) == pytest.approx(
            {"Anthropic": 0.0, "Google": 0.145, "OpenAI": 0.0}, abs=1e-6
        )

    async def test_quota_summary_still_wins_when_it_answers(self) -> None:
        """The old path was not removed: with data, the catalog is never even requested."""
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.url.path)
            return httpx.Response(
                200,
                json={
                    "groups": [
                        {
                            "displayName": "Gemini Models",
                            "buckets": [{"window": "weekly", "remainingFraction": 0.75}],
                        }
                    ]
                },
            )

        credential = Credential(
            provider="google-antigravity", access_token="tok", project_id="proj-1"
        )
        service = MySubsService(
            store=FakeStore(credential),  # type: ignore[arg-type]
            router_source=lambda: None,
            client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

        snapshot = await service.fetch_usage("google-antigravity")

        assert MODELS_PATH not in seen, seen
        # The label carries the group: the real account returns two ("Gemini Models" and
        # "Claude and GPT models") with independent limits, and without a prefix the windows
        # of both collided into a single bar.
        assert used(snapshot) == pytest.approx({"Gemini 7d": 25.0}, abs=1e-6)


class TestEveryQuotaGroupIsShown:
    """`:retrieveUserQuotaSummary` returns **two** groups with independent limits.

    Measured on the real account: "Gemini Models" at 22.02% on the weekly window and "Claude
    and GPT models" at 0%. The previous version picked only the Gemini one, and the card
    implied the subscription has a single quota — anyone using Claude through Antigravity
    could not see theirs.

    Summing them would be worse: they are separate limits, and the sum matches no limit.
    """

    def payload(self) -> dict[str, object]:
        return {
            "groups": [
                {
                    "displayName": "Gemini Models",
                    "buckets": [
                        {"window": "weekly", "remainingFraction": 0.7798252},
                        {"window": "5h", "remainingFraction": 0.9976},
                    ],
                },
                {
                    "displayName": "Claude and GPT models",
                    "buckets": [
                        {"window": "weekly", "remainingFraction": 1.0},
                        {"window": "5h", "remainingFraction": 1.0},
                    ],
                },
            ]
        }

    def test_both_groups_produce_windows(self) -> None:
        from litellm_mysubs.catalog.usage import from_antigravity_summary

        snapshot = from_antigravity_summary(self.payload())
        labels = {w.label for w in snapshot.windows}
        assert labels == {"Gemini 7d", "Gemini 5h", "Claude/GPT 7d", "Claude/GPT 5h"}

    def test_the_values_are_not_mixed_between_groups(self) -> None:
        """Without a prefix, the two `7d` windows collided and one hid the other."""
        from litellm_mysubs.catalog.usage import from_antigravity_summary

        snapshot = from_antigravity_summary(self.payload())
        by_label = {w.label: w.used_percent for w in snapshot.windows}
        assert by_label["Gemini 7d"] == pytest.approx(22.01748, abs=1e-4)
        assert by_label["Claude/GPT 7d"] == pytest.approx(0.0, abs=1e-6)

    def test_an_unknown_group_keeps_its_windows_without_a_prefix(self) -> None:
        """A name the backend changes must not make the bar disappear: the prefix is
        lost, not the data."""
        from litellm_mysubs.catalog.usage import from_antigravity_summary

        snapshot = from_antigravity_summary(
            {"groups": [{"displayName": "Something New", "buckets": [
                {"window": "weekly", "remainingFraction": 0.5}]}]}
        )
        assert [w.label for w in snapshot.windows] == ["7d"]
        assert snapshot.windows[0].used_percent == pytest.approx(50.0, abs=1e-6)


class TestWindowOrder:
    """Short window before the long one, and each group kept together.

    The backend returns the buckets in its own order — measured: the weekly one before the
    5h one — and honouring it left the Antigravity card with its bars inverted relative to
    its neighbours. Comparing two cards at a glance was no longer possible.
    """

    def test_short_window_comes_first(self) -> None:
        from litellm_mysubs.catalog.usage import from_antigravity_summary

        # Backend order: weekly first.
        snapshot = from_antigravity_summary(
            {"groups": [{"displayName": "Gemini Models", "buckets": [
                {"window": "weekly", "remainingFraction": 0.78},
                {"window": "5h", "remainingFraction": 0.99},
            ]}]}
        )
        assert [w.label for w in snapshot.windows] == ["Gemini 5h", "Gemini 7d"]

    def test_each_group_stays_together(self) -> None:
        """Sorting by duration alone interleaved `Claude/GPT 5h` with `Gemini 5h`."""
        from litellm_mysubs.catalog.usage import from_antigravity_summary

        snapshot = from_antigravity_summary(
            {"groups": [
                {"displayName": "Gemini Models", "buckets": [
                    {"window": "weekly", "remainingFraction": 0.78},
                    {"window": "5h", "remainingFraction": 0.99}]},
                {"displayName": "Claude and GPT models", "buckets": [
                    {"window": "weekly", "remainingFraction": 1.0},
                    {"window": "5h", "remainingFraction": 1.0}]},
            ]}
        )
        assert [w.label for w in snapshot.windows] == [
            "Claude/GPT 5h", "Claude/GPT 7d", "Gemini 5h", "Gemini 7d"
        ]

    def test_an_unknown_window_is_kept_at_the_end(self) -> None:
        """A window the backend adds must not disappear from the card."""
        from litellm_mysubs.catalog.usage import from_antigravity_summary

        snapshot = from_antigravity_summary(
            {"groups": [{"displayName": "Gemini Models", "buckets": [
                {"window": "monthly", "remainingFraction": 0.5},
                {"window": "5h", "remainingFraction": 0.9},
            ]}]}
        )
        labels = [w.label for w in snapshot.windows]
        assert labels[0] == "Gemini 5h"
        assert len(labels) == 2, labels
