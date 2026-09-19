"""Subscription usage state: from response headers and from quota endpoints.

Reading the headers costs nothing but requires traffic: with no inference request there is
nothing, and the card stays empty. Hence two paths — the passive one (`from_headers`) and
the active one, which probes each provider's quota endpoint (`usage_probe.probe`). Without
the active one, the account is only known to be exhausted when the first 429 arrives —
which is late for anyone who was counting on it.

The names and the shapes were **measured** against the real proxy, not read from
documentation:

    x-codex-primary-used-percent: 0        x-codex-primary-window-minutes: 300
    x-codex-secondary-used-percent: 19     x-codex-secondary-window-minutes: 10080
    anthropic-ratelimit-unified-5h-utilization: 0.03   (a fraction, not a percentage)
    anthropic-ratelimit-unified-7d-utilization: 0.24

The two scales differ — Codex gives integers from 0 to 100, Anthropic a fraction from 0 to
1 — and treating them as the same showed 0.24% where it is 24%.

All three providers **do** have a queryable quota endpoint, measured with HTTP 200:
`api.anthropic.com/api/oauth/usage` (`from_anthropic_usage`),
`chatgpt.com/backend-api/wham/usage` (`from_codex_usage`) and `:retrieveUserQuotaSummary`
(`from_antigravity_summary`, the same one the Antigravity UI uses). Antigravity is the case
where this is the *only* route: measured against the proxy, it returns no quota headers at
all.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

#: The proxy re-exposes the upstream headers with this prefix.
_PROXY_PREFIX: Final = "llm_provider-"


@dataclass(frozen=True, slots=True)
class Window:
    """One limit window: how much was used and when it resets."""

    label: str
    used_percent: float
    resets_at: float = 0.0

    def resets_in_s(self, *, now: float | None = None) -> float | None:
        if self.resets_at <= 0:
            return None
        return max(0.0, self.resets_at - (time.time() if now is None else now))


@dataclass(frozen=True, slots=True)
class UsageSnapshot:
    """What is known about a subscription's usage, and when it was known.

    `taken_at` is not decoration: a snapshot without an age is read as current state, and a
    three-hour-old value presented as now is the cheapest way to lie. The UI always shows
    the age.
    """

    windows: tuple[Window, ...] = ()
    plan: str = ""
    credits_balance: str = ""
    taken_at: float = 0.0

    @property
    def known(self) -> bool:
        return bool(self.windows) or bool(self.plan)

    def age_s(self, *, now: float | None = None) -> float:
        return (time.time() if now is None else now) - self.taken_at


def _clean(headers: Mapping[str, Any]) -> dict[str, str]:
    """Headers lowercased and stripped of the proxy prefix.

    The same header arrives under different names depending on whether the upstream is
    talked to directly or through the proxy; normalising here avoids two reading paths.
    """
    out: dict[str, str] = {}
    for key, value in headers.items():
        lowered = str(key).lower()
        if lowered.startswith(_PROXY_PREFIX):
            lowered = lowered[len(_PROXY_PREFIX) :]
        out[lowered] = str(value)
    return out


def _number(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _window_label(minutes: float | None, fallback: str) -> str:
    if minutes is None or minutes <= 0:
        return fallback
    if minutes < 60:
        return f"{int(minutes)} min"
    if minutes < 1440:
        return f"{int(minutes // 60)}h"
    return f"{int(minutes // 1440)}d"


def from_codex_headers(headers: Mapping[str, Any], *, now: float | None = None) -> UsageSnapshot:
    """ChatGPT Plus usage from the `x-codex-*` headers.

    The percentages already come in 0-100.
    """
    h = _clean(headers)
    windows: list[Window] = []
    for prefix, fallback in (("primary", "5h"), ("secondary", "7d")):
        used = _number(h.get(f"x-codex-{prefix}-used-percent"))
        if used is None:
            continue
        windows.append(
            Window(
                label=_window_label(_number(h.get(f"x-codex-{prefix}-window-minutes")), fallback),
                used_percent=used,
                resets_at=_number(h.get(f"x-codex-{prefix}-reset-at")) or 0.0,
            )
        )
    plan = h.get("x-codex-plan-type", "")
    if not windows and not plan:
        return UsageSnapshot()
    return UsageSnapshot(
        windows=tuple(windows),
        plan=plan,
        credits_balance=h.get("x-codex-credits-balance", ""),
        taken_at=time.time() if now is None else now,
    )


def from_anthropic_headers(
    headers: Mapping[str, Any], *, now: float | None = None
) -> UsageSnapshot:
    """Claude Max usage from the `anthropic-ratelimit-unified-*` headers.

    Utilization comes as a fraction (0.24 = 24%), unlike Codex. Converting here is what
    lets the UI have a single scale.
    """
    h = _clean(headers)
    windows: list[Window] = []
    for key, label in (("5h", "5h"), ("7d", "7d")):
        used = _number(h.get(f"anthropic-ratelimit-unified-{key}-utilization"))
        if used is None:
            continue
        windows.append(
            Window(
                label=label,
                used_percent=used * 100.0,
                resets_at=_number(h.get(f"anthropic-ratelimit-unified-{key}-reset")) or 0.0,
            )
        )
    if not windows:
        return UsageSnapshot()
    return UsageSnapshot(windows=tuple(windows), taken_at=time.time() if now is None else now)


def from_headers(
    provider: str, headers: Mapping[str, Any], *, now: float | None = None
) -> UsageSnapshot:
    """The provider's snapshot, or an empty one when it publishes nothing.

    Google Antigravity lands here: measured against the proxy, it returns no quota headers.
    An empty `UsageSnapshot()` is the honest answer — `known` at `False` tells the UI to
    write "no data" instead of drawing a bar at zero.
    """
    if provider == "openai-codex":
        return from_codex_headers(headers, now=now)
    if provider == "anthropic":
        return from_anthropic_headers(headers, now=now)
    return UsageSnapshot()


# omp: usage/google-antigravity.ts :: RETRIEVE_USER_QUOTA_SUMMARY_PATH
# omp= RETRIEVE_USER_QUOTA_SUMMARY_PATH = "/v1internal:retrieveUserQuotaSummary"
#: The endpoint the Antigravity UI itself uses. Unlike the model catalog, it reports both
#: windows even when neither is exhausted.
QUOTA_SUMMARY_PATH: Final = "/v1internal:retrieveUserQuotaSummary"

#: The group whose usage matters. The account reports several — "Gemini Models" and
#: "Claude and GPT models" — and adding them up would give a number that corresponds to no
#: limit at all.
_PREFERRED_GROUP: Final = "gemini"


# omp: usage/google-antigravity.ts :: classifyWindow
def _classify_window(raw: str) -> str:
    """Readable window name, from `window` or from `bucketId`."""
    lowered = raw.lower()
    if "week" in lowered or "7d" in lowered:
        return "7d"
    if "5h" in lowered or "five" in lowered:
        return "5h"
    if "day" in lowered or "24h" in lowered:
        return "24h"
    return raw or "?"


def _reset_epoch(raw: object) -> float:
    """`2026-09-23T02:30:06Z` as epoch, or `0.0` when absent."""
    if not isinstance(raw, str) or not raw:
        return 0.0
    from datetime import datetime

    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


# omp: usage/google-antigravity.ts :: buildQuotaSummaryReport
def from_antigravity_summary(
    payload: Mapping[str, Any], *, now: float | None = None
) -> UsageSnapshot:
    """Antigravity usage from `:retrieveUserQuotaSummary`.

    The field is `remainingFraction` — **the inverse** of what the UI shows. A `0.7833` is
    21.7% used, and treating it as "used" showed an almost exhausted account where it is at
    a fifth of the limit.

    **Every** group is shown, prefixing the label with each group's name. The previous
    version picked only the Gemini one and discarded the rest; measured against the real
    account, the endpoint returns two — "Gemini Models" (22.02% for the week) and
    "Claude and GPT models" (0%) — and hiding the second implied the subscription has a
    single quota. Adding them up is what would be wrong: they are independent limits, and
    the sum corresponds to no limit at all.
    """
    groups = payload.get("groups") or payload.get("quotaGroups") or []
    if not isinstance(groups, list) or not groups:
        return UsageSnapshot()

    windows: list[Window] = []
    plans: list[str] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        name = str(group.get("displayName") or "")
        buckets = group.get("buckets") or []
        if not isinstance(buckets, list):
            continue
        prefix = _group_prefix(name)
        for bucket in buckets:
            if not isinstance(bucket, dict):
                continue
            remaining = bucket.get("remainingFraction")
            if not isinstance(remaining, int | float):
                continue
            window = _classify_window(str(bucket.get("window") or bucket.get("bucketId") or ""))
            windows.append(
                Window(
                    label=f"{prefix} {window}" if prefix else window,
                    used_percent=max(0.0, min(100.0, (1.0 - float(remaining)) * 100.0)),
                    resets_at=_reset_epoch(bucket.get("resetTime")),
                )
            )
        if name:
            plans.append(name)
    if not windows:
        return UsageSnapshot()
    # Short window first, as with the other two providers. The backend returns the buckets
    # in its own order — measured: the weekly one before the 5h one — and honouring it left
    # the Antigravity card with its bars inverted relative to its neighbours, forcing the
    # label to be read to compare two cards that should compare at a glance.
    windows.sort(key=_window_order)
    return UsageSnapshot(
        windows=tuple(windows),
        plan=" · ".join(plans),
        taken_at=time.time() if now is None else now,
    )


#: Duration of each window label, in minutes, to sort them from shortest to longest. An
#: unrecognised label goes to the end rather than disappearing.
_WINDOW_MINUTES: Final[dict[str, float]] = {"5h": 300.0, "24h": 1440.0, "7d": 10080.0}


def _window_order(window: Window) -> tuple[str, float, str]:
    """Sort key: group, then duration, then the label.

    The group comes first because a subscription's bars are read as a set: sorting by
    duration alone interleaved `Claude/GPT 5h` with `Gemini 5h` and forced every label to
    be read to know whose the neighbouring bar was.

    Within the group, the short window before the long one — the same thing the other two
    cards do, so they compare at a glance.
    """
    label = window.label
    for name, minutes in _WINDOW_MINUTES.items():
        if label == name:
            return ("", minutes, label)
        if label.endswith(f" {name}"):
            return (label[: -len(name) - 1], minutes, label)
    return (label, float("inf"), label)


def _group_prefix(display_name: str) -> str:
    """Short group label, so it fits next to the window.

    "Gemini Models" → `Gemini`; "Claude and GPT models" → `Claude/GPT`. With no recognised
    name it returns empty and the window goes unprefixed — a label invented from a name the
    backend changed is worse than no label.
    """
    lowered = display_name.lower()
    if "gemini" in lowered:
        return "Gemini"
    if "claude" in lowered or "gpt" in lowered:
        return "Claude/GPT"
    return ""


#: Prefixes of the windows Anthropic's `/api/oauth/usage` returns, and the base label of
#: each. The account also reports suffixed variants (`seven_day_opus`, `seven_day_sonnet`)
#: that are `null` when the plan does not have them — measured on a Max account, where both
#: came back null and only `five_hour`/`seven_day` carried a number.
_ANTHROPIC_WINDOWS: Final[tuple[tuple[str, str], ...]] = (
    ("five_hour", "5h"),
    ("seven_day", "7d"),
)


def _anthropic_label(key: str) -> str | None:
    """`seven_day_opus` -> `7d opus`; `None` for keys that are not windows.

    The label has to fit a narrow card, and an unknown key does not get an invented label:
    the payload carries fields that are not windows and any one of them would draw a
    meaningless bar.
    """
    for prefix, label in _ANTHROPIC_WINDOWS:
        if key == prefix:
            return label
        if key.startswith(f"{prefix}_"):
            return f"{label} {key[len(prefix) + 1 :].replace('_', ' ')}"
    return None


#: How the Quota Dashboard labels the unscoped `kind` values of the `limits` array.
#: `weekly_scoped` is left out because it depends on the model name; a `kind` outside what
#: was measured gets no label.
_ANTHROPIC_LIMIT_LABELS: Final[dict[str, str]] = {
    "session": "5h",
    "weekly_all": "7d",
}


def _anthropic_limit_window(entry: Mapping[str, Any]) -> Window | None:
    """One `limits` entry as a window, or `None` when it is not usable.

    `weekly_scoped` is the only one carrying the model name, in `scope.model.display_name`
    (measured: `"Fable"`, with `scope.model.id` at `null`). Without that name the window
    would be indistinguishable from the global `7d` one, so it is skipped instead of
    duplicating the label.

    `is_active` and `severity` do not filter: Fable came back `is_active:false` and still
    counts 13% against the quota on the dashboard.
    """
    percent = entry.get("percent")
    if not isinstance(percent, int | float) or isinstance(percent, bool):
        return None
    kind = str(entry.get("kind") or "")
    label = _ANTHROPIC_LIMIT_LABELS.get(kind)
    if label is None:
        if kind != "weekly_scoped":
            return None
        scope = entry.get("scope")
        model = scope.get("model") if isinstance(scope, dict) else None
        name = model.get("display_name") if isinstance(model, dict) else None
        if not isinstance(name, str) or not name.strip():
            return None
        label = f"7d {name.strip()}"
    return Window(
        label=label,
        used_percent=float(percent),
        resets_at=_reset_epoch(entry.get("resets_at")),
    )


# omp: usage/claude.ts :: parseBucket, parseUnifiedWindow
def from_anthropic_usage(payload: Mapping[str, Any], *, now: float | None = None) -> UsageSnapshot:
    """Claude Max usage from `GET /api/oauth/usage`.

    The `limits` array is preferred: it is what the Quota Dashboard reads and the only place
    scoped windows appear — measured: the top-level `seven_day_opus`/`seven_day_sonnet` all
    came back `null` while `limits` carried a `weekly_scoped` Fable entry at 13%. There
    `percent` is an integer 0-100 and `resets_at` is ISO-8601
    (`2026-09-23T22:59:59.908951+00:00`).

    Without `limits` — or with it empty — the top-level `five_hour`/`seven_day` keys are
    used, which is the format measured earlier; deleting that path would make the
    regression silent on an account that does not return the array yet.

    Trap measured on the old path: here `utilization` comes as a **percentage** (`15.0` is
    15%), while the same name in the headers — `from_anthropic_headers` — comes as a
    fraction (`0.15`). The same field, two scales: multiplying by 100 on this path showed
    1500%.

    Keys at `null` are windows the account does not have; they are skipped, because a window
    with no number is not a window at zero.
    """
    windows: list[Window] = []
    limits = payload.get("limits")
    if isinstance(limits, list):
        for entry in limits:
            if not isinstance(entry, dict):
                continue
            window = _anthropic_limit_window(entry)
            if window is not None:
                windows.append(window)
    if not windows:
        for key, value in payload.items():
            if not isinstance(value, dict):
                continue
            used = value.get("utilization")
            if not isinstance(used, int | float) or isinstance(used, bool):
                continue
            label = _anthropic_label(str(key))
            if label is None:
                continue
            windows.append(
                Window(
                    label=label,
                    used_percent=float(used),
                    resets_at=_reset_epoch(value.get("resets_at")),
                )
            )
    if not windows:
        return UsageSnapshot()
    # Stable, predictable order in the card: 5h before 7d, variants after the base window.
    windows.sort(key=lambda w: (len(w.label), w.label))
    return UsageSnapshot(windows=tuple(windows), taken_at=time.time() if now is None else now)


# omp: usage/openai-codex.ts :: parseUsagePayload, buildWindowLabel
def from_codex_usage(payload: Mapping[str, Any], *, now: float | None = None) -> UsageSnapshot:
    """ChatGPT Plus usage from `GET /backend-api/wham/usage`.

    `used_percent` is already 0-100 and `reset_at` is epoch in **seconds** — not
    milliseconds, unlike the rest of the ChatGPT backend; dividing it by 1000 put the reset
    in 1970.

    The label comes from `limit_window_seconds` (18000 -> `5h`, 604800 -> `7d`) instead of
    being fixed by position: the plan is what decides the windows, not the field order.
    """
    rate_limit = payload.get("rate_limit")
    windows: list[Window] = []
    if isinstance(rate_limit, dict):
        for key, fallback in (("primary_window", "5h"), ("secondary_window", "7d")):
            window = rate_limit.get(key)
            if not isinstance(window, dict):
                continue
            used = window.get("used_percent")
            if not isinstance(used, int | float) or isinstance(used, bool):
                continue
            seconds = window.get("limit_window_seconds")
            minutes = float(seconds) / 60.0 if isinstance(seconds, int | float) else None
            reset_at = window.get("reset_at")
            windows.append(
                Window(
                    label=_window_label(minutes, fallback),
                    used_percent=float(used),
                    resets_at=float(reset_at) if isinstance(reset_at, int | float) else 0.0,
                )
            )
    plan = str(payload.get("plan_type") or "")
    if not windows and not plan:
        return UsageSnapshot()
    return UsageSnapshot(
        windows=tuple(windows), plan=plan, taken_at=time.time() if now is None else now
    )


#: Labels of the Antigravity catalog families. Measured on `:fetchAvailableModels`: every
#: model carries `modelProvider` as one of these three enums, and that is how the account
#: groups the limits — `apiProvider` is finer-grained (`API_PROVIDER_GOOGLE_GEMINI`) and
#: would split the same family across several bars.
_ANTIGRAVITY_FAMILIES: Final[dict[str, str]] = {
    "MODEL_PROVIDER_ANTHROPIC": "Anthropic",
    "MODEL_PROVIDER_GOOGLE": "Google",
    "MODEL_PROVIDER_OPENAI": "OpenAI",
}


def _family_label(raw: str) -> str:
    """`MODEL_PROVIDER_ANTHROPIC` -> `Anthropic`; unknown -> the suffix in Title Case.

    Discarding a new enum would hide a family the account is already accounting for. A
    `MODEL_PROVIDER_XAI` shows up as `Xai`: ugly, but it shows the limit exists.
    """
    known = _ANTIGRAVITY_FAMILIES.get(raw)
    if known is not None:
        return known
    tail = raw.rpartition("MODEL_PROVIDER_")[2] or raw
    return tail.replace("_", " ").title() or "?"


# omp: usage/google-antigravity.ts :: FETCH_AVAILABLE_MODELS_PATH
# omp= FETCH_AVAILABLE_MODELS_PATH = "/v1internal:fetchAvailableModels"
def from_antigravity_models(
    payload: Mapping[str, Any], *, now: float | None = None
) -> UsageSnapshot:
    """Antigravity usage from `:fetchAvailableModels`.

    Serves when `:retrieveUserQuotaSummary` returns nothing useful. Unlike that one, this
    catalog gives **three families** (Anthropic, Google, OpenAI) instead of time windows —
    which is exactly what the Quota Dashboard shows.

    Three measured facts the code has to respect:

    1. `payload["models"]` is a **dict** `{modelId: {...}}`, not a list. Iterating it as a
       list gave the keys instead of the models and found no `quotaInfo` at all.
    2. `remainingFraction` is what is **left**. Measured `0.99855`, which is 0.14% used —
       not 99.855%, which is what a direct reading put on the bar.
    3. Some models have `quotaInfo` without `resetTime`. They count towards the fraction —
       the limit is real — but they cannot impose a reset the payload does not carry.

    Within a family several models share the limit and report slightly different fractions;
    the **smallest** one (the most consumed) wins, since that is the one that stops the
    family first.
    """
    models = payload.get("models")
    entries: list[Mapping[str, Any]] = []
    if isinstance(models, dict):
        entries = [value for value in models.values() if isinstance(value, dict)]
    elif isinstance(models, list):
        # Defence against a shape change: if a list ever arrives, it is still read.
        entries = [value for value in models if isinstance(value, dict)]

    worst: dict[str, float] = {}
    winner_reset: dict[str, float] = {}
    any_reset: dict[str, float] = {}
    for model in entries:
        quota = model.get("quotaInfo")
        if not isinstance(quota, dict):
            continue
        remaining = quota.get("remainingFraction")
        if not isinstance(remaining, int | float) or isinstance(remaining, bool):
            continue
        family = _family_label(str(model.get("modelProvider") or ""))
        fraction = float(remaining)
        reset = _reset_epoch(quota.get("resetTime"))
        if reset > 0.0:
            seen = any_reset.get(family, 0.0)
            any_reset[family] = reset if seen == 0.0 else min(seen, reset)
        if family not in worst or fraction < worst[family]:
            worst[family] = fraction
            winner_reset[family] = reset

    if not worst:
        return UsageSnapshot()
    windows = tuple(
        Window(
            label=label,
            used_percent=max(0.0, min(100.0, (1.0 - worst[label]) * 100.0)),
            # The most consumed model may be one of those without a `resetTime`; in that
            # case the nearest reset seen in the family stands, instead of none.
            resets_at=winner_reset[label] or any_reset.get(label, 0.0),
        )
        for label in sorted(worst)
    )
    return UsageSnapshot(windows=windows, taken_at=time.time() if now is None else now)
