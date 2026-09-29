"""Connection reopening policy, separated from the transport.

In the original, the decision of what to do with a 401, a 400 or a 429 was embedded inside
the ``httpx`` loops, duplicated between the synchronous and the asynchronous version — and
the two had drifted into different shapes of the same rule. Here the decision is a pure
function over ``(status, body, headers)``, and the transport merely executes it.

Two questions, both answered the way omp answers them:

* **what a refused request means** — `decide_codex`/`decide_antigravity`: refresh, remap,
  redeem, try the next endpoint, or propagate;
* **whether to send it again, and when** — `RetryBudget`, omp's `fetchWithRetry`: a
  retryable status (408, 429, 5xx) or a network failure is re-sent in place after the
  server's own hint (`retry_hint`) or the provider's backoff, a bounded number of times.

That makes the part that matters — *when* a retry happens and why — testable without
opening connections.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from enum import Enum
from typing import Final


class Action(Enum):
    """What to do with a response that is not a success."""

    REFRESH_TOKEN = "refresh_token"
    """Credential rejected: re-read it and try again with the new one."""

    REMAP_MODEL = "remap_model"
    """The account does not serve this name; if it is a known alias, reroute."""

    REDEEM_CREDIT = "redeem_credit"
    """Quota exhausted and there is unused reset credit."""

    FAIL = "fail"
    """Try the next endpoint if there is one, else propagate the upstream error."""

    ABORT = "abort"
    """Propagate at once: trying another endpoint cannot change the answer."""


@dataclass(frozen=True, slots=True)
class Decision:
    action: Action
    reason: str = ""


#: Marker of the ChatGPT account refusing a model.
UNSUPPORTED_MARKER: Final = "is not supported when using Codex"


def is_unsupported_model(body: str) -> bool:
    return UNSUPPORTED_MARKER in str(body)


# omp: fetch-retry.ts :: isRetryableStatus
# omp: error/retryable.ts :: isTransientStatus
def is_retryable_status(status: int) -> bool:
    """408, 429 and every 5xx: the statuses omp re-sends and fails over on."""
    return status >= 500 or status in (408, 429)


def decide_codex(
    status: int,
    body: str = "",
    *,
    can_remap: bool = False,
    can_redeem: bool = False,
) -> Decision:
    """What to do with the final Codex refusal — after the in-place retries ran out.

    ``can_remap`` and ``can_redeem`` are capabilities of the caller, not of global state:
    with no known alias and no credit, the decision has to be to fail.
    """
    if status == 401:
        return Decision(Action.REFRESH_TOKEN, "credential rejected")
    if status == 400 and is_unsupported_model(body):
        if can_remap:
            return Decision(Action.REMAP_MODEL, "known alias of a served model")
        # An arbitrary refused name is the correct answer: substituting another model for
        # it returned 200 with the `model` field echoing the request, and billing started
        # to lie.
        return Decision(Action.FAIL, "the account does not serve this model")
    if status == 429 and can_redeem:
        return Decision(Action.REDEEM_CREDIT, "quota exhausted, reset credit available")
    return Decision(Action.FAIL, f"HTTP {status}")


# omp: providers/google-gemini-cli.ts :: streamGoogleGeminiCli
def decide_antigravity(status: int) -> Decision:
    """What to do with an Antigravity refusal.

    omp fails over to the other endpoint on a transient status only — 408 or 5xx is the
    host's trouble. Any other 4xx is the request's, and the other host would refuse it
    the same way: it propagates from the first host. A model the account does not serve
    (404) or an invalid payload (400) never authorises answering with another model.

    A 429 is the exception, and a deliberate divergence: omp treats it as transient and
    asks the other host. Both hosts front the same account and the same quota, so the
    second one repeats a refusal already known — measured against the real backend, the
    rotation turned an ~11 s failure into ~22 s and changed nothing else. It stays on the
    host that answered, which then gets the in-place retries omp gives its last host.
    """
    if status == 401:
        return Decision(Action.REFRESH_TOKEN, "credential rejected")
    if status == 429:
        return Decision(Action.ABORT, "quota exhausted: the other host serves the same account")
    if is_retryable_status(status):
        return Decision(Action.FAIL, f"HTTP {status}: the other host may answer")
    return Decision(Action.ABORT, f"HTTP {status}: the other host would refuse it too")


@dataclass(frozen=True, slots=True)
class RetryBudget:
    """omp's `fetchWithRetry` options for one provider.

    ``max_attempts`` counts the first send. ``default_delay`` is the wait, in seconds,
    after attempt ``n`` (0-indexed) when the server gave no hint; ``max_delay`` caps every
    wait, and a server hint longer than it is not waited out at all — the refusal is final.
    """

    max_attempts: int
    default_delay: Callable[[int], float]
    max_delay: float

    # omp: fetch-retry.ts :: resolveDefaultDelay
    def delay(self, attempt: int) -> float:
        return min(self.default_delay(attempt), self.max_delay)

    # omp: fetch-retry.ts :: fetchWithRetry
    def after_status(
        self, attempt: int, status: int, headers: Mapping[str, str], body: str
    ) -> float | None:
        """Wait before re-sending after ``status``; ``None`` when this answer is final."""
        if not is_retryable_status(status) or attempt + 1 >= self.max_attempts:
            return None
        hint = retry_hint(headers, body)
        if hint is not None and hint > self.max_delay:
            return None
        return min(self.delay(attempt) if hint is None else hint, self.max_delay)

    def after_network_error(self, attempt: int) -> float | None:
        """Wait before re-sending after the send itself failed; ``None`` when out of tries."""
        if attempt + 1 >= self.max_attempts:
            return None
        return self.delay(attempt)


# omp: providers/openai-codex-responses.ts :: CODEX_MAX_RETRIES
CODEX_MAX_RETRIES: Final = 5
# omp: providers/openai-codex-responses.ts :: CODEX_RETRY_DELAY_MS
CODEX_RETRY_DELAY_S: Final = 0.5
# omp: providers/openai-codex-responses.ts :: CODEX_RATE_LIMIT_BUDGET_MS
CODEX_RATE_LIMIT_BUDGET_S: Final = 300.0

# omp: providers/openai-codex-responses.ts :: openCodexSseEventStream
# omp: providers/openai-codex-responses.ts :: resolveCodexSseMaxAttempts
CODEX_BUDGET: Final = RetryBudget(
    max_attempts=CODEX_MAX_RETRIES + 1,
    default_delay=lambda attempt: CODEX_RETRY_DELAY_S * (attempt + 1),
    max_delay=CODEX_RATE_LIMIT_BUDGET_S,
)

# omp: providers/google-gemini-cli.ts :: MAX_RETRIES
ANTIGRAVITY_MAX_RETRIES: Final = 3
# omp: providers/google-gemini-cli.ts :: BASE_DELAY_MS
ANTIGRAVITY_BASE_DELAY_S: Final = 1.0
# omp: providers/google-gemini-cli.ts :: RATE_LIMIT_BUDGET_MS
ANTIGRAVITY_RATE_LIMIT_BUDGET_S: Final = 300.0

# omp: providers/google-gemini-cli.ts :: streamGoogleGeminiCli
#: The budget of the host that answers last. The others get a single send: a failure
#: there moves on to the next host instead (`decide_antigravity`).
ANTIGRAVITY_BUDGET: Final = RetryBudget(
    max_attempts=ANTIGRAVITY_MAX_RETRIES + 1,
    default_delay=lambda attempt: ANTIGRAVITY_BASE_DELAY_S * 2**attempt,
    max_delay=ANTIGRAVITY_RATE_LIMIT_BUDGET_S,
)


# -- server retry hints -------------------------------------------------------------

# omp: fetch-retry.ts :: QUOTA_RESET_PATTERN
_QUOTA_RESET = re.compile(r"reset after (?:(\d+)h)?(?:(\d+)m)?(\d+(?:\.\d+)?)s", re.I)
# omp: fetch-retry.ts :: PLEASE_RETRY_PATTERN
_PLEASE_RETRY = re.compile(r"Please retry in ([0-9.]+)(ms|s)", re.I)
# omp: fetch-retry.ts :: RETRY_DELAY_FIELD_PATTERN
_RETRY_DELAY_FIELD = re.compile(r'"retryDelay":\s*"([0-9.]+)(ms|s)"', re.I)
# omp: fetch-retry.ts :: TRY_AGAIN_PATTERN
_TRY_AGAIN = re.compile(
    r"try again in\s+~?\s*([0-9.]+)\s*(ms|sec|s|minutes?|mins?|m|hours?|hrs?|h)\b", re.I
)
# omp: fetch-retry.ts :: WILL_RESET_IN_PATTERN
_WILL_RESET_IN = re.compile(
    r"(?:will\s+)?resets?\s+in\s+~?\s*([0-9.]+)\s*"
    r"(ms|sec|s|minutes?|mins?|m|hours?|hrs?|h|days?|d)\b",
    re.I,
)
# omp: fetch-retry.ts :: RESET_IN_HR_MIN_PATTERN
_RESET_IN_HR_MIN = re.compile(
    r"resets?\s+in\s+~?\s*(\d+(?:\.\d+)?)\s*hr\s*(\d+(?:\.\d+)?)\s*min\b", re.I
)
# omp: fetch-retry.ts :: WILL_RESET_AT_PATTERN
_WILL_RESET_AT = re.compile(
    r"(?:will\s+)?reset at\s+([0-9]{4}-[0-9]{2}-[0-9]{2}[ T][0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:?[0-9]{2})?)",
    re.I,
)
# omp: fetch-retry.ts :: CN_RESET_AT_PATTERN
_CN_RESET_AT = re.compile(
    r"将在\s*([0-9]{4}-[0-9]{2}-[0-9]{2}\s+[0-9]{2}:[0-9]{2}:[0-9]{2})\s*重置"
)
# omp: fetch-retry.ts :: RETRY_AFTER_MS_BODY_PATTERN
_RETRY_AFTER_MS_BODY = re.compile(r"\bretry-after-ms\s*[:=]\s*([0-9]+)\b", re.I)
_RETRY_AFTER_BODY = re.compile(r"retry-after\s*[:=]\s*([^\s,;]+)", re.I)
_RESET_MS_BODY = re.compile(r"x-ratelimit-reset-ms\s*[:=]\s*(\d+)", re.I)
_RESET_BODY = re.compile(r"x-ratelimit-reset\s*[:=]\s*(\d+)", re.I)
_HAS_OFFSET = re.compile(r"(?:Z|[+-][0-9]{2}:?[0-9]{2})$", re.I)
_LEADING_INT = re.compile(r"\s*([+-]?\d+)")
_LEADING_FLOAT = re.compile(r"\d+(?:\.\d*)?|\.\d+")

_SECOND: Final = 1000.0
_MINUTE: Final = 60 * _SECOND
_HOUR: Final = 60 * _MINUTE

# omp: fetch-retry.ts :: unitToMs
_UNIT_MS: Final[Mapping[str, float]] = {
    "ms": 1.0,
    "s": _SECOND,
    "sec": _SECOND,
    **dict.fromkeys(("m", "min", "mins", "minute", "minutes"), _MINUTE),
    **dict.fromkeys(("h", "hr", "hrs", "hour", "hours"), _HOUR),
    **dict.fromkeys(("d", "day", "days"), 24 * _HOUR),
}


# omp: fetch-retry.ts :: extractRetryHint
def retry_hint(
    headers: Mapping[str, str], body: str = "", *, now: float | None = None
) -> float | None:
    """The wait the server asked for, in seconds; ``None`` when it named none.

    ``0`` is an answer, not an absence: an explicit ``retry-after: 0`` or a reset time
    already past means "retry now", and collapsing it into ``None`` would make the caller
    sleep its own backoff instead. Headers win outright; within the body the **longest**
    signal wins, because retrying before every window has cleared re-hits a credential
    that is still blocked.
    """
    now_ms = (time.time() if now is None else now) * _SECOND
    from_headers = _header_hint({k.lower(): v for k, v in headers.items()}, now_ms)
    if from_headers is not None:
        return from_headers / _SECOND
    from_body = _body_hint(body, now_ms) if body else None
    return None if from_body is None else from_body / _SECOND


def _header_hint(headers: Mapping[str, str], now_ms: float) -> float | None:
    if value := headers.get("retry-after-ms"):
        ms = _number(value)
        if ms is not None and ms >= 0:
            return ms
    if value := headers.get("retry-after"):
        seconds = _number(value)
        if seconds is not None:
            return max(0.0, seconds * _SECOND)
        if (date := _date_ms(value)) is not None:
            return max(0.0, date - now_ms)
    if value := headers.get("x-ratelimit-reset-ms"):
        reset = _number(value)
        if reset is not None and reset > 0:
            # > 1e12 is epoch ms, > 1e9 epoch seconds, anything smaller a delta in ms.
            if reset <= 1e9:
                return reset
            delta = (reset if reset > 1e12 else reset * _SECOND) - now_ms
            if delta > 0:
                return delta
    if (value := headers.get("x-ratelimit-reset")) and (match := _LEADING_INT.match(value)):
        delta = int(match.group(1)) * _SECOND - now_ms
        if delta > 0:
            return delta
    if value := headers.get("x-ratelimit-reset-after"):
        seconds = _number(value)
        if seconds is not None and seconds > 0:
            return seconds * _SECOND
    return None


def _body_hint(body: str, now_ms: float) -> float | None:
    longest: float | None = None
    # A reset stamp with no timezone is a wall clock in an unknown zone: it only counts
    # when the body carries nothing unambiguous.
    longest_naive: float | None = None
    retry_now = False

    def consider(ms: float | None) -> None:
        nonlocal longest
        if ms is not None and ms > 0 and (longest is None or ms > longest):
            longest = ms

    def consider_clamped(ms: float) -> None:
        nonlocal retry_now
        if ms > 0:
            consider(ms)
        else:
            retry_now = True

    if match := _QUOTA_RESET.search(body):
        hours, minutes = int(match.group(1) or 0), int(match.group(2) or 0)
        consider(((hours * 60 + minutes) * 60 + float(match.group(3))) * _SECOND)
    for pattern in (_WILL_RESET_AT, _CN_RESET_AT):
        if not (match := pattern.search(body)):
            continue
        stamp = match.group(1).replace(" ", "T", 1)
        naive = not _HAS_OFFSET.search(stamp)
        parsed = _date_ms(f"{stamp}Z" if naive else stamp)
        if parsed is not None and parsed > now_ms:
            if naive:
                if longest_naive is None or parsed - now_ms > longest_naive:
                    longest_naive = parsed - now_ms
            else:
                consider(parsed - now_ms)
    if match := _RESET_IN_HR_MIN.search(body):
        hour_count, minute_count = float(match.group(1)), float(match.group(2))
        if hour_count >= 0 and minute_count > 0:
            consider(hour_count * _HOUR + minute_count * _MINUTE)
    for pattern in (_WILL_RESET_IN, _PLEASE_RETRY, _RETRY_DELAY_FIELD, _TRY_AGAIN):
        if match := pattern.search(body):
            value = _leading_float(match.group(1))
            unit = _UNIT_MS.get(match.group(2).lower())
            if value is not None and value > 0 and unit is not None:
                consider(value * unit)
    if match := _RETRY_AFTER_MS_BODY.search(body):
        consider_clamped(float(match.group(1)))
    # Legacy text forms of the headers, competing in the same maximum.
    if match := _RETRY_AFTER_BODY.search(body):
        seconds = _number(match.group(1))
        if seconds is not None:
            consider_clamped(seconds * _SECOND)
        elif (date := _date_ms(match.group(1))) is not None:
            consider_clamped(date - now_ms)
    if match := _RESET_MS_BODY.search(body):
        reset = float(match.group(1))
        consider_clamped(reset - now_ms if reset > 1e12 else reset)
    if match := _RESET_BODY.search(body):
        reset = float(match.group(1))
        consider_clamped(reset * _SECOND - now_ms if reset > 1e9 else reset * _SECOND)
    if longest is not None:
        return longest
    if longest_naive is not None:
        return longest_naive
    return 0.0 if retry_now else None


def _number(text: str) -> float | None:
    """JavaScript's ``Number(text)`` for the forms a header carries: finite, else ``None``."""
    try:
        value = float(text.strip())
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _leading_float(text: str) -> float | None:
    """JavaScript's ``parseFloat``: the number at the start, ignoring what follows."""
    match = _LEADING_FLOAT.match(text)
    return float(match.group()) if match else None


def _date_ms(text: str) -> float | None:
    """``Date.parse`` for the two shapes that reach it: ISO 8601 and the HTTP date."""
    try:
        parsed = datetime.fromisoformat(text.strip())
    except ValueError:
        try:
            parsed = parsedate_to_datetime(text)
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp() * _SECOND
