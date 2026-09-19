"""Usage and finish reason normalization.

The bridges answer the requests themselves, so whatever is not reported here is lost:
LiteLLM falls back to ``token_counter`` estimates and **every cache hit becomes
invisible** in ``/spend/logs``. A subscription account with no cache accounting is an
account that cannot be managed.

Neutral structures instead of LiteLLM's types: the conversion to ``litellm.types.utils``
lives in the transport layer. This keeps the module testable without LiteLLM installed,
and it is also what allows reusing it if the transport changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final


@dataclass(frozen=True, slots=True)
class Usage:
    """Token counts for one turn.

    ``cached_tokens`` is read by LiteLLM's spend logging through an attribute of its own
    (``cache_read_input_tokens``), not through the details wrapper — the conversion handles
    that.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0


def make_usage(
    prompt_tokens: object = 0,
    completion_tokens: object = 0,
    cached_tokens: object = 0,
    reasoning_tokens: object = 0,
    total_tokens: object = None,
) -> Usage:
    """Normalize counts coming off the wire, which arrive in assorted formats and types."""

    def count(value: object) -> int:
        """A broken counter is worth zero: blowing up here lost the whole turn."""
        if value is None or isinstance(value, bool):
            return 0
        if isinstance(value, int):
            return max(0, value)
        if isinstance(value, float):
            return max(0, int(value))
        if isinstance(value, str):
            try:
                return max(0, int(value.strip()))
            except ValueError:
                return 0
        return 0

    prompt = count(prompt_tokens)
    completion = count(completion_tokens)
    total = count(total_tokens) if total_tokens else prompt + completion
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        cached_tokens=count(cached_tokens),
        reasoning_tokens=count(reasoning_tokens),
        total_tokens=total,
    )


# omp: providers/google-shared.ts :: mapStopReason
def google_usage(meta: dict[str, Any]) -> Usage:
    """``promptTokenCount`` **includes** the cached tokens.

    They are subtracted so as not to count them twice, and thoughts count as output.
    """
    cached = meta.get("cachedContentTokenCount") or 0
    thinking = meta.get("thoughtsTokenCount") or 0
    return make_usage(
        prompt_tokens=(meta.get("promptTokenCount") or 0) - cached,
        completion_tokens=(meta.get("candidatesTokenCount") or 0) + thinking,
        cached_tokens=cached,
        reasoning_tokens=thinking,
        total_tokens=meta.get("totalTokenCount"),
    )


def codex_usage(meta: dict[str, Any]) -> Usage:
    """Unlike Google, ``input_tokens`` is **not** reduced by the cached tokens."""
    details = meta.get("input_tokens_details") or {}
    output_details = meta.get("output_tokens_details") or {}
    cached = details.get("cached_tokens")
    if cached is None:
        cached = meta.get("prompt_cache_hit_tokens") or 0
    return make_usage(
        prompt_tokens=meta.get("input_tokens") or 0,
        completion_tokens=meta.get("output_tokens") or 0,
        cached_tokens=cached,
        reasoning_tokens=output_details.get("reasoning_tokens") or 0,
        total_tokens=meta.get("total_tokens"),
    )


# OMP uses an *allow* list, not a deny list: in `mapStopReasonString` only `STOP` and
# `MAX_TOKENS` have a meaning of their own and **everything else is an error**. The
# inverse shape — enumerating the error reasons — misses whatever upstream adds: comparing
# against the source turned up five missing ones (FINISH_REASON_UNSPECIFIED, LANGUAGE,
# IMAGE_OTHER, IMAGE_PROHIBITED_CONTENT, IMAGE_RECITATION), all passing as `stop`.
#
# An unknown reason treated as `stop` hands the client a cut-off response as if it were
# complete; treated as an error, the worst case is that it is too noisy.
NORMAL_FINISH: Final = "STOP"
TRUNCATED_FINISH: Final = "MAX_TOKENS"

#: Reasons where a pending tool call is still the correct outcome of the turn.
_TOOL_CALL_COMPATIBLE: Final[tuple[str, ...]] = ("", NORMAL_FINISH, TRUNCATED_FINISH)


# omp: providers/google-shared.ts :: mapStopReasonString
def google_finish_reason(raw: object, has_tool_calls: bool) -> str:
    """Translate ``candidates[0].finishReason`` into the OpenAI form."""
    reason = str(raw or "").strip().upper()
    if has_tool_calls and reason in _TOOL_CALL_COMPATIBLE:
        return "tool_calls"
    if reason == TRUNCATED_FINISH:
        return "length"
    if reason in ("", NORMAL_FINISH):
        return "stop"
    # `content_filter` is the only OpenAI value that does not lie about a cut imposed by
    # the server; the raw name goes in the in-band error when there is one.
    return "content_filter"


def codex_finish_reason(status: object, has_tool_calls: bool) -> str:
    """``response.incomplete`` means truncation by the output limit.

    Without this a cut-off response reached the client as a clean ``stop``.
    """
    if has_tool_calls:
        return "tool_calls"
    return "length" if str(status or "completed").strip().lower() == "incomplete" else "stop"
