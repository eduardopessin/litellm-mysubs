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
from typing import Any, Literal

#: omp's `StopReason`, restricted to what the wire mappings below produce.
StopReason = Literal["stop", "length", "error"]


@dataclass(frozen=True, slots=True)
class Usage:
    """Token counts for one turn, in the convention of omp's chat-completions server.

    omp keeps ``input`` (uncached prompt) and ``cacheRead`` apart and its
    `openai-chat-server.ts :: buildUsage` emits ``prompt_tokens = input + cacheRead +
    cacheWrite``: **inclusive** of the cached tokens, which are then repeated in
    ``prompt_tokens_details.cached_tokens``. That is also the convention LiteLLM prices
    from — `generic_cost_per_token` bills ``prompt_tokens - cached_tokens`` at the input
    rate and ``cached_tokens`` at the cache-read rate — so a mapper that subtracts the
    cache itself gets it subtracted twice. Measured on 1.101.0, 1000 prompt tokens of
    which 400 cached on ``gemini/gemini-2.5-pro``: the old Google mapper priced the
    prompt at ``0.0003``, omp's formula at ``0.0008``.

    ``cached_tokens`` is read by LiteLLM's spend logging through an attribute of its own
    (``cache_read_input_tokens``), not through the details wrapper — the conversion handles
    that.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0


def _count(value: object) -> int:
    """A counter off the wire as a non-negative int.

    A broken counter is worth zero: blowing up here lost the whole turn.
    """
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


# omp: providers/openai-chat-server.ts :: buildUsage
def _usage(*, uncached: int, cache_read: int, output: int, reasoning: int) -> Usage:
    """omp's ``input``/``cacheRead``/``output`` as the chat counters.

    ``total_tokens`` is recomputed as omp's server does rather than copied from the
    upstream, so it always equals what the other two fields add up to.
    """
    prompt = uncached + cache_read
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=output,
        cached_tokens=cache_read,
        reasoning_tokens=reasoning,
        total_tokens=prompt + output,
    )


# omp: providers/google-shared.ts :: mapGoogleUsage
def google_usage(meta: dict[str, Any]) -> Usage:
    """Gemini ``usageMetadata`` as omp maps it.

    ``promptTokenCount`` includes ``cachedContentTokenCount``. The CCA sometimes omits
    ``promptTokenCount`` or reports a cache count above it, so the prompt falls back to
    ``total - candidates - thoughts`` and the cache is clamped to the prompt: the
    uncached part is never negative. Thoughts count as output.
    """
    candidates = _count(meta.get("candidatesTokenCount"))
    thinking = _count(meta.get("thoughtsTokenCount"))
    total = _count(meta.get("totalTokenCount"))
    prompt = _count(meta.get("promptTokenCount")) or max(0, total - candidates - thinking)
    cache_read = min(_count(meta.get("cachedContentTokenCount")), prompt)
    return _usage(
        uncached=prompt - cache_read,
        cache_read=cache_read,
        output=candidates + thinking,
        reasoning=thinking,
    )


# omp: providers/openai-shared.ts :: populateResponsesUsageFromResponse
# omp: providers/openai-shared.ts :: calculateOpenAIUsageAccounting
def codex_usage(meta: dict[str, Any]) -> Usage:
    """A Responses ``usage`` payload as omp maps it for Codex.

    ``input_tokens`` includes the cached ones. Orchestration tokens are either inside the
    primary counters or beside them, and omp tells which by the reported total; either way
    `calculateUsageCost` bills them at the model's own rates, so they fold into the chat
    counters here. The OpenRouter/DeepSeek cache-write fields omp also reads are not
    ported: the ChatGPT backend does not send them and `Usage` has nowhere to put a write.
    """
    details = meta.get("input_tokens_details") or {}
    output_details = meta.get("output_tokens_details") or {}
    reported_input = _count(meta.get("input_tokens"))
    reported_output = _count(meta.get("output_tokens"))
    cached_raw = details.get("cached_tokens")
    reported_cached = _count(
        cached_raw if cached_raw is not None else meta.get("prompt_cache_hit_tokens")
    )
    orchestration_input = _count(details.get("orchestration_input_tokens"))
    orchestration_input_cached = _count(details.get("orchestration_input_cached_tokens"))
    orchestration_output = _count(output_details.get("orchestration_output_tokens"))
    raw_total = meta.get("total_tokens")
    reported_total = (
        _count(raw_total)
        if isinstance(raw_total, int | float) and not isinstance(raw_total, bool)
        else None
    )
    primary = reported_input + reported_output
    with_separate_orchestration = primary + orchestration_input + orchestration_output
    primary_includes_orchestration = (
        reported_total is not None
        and orchestration_input + orchestration_output > 0
        and abs(reported_total - primary) <= abs(reported_total - with_separate_orchestration)
    )
    orchestration_cached = min(orchestration_input, orchestration_input_cached)
    orchestration_uncached = max(0, orchestration_input - orchestration_cached)
    included = primary_includes_orchestration
    prompt = max(0, reported_input - (orchestration_input if included else 0))
    output = max(0, reported_output - (orchestration_output if included else 0))
    cached = max(0, reported_cached - (orchestration_cached if included else 0))
    return _usage(
        uncached=max(0, prompt - cached) + orchestration_uncached,
        cache_read=cached + orchestration_cached,
        output=output + orchestration_output,
        reasoning=_count(output_details.get("reasoning_tokens")),
    )


# omp: providers/google-shared.ts :: mapStopReasonString
def google_stop_reason(reason: object) -> StopReason:
    """omp's allow list: only ``STOP`` and ``MAX_TOKENS`` are outcomes, the rest errors.

    SAFETY, RECITATION, BLOCKLIST, PROHIBITED_CONTENT, SPII, MALFORMED_FUNCTION_CALL and
    whatever upstream adds next all land on ``error``. The inverse shape — enumerating
    the error reasons — misses what upstream adds: comparing against the source once
    turned up five reasons (FINISH_REASON_UNSPECIFIED, LANGUAGE, IMAGE_OTHER,
    IMAGE_PROHIBITED_CONTENT, IMAGE_RECITATION) that passed as a clean stop.
    """
    if reason == "STOP":
        return "stop"
    if reason == "MAX_TOKENS":
        return "length"
    return "error"


# omp: providers/google-gemini-cli.ts :: streamGoogleGeminiCli
# omp: providers/openai-chat-server.ts :: mapFinishReason
def google_finish_reason(raw: object, has_tool_calls: bool) -> str:
    """The chat finish reason for ``candidates[0].finishReason``.

    Only a benign finish is upgraded by a trailing tool call — a blocked turn stays an
    error even when earlier chunks carried valid calls. An error finish never reaches a
    client through here: `_AntigravityReader.close` raises it, as omp's provider throws
    it, and each route answers with its error shape. What falls through for one is omp's
    `mapFinishReason` default. A turn with no ``finishReason`` at all reads as a stop.
    """
    mapped = google_stop_reason(raw) if raw else "stop"
    if mapped != "error" and has_tool_calls:
        return "tool_calls"
    return "length" if mapped == "length" else "stop"


# omp: providers/openai-shared.ts :: mapOpenAIResponsesStopReason
# omp: providers/openai-shared.ts :: promoteResponsesToolUseStopReason
def codex_finish_reason(status: object, has_tool_calls: bool) -> str:
    """``incomplete`` is truncation by the output limit; any other status a stop.

    Without this a cut-off response reached the client as a clean ``stop``. ``failed``
    and ``cancelled`` arrive as ``response.failed``, which `_CodexReader` raises. A turn
    with tool calls hands them back even when truncated. omp promotes an incomplete turn
    only when every call's arguments closed and ``incomplete_details.reason`` is
    ``max_output_tokens``; the first half holds here because `_CodexReader` records a
    call only on its ``output_item.done``, the reason is not carried to this point.
    """
    if has_tool_calls:
        return "tool_calls"
    return "length" if status == "incomplete" else "stop"
