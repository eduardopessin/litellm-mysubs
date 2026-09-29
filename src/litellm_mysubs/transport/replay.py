"""Which stream events commit an attempt, and which failures omp sends again.

omp re-sends a request that failed mid-stream — but only while the failed attempt is still
*replay-safe*: nothing the user can see has been produced yet. Codex replays a retryable
failure (`#tryRetryProviderError`); the Cloud Code Assist replays an empty ``STOP`` and
fails over to the other host (`streamGoogleGeminiCli`). Past the first text, thinking or
tool call, a replay would hand the client the same prefix twice.

The transport keeps that window with a hold-back: events that deliver nothing — the
``response.created`` preamble, an empty text part, usage — are kept back until one that
does arrives. Then everything held is released in order and the stream goes live; a replay
before that point simply drops what was held, so nothing is ever delivered twice.

These are pure functions over the raw events: the transport asks, and they answer with
what omp's stream processors would have concluded from the same event.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Final

from .retry import is_retryable_status


class Verdict(Enum):
    """What one event means for an attempt nothing has been delivered from yet."""

    HOLD = "hold"
    """Delivers nothing: keep it back; a replay discards it."""

    DELIVER = "deliver"
    """Content, or an outcome omp does not replay: release what was held, go live."""

    RETRY = "retry"
    """A failure omp replays while the attempt is uncommitted."""


def _dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _str(value: object) -> str | None:
    """omp's `optionalCodexString`: a string, or absent."""
    return value if isinstance(value, str) else None


def _first(*values: str | None) -> str | None:
    """JavaScript's ``a ?? b ?? c``: the first one present, even if empty."""
    return next((value for value in values if value is not None), None)


# -- Codex ---------------------------------------------------------------------------

# omp: providers/openai-codex-responses.ts :: CODEX_RETRYABLE_EVENT_CODES
CODEX_RETRYABLE_EVENT_CODES: Final = frozenset({"model_error", "server_error", "internal_error"})
# omp: providers/openai-codex-responses.ts :: CODEX_RETRYABLE_EVENT_MESSAGE
CODEX_RETRYABLE_EVENT_MESSAGE: Final = re.compile(
    r"processing your request|retry your request|temporar(?:y|ily)|overloaded"
    r"|service.?unavailable|internal error|server error",
    re.I,
)
# omp: error/flags.ts :: PYTHON_HTTP2_STREAM_RESET_PATTERN
_PYTHON_HTTP2_STREAM_RESET: Final = re.compile(
    r"<StreamReset stream_id:\d+, error_code:(?:2|7), remote_reset:True>"
)
# omp: error/flags.ts :: PYTHON_HTTP_INCOMPLETE_CHUNK_PATTERN
_PYTHON_HTTP_INCOMPLETE_CHUNK: Final = re.compile(
    r"peer closed connection without sending complete message body \(incomplete chunked read\)"
)

# omp: providers/openai-shared.ts :: OPENAI_RESPONSES_PROGRESS_EVENT_TYPES
_RESPONSES_PROGRESS: Final = frozenset(
    {
        "response.created",
        "response.output_item.added",
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_text.delta",
        "response.reasoning_summary_text.done",
        "response.reasoning_summary_part.done",
        "response.reasoning_text.delta",
        "response.content_part.added",
        "response.output_text.delta",
        "response.refusal.delta",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.custom_tool_call_input.delta",
        "response.custom_tool_call_input.done",
        "response.output_item.done",
        "response.completed",
        "response.incomplete",
        "response.failed",
        "error",
    }
)
# omp: providers/openai-codex-responses.ts :: CODEX_ADDITIONAL_PROGRESS_EVENT_TYPES
_CODEX_ADDITIONAL_PROGRESS: Final = frozenset({"response.done", "response.incomplete"})

_CODEX_FAILURES: Final = ("response.failed", "error")
_CODEX_TERMINALS: Final = ("response.completed", "response.done", "response.incomplete")
_CODEX_DELTAS: Final = (
    "response.output_text.delta",
    "response.refusal.delta",
    "response.reasoning_text.delta",
    "response.reasoning_summary_text.delta",
)
# omp: providers/openai-codex-responses.ts :: createOutputBlockForItem
#: Items that open a tool-call block: visible content the moment they are announced.
_CODEX_TOOL_ITEMS: Final = ("function_call", "computer_call", "custom_tool_call")


# omp: providers/openai-codex-responses.ts :: isCodexStreamProgressEvent
def is_codex_progress(event: dict[str, Any]) -> bool:
    """Whether the event resets the idle watchdog; keep-alives and rate-limit notices don't.

    A Codex response can stay open on those alone, and counting them as activity is what
    would keep a stalled stream alive forever.
    """
    kind = event.get("type")
    return isinstance(kind, str) and (
        kind in _RESPONSES_PROGRESS or kind in _CODEX_ADDITIONAL_PROGRESS
    )


# omp: providers/openai-codex-responses.ts :: isRetryableCodexFailureEvent
def is_retryable_codex_failure(event: dict[str, Any]) -> bool:
    """A ``response.failed``/``error`` the backend itself calls transient."""
    response = _dict(event.get("response"))
    # `event.error ?? event.response?.error`: an empty object still wins.
    error = event["error"] if isinstance(event.get("error"), dict) else _dict(response.get("error"))
    code = _first(_str(error.get("code")), _str(error.get("type")), _str(event.get("code")))
    if code and code.lower() in CODEX_RETRYABLE_EVENT_CODES:
        return True
    message = _first(
        _str(error.get("message")), _str(event.get("message")), _str(response.get("message"))
    )
    return bool(message) and bool(
        CODEX_RETRYABLE_EVENT_MESSAGE.search(message or "")
        or _PYTHON_HTTP2_STREAM_RESET.search(message or "")
        or _PYTHON_HTTP_INCOMPLETE_CHUNK.search(message or "")
    )


def _item_text(item: dict[str, Any]) -> bool:
    """Whether a finished message or reasoning item carries any text of its own."""
    for key in ("content", "summary"):
        parts = item.get(key)
        for part in parts if isinstance(parts, list) else ():
            if _str(_dict(part).get("text")) or _str(_dict(part).get("refusal")):
                return True
    return False


# omp: providers/openai-codex-responses.ts :: tryRetryProviderError
def codex_verdict(event: dict[str, Any]) -> Verdict:
    """omp's replay gate, per event.

    Committing takes a text or reasoning delta of any length — whitespace included, since
    it already reached the consumer as a delta — or a tool call, image or finished text
    the item itself carries. An announced message or reasoning item is not content yet:
    ``output_item.added`` alone is replay-safe.
    """
    kind = event.get("type")
    if kind in _CODEX_FAILURES:
        return Verdict.RETRY if is_retryable_codex_failure(event) else Verdict.DELIVER
    if kind in _CODEX_TERMINALS:
        return Verdict.DELIVER
    if kind in _CODEX_DELTAS:
        return Verdict.DELIVER if _str(event.get("delta")) else Verdict.HOLD
    if kind == "response.reasoning_summary_text.done":
        return Verdict.DELIVER if _str(event.get("text")) else Verdict.HOLD
    item = _dict(event.get("item"))
    if kind == "response.output_item.added":
        return Verdict.DELIVER if item.get("type") in _CODEX_TOOL_ITEMS else Verdict.HOLD
    if kind == "response.output_item.done":
        visible = (
            item.get("type") in _CODEX_TOOL_ITEMS
            or (item.get("type") == "image_generation_call" and bool(item.get("result")))
            or _item_text(item)
        )
        return Verdict.DELIVER if visible else Verdict.HOLD
    return Verdict.HOLD


# -- Cloud Code Assist (Antigravity) ----------------------------------------------

# omp: providers/google-shared.ts :: mapStopReasonString
#: Finish reasons omp reads as a normal end; every other one is a failed generation.
GOOGLE_OK_FINISHES: Final = ("STOP", "MAX_TOKENS")


def _candidate(event: dict[str, Any]) -> dict[str, Any]:
    """``candidates[0]``, the only one omp reads."""
    candidates = _dict(event.get("response")).get("candidates")
    return _dict(candidates[0]) if isinstance(candidates, list) and candidates else {}


def _google_parts(event: dict[str, Any]) -> list[dict[str, Any]]:
    parts = _dict(_candidate(event).get("content")).get("parts")
    return [_dict(part) for part in parts] if isinstance(parts, list) else []


def google_finish(event: dict[str, Any]) -> str | None:
    """The event's ``finishReason``, if it carries one."""
    return _str(_candidate(event).get("finishReason"))


# omp: providers/google-shared.ts :: isThinkingPart
# omp: providers/google-shared.ts :: hasMeaningfulGoogleContent
def google_meaningful(event: dict[str, Any]) -> bool:
    """A tool call or visible text that is not only whitespace: an answer worth delivering."""
    for part in _google_parts(event):
        if part.get("functionCall"):
            return True
        text = _str(part.get("text"))
        if part.get("thought") is not True and text and text.strip():
            return True
    return False


def _google_thinking(event: dict[str, Any]) -> bool:
    """Thinking omp would keep (`hasThinkingOutput`): a thought-only STOP is never replayed."""
    for part in _google_parts(event):
        if part.get("thought") is True and (
            (_str(part.get("text")) or "").strip() or part.get("thoughtSignature")
        ):
            return True
    return False


# omp: providers/google-gemini-cli.ts :: streamGoogleGeminiCli
def antigravity_verdict(event: dict[str, Any]) -> Verdict:
    """omp's per-chunk handling, reduced to the replay gate.

    ``RETRY`` is the in-band error with a transient code: omp throws it with that status
    and moves to the other endpoint. A 429 in band stays with the host that sent it, for
    the reason `retry.decide_antigravity` gives. An empty ``STOP`` is held: whether it is
    replayed is decided when the stream ends empty.
    """
    error = event.get("error")
    # JavaScript truthiness: `{}` is an error too.
    if isinstance(error, dict) or error:
        code = _dict(error).get("code")
        transient = (
            isinstance(code, int)
            and not isinstance(code, bool)
            and code != 429
            and is_retryable_status(code)
        )
        return Verdict.RETRY if transient else Verdict.DELIVER
    response = _dict(event.get("response"))
    if not response.get("candidates") and _dict(response.get("promptFeedback")).get("blockReason"):
        return Verdict.DELIVER
    if google_meaningful(event) or _google_thinking(event):
        return Verdict.DELIVER
    finish = google_finish(event)
    if finish and finish != "STOP":
        return Verdict.DELIVER
    return Verdict.HOLD
