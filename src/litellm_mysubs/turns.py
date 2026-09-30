"""Reading a provider stream into a turn, and shaping that turn the way LiteLLM expects.

Nothing here knows how to open a connection or how to patch LiteLLM: events go in, a
`_Turn` and the chunks/responses built from it come out. That is what lets the streaming
and non-streaming paths share one interpretation routine — having two is what made the
original's sync and async versions drift apart.

The two readers are ports of omp's stream handlers: `_CodexReader` of
``providers/openai-codex-responses.ts :: CodexStreamProcessor`` (its event handling, not
its transport recovery, which lives in `transport/`), `_AntigravityReader` of
``providers/google-gemini-cli.ts :: streamGoogleGeminiCli``. Both accumulate what omp's
``AssistantMessage`` holds — ordered content blocks, a stop reason, usage — and emit, as
they go, the chat-completions deltas omp's ``openai-chat-server.ts :: encodeStream``
writes for the same provider events. Both run every reasoning delta through omp's
thinking-loop guard (`thinking_loop.LoopGuard`), which omp's ``stream()`` wraps around
every provider; the visible text is not judged (see `LoopGuard` for why).

`remember_signature` is injected rather than imported: the signature cache lives in
`plugin.py`'s process state, and importing it back would close a cycle.
"""

from __future__ import annotations

import itertools
import json
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Final, Literal

import litellm
from litellm.types.utils import Delta, ModelResponse, ModelResponseStream, StreamingChoices

from .wire import antigravity, codex, planning_leak, thinking_loop, thinking_markup
from .wire.usage import Usage, codex_usage, google_stop_reason, google_usage

#: omp's `StopReason`, as far as a stream that did not fail can end.
StopReason = Literal["stop", "length", "toolUse", "error"]


def _discard_signature(call_id: str, signature: str) -> None:
    """No-op sink: a reader used without `plugin.py` has nowhere to keep signatures."""


#: Replaced by `set_signature_sink`; see the module docstring.
remember_signature: Callable[[str, str], None] = _discard_signature


def set_signature_sink(sink: Callable[[str, str], None]) -> None:
    """Points `remember_signature` at the process-wide cache."""
    global remember_signature
    remember_signature = sink


# -- the turn --------------------------------------------------------------------


class _Turn:
    """What a stream produced, in the terms of omp's ``AssistantMessage``.

    ``content`` holds the blocks in the order they opened: ``{"type": "text", "text"}``,
    ``{"type": "thinking", "thinking"}``, ``{"type": "toolCall", "id", "name",
    "arguments"}`` with the arguments as the JSON text a chat client receives. The same
    object serves both paths: the non-streaming one reads it at the end, the streaming
    one emits deltas as it fills. ``usage`` is set as soon as the upstream reports it, so
    a turn that then fails still knows what it cost.
    """

    __slots__ = ("content", "stop_reason", "usage", "usage_meta")

    def __init__(self) -> None:
        self.content: list[dict[str, Any]] = []
        self.stop_reason: StopReason = "stop"
        self.usage: Usage | None = None
        #: The upstream's own usage payload, as last reported.
        self.usage_meta: dict[str, Any] = {}

    def text(self) -> str:
        return "".join(block["text"] for block in self.content if block["type"] == "text")

    def reasoning(self) -> str:
        return "".join(block["thinking"] for block in self.content if block["type"] == "thinking")

    def tool_calls(self) -> list[dict[str, Any]]:
        return [
            {
                "id": block["id"],
                "type": "function",
                "function": {"name": block["name"], "arguments": block["arguments"]},
            }
            for block in self.content
            if block["type"] == "toolCall"
        ]

    def has_tool_calls(self) -> bool:
        return any(block["type"] == "toolCall" for block in self.content)


# -- failures --------------------------------------------------------------------


class StreamError(RuntimeError):
    """Failure inside a stream with HTTP 200.

    Both Codex (``response.failed``) and the CCA (in-band ``error``) report errors in the
    body of a successful response. Swallowing them delivered an empty turn as success.

    ``status`` is the one the upstream put in the event, when it put one. Without it the
    failure is what omp's gateway makes of a provider failure with no status — 502
    ``upstream_error`` — and LiteLLM's proxy reads both off the exception (`status_code`,
    `type`); `observability._as_litellm_error` classifies by the message the way omp's
    `classifyGatewayError` does before the error leaves.
    """

    status_code: int = 502
    type: str = "upstream_error"

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class GenerationFailedError(StreamError):
    """The upstream ended the turn with a finish reason omp counts as an error."""


class ThinkingLoopError(StreamError):
    """omp's loop guard tripped: the turn is ended before the rest of it is billed."""


# omp: providers/openai-codex-responses.ts :: CodexWhitespaceToolCallLoopError
class WhitespaceLoopError(StreamError):
    """Codex streamed whitespace into a call's arguments without end.

    ``replayable`` is omp's condition for asking again: besides the degenerate call, which
    is dropped, the turn produced nothing but reasoning, and no call had finished.
    """

    def __init__(self, message: str, *, replayable: bool) -> None:
        super().__init__(message)
        self.replayable = replayable


# omp: providers/openai-chat-server.ts :: stringifyArgs
def _stringify_args(arguments: object) -> str:
    """Tool-call arguments as omp's chat server writes them: ``JSON.stringify``."""
    try:
        return json.dumps(arguments, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"


# omp: providers/openai-shared.ts :: finalizeToolCallArgumentsDone
def _final_arguments(raw: str) -> str:
    """The authoritative arguments of a finished call, as a chat client receives them.

    omp parses them (``parseStreamingJson``) and re-serializes the object. What will not
    parse is kept verbatim here: omp's relaxed repair parser is not ported, and handing
    the client the upstream's own text beats replacing it with ``{}``.
    """
    stripped = raw.lstrip()
    if not stripped:
        return "{}"
    try:
        return _stringify_args(json.loads(stripped))
    except ValueError:
        return raw


# -- Codex -----------------------------------------------------------------------


# omp: providers/openai-codex-responses.ts :: CODEX_WHITESPACE_TOOL_CALL_ARGUMENT_DELTA_EVENT_LIMIT
WHITESPACE_DELTA_EVENT_LIMIT: Final = 256
# omp: providers/openai-codex-responses.ts :: CODEX_WHITESPACE_TOOL_CALL_ARGUMENT_DELTA_CHAR_LIMIT
WHITESPACE_DELTA_CHAR_LIMIT: Final = 16 * 1024

#: `response.status` values omp accepts (`parseCodexResponseStatus`).
_RESPONSE_STATUSES: Final = frozenset(
    ("completed", "failed", "in_progress", "cancelled", "queued", "incomplete")
)
_TERMINAL_EVENTS: Final = frozenset(("response.completed", "response.done", "response.incomplete"))


# omp: providers/openai-codex-responses.ts :: isJsonWhitespaceOnly
def _is_json_whitespace_only(value: str) -> bool:
    """Only JSON's four whitespace characters; an empty delta counts, as in omp."""
    return all(char in "\t\n\r " for char in value)


def _string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _integer(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return int(value)


def _error_detail(value: object) -> dict[str, str | None] | None:
    """omp's ``codexErrorDetailSchema``: an object's string ``code``/``type``/``message``."""
    if not isinstance(value, dict):
        return None
    return {key: _string(value.get(key)) for key in ("code", "type", "message")}


def _first(*values: str | None) -> str | None:
    """JavaScript's ``a ?? b ?? c``."""
    return next((value for value in values if value is not None), None)


def _truncated_json(event: dict[str, Any]) -> str:
    raw = json.dumps(event, separators=(",", ":"), ensure_ascii=False, default=str)
    return raw if len(raw) <= 800 else f"{raw[:800]}…[truncated {len(raw) - 800}]"


# omp: providers/openai-codex-responses.ts :: formatCodexFailure
def _format_codex_failure(event: dict[str, Any]) -> str:
    response = event.get("response")
    response = response if isinstance(response, dict) else {}
    error = _error_detail(event.get("error")) or _error_detail(response.get("error")) or {}
    message = _first(error.get("message"), _string(event.get("message")))
    message = _first(message, _string(response.get("message")))
    code = _first(error.get("code"), error.get("type"), _string(event.get("code")))
    status = _first(_string(response.get("status")), _string(event.get("status")))
    meta = [f"{name}={value}" for name, value in (("code", code), ("status", status)) if value]
    if message:
        return f"Codex response failed: {message}" + (f" ({', '.join(meta)})" if meta else "")
    if meta:
        return f"Codex response failed ({', '.join(meta)})"
    return f"Codex response failed: {_truncated_json(event)}"


# omp: providers/openai-codex-responses.ts :: createCodexProviderStreamError
# omp: providers/openai-codex-responses.ts :: formatCodexErrorEvent
def _codex_stream_error(event: dict[str, Any]) -> StreamError:
    """The failure an ``error`` or ``response.failed`` event carries, worded as omp's."""
    detail = _format_codex_failure(event)
    if event.get("type") == "error":
        detail = detail.replace("response failed", "error event", 1)
    return StreamError(detail)


# omp: providers/openai-shared.ts :: finalizeReasoningThinking
def _final_thinking(item: dict[str, Any], streamed: str) -> str:
    """The summary parts joined, else the reasoning text, else what streamed."""
    summary = item.get("summary")
    if isinstance(summary, list):
        joined = "\n\n".join(
            str(part.get("text") or "") for part in summary if isinstance(part, dict)
        )
        if joined:
            return joined
    content = item.get("content")
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict) and first.get("type") == "reasoning_text":
            text = str(first.get("text") or "")
            if text:
                return text
    return streamed


# omp: providers/openai-shared.ts :: finalizeMessageText
def _final_text(item: dict[str, Any], streamed: str) -> str:
    """The terminal content is authoritative when the item carries any."""
    content = item.get("content")
    if not isinstance(content, list) or not content:
        return streamed
    return "".join(
        str((part.get("text") if part.get("type") == "output_text" else part.get("refusal")) or "")
        for part in content
        if isinstance(part, dict)
    )


# omp: providers/openai-codex-responses.ts :: createOutputBlockForItem
def _block_for_item(item: dict[str, Any]) -> dict[str, Any] | None:
    """The content block an output item opens; ``None`` for items with no block.

    Tool calls keep their raw argument text under ``partial`` while it streams, as omp's
    ``kStreamingPartialJson``; ``None`` there means the arguments were finalized.
    """
    kind = item.get("type")
    if kind == "reasoning":
        return {"type": "thinking", "thinking": ""}
    if kind == "message":
        return {"type": "text", "text": ""}
    call_id = codex.composite_call_id(_string(item.get("call_id")), _string(item.get("id")))
    if kind == "function_call":
        return {
            "type": "toolCall",
            "id": call_id,
            "name": str(item.get("name") or ""),
            "arguments": "{}",
            "partial": str(item.get("arguments") or ""),
            "custom": False,
        }
    if kind == "computer_call":
        return {
            "type": "toolCall",
            "id": call_id,
            "name": "computer",
            "arguments": "{}",
            "partial": "",
            "custom": False,
        }
    if kind == "custom_tool_call":
        raw = str(item.get("input") or "")
        return {
            "type": "toolCall",
            "id": call_id,
            "name": str(item.get("name") or ""),
            "arguments": _stringify_args({"input": raw}),
            "partial": raw,
            "custom": True,
        }
    return None


@dataclass(slots=True)
class _OpenItem:
    """omp's ``CodexOpenItem``: an output item between its ``added`` and ``done``."""

    item: dict[str, Any]
    block: dict[str, Any] | None
    item_id: str | None
    output_index: int | None
    #: Position of the call in the chat ``tool_calls`` array, for tool-call items.
    tool_index: int | None = None
    #: Whether any argument bytes left in a delta; omp's ``hasArgumentBytes``.
    sent_arguments: bool = False


@dataclass(slots=True)
class _WhitespaceRun:
    """omp's ``CodexWhitespaceToolCallArgumentsDeltaState``."""

    item_id: str
    output_index: int | None
    events: int
    chars: int
    first_sequence: int | None
    last_sequence: int | None = None


# omp: providers/openai-codex-responses.ts :: CodexStreamProcessor
# omp: providers/openai-codex-responses.ts :: CodexStreamRuntime
class _CodexReader:
    """Translates Responses API events into chat deltas, filling a `_Turn`.

    A ``feed``/``close`` interface instead of a generator over an iterable: the same object
    serves the streaming path (what ``feed`` returns is emitted) and the non-streaming one
    (it is discarded), without an event that produces several chunks being held back.
    ``done`` turns true at the terminal event, after which omp reads no further.

    Events reach the item they name: ``item_id`` first, ``output_index`` for items the
    wire leaves without an id, the latest added item only for events that carry neither.
    A delta for an item that already closed is dropped instead of landing on a sibling.
    """

    __slots__ = (
        "_current",
        "_finished_call",
        "_guard",
        "_open_by_id",
        "_open_by_index",
        "_tool_count",
        "_turn",
        "_whitespace",
        "done",
    )

    def __init__(self, turn: _Turn, *, wire_model: str = "") -> None:
        self._turn = turn
        self._guard = thinking_loop.LoopGuard(wire_model)
        self._open_by_id: dict[str, _OpenItem] = {}
        self._open_by_index: dict[int, _OpenItem] = {}
        self._current: _OpenItem | None = None
        self._whitespace: _WhitespaceRun | None = None
        self._tool_count = 0
        #: A call reached ``output_item.done``: omp's ``canSafelyReplayWebsocketOverSse``.
        self._finished_call = False
        self.done = False

    # omp: providers/openai-codex-responses.ts :: CodexStreamProcessor.handleStreamEvent
    def feed(self, event: dict[str, Any]) -> list[ModelResponseStream]:
        kind = event.get("type")
        if not isinstance(kind, str) or not kind:
            return []
        if kind == "response.output_item.added":
            self._whitespace = None
            return self._item_added(event)
        if kind == "response.reasoning_summary_part.added":
            entry = self._open_item_for(event)
            part = event.get("part")
            if (
                entry is not None
                and entry.item.get("type") == "reasoning"
                and isinstance(part, dict)
                and part.get("type") == "summary_text"
            ):
                entry.item.setdefault("summary", []).append(
                    {**part, "text": str(part.get("text") or "")}
                )
            return []
        if kind == "response.reasoning_summary_text.delta":
            return self._summary_delta(event)
        if kind == "response.reasoning_text.delta":
            entry = self._open_item_for(event)
            if entry is None or entry.item.get("type") != "reasoning" or entry.block is None:
                return []
            return self._thinking(entry.block, str(event.get("delta") or ""))
        if kind == "response.reasoning_summary_part.done":
            return self._summary_part_done(event)
        if kind == "response.content_part.added":
            entry = self._open_item_for(event)
            part = event.get("part")
            if (
                entry is not None
                and entry.item.get("type") == "message"
                and isinstance(part, dict)
                and part.get("type") in ("output_text", "refusal")
            ):
                entry.item.setdefault("content", []).append(dict(part))
            return []
        if kind in ("response.output_text.delta", "response.refusal.delta"):
            part_type = "refusal" if kind == "response.refusal.delta" else "output_text"
            return self._text_delta(event, part_type)
        if kind in (
            "response.function_call_arguments.delta",
            "response.custom_tool_call_input.delta",
        ):
            item_type = (
                "function_call" if kind.startswith("response.function") else "custom_tool_call"
            )
            return self._arguments_delta(event, item_type)
        if kind in (
            "response.function_call_arguments.done",
            "response.custom_tool_call_input.done",
        ):
            self._whitespace = None
            self._arguments_done(event, custom=kind.startswith("response.custom"))
            return []
        if kind == "response.output_item.done":
            self._whitespace = None
            return self._item_done(event)
        if kind in _TERMINAL_EVENTS:
            self._completed(event)
            return []
        if kind in ("error", "response.failed"):
            raise _codex_stream_error(event)
        return []

    def close(self) -> list[ModelResponseStream]:
        """Only a terminal event closes the response; a failed one fails the turn.

        A stream cut before that is a transport failure: returning it as success delivered
        truncated output as if it were complete.
        """
        if not self.done:
            raise StreamError("Codex stream ended before terminal completion event")
        if self._turn.stop_reason == "error":
            raise StreamError("Codex response failed")
        if detail := self._guard.done():
            raise ThinkingLoopError(thinking_loop.loop_error_message(detail))
        return []

    # -- routing -----------------------------------------------------------------

    # omp: providers/openai-codex-responses.ts :: CodexStreamRuntime.openItemForEvent
    def _open_item_for(self, event: dict[str, Any]) -> _OpenItem | None:
        item_id = _string(event.get("item_id"))
        if item_id:
            return self._open_by_id.get(item_id)
        output_index = _integer(event.get("output_index"))
        if output_index is not None:
            return self._open_by_index.get(output_index)
        return self._current

    def _close_item(self, entry: _OpenItem | None) -> None:
        if entry is None:
            return
        if entry.item_id:
            self._open_by_id.pop(entry.item_id, None)
        if entry.output_index is not None:
            self._open_by_index.pop(entry.output_index, None)
        if self._current is entry:
            self._current = None

    # -- events ------------------------------------------------------------------

    def _item_added(self, event: dict[str, Any]) -> list[ModelResponseStream]:
        raw = event.get("item")
        item = dict(raw) if isinstance(raw, dict) else {}
        block = _block_for_item(item)
        if block is not None:
            self._turn.content.append(block)
        entry = _OpenItem(
            item=item,
            block=block,
            item_id=_string(item.get("id")) or None,
            output_index=_integer(event.get("output_index")),
        )
        self._current = entry
        if entry.item_id:
            self._open_by_id[entry.item_id] = entry
        if entry.output_index is not None:
            self._open_by_index[entry.output_index] = entry
        if block is None or block["type"] != "toolCall":
            return []
        entry.tool_index = self._tool_count
        self._tool_count += 1
        return [_tool_open_chunk(entry.tool_index, block["id"], block["name"])]

    # omp: providers/openai-shared.ts :: appendReasoningSummaryTextDelta
    def _summary_delta(self, event: dict[str, Any]) -> list[ModelResponseStream]:
        entry = self._open_item_for(event)
        delta = str(event.get("delta") or "")
        if entry is None or entry.item.get("type") != "reasoning" or entry.block is None:
            return []
        if not delta:
            return []
        # Codex passes no summary index: the delta extends the latest part.
        summary = entry.item.setdefault("summary", [])
        if not summary:
            summary.append({"type": "summary_text", "text": ""})
        summary[-1]["text"] = str(summary[-1].get("text") or "") + delta
        return self._thinking(entry.block, delta)

    # omp: providers/openai-shared.ts :: appendReasoningSummaryPartDone
    def _summary_part_done(self, event: dict[str, Any]) -> list[ModelResponseStream]:
        """A finished summary part is followed by a paragraph break, as omp emits it."""
        entry = self._open_item_for(event)
        if entry is None or entry.item.get("type") != "reasoning" or entry.block is None:
            return []
        summary = entry.item.get("summary")
        if not isinstance(summary, list) or not summary:
            return []
        summary[-1]["text"] = str(summary[-1].get("text") or "") + "\n\n"
        return self._thinking(entry.block, "\n\n")

    def _thinking(self, block: dict[str, Any], delta: str) -> list[ModelResponseStream]:
        if not delta:
            return []
        if detail := self._guard.thinking_delta(delta):
            raise ThinkingLoopError(thinking_loop.loop_error_message(detail))
        block["thinking"] += delta
        return [_delta_chunk(Delta(reasoning_content=delta))]

    # omp: providers/openai-shared.ts :: appendMessageTextDelta
    def _text_delta(self, event: dict[str, Any], part_type: str) -> list[ModelResponseStream]:
        """`refusal` is visible text too: without it a refused turn read as empty."""
        entry = self._open_item_for(event)
        if entry is None or entry.item.get("type") != "message" or entry.block is None:
            return []
        delta = str(event.get("delta") or "")
        if not delta:
            return []
        self._guard.text_delta(delta)
        content = entry.item.setdefault("content", [])
        field = "text" if part_type == "output_text" else "refusal"
        if not content or not isinstance(content[-1], dict) or content[-1].get("type") != part_type:
            content.append({"type": part_type, field: ""})
        content[-1][field] = str(content[-1].get(field) or "") + delta
        entry.block["text"] += delta
        return [_delta_chunk(Delta(content=delta))]

    # omp: providers/openai-codex-responses.ts :: CodexStreamRuntime.handleToolCallArgumentsDelta
    # omp: providers/openai-codex-responses.ts :: CodexStreamRuntime.handleCustomToolCallInputDelta
    # omp: providers/openai-shared.ts :: accumulateToolCallArgumentsDelta
    def _arguments_delta(self, event: dict[str, Any], item_type: str) -> list[ModelResponseStream]:
        delta = str(event.get("delta") or "")
        # Observed before the item is looked up: degenerate frames keep arriving after
        # the item closed, and dropping them unobserved reopens the endless stream.
        if interruption := self._observe_whitespace(event, delta):
            self._drop_degenerate_call()
            replayable = not self._finished_call and all(
                block["type"] == "thinking" for block in self._turn.content
            )
            raise WhitespaceLoopError(interruption, replayable=replayable)
        entry = self._open_item_for(event)
        if entry is None or entry.item.get("type") != item_type or entry.block is None:
            return []
        if not delta or entry.tool_index is None:
            return []
        block = entry.block
        block["partial"] = str(block.get("partial") or "") + delta
        if block["custom"]:
            block["arguments"] = _stringify_args({"input": block["partial"]})
        entry.sent_arguments = True
        return [_tool_delta_chunk(entry.tool_index, delta)]

    def _drop_degenerate_call(self) -> None:
        """omp's ``#dropTrailingDegenerateToolCall``: the looping call never reaches the
        answer; what came before it is kept."""
        entry = self._current
        block = entry.block if entry is not None else None
        content = self._turn.content
        if block is not None and block["type"] == "toolCall" and content and content[-1] is block:
            content.pop()
        self._close_item(entry)

    def _observe_whitespace(self, event: dict[str, Any], delta: str) -> str | None:
        """The backend sometimes loops on whitespace-only argument deltas and never ends.

        omp's ``CodexStreamRuntime.observeWhitespaceToolCallArgumentsDelta``, under the
        class's anchor (the method's own would not fit on one line).

        Consecutive ones for the same item are counted; any other delta resets the run.
        """
        if not _is_json_whitespace_only(delta):
            self._whitespace = None
            return None
        item_id = (
            _string(event.get("item_id"))
            or (_string(self._current.item.get("id")) if self._current is not None else None)
            or ""
        )
        output_index = _integer(event.get("output_index"))
        sequence = _integer(event.get("sequence_number"))
        run = self._whitespace
        if run is None or run.item_id != item_id or run.output_index != output_index:
            run = _WhitespaceRun(item_id, output_index, 0, 0, sequence)
            self._whitespace = run
        run.events += 1
        run.chars += len(delta)
        run.last_sequence = sequence
        if run.events < WHITESPACE_DELTA_EVENT_LIMIT and run.chars < WHITESPACE_DELTA_CHAR_LIMIT:
            return None
        item_label = f" for item {item_id}" if item_id else ""
        sequence_label = (
            ""
            if run.first_sequence is None or run.last_sequence is None
            else f", sequence {run.first_sequence}..{run.last_sequence}"
        )
        return (
            f"Interrupted OpenAI Codex response after {run.events} consecutive "
            f"whitespace-only tool-call argument delta events ({run.chars} chars"
            f"{sequence_label}){item_label}."
        )

    # omp: providers/openai-codex-responses.ts :: CodexStreamRuntime.handleToolCallArgumentsDone
    # omp: providers/openai-codex-responses.ts :: CodexStreamRuntime.handleCustomToolCallInputDone
    def _arguments_done(self, event: dict[str, Any], *, custom: bool) -> None:
        entry = self._open_item_for(event)
        item_type = "custom_tool_call" if custom else "function_call"
        if entry is None or entry.item.get("type") != item_type or entry.block is None:
            return
        value = event.get("input" if custom else "arguments")
        if not isinstance(value, str):
            return
        block = entry.block
        if custom:
            block["partial"] = value
            block["arguments"] = _stringify_args({"input": value})
        else:
            block["arguments"] = _final_arguments(value)
            block["partial"] = None

    # omp: providers/openai-codex-responses.ts :: CodexStreamProcessor.handleOutputItemDone
    def _item_done(self, event: dict[str, Any]) -> list[ModelResponseStream]:
        raw = event.get("item")
        if not isinstance(raw, dict):
            return []
        item = dict(raw)
        item_id = _string(item.get("id"))
        entry = (self._open_by_id.get(item_id) if item_id else None) or self._open_item_for(event)
        block = entry.block if entry is not None else None
        kind = item.get("type")
        self._close_item(entry)
        if block is None:
            return []
        if kind == "reasoning" and block["type"] == "thinking":
            block["thinking"] = _final_thinking(item, block["thinking"])
            if detail := self._guard.thinking_end():
                raise ThinkingLoopError(thinking_loop.loop_error_message(detail))
            return []
        if kind == "message" and block["type"] == "text":
            block["text"] = _final_text(item, block["text"])
            return []
        if block["type"] != "toolCall" or entry is None or entry.tool_index is None:
            return []
        if kind == "function_call":
            block["arguments"] = _final_arguments(str(item.get("arguments") or "{}"))
        elif kind == "computer_call":
            block["arguments"] = "{}"
        elif kind == "custom_tool_call":
            partial = block.get("partial")
            raw_input = partial if partial else str(item.get("input") or "")
            block["arguments"] = _stringify_args({"input": raw_input})
        else:
            return []
        block["partial"] = None
        self._finished_call = True
        return self._correct_arguments(entry, block)

    # omp: providers/openai-chat-server.ts :: encodeStream
    def _correct_arguments(
        self, entry: _OpenItem, block: dict[str, Any]
    ) -> list[ModelResponseStream]:
        """The final arguments, for a call whose deltas never carried them.

        omp's chat encoder corrects a call at ``toolcall_end`` only where what streamed was
        empty, since a client concatenates each field. Codex sends some calls whole in
        ``output_item.done``; without this, a streamed call reached the client with no
        arguments. The id and name are not corrected: both come with ``output_item.added``,
        and the Messages and Responses encoders read an id as a new block.
        """
        if entry.sent_arguments or entry.tool_index is None:
            return []
        entry.sent_arguments = True
        return [_tool_delta_chunk(entry.tool_index, block["arguments"])]

    # omp: providers/openai-codex-responses.ts :: CodexStreamProcessor.handleResponseCompleted
    # omp: providers/openai-shared.ts :: mapOpenAIResponsesStopReason
    # omp: providers/openai-shared.ts :: promoteResponsesToolUseStopReason
    # omp: providers/openai-shared.ts :: finalizePendingResponsesToolCalls
    def _completed(self, event: dict[str, Any]) -> None:
        self.done = True
        turn = self._turn
        raw = event.get("response")
        response = raw if isinstance(raw, dict) else {}
        usage = response.get("usage")
        if isinstance(usage, dict):
            turn.usage_meta = usage
            turn.usage = codex_usage(usage)
        raw_status = response.get("status")
        status = (
            raw_status if isinstance(raw_status, str) and raw_status in _RESPONSE_STATUSES else None
        )
        details = response.get("incomplete_details")
        reason = details.get("reason") if isinstance(details, dict) else None
        # Steering stops a response at an output boundary: it ends like a completed one.
        steered = status == "incomplete" and reason == "steered"
        promote_incomplete = (
            status == "incomplete"
            and reason == "max_output_tokens"
            and self._executable_incomplete_tool_calls()
        )
        for block in turn.content:
            if block["type"] == "toolCall" and block.get("partial"):
                partial = block["partial"]
                block["arguments"] = (
                    _stringify_args({"input": partial})
                    if block["custom"]
                    else _final_arguments(partial)
                )
            if block["type"] == "toolCall":
                block["partial"] = None
        turn.stop_reason = _responses_stop_reason("completed" if steered else status)
        if turn.has_tool_calls() and (
            turn.stop_reason == "stop" or (promote_incomplete and turn.stop_reason == "length")
        ):
            turn.stop_reason = "toolUse"

    # omp: providers/openai-shared.ts :: hasExecutableIncompleteResponsesToolCalls
    def _executable_incomplete_tool_calls(self) -> bool:
        """Whether a turn cut by the output limit can still hand back its calls.

        Only calls whose arguments are complete JSON count, and a call already closed by
        ``output_item.done`` does not: omp does not take that event as proof of
        completion. ``json.loads`` stands for omp's ``classifyJsonPrefix(...) ===
        "complete"``.
        """
        has_call = False
        for block in self._turn.content:
            if block["type"] != "toolCall":
                continue
            has_call = True
            partial = block.get("partial")
            if block["custom"] or partial is None:
                return False
            try:
                json.loads(partial)
            except ValueError:
                return False
        return has_call


def _responses_stop_reason(status: str | None) -> StopReason:
    """omp's `mapOpenAIResponsesStopReason`; an unknown status degrades to a stop."""
    if status == "incomplete":
        return "length"
    if status in ("failed", "cancelled"):
        return "error"
    return "stop"


# -- Antigravity -----------------------------------------------------------------


def _js_truthy(value: object) -> bool:
    """JavaScript truthiness: an empty object or list is still true."""
    if isinstance(value, dict | list):
        return True
    return bool(value)


def _raise_in_band(event: dict[str, Any]) -> None:
    """The CCA returns errors inside the stream with HTTP 200; omp throws on any of them."""
    error = event.get("error")
    if not _js_truthy(error):
        return
    fields = error if isinstance(error, dict) else {}
    detail = fields.get("message") or fields.get("status") or "unknown error"
    code = _integer(fields.get("code"))
    raise StreamError(
        f"Cloud Code Assist stream error: {detail}",
        status=code if code is not None and code >= 400 else None,
    )


_tool_call_counter: Iterator[int] = itertools.count(1)


# omp: providers/google-shared.ts :: nextToolCallId
def _next_tool_call_id(name: str) -> str:
    """An id for a call the upstream sent without one, or with one already used."""
    return f"{name}_{int(time.time() * 1000)}_{next(_tool_call_counter)}"


# omp: providers/google-shared.ts :: hasMeaningfulGoogleContent
def _has_meaningful_content(turn: _Turn) -> bool:
    return any(
        block["type"] == "toolCall" or (block["type"] == "text" and block["text"].strip())
        for block in turn.content
    )


# omp: providers/google-gemini-cli.ts :: streamGoogleGeminiCli
class _AntigravityReader:
    """Translates ``:streamGenerateContent`` events into chat deltas, filling a `_Turn`.

    Visible text runs, in omp's order, through the planning-leak buffer (flash models
    only) and then the reasoning-markup healer, so a leaked ``<thinking>`` section reaches
    the client as reasoning. A Cloud Code stream is read to its end: ``done`` never turns
    true before it.
    """

    __slots__ = (
        "_block",
        "_buffering",
        "_error_message",
        "_guard",
        "_healing",
        "_leak_model",
        "_saw_finish",
        "_text_buffer",
        "_tool_count",
        "_tool_names",
        "_turn",
        "_wire_model",
        "done",
    )

    def __init__(
        self, turn: _Turn, *, wire_model: str, tool_names: frozenset[str] = frozenset()
    ) -> None:
        self._turn = turn
        self._wire_model = wire_model
        self._tool_names = tool_names
        self._guard = thinking_loop.LoopGuard(wire_model)
        self._healing = thinking_markup.StreamMarkupHealing()
        self._leak_model = planning_leak.is_flash_leak_model(wire_model)
        self._block: dict[str, Any] | None = None
        self._buffering = False
        self._text_buffer = ""
        self._saw_finish = False
        self._error_message = ""
        self._tool_count = 0
        self.done = False

    def feed(self, event: dict[str, Any]) -> list[ModelResponseStream]:
        _raise_in_band(event)
        response = event.get("response")
        if not isinstance(response, dict):
            return []
        candidates = response.get("candidates")
        candidates = candidates if isinstance(candidates, list) else []
        feedback = response.get("promptFeedback")
        if not candidates and isinstance(feedback, dict) and feedback.get("blockReason"):
            detail = feedback.get("blockReasonMessage")
            raise StreamError(
                f"Request blocked by Google ({feedback['blockReason']})"
                + (f": {detail}" if detail else "")
            )
        meta = response.get("usageMetadata")
        candidate = candidates[0] if candidates and isinstance(candidates[0], dict) else None
        chunks: list[ModelResponseStream] = []
        if candidate is not None:
            content = candidate.get("content")
            parts = content.get("parts") if isinstance(content, dict) else None
            parts = (
                [part for part in parts if isinstance(part, dict)]
                if isinstance(parts, list)
                else []
            )
            finish = candidate.get("finishReason")
            if finish:
                # The event that closes the turn is the only one carrying the full
                # `usageMetadata`, and it is the one the retirement notice comes in
                # (measured: a single event, with text, `finishReason: STOP` and
                # `total_tokens=0`). Guarding here, **before** emitting what this event
                # carries, is what keeps the notice from going out as content.
                pending = "".join(str(part.get("text") or "") for part in parts)
                self._guard_retired(pending, meta if isinstance(meta, dict) else None)
            for part in parts:
                chunks.extend(self._part(part))
            if finish:
                self._finish(finish)
        if isinstance(meta, dict):
            self._turn.usage_meta = meta
            self._turn.usage = google_usage(meta)
        return chunks

    def close(self) -> list[ModelResponseStream]:
        """Releases what the filters held back, then fails the turn where omp throws.

        In omp's order: an error finish (SAFETY, RECITATION, MALFORMED_FUNCTION_CALL, ...),
        then a turn with nothing to deliver (no text, no tool call) unless the output
        limit cut it, then a stream that
        ended without any finish at all — a connection dropped mid-answer is not an
        answer. The usage has arrived by then, and whatever text already left stays with
        the client, which then reads the failure — not a turn that merely stopped.
        """
        chunks: list[ModelResponseStream] = []
        if self._buffering and self._text_buffer:
            outcome, visible = planning_leak.consume_planning_buffer(
                self._text_buffer, self._tool_names, final=True
            )
            if outcome != "incomplete":
                chunks.extend(self._feed_visible(visible))
        self._buffering = False
        self._text_buffer = ""
        chunks.extend(self._flush_visible())
        self._end_block()
        # Safety net: if the notice arrives with no `finishReason` in the same event, or
        # spread over several, `feed` never saw it whole. Here the turn is complete and the
        # final `usageMetadata` has arrived.
        self._guard_retired()
        turn = self._turn
        if turn.stop_reason == "error":
            raise GenerationFailedError(self._error_message)
        # Deliberate divergence: omp fails any turn without content, a MAX_TOKENS one
        # included. A client that set the output ceiling asked for the cut; measured live
        # on gemini-3-flash, "Write a 600-word essay about bridges." with max_tokens=64
        # spent the budget on reasoning, and 0.1.16 answered 200 with `length` where the
        # port answered 502. A cut turn stays a `length` stop, empty or not.
        if turn.stop_reason != "length" and not _has_meaningful_content(turn):
            thought_only = any(
                block["type"] == "thinking" and block["thinking"].strip() for block in turn.content
            )
            raise StreamError(
                "Cloud Code Assist API returned a thought-only response without final output"
                if thought_only
                else "Cloud Code Assist API returned an empty response"
            )
        if not self._saw_finish:
            raise StreamError(
                "Cloud Code Assist stream ended without a finish reason "
                "(connection dropped or response truncated)"
            )
        if detail := self._guard.done():
            raise ThinkingLoopError(thinking_loop.loop_error_message(detail))
        return chunks

    def _guard_retired(self, pending: str = "", meta: dict[str, Any] | None = None) -> None:
        """A retired model answers 200 with a notice; accepting it put it in the history.

        No omp counterpart: the notice is specific to the CCA catalog. The check is over
        the turn's **accumulated** text plus what has not been emitted yet: the notice can
        arrive split across parts, and neither half alone matches the markers.
        """
        antigravity.raise_if_retired(
            self._turn.text() + pending,
            meta if meta is not None else self._turn.usage_meta,
            self._wire_model,
        )

    def _part(self, part: dict[str, Any]) -> list[ModelResponseStream]:
        chunks: list[ModelResponseStream] = []
        text = part.get("text")
        if isinstance(text, str) and text:
            # omp's `isThinkingPart`: the flag itself, not a signature.
            if part.get("thought") is True:
                chunks.extend(self._flush_visible())
                chunks.extend(self._thinking(text))
            else:
                chunks.extend(self._visible_part(text))
        call = part.get("functionCall")
        if _js_truthy(call):
            chunks.extend(self._flush_visible())
            self._end_block()
            # omp drops a planning buffer still held when a call arrives.
            self._buffering = False
            self._text_buffer = ""
            chunks.extend(self._tool_call(call if isinstance(call, dict) else {}, part))
        return chunks

    def _visible_part(self, text: str) -> list[ModelResponseStream]:
        """Flash models' visible text may open a planning object; it is held until known."""
        if self._buffering:
            self._text_buffer += text
        elif self._leak_model and text.lstrip().startswith("{"):
            self._buffering = True
            self._text_buffer = text
        else:
            return self._feed_visible(text)
        outcome, visible = planning_leak.consume_planning_buffer(
            self._text_buffer, self._tool_names
        )
        if outcome == "incomplete":
            return []
        self._buffering = False
        self._text_buffer = ""
        return self._feed_visible(visible)

    def _feed_visible(self, text: str) -> list[ModelResponseStream]:
        return self._healed(self._healing.feed_events(text))

    def _flush_visible(self) -> list[ModelResponseStream]:
        return self._healed(self._healing.flush_events())

    def _healed(self, events: list[thinking_markup.HealingEvent]) -> list[ModelResponseStream]:
        chunks: list[ModelResponseStream] = []
        for kind, value in events:
            chunks.extend(self._thinking(value) if kind == "thinking" else self._text(value))
        return chunks

    def _start_block(self, kind: Literal["text", "thinking"]) -> dict[str, Any]:
        block = self._block
        if block is None or block["type"] != kind:
            self._end_block()
            block = {"type": kind, kind: ""}
            self._turn.content.append(block)
            self._block = block
        return block

    def _end_block(self) -> None:
        """omp's ``endCurrentBlock``; the end of a reasoning block flushes the guard."""
        block, self._block = self._block, None
        if (
            block is not None
            and block["type"] == "thinking"
            and (detail := self._guard.thinking_end())
        ):
            raise ThinkingLoopError(thinking_loop.loop_error_message(detail))

    def _thinking(self, delta: str) -> list[ModelResponseStream]:
        if not delta:
            return []
        block = self._start_block("thinking")
        if detail := self._guard.thinking_delta(delta):
            raise ThinkingLoopError(thinking_loop.loop_error_message(detail))
        block["thinking"] += delta
        return [_delta_chunk(Delta(reasoning_content=delta))]

    def _text(self, delta: str) -> list[ModelResponseStream]:
        if not delta:
            return []
        block = self._start_block("text")
        self._guard.text_delta(delta)
        block["text"] += delta
        return [_delta_chunk(Delta(content=delta))]

    def _tool_call(self, call: dict[str, Any], part: dict[str, Any]) -> list[ModelResponseStream]:
        """A whole call per part; an id missing or already used gets a fresh one."""
        provided = call.get("id")
        name = str(call.get("name") or "")
        taken = any(
            block["type"] == "toolCall" and block["id"] == provided for block in self._turn.content
        )
        call_id = _next_tool_call_id(name or "tool") if not provided or taken else str(provided)
        args = call.get("args")
        arguments = _stringify_args(args if args is not None else {})
        self._turn.content.append(
            {"type": "toolCall", "id": call_id, "name": name, "arguments": arguments}
        )
        if signature := part.get("thoughtSignature"):
            remember_signature(call_id, str(signature))
        index = self._tool_count
        self._tool_count += 1
        return [_tool_open_chunk(index, call_id, name), _tool_delta_chunk(index, arguments)]

    def _finish(self, finish: object) -> None:
        """Only a benign finish is upgraded by a tool call: a blocked turn stays an error."""
        self._saw_finish = True
        mapped = google_stop_reason(finish)
        if mapped in ("stop", "length") and self._turn.has_tool_calls():
            self._turn.stop_reason = "toolUse"
            return
        self._turn.stop_reason = mapped
        if mapped == "error":
            self._error_message = f"Generation failed with finish reason: {finish}"


# -- the shape LiteLLM expects -------------------------------------------------


def _delta_chunk(delta: Delta) -> ModelResponseStream:
    return ModelResponseStream(choices=[StreamingChoices(index=0, delta=delta, finish_reason=None)])


def _tool_open_chunk(index: int, call_id: str, name: str) -> ModelResponseStream:
    """Opens a tool call. ``role`` rides here because this may be the turn's first chunk."""
    return _delta_chunk(
        Delta(
            role="assistant",
            tool_calls=[
                {
                    "index": index,
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": ""},
                }
            ],
        )
    )


def _tool_delta_chunk(index: int, arguments: str) -> ModelResponseStream:
    return _delta_chunk(Delta(tool_calls=[{"index": index, "function": {"arguments": arguments}}]))


def _finish_chunk(reason: str) -> ModelResponseStream:
    return ModelResponseStream(
        choices=[StreamingChoices(index=0, delta=Delta(), finish_reason=reason)]
    )


def _usage_chunk(usage: Usage) -> ModelResponseStream:
    """Final chunk with the real usage; without it LiteLLM estimates by token counting.

    ``choices`` carries one empty entry instead of being ``[]``: the ``/v1/responses``
    route's iterator does ``chunk.choices[0].delta`` with no guard, and an empty list kills
    the stream before the terminal event — the client waits forever.
    """
    chunk = ModelResponseStream(
        choices=[StreamingChoices(index=0, delta=Delta(), finish_reason=None)]
    )
    chunk.usage = _litellm_usage(usage)
    return chunk


def _litellm_usage(usage: Usage) -> litellm.Usage:
    """``cached_tokens`` also has to go on the attribute the spend logging reads."""
    out = litellm.Usage(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        prompt_tokens_details={"cached_tokens": usage.cached_tokens},
    )
    out.cache_read_input_tokens = usage.cached_tokens
    return out


# omp: providers/openai-chat-server.ts :: mapFinishReason
def finish_reason(turn: _Turn) -> str:
    """The chat ``finish_reason`` for how the turn ended."""
    if turn.stop_reason == "toolUse" or (turn.has_tool_calls() and turn.stop_reason == "stop"):
        return "tool_calls"
    return "length" if turn.stop_reason == "length" else "stop"


# omp: providers/openai-chat-server.ts :: encodeResponse
def _message(turn: _Turn) -> dict[str, Any]:
    """Assistant message in the OpenAI shape: text beside the calls, reasoning apart."""
    text = turn.text()
    message: dict[str, Any] = {"role": "assistant", "content": text or None}
    reasoning = turn.reasoning()
    if reasoning:
        message["reasoning_content"] = reasoning
    calls = turn.tool_calls()
    if calls:
        message["tool_calls"] = calls
    return message


def _model_response(model: str, turn: _Turn) -> ModelResponse:
    return ModelResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[{"index": 0, "message": _message(turn), "finish_reason": finish_reason(turn)}],
        usage=_litellm_usage(turn.usage or Usage()),
    )
