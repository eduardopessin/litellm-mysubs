"""Reading a provider stream into a turn, and shaping that turn the way LiteLLM expects.

Extracted from `plugin.py`, which had grown to cover six unrelated jobs at once. Nothing
here knows how to open a connection or how to patch LiteLLM: events go in, a `_Turn` and
the chunks/responses built from it come out. That is what lets the streaming and
non-streaming paths share one interpretation routine — having two is what made the
original's sync and async versions drift apart.

`remember_signature` is injected rather than imported: the signature cache lives in
`plugin.py`'s process state, and importing it back would close a cycle.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from typing import Any

import litellm
from litellm.types.utils import Delta, ModelResponse, ModelResponseStream, StreamingChoices

from .wire import antigravity, codex, planning_leak, thinking_loop
from .wire.usage import Usage


def _discard_signature(call_id: str, signature: str) -> None:
    """No-op sink: a reader used without `plugin.py` has nowhere to keep signatures."""


#: Replaced by `set_signature_sink`; see the module docstring.
remember_signature: Callable[[str, str], None] = _discard_signature


def set_signature_sink(sink: Callable[[str, str], None]) -> None:
    """Points `remember_signature` at the process-wide cache."""
    global remember_signature
    remember_signature = sink


# -- event interpretation ------------------------------------------------------


class _Turn:
    """Accumulator for what a stream of events produced.

    The same object serves both paths: in the non-streaming one it is read at the end, in
    the streaming one it is emitted as it goes. Having two interpretation routines is what
    made the original's synchronous and asynchronous versions diverge.
    """

    __slots__ = (
        "finish_raw",
        "reasoning",
        "response_payload",
        "terminal",
        "text",
        "tool_calls",
        "usage_meta",
    )

    def __init__(self) -> None:
        self.text: list[str] = []
        self.reasoning: list[str] = []
        self.tool_calls: list[dict[str, Any]] = []
        self.usage_meta: dict[str, Any] = {}
        self.finish_raw: object = None
        self.terminal = False
        #: The terminal event's own `response` object, kept verbatim.
        #:
        #: Codex speaks the Responses API natively, so `/v1/responses` is served by handing
        #: this back rather than rebuilding it from the accumulated text — see
        #: `_codex_responses_turn`. The chat path ignores it.
        self.response_payload: dict[str, Any] = {}


class StreamError(RuntimeError):
    """Failure inside a stream with HTTP 200.

    Both Codex (``response.failed``) and the CCA (in-band ``error``) report errors in the
    body of a successful response. Swallowing them delivered an empty turn as success.
    """


class _CodexReader:
    """Translates Responses API events into OpenAI chunks, updating a `_Turn`.

    A ``feed``/``close`` interface instead of a generator over an iterable: the same object
    serves the streaming path (what ``feed`` returns is emitted) and the non-streaming one
    (it is discarded), without an event that produces several chunks being held back.
    """

    __slots__ = ("_active", "_index_of", "_turn", "_ws_bytes", "_ws_events")

    def __init__(self, turn: _Turn) -> None:
        self._turn = turn
        self._active: dict[str, dict[str, Any]] = {}
        self._index_of: dict[str, int] = {}
        self._ws_events = 0
        self._ws_bytes = 0

    def feed(self, event: dict[str, Any]) -> list[ModelResponseStream]:
        kind = event.get("type")
        turn = self._turn

        if kind == "response.output_item.added":
            item = event.get("item") or {}
            if item.get("type") != "function_call":
                return []
            item_id = str(item.get("id") or "")
            call_id = codex.composite_call_id(item.get("call_id"), item.get("id"))
            name = str(item.get("name") or "")
            index = len(self._index_of)
            self._index_of[item_id] = index
            self._active[item_id] = {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": ""},
            }
            return [_tool_open_chunk(index, call_id, name)]

        if kind == "response.function_call_arguments.delta":
            item_id = str(event.get("item_id") or "")
            delta = str(event.get("delta") or "")
            if not delta:
                return []
            # The backend sometimes enters a loop emitting only whitespace in the
            # arguments; with no brake the stream never closes. OMP limits: 256 events /
            # 16 KB.
            if not delta.strip():
                self._ws_events += 1
                self._ws_bytes += len(delta)
                if self._ws_events > 256 or self._ws_bytes > 16384:
                    raise StreamError("Codex: whitespace loop in tool call arguments")
            if item_id in self._active:
                self._active[item_id]["function"]["arguments"] += delta
            return [_tool_delta_chunk(self._index_of.get(item_id, 0), delta)]

        if kind == "response.output_item.done":
            item = event.get("item") or {}
            item_id = str(item.get("id") or "")
            if item.get("type") == "function_call" and item_id in self._active:
                call = self._active.pop(item_id)
                if item.get("arguments"):
                    call["function"]["arguments"] = str(item["arguments"])
                turn.tool_calls.append(call)
            return []

        if kind in ("response.reasoning_text.delta", "response.reasoning_summary_text.delta"):
            delta = str(event.get("delta") or "")
            if not delta:
                return []
            turn.reasoning.append(delta)
            return [_delta_chunk(Delta(reasoning_content=delta))]

        if kind in ("response.output_text.delta", "response.refusal.delta"):
            # OMP treats `refusal` as visible text; without this branch a refused turn
            # reached the client with empty content and a clean stop.
            delta = str(event.get("delta") or "")
            if not delta:
                return []
            turn.text.append(delta)
            return [_delta_chunk(Delta(content=delta))]

        if kind in ("response.completed", "response.incomplete"):
            payload = event.get("response") or {}
            turn.terminal = True
            turn.usage_meta = payload.get("usage") or {}
            turn.finish_raw = payload.get("status") or (
                "incomplete" if kind == "response.incomplete" else "completed"
            )
            if isinstance(payload, dict):
                turn.response_payload = payload
            return []

        if kind in ("response.failed", "error"):
            payload = event.get("response") or {}
            detail = payload.get("error") or event.get("message") or "unknown error"
            raise StreamError(f"Codex: {detail}")

        return []

    def close(self) -> list[ModelResponseStream]:
        """Only `response.completed`/`response.incomplete` close the response.

        A stream cut before that is a transport failure: returning it as success delivered
        truncated output as if it were complete.
        """
        if not self._turn.terminal:
            raise StreamError("Codex: stream ended without response.completed/response.incomplete")
        return []


def _raise_in_band(event: dict[str, Any]) -> None:
    """The CCA returns errors inside the stream with HTTP 200."""
    error = event.get("error")
    if isinstance(error, dict) and int(error.get("code") or 0) >= 400:
        raise StreamError(f"Antigravity {error.get('code')}: {error.get('message') or error}")
    feedback = (event.get("response") or {}).get("promptFeedback")
    if isinstance(feedback, dict) and feedback.get("blockReason"):
        raise StreamError(f"Antigravity: content blocked ({feedback['blockReason']})")


class _AntigravityReader:
    """Translates ``:streamGenerateContent`` events, updating a `_Turn`."""

    __slots__ = ("_guard", "_leak", "_tool_index", "_turn", "_wire_model")

    def __init__(self, turn: _Turn, *, wire_model: str) -> None:
        self._turn = turn
        self._wire_model = wire_model
        self._guard = thinking_loop.guard_for(wire_model)
        self._leak = (
            planning_leak.PlanningLeakFilter()
            if planning_leak.is_flash_leak_model(wire_model)
            else None
        )
        self._tool_index = 0

    def feed(self, event: dict[str, Any]) -> list[ModelResponseStream]:
        _raise_in_band(event)
        turn = self._turn
        payload = event.get("response") or {}
        turn.usage_meta = payload.get("usageMetadata") or turn.usage_meta
        candidates = payload.get("candidates") or []
        if not candidates:
            return []
        turn.finish_raw = candidates[0].get("finishReason") or turn.finish_raw

        parts = (candidates[0].get("content") or {}).get("parts") or []
        if candidates[0].get("finishReason"):
            # The event that closes the turn is the only one carrying the full
            # `usageMetadata`, and it is the one the retirement notice comes in (measured:
            # a single event, with text, `finishReason: STOP` and `total_tokens=0`).
            # Guarding here, **before** emitting what this event carries, is what keeps the
            # notice from going out as content.
            self._guard_retired("".join(str(part.get("text") or "") for part in parts))

        chunks: list[ModelResponseStream] = []
        for part in parts:
            text = str(part.get("text") or "")
            if text:
                chunks.extend(self._text(text, thought=bool(part.get("thought"))))
            call = part.get("functionCall")
            if call:
                chunks.extend(self._call(call, part.get("thoughtSignature")))
        return chunks

    def _guard_retired(self, pending: str = "") -> None:
        """A retired model answers 200 with a notice; accepting it put it in the history.

        The check is over the turn's **accumulated** text plus what has not been emitted
        yet: the notice can arrive split across parts, and neither half alone matches the
        markers.
        """
        antigravity.raise_if_retired(
            "".join(self._turn.text) + pending, self._turn.usage_meta, self._wire_model
        )

    def _text(self, text: str, *, thought: bool) -> list[ModelResponseStream]:
        turn = self._turn
        if thought:
            if self._guard is not None and (reason := self._guard.feed(text)):
                raise thinking_loop.ThinkingLoopError(
                    f"Antigravity: reasoning loop ({reason}) after "
                    f"{self._guard.chars} chars on {self._wire_model}; aborted instead of "
                    "billing the rest"
                )
            turn.reasoning.append(text)
            return [_delta_chunk(Delta(reasoning_content=text))]
        visible = self._leak.feed(text) if self._leak is not None else text
        if not visible:
            return []
        turn.text.append(visible)
        return [_delta_chunk(Delta(content=visible))]

    def _call(self, call: dict[str, Any], signature: object) -> list[ModelResponseStream]:
        call_id = str(call.get("id") or f"call_{uuid.uuid4().hex[:8]}")
        if signature:
            remember_signature(call_id, str(signature))
        name = str(call.get("name") or "")
        arguments = json.dumps(call.get("args") or {})
        self._turn.tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
        index = self._tool_index
        self._tool_index += 1
        return [_tool_open_chunk(index, call_id, name), _tool_delta_chunk(index, arguments)]

    def close(self) -> list[ModelResponseStream]:
        """Flushes what the leak filter held back and turned out not to be planning."""
        chunks: list[ModelResponseStream] = []
        if self._leak is not None and (tail := self._leak.flush()):
            self._turn.text.append(tail)
            chunks.append(_delta_chunk(Delta(content=tail)))
        # Safety net: if the notice arrives with no `finishReason` in the same event, or
        # spread over several, `feed` never saw it whole. Here the turn is complete and the
        # final `usageMetadata` has arrived. Fires at most once per response — if `feed`
        # already raised, this line is never reached.
        self._guard_retired()
        return chunks


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


def _message(turn: _Turn) -> dict[str, Any]:
    """Assistant message in the OpenAI shape, with the reasoning in the standard field."""
    message: dict[str, Any] = {"role": "assistant"}
    if turn.tool_calls:
        message["tool_calls"] = turn.tool_calls
    else:
        message["content"] = "".join(turn.text)
    if turn.reasoning:
        message["reasoning_content"] = "".join(turn.reasoning)
    return message


def _model_response(model: str, turn: _Turn, *, finish_reason: str, usage: Usage) -> ModelResponse:
    return ModelResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[{"index": 0, "message": _message(turn), "finish_reason": finish_reason}],
        usage=_litellm_usage(usage),
    )


