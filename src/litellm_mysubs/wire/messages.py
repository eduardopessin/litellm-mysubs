"""Anthropic Messages ⇄ the canonical turn, so ``/v1/messages`` reaches every subscription.

Anthropic-native clients speak Messages. Serving only chat-completions and Responses makes
each of them adapt, which is what a proxy exists to avoid. Measured on the live gateway
before this module existed::

    /v1/messages  mysubs/claudecode/*  401 Missing Anthropic API Key
    /v1/messages  mysubs/codex/*       401 AuthenticationError

Only the **envelope** is translated here. The provider wire stays each provider's own:
`plugin.py` feeds the converted turns to `_codex_spec` / `_antigravity_spec`, so Codex still
gets the Responses shape and Antigravity still gets Cloud Code — including everything the
0.1.5 fixes put there (`tool_use.id`, the thinking budget bounds). Claude Max needs no
conversion at all: Messages *is* its wire.

The pivot is the same message list the chat route already uses, so a block shape that works
on one route cannot break on another.
"""

from __future__ import annotations

import json
import uuid
from typing import Any


def _text_of(content: Any) -> str:
    """The plain text of a Messages `content`, which is a string or a block list."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        str(block.get("text") or "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def to_chat_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Messages turns as the canonical list the provider spec builders consume.

    `system` is a top-level field in Messages and a leading message in the canonical form,
    so it moves rather than being dropped — a system prompt silently lost is worse than a
    rejected request.

    `tool_use` and `tool_result` blocks carry ids that the providers need: Vertex requires
    `tool_use.id` (see the 0.1.5 fix) and Codex pairs its outputs by `call_id`. They are
    preserved verbatim rather than regenerated.
    """
    out: list[dict[str, Any]] = []

    system = payload.get("system")
    if system:
        out.append({"role": "system", "content": _text_of(system)})

    for message in payload.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user")
        content = message.get("content")

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            continue

        # A single Messages turn can hold text, tool calls and tool results at once; the
        # canonical form splits those across messages, and tool results are their own role.
        text_parts: list[Any] = []
        tool_calls: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []

        for block in content:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text":
                text_parts.append({"type": "text", "text": str(block.get("text") or "")})
            elif kind == "image":
                source = block.get("source") or {}
                if source.get("type") == "base64":
                    url = f"data:{source.get('media_type')};base64,{source.get('data')}"
                    text_parts.append({"type": "image_url", "image_url": {"url": url}})
            elif kind == "tool_use":
                tool_calls.append(
                    {
                        "id": str(block.get("id") or ""),
                        "type": "function",
                        "function": {
                            "name": str(block.get("name") or ""),
                            "arguments": json.dumps(block.get("input") or {}),
                        },
                    }
                )
            elif kind == "tool_result":
                results.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(block.get("tool_use_id") or ""),
                        "content": _text_of(block.get("content")),
                    }
                )

        if text_parts or tool_calls:
            turn: dict[str, Any] = {"role": role}
            if text_parts:
                turn["content"] = text_parts if len(text_parts) > 1 else text_parts[0]["text"]
            else:
                turn["content"] = None
            if tool_calls:
                turn["tool_calls"] = tool_calls
            out.append(turn)
        out.extend(results)

    return out


def to_tools(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Messages tool declarations as the canonical ones.

    Messages nests the schema under `input_schema`; the canonical form nests the whole
    declaration under `function`.
    """
    tools: list[dict[str, Any]] = []
    for tool in payload.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": str(tool.get("name") or ""),
                    "description": str(tool.get("description") or ""),
                    "parameters": tool.get("input_schema") or {},
                },
            }
        )
    return tools


# omp: providers/anthropic-messages-server.ts :: mapToolChoice
def to_tool_choice(payload: dict[str, Any]) -> object | None:
    """Messages ``tool_choice`` as the canonical (chat-completions) one, or ``None``.

    Messages spells it ``{"type": "auto" | "any" | "none"}`` or ``{"type": "tool", "name"}``;
    the canonical form is ``"auto"``, ``"required"``, ``"none"`` or ``{"type": "function",
    "function": {"name"}}``, which the Codex and Antigravity builders already translate.
    Passing the Messages shape through was answered by Codex with 400 ``Invalid value:
    'auto'`` (``'any'``, ``'tool'``) — every explicit choice failed, measured on the live
    gateway on 0.1.14. ``disable_parallel_tool_use`` has no counterpart in either builder
    and is not carried.
    """
    choice = payload.get("tool_choice")
    if not isinstance(choice, dict):
        return None
    kind = choice.get("type")
    if kind == "auto":
        return "auto"
    if kind == "any":
        return "required"
    if kind == "none":
        return "none"
    if kind == "tool" and choice.get("name"):
        return {"type": "function", "function": {"name": str(choice["name"])}}
    return None


# -- outbound: ported from omp's Anthropic Messages server ---------------------
#
# omp serves Anthropic Messages from other providers' streams with the same shapes this
# route needs, so the encoding is theirs. The one structural difference is the input: omp
# drives `encodeStream` from explicit `*_start`/`*_delta`/`*_end` events, while this
# plugin's readers produce canonical LiteLLM chunks with no start or end markers. So a
# block here opens when the kind of content changes and closes when the next one opens,
# which is the same sequence for the same turn.


# omp: providers/anthropic-messages-server.ts :: newMessageId
def new_message_id() -> str:
    """Anthropic's `msg_` prefix and 24 hex digits.

    A client that matches on the prefix would not recognise the `chatcmpl-` id the chat
    route produces, and the two routes must be indistinguishable from the outside.
    """
    return f"msg_{uuid.uuid4().hex[:24]}"


# omp: providers/anthropic-messages-server.ts :: mapStopReasonOut
def map_stop_reason_out(reason: object, has_tool_use: bool) -> str:
    """The Anthropic stop reason for a canonical finish reason.

    The canonical reasons are OpenAI's, which is what the readers produce: `length` and
    `tool_calls` are omp's `length` and `toolUse`. The fallback is omp's too — a turn that
    hands a tool back owes the client `tool_use` even when the upstream closed it with a
    plain stop, or the Anthropic loop (run tools while `stop_reason == "tool_use"`) never
    runs the call.

    A turn the upstream stopped for its own reasons (Google's SAFETY, RECITATION, ...) never
    gets here: as in omp, whose `encodeResponse` throws on an `error` stop, the reader
    raises it and the client gets the `error` event or an error response instead of a
    turn that merely ended.
    """
    if reason == "length":
        return "max_tokens"
    if reason == "tool_calls":
        return "tool_use"
    return "tool_use" if has_tool_use else "end_turn"


# omp: providers/anthropic-messages-server.ts :: encodeUsage
def _usage(usage: Any) -> dict[str, int]:
    """Canonical usage as Messages counts: ``input_tokens`` excludes the cache reads.

    Both mappers in `wire/usage.py` put the cached tokens inside ``prompt_tokens`` (omp's
    chat convention), so omp's ``input`` is the difference. The spend row of a streamed
    turn is priced from these fields (`observability._messages_response`), which is why
    the cache has to be stated rather than folded into ``input_tokens`` at the full rate.
    Neither upstream reports a cache write.
    """
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    details = getattr(usage, "prompt_tokens_details", None)
    cache_read = min(int(getattr(details, "cached_tokens", 0) or 0), prompt)
    return {
        "input_tokens": prompt - cache_read,
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": 0,
    }


# omp: providers/anthropic-messages-server.ts :: encodeResponse, encodeContentBlocks
def encode_response(response: Any, model: str) -> dict[str, Any]:
    """A canonical response as a Messages payload."""
    choice = (getattr(response, "choices", None) or [None])[0]
    message = getattr(choice, "message", None)

    blocks: list[dict[str, Any]] = []
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning:
        blocks.append({"type": "thinking", "thinking": str(reasoning)})
    text = getattr(message, "content", None)
    if text:
        blocks.append({"type": "text", "text": str(text)})
    for call in getattr(message, "tool_calls", None) or []:
        function = getattr(call, "function", None)
        raw = getattr(function, "arguments", "") or "{}"
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # An unparseable argument string is the model's, not ours: it travels as-is
            # rather than being dropped, so the client sees what was actually produced.
            parsed = {"__raw": raw}
        blocks.append(
            {
                "type": "tool_use",
                "id": str(getattr(call, "id", "") or f"toolu_{uuid.uuid4().hex[:16]}"),
                "name": str(getattr(function, "name", "") or ""),
                "input": parsed,
            }
        )

    has_tool_use = any(block["type"] == "tool_use" for block in blocks)
    return {
        "id": new_message_id(),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": map_stop_reason_out(getattr(choice, "finish_reason", None), has_tool_use),
        "stop_sequence": None,
        "usage": _usage(getattr(response, "usage", None)),
    }


# omp: providers/anthropic-messages-server.ts :: sseFrame
def sse_frame(event: str, data: dict[str, Any]) -> bytes:
    """One event as the SSE frame a Messages client parses, ``event:`` line included.

    The Anthropic SDKs dispatch on the SSE ``event`` field and drop a frame without one —
    ``anthropic/_streaming.py`` yields only when ``sse.event`` is ``message_start``,
    ``content_block_delta`` and so on. LiteLLM's proxy writes a dict chunk as a bare
    ``data:`` line (`ProxyBaseLLMRequestProcessing.return_sse_chunk`), so the dicts this
    route used to hand it reached `client.messages.stream(...)` as nothing at all: measured
    through the real proxy app and the real SDK, `get_final_message()` failed with no
    snapshot while ``curl`` showed every line. Bytes pass through that serializer as-is.
    """
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


# omp: providers/anthropic-messages-server.ts :: encodeStream
def stream_error(message: str) -> dict[str, Any]:
    """The `error` event omp's encoder sends when the stream fails after it opened.

    Without it the SDK sees the stream simply end: LiteLLM's own error frame is a bare
    ``data:`` line, which the SDK drops like any other (see `sse_frame`).
    """
    return {"type": "error", "error": {"type": "api_error", "message": message}}


# omp: providers/anthropic-messages-server.ts :: encodeStream, closeBlock
class MessagesStreamEncoder:
    """The Anthropic event sequence for one streamed turn, each block as it arrives.

    Block indices are **ours**: the upstreams never send them, so numbering blocks as they
    open is no more invented than numbering them at the end, and it is what lets each block
    leave as it arrives. Thinking and tool calls stream like text: both readers emit them
    as canonical deltas (`reasoning_content`, `tool_calls`) while the turn runs, and
    relaying the text alone made a streamed tool call vanish — the client got
    `stop_reason: tool_use` and no `tool_use` block to answer.
    """

    __slots__ = ("_index", "_open", "_tool_blocks")

    def __init__(self) -> None:
        #: Index of the open block, or of the next one when none is open.
        self._index = 0
        #: What the open block holds: ``"text"``, ``"thinking"``, ``"tool_use:<n>"``.
        self._open: str | None = None
        #: Canonical tool-call index -> the block index it was given.
        self._tool_blocks: dict[int, int] = {}

    def start(self, model: str) -> dict[str, Any]:
        """`message_start`: the envelope, with no content and zero usage yet."""
        return {
            "type": "message_start",
            "message": {
                "id": new_message_id(),
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": _usage(None),
            },
        }

    def text(self, text: str) -> list[dict[str, Any]]:
        out = self._switch("text", {"type": "text", "text": ""})
        out.append(self._delta(self._index, {"type": "text_delta", "text": text}))
        return out

    def thinking(self, text: str) -> list[dict[str, Any]]:
        # omp opens with no `signature` and sends a `signature_delta` only when the block
        # has one. Neither upstream hands the readers an Anthropic signature, so none goes
        # out rather than an invented one.
        out = self._switch("thinking", {"type": "thinking", "thinking": ""})
        out.append(self._delta(self._index, {"type": "thinking_delta", "thinking": text}))
        return out

    def tool_call(self, index: int, call_id: str, name: str) -> list[dict[str, Any]]:
        """Opens a ``tool_use`` block. ``input`` starts empty and arrives as JSON deltas.

        The id travels verbatim, as on the non-streamed path: Codex pairs its outputs by
        the composite id and Vertex requires ``tool_use.id`` on the way back.
        """
        out = self._switch(
            f"tool_use:{index}",
            {"type": "tool_use", "id": call_id, "name": name, "input": {}},
        )
        self._tool_blocks[index] = self._index
        return out

    def tool_arguments(self, index: int, partial_json: str) -> list[dict[str, Any]]:
        # Both readers send a call's arguments only after its opening chunk, so the block
        # exists; a fragment for a call that never opened is a reader defect and raises
        # here rather than landing on whatever block happens to be open.
        block = self._tool_blocks[index]
        return [self._delta(block, {"type": "input_json_delta", "partial_json": partial_json})]

    def done(self, reason: object, usage: Any) -> list[dict[str, Any]]:
        """omp's `done`: close what is open, then `message_delta` and `message_stop`."""
        return [
            *self.close_block(),
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": map_stop_reason_out(reason, bool(self._tool_blocks)),
                    "stop_sequence": None,
                },
                "usage": _usage(usage),
            },
            {"type": "message_stop"},
        ]

    def close_block(self) -> list[dict[str, Any]]:
        if self._open is None:
            return []
        self._open = None
        out = [{"type": "content_block_stop", "index": self._index}]
        self._index += 1
        return out

    def _switch(self, key: str, content_block: dict[str, Any]) -> list[dict[str, Any]]:
        if self._open == key:
            return []
        out = self.close_block()
        self._open = key
        out.append(
            {"type": "content_block_start", "index": self._index, "content_block": content_block}
        )
        return out

    @staticmethod
    def _delta(index: int, delta: dict[str, Any]) -> dict[str, Any]:
        return {"type": "content_block_delta", "index": index, "delta": delta}
