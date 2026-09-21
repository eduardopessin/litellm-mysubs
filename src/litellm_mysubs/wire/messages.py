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
import time
import uuid
from typing import Any, Final

#: Anthropic's stop reasons, keyed by the OpenAI finish reason the readers produce.
_STOP_REASON: Final = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "stop_sequence",
}


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


def from_model_response(response: Any, model: str) -> dict[str, Any]:
    """A canonical response as a Messages payload.

    The id is minted here with Anthropic's `msg_` prefix: a client that matches on it would
    not recognise the `chatcmpl-` one the chat route produces, and the two routes must be
    indistinguishable from the outside.
    """
    choice = (getattr(response, "choices", None) or [None])[0]
    message = getattr(choice, "message", None)
    usage = getattr(response, "usage", None)

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

    finish = str(getattr(choice, "finish_reason", "") or "stop")
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": _STOP_REASON.get(finish, "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        },
    }


def sse(event: str, data: dict[str, Any]) -> dict[str, Any]:
    """One Messages stream event, in the shape the Anthropic SDK expects."""
    return {"type": event, **data}


def stream_events(payload: dict[str, Any], model: str) -> list[dict[str, Any]]:
    """A finished Messages payload as the event sequence a streaming client expects.

    The turn is produced whole and then replayed as events. Anthropic's sequence is
    `message_start` → per-block `start`/`delta`/`stop` → `message_delta` → `message_stop`,
    and a client that tracks block indices needs them contiguous — which is why they are
    numbered here rather than taken from the provider.
    """
    events: list[dict[str, Any]] = [
        sse(
            "message_start",
            {
                "message": {
                    **{k: v for k, v in payload.items() if k != "content"},
                    "content": [],
                }
            },
        )
    ]

    for index, block in enumerate(payload.get("content") or []):
        kind = block.get("type")
        if kind == "tool_use":
            events.append(
                sse(
                    "content_block_start",
                    {
                        "index": index,
                        "content_block": {**block, "input": {}},
                    },
                )
            )
            events.append(
                sse(
                    "content_block_delta",
                    {
                        "index": index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(block.get("input") or {}),
                        },
                    },
                )
            )
        else:
            field = "thinking" if kind == "thinking" else "text"
            events.append(
                sse(
                    "content_block_start",
                    {"index": index, "content_block": {"type": kind, field: ""}},
                )
            )
            events.append(
                sse(
                    "content_block_delta",
                    {
                        "index": index,
                        "delta": {f"{field}_delta": block.get(field, ""), "type": f"{field}_delta"},
                    },
                )
            )
        events.append(sse("content_block_stop", {"index": index}))

    events.append(
        sse(
            "message_delta",
            {
                "delta": {
                    "stop_reason": payload.get("stop_reason"),
                    "stop_sequence": payload.get("stop_sequence"),
                },
                "usage": payload.get("usage") or {},
            },
        )
    )
    events.append(sse("message_stop", {}))
    return events


def request_id() -> str:
    """Correlation id for a Messages turn, in the upstream's own shape."""
    return f"req_{int(time.time())}_{uuid.uuid4().hex[:12]}"
