"""OpenAI Responses ⇄ the canonical turn, so ``/v1/responses`` reaches every subscription.

The counterpart of `messages.py` for the Responses dialect, and a port of omp's Responses
server (`openai-responses-server.ts`), which serves this API from other providers' streams
with the shapes this route needs. Only the **envelope** is translated: the request becomes
the canonical message list `dispatch` already consumes, and each provider keeps its own
wire underneath — Codex still goes out as Responses and Antigravity as Cloud Code.

The route used to hand the `input` items to the providers' chat converters as they came,
on the theory that they were already the shape those produce. They are not: an item with no
``role`` (`function_call`, `function_call_output`) read as an empty user turn and was
dropped, and neither `instructions` nor `reasoning` was read at all. A client replaying a
tool exchange sent the model a conversation with the call and its result cut out.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

# -- inbound: omp's `parseRequest` ---------------------------------------------
#
# omp parses into its own `Context`; the pivot here is the canonical chat list, so the
# mapping is item for item into chat roles. Three differences, each because the canonical
# list has no slot for what omp keeps:
#
# - `reasoning` items are not bridged. omp keeps them as a signed `thinking` block that its
#   OpenAI providers replay; here the Codex request builder replays the reasoning of the
#   responses it kept itself (`codex.NativeTurn`), matched by call ids or answer text, and
#   a client's own reasoning items have nowhere to go in the canonical list.
# - `custom_tool_call`, `computer_call` and their outputs are not bridged: the canonical
#   tool call is a function call, and neither upstream here serves those tool kinds.
# - `function_call.arguments` stays the JSON string it arrived as. omp parses it because
#   its `ToolCall` holds an object; the canonical call holds the string both wires consume.
#
# `tools` and `tool_choice` need no translation, so omp's `buildTools`/`mapToolChoice` have
# no counterpart: both spec builders already read the flat Responses shapes
# (`{"type": "function", "name": ...}`), and Codex, a Responses backend itself, keeps the
# hosted tools omp's generic bridge has to drop.


# omp: providers/openai-responses-server.ts :: outputTextOf
def _output_text_of(content: Any) -> str:
    """An assistant item's text; a refusal is kept so the replayed history still says so."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") in ("output_text", "text"):
            parts.append(str(block.get("text") or ""))
        elif block.get("type") == "refusal":
            parts.append(f"[refusal: {block.get('refusal') or ''}]")
    return "".join(parts)


def _input_text_of(content: Any) -> str:
    """A system item's text, flattened as omp's `inputContentParts` does for its prompt."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        str(block.get("text") or "")
        for block in content
        if isinstance(block, dict) and block.get("type") in ("input_text", "text")
    )


def _has_native_refs(content: Any) -> bool:
    return isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") in ("input_image", "input_file")
        for block in content
    )


# omp: providers/openai-responses-server.ts :: ensureAssistantPlaceholder
def _assistant_placeholder(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """The assistant turn a call item belongs to: the previous one, or a new empty one."""
    if messages and messages[-1].get("role") == "assistant":
        return messages[-1]
    placeholder: dict[str, Any] = {"role": "assistant", "content": None}
    messages.append(placeholder)
    return placeholder


# omp: providers/openai-responses-server.ts :: findToolNameById
def _tool_name(messages: list[dict[str, Any]], call_id: str) -> str:
    """The name of the call a result answers. Antigravity's `functionResponse` needs it."""
    for message in reversed(messages):
        for call in message.get("tool_calls") or []:
            if call.get("id") == call_id:
                return str(call["function"]["name"])
    return ""


# omp: providers/openai-responses-server.ts :: parseRequest
def to_chat_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """A Responses request's `instructions` and `input` as the canonical message list.

    System prompts lead, one message each, in order — `instructions` first, then the
    ``system`` items — because that is where Codex's builder takes its ``instructions``
    from. User and developer content travels verbatim: the part types it holds
    (`input_text`, `input_image`) are ones both converters read, which is what omp's
    `providerPayload` preserves the native item for.
    """
    system: list[str] = []
    messages: list[dict[str, Any]] = []

    instructions = payload.get("instructions")
    if isinstance(instructions, str) and instructions:
        system.append(instructions)

    value = payload.get("input")
    if isinstance(value, str):
        messages.append({"role": "user", "content": value})
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        # Items may omit `type` and rely on `role` (the convenience shape).
        kind = item.get("type") or ("message" if "role" in item else None)
        if kind == "message":
            role = item.get("role")
            content = item.get("content")
            if role == "system":
                if _has_native_refs(content):
                    messages.append({"role": "developer", "content": content})
                elif text := _input_text_of(content):
                    system.append(text)
            elif role in ("user", "developer"):
                messages.append({"role": role, "content": content})
            elif role == "assistant":
                messages.append({"role": "assistant", "content": _output_text_of(content)})
        elif kind == "function_call":
            call_id = str(item.get("call_id") or "")
            placeholder = _assistant_placeholder(messages)
            placeholder.setdefault("tool_calls", []).append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": str(item.get("name") or ""),
                        "arguments": str(item.get("arguments") or "{}"),
                    },
                }
            )
        elif kind == "function_call_output":
            call_id = str(item.get("call_id") or "")
            output = item.get("output")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": _tool_name(messages, call_id),
                    "content": output if isinstance(output, list) else str(output or ""),
                }
            )
        # Other item types are tolerated but not bridged, as in omp.

    return [*({"role": "system", "content": text} for text in system), *messages]


# omp: providers/openai-responses-server.ts :: parseRequest
def to_options(payload: dict[str, Any]) -> dict[str, Any]:
    """The request options the spec builders read under their canonical names.

    `reasoning` travels whole, not as its `effort` alone: Codex's builder accepts the
    object and honours the requested `summary` level, which omp's generic bridge has no
    slot for. `max_output_tokens` is the Responses spelling of the output ceiling
    Antigravity's builder reads as `max_tokens`.
    """
    options: dict[str, Any] = {}
    reasoning = payload.get("reasoning")
    if isinstance(reasoning, dict):
        options["reasoning_effort"] = reasoning
    if payload.get("max_output_tokens") is not None:
        options["max_tokens"] = payload["max_output_tokens"]
    return options


# -- outbound: ported from omp's Responses server ------------------------------
#
# omp drives `encodeStream` from explicit `*_start`/`*_delta`/`*_end` events; this plugin's
# readers produce canonical LiteLLM chunks with no start or end markers. So an item opens
# when the kind of content changes and is finished — `*_end` and `closeOpen` together —
# when the next one opens or the turn ends. omp leaves a message item open until the turn
# ends because its text parts can resume the same message (`textSignature`); the canonical
# chunks carry no such identity, so each item closes before the next opens, which is also
# the order the real API emits them in.


# omp: providers/openai-responses-server.ts :: makeRespId
def make_resp_id() -> str:
    return f"resp_{uuid.uuid4().hex}"


# omp: providers/openai-responses-server.ts :: makeMsgId
def make_msg_id() -> str:
    return f"msg_{uuid.uuid4().hex}"


# omp: providers/openai-responses-server.ts :: makeReasoningId
def make_reasoning_id() -> str:
    return f"rs_{uuid.uuid4().hex}"


# omp: providers/openai-responses-server.ts :: makeFuncCallId
def make_func_call_id() -> str:
    return f"fc_{uuid.uuid4().hex}"


# omp: providers/openai-responses-server.ts :: wireCallId
def wire_call_id(call_id: str) -> str:
    """Only the ``call_id`` half of the composite ``call_id|item_id`` goes on the wire.

    The Codex reader mints the composite so a follow-up chat turn can be paired, but
    third-party clients validate `call_id` against ``^[a-zA-Z0-9_-]+$`` or echo it to other
    backends, and ``|`` fails both.
    """
    return call_id.split("|", 1)[0]


# omp: providers/openai-responses-server.ts :: responseStatusForStopReason
def response_status(finish_reason: object) -> str:
    """The Responses status for a canonical finish reason.

    `length` is omp's `length`. omp's `error` and `aborted` have no canonical reason: a
    failed turn reaches the encoder as an exception, and `failed` answers it.
    """
    return "incomplete" if finish_reason == "length" else "completed"


# omp: providers/openai-responses-server.ts :: incompleteDetailsForStatus
def incomplete_details(status: str) -> dict[str, str] | None:
    return {"reason": "max_output_tokens"} if status == "incomplete" else None


# omp: providers/openai-responses-server.ts :: buildUsage
def build_usage(usage: Any) -> dict[str, Any]:
    """Canonical usage as Responses counts.

    `prompt_tokens` already includes the cached tokens, which is omp's
    ``input + cacheRead + cacheWrite``; the cached ones are then a detail *of* the input,
    and that detail is what prices them at the cache rate in the spend row.
    """
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    output = int(getattr(usage, "completion_tokens", 0) or 0)
    cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0)
    reasoning = getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", 0)
    return {
        "input_tokens": prompt,
        "input_tokens_details": {"cached_tokens": int(cached or 0)},
        "output_tokens": output,
        "output_tokens_details": {"reasoning_tokens": int(reasoning or 0)},
        "total_tokens": prompt + output,
    }


# omp: providers/openai-responses-server.ts :: buildResponseEnvelope
def build_response_envelope(
    *,
    response_id: str,
    created_at: int,
    model: str,
    status: str,
    output: list[dict[str, Any]],
    usage: dict[str, Any] | None,
    error_message: str | None = None,
) -> dict[str, Any]:
    envelope: dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "model": model,
        "output": output,
        "usage": usage,
        "incomplete_details": incomplete_details(status),
    }
    if status == "failed":
        envelope["error"] = {"message": error_message or "response failed"}
    return envelope


# omp: providers/openai-responses-server.ts :: OpenMessage, OpenReasoning, OpenFunctionCall
@dataclass(slots=True)
class _OpenItem:
    """The item being streamed: what it is, where it sits, and what it has said so far."""

    kind: str
    item_id: str
    output_index: int
    text: str = ""
    call_id: str = ""
    name: str = ""


# omp: providers/openai-responses-server.ts :: encodeStream
class ResponsesStreamEncoder:
    """The Responses event sequence for one turn, each item as it arrives.

    Every event carries `sequence_number`, strictly increasing across the whole stream:
    the SDK declares it on every event type, and a client that reorders on it needs it on
    the items appended after the text as much as on the text.

    The terminal response lists the items exactly as they were streamed — same ids, same
    order — which is omp's `finishedItems`. omp rebuilds the terminal list from its final
    message instead, minting fresh ids for items that carried none; there is no final
    message here other than the stream itself, so the ids a client saw stay the ids it
    gets.
    """

    __slots__ = (
        "_created_at",
        "_model",
        "_open",
        "_output_index",
        "_sequence",
        "_tool_items",
        "finished_items",
        "response_id",
    )

    def __init__(self, model: str) -> None:
        self.response_id = make_resp_id()
        self._model = model
        self._created_at = int(time.time())
        self._sequence = 0
        self._output_index = 0
        self._open: _OpenItem | None = None
        #: Canonical tool-call index -> the open call it streams into.
        self._tool_items: dict[int, _OpenItem] = {}
        self.finished_items: list[dict[str, Any]] = []

    def start(self) -> list[dict[str, Any]]:
        """`response.created` and `response.in_progress`: the envelope, with no output.

        Both, as the real API sends them: some clients gate on `in_progress` before
        reading items.
        """
        return [
            self._emit("response.created", {"response": self._snapshot("in_progress")}),
            self._emit("response.in_progress", {"response": self._snapshot("in_progress")}),
        ]

    def text(self, text: str) -> list[dict[str, Any]]:
        out = [] if self._is_open("message") else [*self.close_open(), *self._open_message()]
        cur = self._current()
        cur.text += text
        out.append(
            self._emit(
                "response.output_text.delta",
                {**self._text_address(cur), "delta": text, "logprobs": []},
            )
        )
        return out

    def thinking(self, text: str) -> list[dict[str, Any]]:
        out = [] if self._is_open("reasoning") else [*self.close_open(), *self._open_reasoning()]
        cur = self._current()
        cur.text += text
        out.append(
            self._emit(
                "response.reasoning_summary_text.delta",
                {**self._summary_address(cur), "delta": text},
            )
        )
        return out

    def tool_call(self, index: int, call_id: str, name: str) -> list[dict[str, Any]]:
        """Opens a `function_call` item; its arguments arrive as `tool_arguments` deltas."""
        out = self.close_open()
        cur = _OpenItem(
            kind="function_call",
            item_id=make_func_call_id(),
            output_index=self._allocate_output_index(),
            call_id=wire_call_id(call_id),
            name=name,
        )
        item = {
            "type": "function_call",
            "id": cur.item_id,
            "call_id": cur.call_id,
            "name": name,
            "arguments": "",
            "status": "in_progress",
        }
        out.append(
            self._emit(
                "response.output_item.added", {"output_index": cur.output_index, "item": item}
            )
        )
        self._open = cur
        self._tool_items[index] = cur
        return out

    def tool_arguments(self, index: int, partial_json: str) -> list[dict[str, Any]]:
        # Both readers send a call's arguments right after its opening chunk, so the call
        # is open; a fragment for one that is not is a reader defect and raises here rather
        # than landing on whatever item happens to be open.
        cur = self._tool_items[index]
        cur.text += partial_json
        return [
            self._emit(
                "response.function_call_arguments.delta",
                {"item_id": cur.item_id, "output_index": cur.output_index, "delta": partial_json},
            )
        ]

    def done(self, finish_reason: object, usage: Any) -> list[dict[str, Any]]:
        """Closes what is open, then `response.completed` or `response.incomplete`."""
        out = self.close_open()
        status = response_status(finish_reason)
        name = "response.incomplete" if status == "incomplete" else "response.completed"
        response = build_response_envelope(
            response_id=self.response_id,
            created_at=self._created_at,
            model=self._model,
            status=status,
            output=list(self.finished_items),
            usage=build_usage(usage),
        )
        out.append(self._emit(name, {"response": response}))
        return out

    def failed(self, message: str) -> list[dict[str, Any]]:
        """omp's failure path: close what is open, then `response.failed` with the reason.

        What streamed before the failure is kept in the output, as omp keeps its
        `finishedItems`: the client has already rendered it.
        """
        out = self.close_open()
        response = build_response_envelope(
            response_id=self.response_id,
            created_at=self._created_at,
            model=self._model,
            status="failed",
            output=list(self.finished_items),
            usage=None,
            error_message=message or "stream failed",
        )
        out.append(self._emit("response.failed", {"response": response}))
        return out

    # omp: providers/openai-responses-server.ts :: closeOpen, closeFunctionCall
    def close_open(self) -> list[dict[str, Any]]:
        """The open item's `*_end` events and its `output_item.done`, or nothing."""
        cur = self._open
        if cur is None:
            return []
        self._open = None
        if cur.kind == "message":
            part = {"type": "output_text", "text": cur.text, "annotations": []}
            item: dict[str, Any] = {
                "type": "message",
                "id": cur.item_id,
                "status": "completed",
                "role": "assistant",
                "content": [part],
            }
            out = [
                self._emit(
                    "response.output_text.done",
                    {**self._text_address(cur), "text": cur.text, "logprobs": []},
                ),
                # `logprobs` is not in omp's part. LiteLLM's `ContentPartDoneEvent` requires
                # it, and the real API sends it; without it the event fails validation.
                self._emit(
                    "response.content_part.done",
                    {**self._text_address(cur), "part": {**part, "logprobs": []}},
                ),
            ]
        elif cur.kind == "reasoning":
            summary = {"type": "summary_text", "text": cur.text}
            item = {"type": "reasoning", "id": cur.item_id, "summary": [summary]}
            out = [
                self._emit(
                    "response.reasoning_summary_text.done",
                    {**self._summary_address(cur), "text": cur.text},
                ),
                self._emit(
                    "response.reasoning_summary_part.done",
                    {**self._summary_address(cur), "part": summary},
                ),
            ]
        else:
            # omp falls back to the finished call's arguments when nothing streamed; here
            # the canonical chunks are the only source, so an argument-less call is `{}`.
            arguments = cur.text or "{}"
            item = {
                "type": "function_call",
                "id": cur.item_id,
                "call_id": cur.call_id,
                "name": cur.name,
                "arguments": arguments,
                "status": "completed",
            }
            out = [
                self._emit(
                    "response.function_call_arguments.done",
                    {
                        "item_id": cur.item_id,
                        "output_index": cur.output_index,
                        "arguments": arguments,
                        "name": cur.name,
                    },
                )
            ]
            self._tool_items = {
                index: call for index, call in self._tool_items.items() if call is not cur
            }
        out.append(
            self._emit(
                "response.output_item.done", {"output_index": cur.output_index, "item": item}
            )
        )
        self.finished_items.append(item)
        return out

    # omp: providers/openai-responses-server.ts :: openMessage
    def _open_message(self) -> list[dict[str, Any]]:
        cur = _OpenItem(
            kind="message", item_id=make_msg_id(), output_index=self._allocate_output_index()
        )
        self._open = cur
        item = {
            "type": "message",
            "id": cur.item_id,
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        return [
            self._emit(
                "response.output_item.added", {"output_index": cur.output_index, "item": item}
            ),
            self._emit(
                "response.content_part.added",
                {
                    **self._text_address(cur),
                    "part": {"type": "output_text", "text": "", "annotations": []},
                },
            ),
        ]

    # omp: providers/openai-responses-server.ts :: openReasoning
    def _open_reasoning(self) -> list[dict[str, Any]]:
        cur = _OpenItem(
            kind="reasoning",
            item_id=make_reasoning_id(),
            output_index=self._allocate_output_index(),
        )
        self._open = cur
        item = {"type": "reasoning", "id": cur.item_id, "summary": []}
        return [
            self._emit(
                "response.output_item.added", {"output_index": cur.output_index, "item": item}
            ),
            self._emit(
                "response.reasoning_summary_part.added",
                {**self._summary_address(cur), "part": {"type": "summary_text", "text": ""}},
            ),
        ]

    def _is_open(self, kind: str) -> bool:
        return self._open is not None and self._open.kind == kind

    def _current(self) -> _OpenItem:
        assert self._open is not None
        return self._open

    def _allocate_output_index(self) -> int:
        index = self._output_index
        self._output_index += 1
        return index

    def _snapshot(self, status: str) -> dict[str, Any]:
        return build_response_envelope(
            response_id=self.response_id,
            created_at=self._created_at,
            model=self._model,
            status=status,
            output=[],
            usage=None,
        )

    def _emit(self, name: str, data: dict[str, Any]) -> dict[str, Any]:
        event = {"type": name, "sequence_number": self._sequence, **data}
        self._sequence += 1
        return event

    @staticmethod
    def _text_address(cur: _OpenItem) -> dict[str, Any]:
        return {"item_id": cur.item_id, "output_index": cur.output_index, "content_index": 0}

    @staticmethod
    def _summary_address(cur: _OpenItem) -> dict[str, Any]:
        return {"item_id": cur.item_id, "output_index": cur.output_index, "summary_index": 0}
