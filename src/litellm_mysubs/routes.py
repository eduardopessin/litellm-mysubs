"""The three routes a client can speak, and the turn builders behind them.

Extracted from `plugin.py`. Each `dispatch*` answers one wire dialect and returns `None`
when the model is not ours — the only way to say "not mine" without fabricating a
response, since the caller then delegates to the original.

    /v1/chat/completions  ->  dispatch
    /v1/responses         ->  dispatch_responses
    /v1/messages          ->  dispatch_messages

A client should not have to know which subscription is behind a model name, so all three
serve all three providers. What differs is only the envelope: the turns converge on one
canonical list, and each provider keeps its own wire underneath, which is why a fix to one
path cannot leave another behind.

Nothing here patches LiteLLM or reads process state; `plugin.py` owns that.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any, Protocol, cast

from litellm.types.utils import ModelResponse, ModelResponseStream

from .credentials.store import ProviderId
from .observability import (
    _as_litellm_error,
    _logged,
    _logged_messages,
    _logged_stream,
    _stamp_logging_identity,
    _translate_errors,
)
from .observability import _wrap_stream as _wrap_stream
from .specs import _antigravity_spec, _codex_spec, _transport
from .transport.client import UpstreamError
from .turns import (
    _AntigravityReader,
    _CodexReader,
    _finish_chunk,
    _model_response,
    _Turn,
    _usage_chunk,
)
from .wire import anthropic, codex, messages
from .wire.usage import Usage, codex_finish_reason, codex_usage, google_finish_reason, google_usage

if TYPE_CHECKING:
    from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse


def is_gemini_model(model: str) -> bool:
    """Models served by the Google Antigravity subscription.

    No OMP anchor on purpose: there the distinction is made by ``model.provider`` in a
    typed catalog (`google-gemini-cli.ts`), not by a predicate over the name. Here the name
    is all that arrives from the client.
    """
    lowered = str(model).lower()
    return "gemini" in lowered or "antigravity" in lowered


class _Reader(Protocol):
    def feed(self, event: dict[str, Any]) -> list[ModelResponseStream]: ...
    def close(self) -> list[ModelResponseStream]: ...


async def _drive(events: AsyncIterator[dict[str, Any]], reader: _Reader) -> None:
    """Non-streaming path: consumes everything through the same reader, discarding chunks.

    A single interpretation routine, shared with the streaming path — having two is what
    made the original's synchronous and asynchronous versions diverge.
    """
    async for event in events:
        reader.feed(event)
    reader.close()


async def _pump(
    events: AsyncIterator[dict[str, Any]], reader: _Reader
) -> AsyncIterator[ModelResponseStream]:
    """Streaming path: emits each event's chunks as they arrive."""
    async for event in events:
        for chunk in reader.feed(event):
            yield chunk
    for chunk in reader.close():
        yield chunk


async def _codex_turn(model: str, messages: list[Any], extra: dict[str, Any]) -> ModelResponse:
    spec = await _codex_spec(model, messages, extra)
    turn = _Turn()
    await _drive(_transport().stream(spec), _CodexReader(turn))
    return _model_response(
        model,
        turn,
        finish_reason=codex_finish_reason(turn.finish_raw, bool(turn.tool_calls)),
        usage=codex_usage(turn.usage_meta),
    )


def _responses_input(kwargs: dict[str, Any]) -> list[Any]:
    """``/v1/responses`` carries ``input``, not ``messages``.

    A plain string is the documented shorthand for a single user turn, and the item form is
    already what ``messages_to_input`` produces on the way out, so both are handed to the
    existing body builder unchanged rather than being converted twice.
    """
    value = kwargs.get("input")
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if isinstance(value, list):
        return list(value)
    return []


async def _codex_responses_turn(
    model: str, kwargs: dict[str, Any]
) -> ResponsesAPIResponse:
    """Serves ``/v1/responses`` from the subscription's own Responses payload.

    Codex **is** a Responses API endpoint, so the terminal event's `response` object is the
    shape this route has to return and is kept as the base — status, `previous_response_id`
    and the rest are the upstream's.

    What the terminal event does **not** carry is `output`. Measured against the real
    endpoint: the items arrive in the `response.output_item.done` events during the stream,
    and `response.completed` closes the turn without repeating them, so a straight
    passthrough returned `output: []` with a non-zero `output_tokens` — a completed turn
    whose text had vanished. They are rebuilt here from what the reader accumulated, which
    is the same source the chat path uses, so the two routes cannot disagree.

    Usage is rebuilt for a second reason: an absent or partial `usage` fails
    `ResponseAPIUsage` validation, which requires all three counters. Going through
    `codex_usage` is also what makes this route's spend log match the chat route's.
    """
    from litellm.types.llms.openai import ResponsesAPIResponse

    spec = await _codex_spec(model, _responses_input(kwargs), kwargs)
    turn = _Turn()
    await _drive(_transport().stream(spec), _CodexReader(turn))

    payload = dict(turn.response_payload)
    # `model` echoes the wire name; the caller asked for the public one and the spend log
    # reads this field.
    payload["model"] = model
    payload.setdefault("id", f"resp_{uuid.uuid4().hex[:24]}")
    payload.setdefault("created_at", int(time.time()))
    payload.setdefault("object", "response")
    if not payload.get("output"):
        payload["output"] = _responses_output(turn)
    payload["usage"] = _responses_usage(codex_usage(turn.usage_meta))
    return ResponsesAPIResponse(**payload)


def _responses_output(turn: _Turn) -> list[dict[str, Any]]:
    """``output`` items for a turn the terminal event did not carry them for.

    Reasoning comes first, then the message, then the tool calls — the order the Responses
    API documents and the order a client replays them in. Tool calls carry the composite
    id the chat path also emits, so a follow-up turn matches its output to the right call.
    """
    items: list[dict[str, Any]] = []
    reasoning = "".join(turn.reasoning)
    if reasoning:
        items.append(
            {
                "type": "reasoning",
                "id": f"rs_{uuid.uuid4().hex[:24]}",
                "summary": [{"type": "summary_text", "text": reasoning}],
            }
        )
    text = "".join(turn.text)
    if text:
        items.append(
            {
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        )
    for call in turn.tool_calls:
        function = call.get("function") or {}
        items.append(
            {
                "type": "function_call",
                "id": f"fc_{uuid.uuid4().hex[:24]}",
                "call_id": str(call.get("id") or ""),
                "name": str(function.get("name") or ""),
                "arguments": str(function.get("arguments") or ""),
                "status": "completed",
            }
        )
    return items


def _responses_usage(usage: Usage) -> ResponseAPIUsage:
    """``ResponseAPIUsage`` counters, which are named differently from the chat ones.

    All three are required by the model, so they are always supplied — a turn whose
    upstream reported nothing bills zero rather than failing to construct.
    """
    from litellm.types.llms.openai import ResponseAPIUsage

    return ResponseAPIUsage(
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
    )


async def _codex_responses_stream(
    model: str, kwargs: dict[str, Any]
) -> AsyncIterator[Any]:
    """``/v1/responses`` with ``stream: true``, as Responses API events.

    The Responses SSE protocol is not the chat chunk sequence: a client reads items, not
    deltas on a choice. The order below was **captured from this proxy's own native path**
    (`qwen-agent-coder`, `/v1/responses`, `stream: true`) rather than assumed, because the
    previous attempt at this route shipped a payload shape that the real endpoint never
    sends:

        response.created -> response.in_progress
        -> output_item.added -> content_part.added
        -> output_text.delta* -> output_text.done -> content_part.done
        -> output_item.done
        -> response.completed

    `item_id`/`output_index`/`content_index` tie the parts to their item — a client that
    tracks them needs all three.

    Events are emitted as LiteLLM's **typed** event models, not dicts. The proxy serialises
    a stream chunk with `_serialize_streaming_chunk`, which calls `.model_dump_json()`; a
    plain dict falls through to `str()` and reaches the client as a Python repr with single
    quotes, which no JSON parser accepts. Measured against a real proxy before this was
    written the second time.

    Text is emitted as it arrives; reasoning and tool calls are emitted as completed items
    once the turn closes. Reasoning deltas are deliberately not streamed: `_CodexReader`
    accumulates them for the chat path, and replaying them as `reasoning_text.delta` would
    mean a second interpretation of the same events — the divergence the module docstring
    exists to prevent.
    """
    spec = await _codex_spec(model, _responses_input(kwargs), kwargs)
    turn = _Turn()
    async for out in _responses_events(
        model, spec, turn, _CodexReader(turn), codex_usage, turn.response_payload
    ):
        yield out


async def _antigravity_responses_stream(
    model: str, kwargs: dict[str, Any]
) -> AsyncIterator[Any]:
    """``/v1/responses`` with ``stream: true`` for Antigravity, incrementally.

    The first version served this cell by awaiting the whole chat turn and replaying it as
    two events. It answered correctly and priced correctly, and it was not a stream.
    Measured on the live gateway against Codex on the same route and prompt::

        codex   events=7117  deltas=7107  ttft=    30ms
        gemini  events=   2  deltas=   0  ttft= 20911ms

    A client reading `stream.text_deltas` got nothing until the turn had finished. Same
    route, same client, two different contracts — which is the one thing this plugin
    exists to prevent.

    Nothing about the incremental form is Codex-specific: `_ResponsesStreamState` owns
    every id and index, and `_chunk_text` reads the canonical chunk both readers emit. So
    this is the same driver with the other provider's spec and reader, not a second
    implementation.
    """
    messages = _responses_input(kwargs)
    spec = await _antigravity_spec(model, messages, kwargs)
    turn = _Turn()
    reader = _AntigravityReader(turn, wire_model=str(spec.body.get("model") or model))
    async for out in _responses_events(model, spec, turn, reader, google_usage, {}):
        yield out


async def _responses_events(
    model: str,
    spec: Any,
    turn: _Turn,
    reader: Any,
    usage_of: Callable[[dict[str, Any]], Any],
    base_payload: dict[str, Any],
) -> AsyncIterator[Any]:
    """Drives one upstream stream and emits the Responses event sequence for it.

    Provider-specific parts are the three arguments: the request spec, the reader that
    turns wire events into canonical chunks, and the usage mapper. Everything else — the
    event order, the item ids, the indices and the strictly increasing sequence numbers —
    belongs to `_ResponsesStreamState` and is identical for both subscriptions, which is
    what keeps the two cells of this route telling a client the same story.

    `base_payload` is the upstream's own response object where one exists (Codex answers
    in Responses shape) and empty where it does not (Cloud Code answers in
    `candidates`/`parts`); the envelope is completed from the stream state either way.
    """
    state = _ResponsesStreamState(model=model)
    for out in state.created():
        yield out

    async for event in _transport().stream(spec):
        for chunk in reader.feed(event):
            text = _chunk_text(chunk)
            if not text:
                continue
            for out in state.open_message():
                yield out
            yield state.delta(text)
    reader.close()

    for out in state.close_message():
        yield out

    # Reasoning and tool calls are known only once the turn has closed, so they are
    # announced and completed back to back rather than streamed.
    for item in _responses_output(turn):
        if item.get("type") == "message":
            continue  # already streamed above
        for out in state.whole_item(item):
            yield out

    payload = dict(base_payload)
    payload["model"] = model
    payload["id"] = state.response_id
    payload.setdefault("created_at", state.created_at)
    payload.setdefault("object", "response")
    payload["status"] = payload.get("status") or "completed"
    payload["output"] = state.items
    payload["usage"] = _responses_usage(usage_of(turn.usage_meta)).model_dump()
    yield state.completed(payload)


def _chunk_text(chunk: ModelResponseStream) -> str:
    """Visible text on a chunk, ignoring reasoning and tool deltas."""
    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return ""
    delta = getattr(choices[0], "delta", None)
    return str(getattr(delta, "content", "") or "") if delta is not None else ""



class _ResponsesStreamState:
    """Sequence numbers, item ids and indices for one Responses stream.

    Kept in an object because every event carries `sequence_number`, and a client that
    reorders on it needs the numbering to be strictly increasing across the whole stream —
    including the items appended after the text has finished.
    """

    __slots__ = (
        "created_at",
        "items",
        "message_id",
        "message_open",
        "model",
        "output_index",
        "response_id",
        "text",
    )

    def __init__(self, model: str) -> None:
        self.model = model
        self.response_id = f"resp_{uuid.uuid4().hex[:24]}"
        self.created_at = int(time.time())
        self.output_index = 0
        self.message_id = f"msg_{uuid.uuid4().hex[:24]}"
        self.message_open = False
        self.text: list[str] = []
        self.items: list[dict[str, Any]] = []

    def _events(self) -> Any:
        from litellm.types.llms.openai import ResponsesAPIStreamEvents

        return ResponsesAPIStreamEvents

    def envelope(self, status: str) -> dict[str, Any]:
        return {
            "id": self.response_id,
            "created_at": self.created_at,
            "model": self.model,
            "object": "response",
            "status": status,
            "output": list(self.items),
        }

    def created(self) -> list[Any]:
        from litellm.types.llms.openai import (
            ResponseCreatedEvent,
            ResponseInProgressEvent,
            ResponsesAPIResponse,
        )

        kinds = self._events()
        envelope = ResponsesAPIResponse.model_validate(self.envelope("in_progress"))
        return [
            ResponseCreatedEvent(type=kinds.RESPONSE_CREATED, response=envelope),
            ResponseInProgressEvent(type=kinds.RESPONSE_IN_PROGRESS, response=envelope),
        ]

    def delta(self, text: str) -> Any:
        from litellm.types.llms.openai import OutputTextDeltaEvent

        self.text.append(text)
        return OutputTextDeltaEvent(
            type=self._events().OUTPUT_TEXT_DELTA,
            item_id=self.message_id,
            output_index=self.output_index,
            content_index=0,
            delta=text,
        )

    def open_message(self) -> list[Any]:
        """`output_item.added` + `content_part.added`, once, before the first delta."""
        if self.message_open:
            return []
        from litellm.types.llms.openai import ContentPartAddedEvent, OutputItemAddedEvent

        self.message_open = True
        kinds = self._events()
        return [
            OutputItemAddedEvent(
                type=kinds.OUTPUT_ITEM_ADDED,
                output_index=self.output_index,
                item=({
                    "id": self.message_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [],
                }),  # type: ignore[arg-type]  # pydantic coerces the dict; the annotation is narrower
            ),
            ContentPartAddedEvent(
                type=kinds.CONTENT_PART_ADDED,
                item_id=self.message_id,
                output_index=self.output_index,
                content_index=0,
                part=({"type": "output_text", "text": "", "annotations": [], "logprobs": []}),  # type: ignore[arg-type]  # pydantic coerces the dict; the annotation is narrower
            ),
        ]

    def close_message(self) -> list[Any]:
        """`output_text.done` + `content_part.done` + `output_item.done`."""
        if not self.message_open:
            return []
        from litellm.types.llms.openai import (
            ContentPartDoneEvent,
            OutputItemDoneEvent,
            OutputTextDoneEvent,
        )

        kinds = self._events()
        text = "".join(self.text)
        item: dict[str, Any] = {
            "id": self.message_id,
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }
        self.items.append(item)
        out = [
            OutputTextDoneEvent(
                type=kinds.OUTPUT_TEXT_DONE,
                item_id=self.message_id,
                output_index=self.output_index,
                content_index=0,
                text=text,
            ),
            ContentPartDoneEvent(
                type=kinds.CONTENT_PART_DONE,
                item_id=self.message_id,
                output_index=self.output_index,
                content_index=0,
                part=({"type": "output_text", "text": text, "annotations": [], "logprobs": []}),  # type: ignore[arg-type]  # pydantic coerces the dict; the annotation is narrower
            ),
            OutputItemDoneEvent(
                type=kinds.OUTPUT_ITEM_DONE, output_index=self.output_index, item=item  # type: ignore[arg-type]
            ),
        ]
        self.output_index += 1
        self.message_open = False
        return out

    def whole_item(self, item: dict[str, Any]) -> list[Any]:
        """An item known only at the end: announced and completed back to back."""
        from litellm.types.llms.openai import OutputItemAddedEvent, OutputItemDoneEvent

        kinds = self._events()
        self.items.append(item)
        out = [
            OutputItemAddedEvent(
                type=kinds.OUTPUT_ITEM_ADDED, output_index=self.output_index, item=item  # type: ignore[arg-type]
            ),
            OutputItemDoneEvent(
                type=kinds.OUTPUT_ITEM_DONE, output_index=self.output_index, item=item  # type: ignore[arg-type]
            ),
        ]
        self.output_index += 1
        return out

    def completed(self, payload: dict[str, Any]) -> Any:
        from litellm.types.llms.openai import ResponseCompletedEvent, ResponsesAPIResponse

        return ResponseCompletedEvent(
            type=self._events().RESPONSE_COMPLETED,
            response=ResponsesAPIResponse.model_validate(payload),
        )


async def _antigravity_turn(
    model: str, messages: list[Any], extra: dict[str, Any]
) -> ModelResponse:
    spec = await _antigravity_spec(model, messages, extra)
    turn = _Turn()
    reader = _AntigravityReader(turn, wire_model=str(spec.body.get("model") or model))
    await _drive(_transport().stream(spec), reader)
    return _model_response(
        model,
        turn,
        finish_reason=google_finish_reason(turn.finish_raw, bool(turn.tool_calls)),
        usage=google_usage(turn.usage_meta),
    )


async def _codex_stream(
    model: str, messages: list[Any], extra: dict[str, Any]
) -> AsyncIterator[ModelResponseStream]:
    spec = await _codex_spec(model, messages, extra)
    turn = _Turn()
    async for chunk in _pump(_transport().stream(spec), _CodexReader(turn)):
        yield chunk
    yield _finish_chunk(codex_finish_reason(turn.finish_raw, bool(turn.tool_calls)))
    yield _usage_chunk(codex_usage(turn.usage_meta))


async def _antigravity_stream(
    model: str, messages: list[Any], extra: dict[str, Any]
) -> AsyncIterator[ModelResponseStream]:
    spec = await _antigravity_spec(model, messages, extra)
    turn = _Turn()
    reader = _AntigravityReader(turn, wire_model=str(spec.body.get("model") or model))
    async for chunk in _pump(_transport().stream(spec), reader):
        yield chunk
    yield _finish_chunk(google_finish_reason(turn.finish_raw, bool(turn.tool_calls)))
    yield _usage_chunk(google_usage(turn.usage_meta))


async def _antigravity_responses(model: str, kwargs: dict[str, Any]) -> Any:
    """Serves ``/v1/responses`` for Antigravity by translating its chat turn.

    Codex is natively a Responses endpoint, so `_codex_responses_turn` keeps the
    upstream's own object. Gemini is not: Cloud Code answers in `candidates`/`parts`, and
    there is no Responses payload to pass through. Rather than hand-build items — which is
    what made delegating look preferable — the turn is produced by the existing chat path
    and handed to `LiteLLMCompletionResponsesConfig`, the same transform the proxy applies
    to every other chat-backed model on this route.

    Going through `_logged` is the point: it is what prices the turn under the wire
    identity. The delegated version could not, because the native path costs from the
    response object, which carries the public name and no rate.

    Streaming replays the finished turn, as `/v1/messages` does for the same reason: the
    event sequence is reconstructed from a complete answer rather than interleaved with a
    second reading of the upstream stream.
    """
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    turn_input = _responses_input(kwargs)
    extra = dict(kwargs)
    extra.pop("stream", None)
    response = await _logged(_antigravity_turn(model, turn_input, extra), kwargs)
    transform = (
        LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response
    )
    answer = transform(
        request_input=kwargs.get("input") or "",
        responses_api_request=cast("Any", kwargs),
        chat_completion_response=response,
    )
    return answer


async def dispatch_responses(*, provider: ProviderId | None = None, **kwargs: Any) -> Any:
    """``/v1/responses`` counterpart of `dispatch`; ``None`` means "not mine".

    Anthropic has no branch for the same reason it has none in `dispatch` — LiteLLM's
    native path serves it with the token `_delegate_kwargs` injects, and that path already
    answers this route correctly.

    Antigravity used to be delegated too, on the grounds that returning Gemini through a
    Responses object would mean inventing item structure the upstream never sent. The
    objection was right; the conclusion was wrong, because delegating does not stop the
    turn from spending the subscription. Measured on the live gateway, same model and the
    same 122+116 tokens across all six routes::

        acompletion         gemini/gemini-3.6-flash       prov=gemini   0.0005265
        anthropic_messages  gemini/gemini-3.6-flash       prov=gemini   0.0005265
        aresponses          gemini/gemini-3.6-flash-low   prov=         0.0

    Stamping the identity before handing off was not enough: the provider stuck, the
    model name did not. The native path prices from the **response**, which carries the
    public name the client asked for and ``cost: None`` — the same mechanism as
    BerriAI/litellm#42161, on this route.

    So the turn is served here. Non-streaming is shaped with LiteLLM's own
    `LiteLLMCompletionResponsesConfig`, the transform the proxy applies to every other
    chat-backed model on this route; streaming goes through `_antigravity_responses_stream`
    and emits the same event sequence Codex does, because a client on this route must not
    be able to tell which subscription answered.

    The `try` is not decoration: `dispatch` has had it since the chat route existed, and
    without it a quota refusal from the same transport reaches the client as HTTP 500
    instead of 429 — a client cannot back off on a 500. Delegating used to hide that,
    because LiteLLM's native path normalised the error on the way out.
    """
    model = str(kwargs.get("model") or "")
    try:
        if provider == "google-antigravity" or (provider is None and is_gemini_model(model)):
            _stamp_logging_identity(model, kwargs)
            if kwargs.get("stream"):
                return _logged_stream(_antigravity_responses_stream(model, kwargs), kwargs)
            return await _antigravity_responses(model, kwargs)
        if provider == "openai-codex" or (provider is None and codex.is_codex_model(model)):
            _stamp_logging_identity(model, kwargs)
            if kwargs.get("stream"):
                return _logged_stream(_codex_responses_stream(model, kwargs), kwargs)
            return await _logged(_codex_responses_turn(model, kwargs), kwargs)
    except UpstreamError as error:
        raise _as_litellm_error(error, model, kwargs) from error
    return None


async def dispatch_messages(*, provider: ProviderId | None = None, **kwargs: Any) -> Any:
    """``/v1/messages`` counterpart of `dispatch`; ``None`` means "not mine".

    All three subscriptions are served, because a client that speaks Messages should not
    have to know which one is behind a model name. Only the **envelope** is translated:
    the turns are converted to the canonical list `dispatch` already consumes, and each
    provider keeps its own wire — so Codex still goes out as Responses and Antigravity as
    Cloud Code, carrying every fix those paths have.

    Claude Max is the degenerate case: Messages *is* its wire, and LiteLLM's native path
    answers it correctly once `_delegate_kwargs` has injected the OAuth token. Returning
    `None` routes it there, exactly as `dispatch` does for chat.

    Streaming is incremental, as on the other two routes. The replay this used to do —
    produce the turn whole, then emit it as events — was defended on the grounds that a
    translated stream would have to invent block indices mid-flight. The indices are ours
    either way: the upstreams do not send them, so numbering blocks as they open is no
    more invented than numbering them at the end, and it is what lets text leave as it
    arrives. Measured on the live gateway before this changed: Codex answered with 9
    events and 30.7 s to first byte, Antigravity with 6 events and 6.9 s, against the 66
    events and 0.78 s a natively-served Claude turn delivered on the same route.
    """
    model = str(kwargs.get("model") or "")
    if provider == "anthropic" or (provider is None and anthropic.is_anthropic_model(model)):
        return None

    # Stamped before serving, as on the other two routes: the spend log reads the identity
    # off the logging object, and without it the row lands with the public name and no
    # provider — no icon, and no rate in the price map to bill it against.
    _stamp_logging_identity(model, kwargs)
    payload = dict(kwargs)
    converted = dict(kwargs)
    converted["messages"] = messages.to_chat_messages(payload)
    tools = messages.to_tools(payload)
    if tools:
        converted["tools"] = tools
    converted.pop("system", None)
    converted.pop("stream", None)

    is_gemini = provider == "google-antigravity" or (provider is None and is_gemini_model(model))
    is_codex = provider == "openai-codex" or (provider is None and codex.is_codex_model(model))
    if not is_gemini and not is_codex:
        return None

    try:
        if kwargs.get("stream"):
            chunks = (
                _antigravity_stream(model, converted["messages"], converted)
                if is_gemini
                else _codex_stream(model, converted["messages"], converted)
            )
            return _wrap_messages_stream(chunks, model, kwargs)
        turn = (
            _antigravity_turn(model, converted["messages"], converted)
            if is_gemini
            else _codex_turn(model, converted["messages"], converted)
        )
        response = await _logged(turn, kwargs)
    except UpstreamError as error:
        raise _as_litellm_error(error, model, kwargs) from error
    return messages.from_model_response(response, model)


async def _messages_events(
    chunks: AsyncIterator[ModelResponseStream], model: str, kwargs: dict[str, Any]
) -> AsyncIterator[dict[str, Any]]:
    """Relays canonical chunks as the Anthropic event sequence, text first.

    Both subscriptions reach this through their own reader, so the sequence a client sees
    does not depend on which one answered. Reasoning and tool calls are known only once
    the turn closes and are emitted as whole blocks after the text — the same division the
    Responses route makes, because neither upstream streams them in a form that can be
    replayed without a second interpretation of the same events.

    The chunks carry the finished turn's own finish reason and usage on their tail, which
    `_finish_chunk` and `_usage_chunk` appended; they are read here rather than rebuilt,
    so a streamed turn and a non-streamed one cannot disagree about either.
    """
    state = messages.MessagesStreamState()
    envelope = messages.envelope(model)
    text_parts: list[str] = []
    tail: dict[str, Any] = {}

    # Opened before the first chunk, not on it. `message_start` carries no content — it is
    # the envelope, and a client reads the message id and the model from it. Holding it
    # back until the model spoke made the route look slower than it is: measured against
    # `/v1/responses`, which emits `response.created` immediately, 4003 ms to first event
    # against 32 ms for the same prompt and ceiling.
    yield state.start(envelope)

    async for chunk in chunks:
        _absorb_tail(tail, chunk)
        text = _chunk_text(chunk)
        if not text:
            continue
        for out in state.open_text():
            yield out
        text_parts.append(text)
        yield state.delta(text)

    for out in state.close_text():
        yield out

    payload = {
        **envelope,
        "content": [{"type": "text", "text": "".join(text_parts)}],
        "stop_reason": messages.stop_reason(tail.get("finish_reason")),
        "usage": tail.get("usage") or {},
    }
    for out in state.finish(payload):
        yield out


def _absorb_tail(tail: dict[str, Any], chunk: ModelResponseStream) -> None:
    """Keeps the finish reason and usage the trailing chunks carry.

    `_finish_chunk` and `_usage_chunk` close every stream this plugin produces, so the
    turn's own numbers are already on the wire — reading them here is what keeps a
    streamed Messages turn agreeing with the non-streamed one about `stop_reason` and
    token counts.
    """
    usage = getattr(chunk, "usage", None)
    if usage is not None:
        tail["usage"] = {
            "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        }
    choices = getattr(chunk, "choices", None) or []
    if choices and getattr(choices[0], "finish_reason", None):
        tail["finish_reason"] = choices[0].finish_reason


def _wrap_messages_stream(
    chunks: AsyncIterator[ModelResponseStream], model: str, kwargs: dict[str, Any]
) -> AsyncIterator[dict[str, Any]]:
    """The Messages event stream, with the turn's spend row dispatched at its end.

    The chat route gets its row from `CustomStreamWrapper`; this route never reaches that
    wrapper, because it emits Anthropic events rather than chat chunks. `_logged_messages`
    closes the same accounting hole `_logged_stream` closes on the Responses route.
    """
    return _logged_messages(
        _messages_events(_translate_errors(chunks, model, kwargs), model, kwargs),
        model,
        kwargs,
    )


async def dispatch(*, provider: ProviderId | None = None, **kwargs: Any) -> Any:
    """Serves the request if the model belongs to one of our subscriptions; ``None`` if not.

    ``None`` is the only way to say "not mine" without fabricating a response: the caller
    delegates to the original. One of our models that the upstream refuses propagates the
    error — `RemapRequired` and `RedeemRequired` included, for the reasons at the top of
    the module.

    ``provider`` comes from the deployment's `model_info.mysubs_provider` when the request
    goes through the Router, and **wins** over the name heuristic. It is what keeps a
    `claude-sonnet-4-6` served by Antigravity from being treated as Anthropic, or a
    `gpt-oss-120b-medium` from the same account from ending up at Codex — measured: seven
    of the 32 models in the real catalog dispatched to the wrong place.
    """
    model = str(kwargs.get("model") or "")
    messages = kwargs.get("messages") or []
    streaming = bool(kwargs.get("stream"))

    try:
        if provider == "google-antigravity" or (provider is None and is_gemini_model(model)):
            # Stamped before serving, not after: the spend log reads the identity off the
            # logging object, and the streaming path hands that object to the wrapper.
            _stamp_logging_identity(model, kwargs)
            if streaming:
                return _wrap_stream(
                    _translate_errors(_antigravity_stream(model, messages, kwargs), model, kwargs),
                    model,
                    kwargs,
                )
            return await _logged(_antigravity_turn(model, messages, kwargs), kwargs)

        # After Gemini: `codex.is_codex_model` matches any name containing "gpt-", and a
        # hypothetical "gemini-gpt" belongs to Google.
        if provider == "openai-codex" or (provider is None and codex.is_codex_model(model)):
            _stamp_logging_identity(model, kwargs)
            if streaming:
                return _wrap_stream(
                    _translate_errors(_codex_stream(model, messages, kwargs), model, kwargs),
                    model,
                    kwargs,
                )
            return await _logged(_codex_turn(model, messages, kwargs), kwargs)
    except UpstreamError as error:
        raise _as_litellm_error(error, model, kwargs) from error

    # `anthropic` has no branch of its own: it is served by LiteLLM's native path with the
    # prompt and the token that `_delegate_kwargs` injects. Returning `None` is what routes
    # it there.
    return None


# -- logging, cost identity, error translation: see `observability.py` ---------

