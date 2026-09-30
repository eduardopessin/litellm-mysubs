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

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from typing import Any, Final, NoReturn, Protocol, TypeVar, cast

from litellm.types.utils import ModelResponse, ModelResponseStream

from .credentials.store import ProviderId
from .observability import (
    TRANSLATED_ERRORS,
    _as_litellm_error,
    _logged,
    _logged_messages,
    _logged_stream,
    _record_failed_usage,
    _stamp_logging_identity,
    _translate_errors,
)
from .observability import _wrap_stream as _wrap_stream
from .specs import _antigravity_spec, _codex_spec, _transport
from .transport.client import RequestSpec
from .turns import (
    MalformedCallError,
    ThinkingLoopError,
    UnansweredTurnError,
    WhitespaceLoopError,
    _AntigravityReader,
    _CodexReader,
    _finish_chunk,
    _litellm_usage,
    _model_response,
    _Turn,
    _usage_chunk,
    finish_reason,
)
from .wire import anthropic, codex, messages, responses
from .wire.antigravity import AntigravitySession
from .wire.usage import Usage

_T = TypeVar("_T")


def is_gemini_model(model: str) -> bool:
    """Models served by the Google Antigravity subscription.

    No OMP anchor on purpose: there the distinction is made by ``model.provider`` in a
    typed catalog (`google-gemini-cli.ts`), not by a predicate over the name. Here the name
    is all that arrives from the client.
    """
    lowered = str(model).lower()
    return "gemini" in lowered or "antigravity" in lowered


class _Reader(Protocol):
    #: The terminal event has been read; omp reads no further.
    done: bool

    def feed(self, event: dict[str, Any]) -> list[ModelResponseStream]: ...
    def close(self) -> list[ModelResponseStream]: ...


class _ItemEncoder(Protocol):
    """What `_chunk_events` drives: the Messages and the Responses stream encoders."""

    def thinking(self, text: str) -> list[dict[str, Any]]: ...
    def text(self, text: str) -> list[dict[str, Any]]: ...
    def tool_call(self, index: int, call_id: str, name: str) -> list[dict[str, Any]]: ...
    def tool_arguments(self, index: int, partial_json: str) -> list[dict[str, Any]]: ...


async def _release(events: AsyncIterator[dict[str, Any]]) -> None:
    """Closes the upstream stream the moment its reader stops consuming it.

    A bare ``async for`` leaves the generator suspended when the reader raises — the
    whitespace brake, the reasoning-loop guard, an in-band error — and with it the
    transport's HTTP response. Measured through the proxy: the upstream was still open a
    second later and after a ``gc.collect()``, so the subscription went on generating (and
    billing) the turn the brake had just refused. omp aborts the upstream request when its
    guard trips (``utils/thinking-loop.ts :: guardThinkingLoopStream``).
    """
    aclose = getattr(events, "aclose", None)
    if aclose is not None:
        await aclose()


async def _drive(events: AsyncIterator[dict[str, Any]], reader: _Reader) -> None:
    """Non-streaming path: consumes everything through the same reader, discarding chunks.

    A single interpretation routine, shared with the streaming path — having two is what
    made the original's synchronous and asynchronous versions diverge.
    """
    try:
        async for event in events:
            reader.feed(event)
            if reader.done:
                break
    finally:
        await _release(events)
    reader.close()


async def _pump(
    events: AsyncIterator[dict[str, Any]], reader: _Reader
) -> AsyncIterator[ModelResponseStream]:
    """Streaming path: emits each event's chunks as they arrive."""
    try:
        async for event in events:
            for chunk in reader.feed(event):
                yield chunk
            if reader.done:
                break
    finally:
        await _release(events)
    for chunk in reader.close():
        yield chunk


def _record_failure(turn: _Turn, error: Exception, model: str, extra: dict[str, Any]) -> None:
    """Bills a failed turn for what the upstream reported it cost before failing.

    omp's gateway records the usage of every finished turn, failed ones included
    (``recordGatewayUsage`` runs before the stop reason is looked at): a Gemini turn
    stopped for SAFETY, or cut before its finish, has consumed the subscription all the
    same. The exception is omp's loop guard, which ends the turn with an empty message and
    zero usage; so does this. The Codex whitespace brake cannot carry any: it trips before
    the terminal event, the only one Codex reports usage in.
    """
    if turn.usage is None or isinstance(error, ThinkingLoopError | WhitespaceLoopError):
        return
    _record_failed_usage(_litellm_usage(turn.usage), model, extra)


async def _served_turn(
    spec: RequestSpec, reader: _Reader, turn: _Turn, model: str, extra: dict[str, Any]
) -> ModelResponse:
    try:
        await _drive(_transport().stream(spec), reader)
    except Exception as error:
        _record_failure(turn, error, model, extra)
        raise
    return _model_response(model, turn)


async def _served_stream(
    spec: RequestSpec, reader: _Reader, turn: _Turn, model: str, extra: dict[str, Any]
) -> AsyncIterator[ModelResponseStream]:
    try:
        async for chunk in _pump(_transport().stream(spec), reader):
            yield chunk
    except Exception as error:
        _record_failure(turn, error, model, extra)
        raise
    yield _finish_chunk(finish_reason(turn))
    yield _usage_chunk(turn.usage or Usage())


def _antigravity_tool_names(spec: RequestSpec) -> frozenset[str]:
    """The declared tool names, as the model sees them: the planning filter matches them."""
    request = spec.body.get("request")
    tools = request.get("tools") if isinstance(request, dict) else None
    return frozenset(
        str(declaration.get("name"))
        for tool in tools or []
        if isinstance(tool, dict)
        for declaration in tool.get("functionDeclarations") or []
        if isinstance(declaration, dict) and declaration.get("name")
    )


def _codex_reader(spec: RequestSpec, model: str, turn: _Turn) -> _CodexReader:
    return _CodexReader(turn, wire_model=str(spec.body.get("model") or model))


def _antigravity_reader(
    spec: RequestSpec, model: str, turn: _Turn, session: AntigravitySession
) -> _AntigravityReader:
    return _AntigravityReader(
        turn,
        wire_model=str(spec.body.get("model") or model),
        tool_names=_antigravity_tool_names(spec),
        session=session,
    )


# omp: stream.ts :: THINKING_LOOP_MAX_ATTEMPTS
THINKING_LOOP_MAX_ATTEMPTS: Final = 3
# omp: stream.ts :: THINKING_LOOP_RETRY_BASE_DELAY_MS
THINKING_LOOP_RETRY_BASE_DELAY: Final = 0.5
# omp: stream.ts :: THINKING_LOOP_RETRY_MAX_DELAY_MS
THINKING_LOOP_RETRY_MAX_DELAY: Final = 8.0


# omp: stream.ts :: resolveWithThinkingLoopRetries
async def _resampling_loops(attempt: Callable[[], Awaitable[_T]]) -> _T:
    """A non-streamed turn the loop guard stopped is asked again, up to three attempts.

    omp's gateway answers a non-streamed request through ``completeSimple``, which
    re-samples a thinking-loop stall; nothing of the failed attempt has reached the client
    yet, so the retry is invisible to it. A streamed turn is not retried, there or here:
    its reasoning already went out.

    Diverges from omp by also re-sampling a Cloud Code turn that stopped unanswered or with
    a malformed call (`UnansweredTurnError`, `MalformedCallError`): the same reasoning holds
    — nothing reached the client — and the live measurement in `UnansweredTurnError` shows
    these are sampling failures a fresh attempt usually avoids.
    """
    for number in range(1, THINKING_LOOP_MAX_ATTEMPTS + 1):
        try:
            return await attempt()
        except (ThinkingLoopError, UnansweredTurnError, MalformedCallError):
            if number >= THINKING_LOOP_MAX_ATTEMPTS:
                raise
        await asyncio.sleep(
            min(THINKING_LOOP_RETRY_BASE_DELAY * 2 ** (number - 1), THINKING_LOOP_RETRY_MAX_DELAY)
        )
    raise AssertionError("unreachable")  # pragma: no cover


# omp: providers/openai-codex-responses.ts :: CODEX_WHITESPACE_LOOP_RETRY_LIMIT
WHITESPACE_LOOP_RETRY_LIMIT: Final = 2
# omp: providers/openai-codex-responses.ts :: CODEX_WHITESPACE_LOOP_RETRY_DELAY_MS
WHITESPACE_LOOP_RETRY_DELAY: Final = 0.25


async def _replaying_whitespace_loops(attempt: Callable[[], Awaitable[_T]]) -> _T:
    """A non-streamed Codex turn the whitespace brake stopped is asked again, twice at most.

    omp's ``CodexStreamProcessor`` (``#tryRecoverWhitespaceToolCallLoop``) replays the
    request when the looping call was the only thing produced besides reasoning: sampling
    usually breaks the loop on a fresh attempt. Nothing of a non-streamed turn has reached
    the client, so the replay is invisible to it. A streamed turn is not replayed: the
    call's opening chunk already went out, and a second one would read as another call.
    """
    for replay in range(WHITESPACE_LOOP_RETRY_LIMIT + 1):
        try:
            return await attempt()
        except WhitespaceLoopError as error:
            if not error.replayable or replay >= WHITESPACE_LOOP_RETRY_LIMIT:
                raise
        await asyncio.sleep(WHITESPACE_LOOP_RETRY_DELAY * (replay + 1))
    raise AssertionError("unreachable")  # pragma: no cover


async def _codex_turn(model: str, messages: list[Any], extra: dict[str, Any]) -> ModelResponse:
    async def attempt() -> ModelResponse:
        spec = await _codex_spec(model, messages, extra)
        turn = _Turn()
        return await _served_turn(spec, _codex_reader(spec, model, turn), turn, model, extra)

    return await _resampling_loops(lambda: _replaying_whitespace_loops(attempt))


async def _antigravity_turn(
    model: str, messages: list[Any], extra: dict[str, Any]
) -> ModelResponse:
    async def attempt() -> ModelResponse:
        spec, session = await _antigravity_spec(model, messages, extra)
        turn = _Turn()
        reader = _antigravity_reader(spec, model, turn, session)
        return await _served_turn(spec, reader, turn, model, extra)

    return await _resampling_loops(attempt)


async def _codex_stream(
    model: str, messages: list[Any], extra: dict[str, Any]
) -> AsyncIterator[ModelResponseStream]:
    spec = await _codex_spec(model, messages, extra)
    turn = _Turn()
    async for chunk in _served_stream(spec, _codex_reader(spec, model, turn), turn, model, extra):
        yield chunk


async def _antigravity_stream(
    model: str, messages: list[Any], extra: dict[str, Any]
) -> AsyncIterator[ModelResponseStream]:
    spec, session = await _antigravity_spec(model, messages, extra)
    turn = _Turn()
    reader = _antigravity_reader(spec, model, turn, session)
    async for chunk in _served_stream(spec, reader, turn, model, extra):
        yield chunk


def _raise_translated(error: Exception, model: str, kwargs: dict[str, Any]) -> NoReturn:
    """Re-raises an upstream failure as the exception LiteLLM has a class for."""
    translated = _as_litellm_error(error, model, kwargs)
    if translated is error:
        raise error
    raise translated from error


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

    So the turn is served here, for both subscriptions the same way, as `dispatch_messages`
    serves Messages: the request becomes the canonical turn (`responses.to_chat_messages`,
    omp's `parseRequest`), each provider answers on its own wire, and its canonical chunks
    are encoded as the Responses API by omp's encoder (`responses.ResponsesStreamEncoder`).
    Codex used to keep its upstream's own response object instead, since it answers in
    Responses shape; that object never carried `output`, echoed the wire model and the
    upstream's id, and made a Codex turn read differently from an Antigravity one on the
    same route — which is the one thing a client must not be able to tell.

    The `try` is not decoration: `dispatch` has had it since the chat route existed, and
    without it a quota refusal from the same transport reaches the client as HTTP 500
    instead of 429 — a client cannot back off on a 500. Delegating used to hide that,
    because LiteLLM's native path normalised the error on the way out.
    """
    model = str(kwargs.get("model") or "")
    is_gemini = provider == "google-antigravity" or (provider is None and is_gemini_model(model))
    is_codex = provider == "openai-codex" or (provider is None and codex.is_codex_model(model))
    if not is_gemini and not is_codex:
        return None

    _stamp_logging_identity(model, kwargs)
    converted = {
        key: value
        for key, value in kwargs.items()
        if key not in ("input", "instructions", "reasoning", "max_output_tokens", "stream")
    }
    converted.update(responses.to_options(kwargs))
    turn_messages = responses.to_chat_messages(kwargs)

    def chunks() -> AsyncIterator[ModelResponseStream]:
        return (
            _antigravity_stream(model, turn_messages, converted)
            if is_gemini
            else _codex_stream(model, turn_messages, converted)
        )

    try:
        if kwargs.get("stream"):
            return _logged_stream(
                _responses_events(_translate_errors(chunks(), model, kwargs), model), kwargs
            )
        return await _logged(
            _resampling_loops(
                lambda: _replaying_whitespace_loops(lambda: _responses_turn(chunks(), model))
            ),
            kwargs,
        )
    except TRANSLATED_ERRORS as error:
        _raise_translated(error, model, kwargs)


# omp: providers/openai-responses-server.ts :: sseEvent
def _litellm_event(event: dict[str, Any]) -> Any:
    """One encoded event as the typed object LiteLLM's Responses stream carries.

    omp frames each event itself, ``event:`` line included. Here the frames are LiteLLM's:
    the Router only takes a stream that is a `BaseResponsesAPIStreamingIterator`
    (`_logged_stream`), the success handler only assembles a turn whose terminal event is a
    `ResponseCompletedEvent`, and the proxy writes a typed chunk as a bare ``data:`` line
    (`_serialize_streaming_chunk` -> `model_dump_json`). A dict would reach the client as a
    Python repr. The missing ``event:`` line costs an OpenAI client nothing: the SDK
    dispatches on the payload's `type` (`openai/_streaming.py`), and the payload carries it.

    The class is the one LiteLLM picks for the same event arriving from OpenAI itself
    (`OpenAIResponsesAPIConfig.get_event_model_class`), so these events serialise exactly
    as a natively-served OpenAI stream does; its models allow extra fields, so
    `sequence_number` and `logprobs` survive.
    """
    from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig

    return OpenAIResponsesAPIConfig.get_event_model_class(event["type"])(**event)


async def _responses_events(
    chunks: AsyncIterator[ModelResponseStream], model: str
) -> AsyncIterator[Any]:
    """Relays canonical chunks as the Responses event sequence, each item as it arrives.

    `response.created` leaves before the first chunk, not on it: it is the envelope, and a
    client that waits on it would otherwise wait on the model.

    A failure after the stream opened goes out as omp's `response.failed` — with what had
    streamed so far closed into its output — and is then re-raised. The event is what a
    client reading the terminal event keys on; the exception is what still reaches
    LiteLLM's failure hook, whose own ``data: {"error": ...}`` frame follows it and makes
    the SDK raise. Without the event, that frame was the only sign the turn had failed.
    """
    encoder = responses.ResponsesStreamEncoder(model)
    tail: dict[str, Any] = {}
    try:
        for event in encoder.start():
            yield _litellm_event(event)
        async for chunk in chunks:
            _absorb_tail(tail, chunk)
            for event in _chunk_events(encoder, chunk):
                yield _litellm_event(event)
        for event in encoder.done(tail.get("finish_reason"), tail.get("usage")):
            yield _litellm_event(event)
    except Exception as error:
        for event in encoder.failed(str(getattr(error, "message", None) or error)):
            yield _litellm_event(event)
        raise


# omp: providers/openai-responses-server.ts :: encodeResponse
async def _responses_turn(chunks: AsyncIterator[ModelResponseStream], model: str) -> Any:
    """The non-streamed answer: the response the streamed one ends with.

    omp builds both from the same item builders; here both come from the same encoder
    over the same canonical chunks, so a streamed and a non-streamed turn cannot disagree
    about the items, their order, the status or the usage.
    """
    from litellm.types.llms.openai import ResponsesAPIResponse

    encoder = responses.ResponsesStreamEncoder(model)
    tail: dict[str, Any] = {}
    async for chunk in chunks:
        _absorb_tail(tail, chunk)
        _chunk_events(encoder, chunk)
    terminal = encoder.done(tail.get("finish_reason"), tail.get("usage"))[-1]
    return ResponsesAPIResponse.model_validate(terminal["response"])


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
    choice = messages.to_tool_choice(payload)
    if choice is None:
        converted.pop("tool_choice", None)
    else:
        converted["tool_choice"] = choice
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
    except TRANSLATED_ERRORS as error:
        _raise_translated(error, model, kwargs)
    return messages.encode_response(response, model)


async def _messages_events(
    chunks: AsyncIterator[ModelResponseStream], model: str, kwargs: dict[str, Any]
) -> AsyncIterator[dict[str, Any]]:
    """Relays canonical chunks as the Anthropic event sequence, each block as it arrives.

    Both subscriptions reach this through their own reader, so the sequence a client sees
    does not depend on which one answered. Reasoning, text and tool calls all arrive as
    chunk deltas while the turn runs, and each becomes its own block — relaying the text
    alone, as this did, left a streamed tool call with ``stop_reason: tool_use`` and no
    ``tool_use`` block, while the non-streamed call returned it.

    The chunks carry the finished turn's own finish reason and usage on their tail, which
    `_finish_chunk` and `_usage_chunk` appended; they are read here rather than rebuilt,
    so a streamed turn and a non-streamed one cannot disagree about either.
    """
    encoder = messages.MessagesStreamEncoder()
    tail: dict[str, Any] = {}

    # Opened before the first chunk, not on it. `message_start` carries no content — it is
    # the envelope, and a client reads the message id and the model from it. Holding it
    # back until the model spoke made the route look slower than it is: measured against
    # `/v1/responses`, which emits `response.created` immediately, 4003 ms to first event
    # against 32 ms for the same prompt and ceiling.
    yield encoder.start(model)

    async for chunk in chunks:
        _absorb_tail(tail, chunk)
        for out in _chunk_events(encoder, chunk):
            yield out

    for out in encoder.done(tail.get("finish_reason"), tail.get("usage")):
        yield out


def _chunk_events(encoder: _ItemEncoder, chunk: ModelResponseStream) -> list[dict[str, Any]]:
    """The events one canonical chunk produces, in the order the readers emit them.

    Shared by the Messages and Responses routes, so a chunk means the same thing to both.
    """
    # Every chunk the readers and `_finish_chunk`/`_usage_chunk` build has one choice.
    delta = chunk.choices[0].delta
    out: list[dict[str, Any]] = []
    if reasoning := getattr(delta, "reasoning_content", None):
        out.extend(encoder.thinking(str(reasoning)))
    if text := getattr(delta, "content", None):
        out.extend(encoder.text(str(text)))
    for call in getattr(delta, "tool_calls", None) or []:
        function = call.function
        # Only the opening chunk carries the id; the argument chunks carry the index.
        if call.id:
            out.extend(encoder.tool_call(call.index, call.id, str(function.name or "")))
        if function.arguments:
            out.extend(encoder.tool_arguments(call.index, function.arguments))
    return out


def _absorb_tail(tail: dict[str, Any], chunk: ModelResponseStream) -> None:
    """Keeps the finish reason and usage the trailing chunks carry.

    `_finish_chunk` and `_usage_chunk` close every stream this plugin produces, so the
    turn's own numbers are already on the wire — reading them here is what keeps a
    streamed Messages turn agreeing with the non-streamed one about `stop_reason` and
    token counts.
    """
    usage = getattr(chunk, "usage", None)
    if usage is not None:
        tail["usage"] = usage
    choices = getattr(chunk, "choices", None) or []
    if choices and getattr(choices[0], "finish_reason", None):
        tail["finish_reason"] = choices[0].finish_reason


async def _wrap_messages_stream(
    chunks: AsyncIterator[ModelResponseStream], model: str, kwargs: dict[str, Any]
) -> AsyncIterator[bytes]:
    """The Messages stream as SSE frames, with the turn's spend row dispatched at its end.

    The chat route gets its row from `CustomStreamWrapper`; this route never reaches that
    wrapper, because it emits Anthropic events rather than chat chunks. `_logged_messages`
    closes the same accounting hole `_logged_stream` closes on the Responses route.

    Frames, not dicts: see `messages.sse_frame` for why a dict reached the Anthropic SDK as
    an empty stream. Framing runs outside the accounting so `_logged_messages` still reads
    the events themselves, and the explicit `aclose` hands it a client disconnect at once
    — its `finally` is what bills a turn the client stopped reading.

    A failure after the stream opened goes out as omp's `error` event and is then
    re-raised: the event is what the SDK raises on, and the exception is what still
    reaches LiteLLM's failure hook — whose own error frame the SDK drops unread.
    """
    # `_logged_messages` is an async generator; its declared type is only the iterator.
    events = cast(
        "AsyncGenerator[dict[str, Any], None]",
        _logged_messages(
            _messages_events(_translate_errors(chunks, model, kwargs), model, kwargs),
            model,
            kwargs,
        ),
    )
    try:
        async for event in events:
            yield messages.sse_frame(event["type"], event)
    except Exception as error:
        failure = messages.stream_error(str(getattr(error, "message", None) or error))
        yield messages.sse_frame("error", failure)
        raise
    finally:
        await events.aclose()


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
    except TRANSLATED_ERRORS as error:
        _raise_translated(error, model, kwargs)

    # `anthropic` has no branch of its own: it is served by LiteLLM's native path with the
    # prompt and the token that `_delegate_kwargs` injects. Returning `None` is what routes
    # it there.
    return None


# -- logging, cost identity, error translation: see `observability.py` ---------

