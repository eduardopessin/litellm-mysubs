"""Spend logging, cost identity and error translation for the three routes.

Extracted from `plugin.py`. None of it touches process state: every function takes the
caller's `kwargs` and works from what is in them, which is why it moved out cleanly.

The three jobs here are what makes a served request indistinguishable from a native one
from the outside:

- **identity** — the wire `(model, provider)` pair, so the spend log can price the call
  and the UI can draw a provider icon;
- **logging** — firing the success handler on paths that never reach LiteLLM's own
  `@client` wrapper, which is every path this plugin serves;
- **errors** — mapping an upstream refusal onto the exception LiteLLM has a class for, so
  a 429 stays a 429 instead of collapsing into `internal_server_error`.
"""

from __future__ import annotations

import contextlib
import datetime
import time
import uuid
from collections.abc import AsyncIterator, Coroutine, Mapping
from types import MappingProxyType
from typing import Any, Final

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.types.utils import ModelResponseStream

from .transport.client import RedeemRequired, RemapRequired, UpstreamError

#: Operator-facing log. `litellm_mysubs.*` loggers are silent inside the proxy: the root
#: logger has no handlers, so a plain `getLogger(__name__)` emits nowhere — measured, and
#: the reason a diagnostic build looked like the code never ran. `verbose_proxy_logger`
#: is the one with a handler, and it is where an operator already looks.
_LOG: Final = verbose_proxy_logger

#: Carries the deployment's wire model name across the dispatch boundary.
_WIRE_MODEL_KEY: Final = "mysubs_wire_model"


async def _translate_errors(
    chunks: AsyncIterator[ModelResponseStream], model: str, kwargs: dict[str, Any]
) -> AsyncIterator[ModelResponseStream]:
    """Same translation as `dispatch`, for the streaming path.

    The generator is built before `dispatch` returns but only starts running when the
    wrapper pulls the first chunk, so the `try` around the call site never sees the
    upstream refusal — `_open` runs inside the first `__anext__`. Wrapping the iterator is
    what puts the translation where the error actually surfaces.
    """
    try:
        async for chunk in chunks:
            yield chunk
    except UpstreamError as error:
        raise _as_litellm_error(error, model, kwargs) from error


def _as_litellm_error(error: UpstreamError, model: str, kwargs: dict[str, Any]) -> Exception:
    """Translates an upstream refusal into the exception LiteLLM understands.

    `UpstreamError` already carries the real status, but the proxy has no class for it: an
    unrecognised exception is reported as `internal_server_error` with HTTP 500, and the
    upstream status survives only as text inside the message. Measured on the live
    gateway, a quota refusal reached the client as::

        HTTP 500  {"error": {"type": "internal_server_error", "code": "500",
                             "message": "HTTP 429: ... RESOURCE_EXHAUSTED ..."}}

    A client cannot back off on a 500. It can on a 429, and backing off is the one correct
    response to a quota refusal — so the status is mapped, never invented: anything that is
    not a status LiteLLM has a class for propagates unchanged.

    `RemapRequired` and `RedeemRequired` are excluded even though the latter is also a 429:
    they are signals to `plugin.py`, not answers to the client, and the module docstring's
    rule that they propagate as themselves is what lets a caller act on them.
    """
    if isinstance(error, RemapRequired | RedeemRequired):
        return error
    _, family = _cost_identity(model, kwargs)
    if error.status == 429:
        return litellm.exceptions.RateLimitError(
            message=str(error), llm_provider=family, model=model
        )
    return error


async def _logged(turn: Coroutine[Any, Any, Any], kwargs: dict[str, Any]) -> Any:
    """Awaits a non-streaming turn and dispatches success logging for it.

    Chat streaming already logs: `_wrap_stream` hands the response to
    `CustomStreamWrapper`, which dispatches the success handler at end of stream. A
    non-streaming turn returns bare straight out of `dispatch` — a `ModelResponse` on the
    chat route, a `ResponsesAPIResponse` on ``/v1/responses`` — so it never reaches the
    `@client` wrapper in ``litellm.utils`` that would normally call
    ``logging_obj.async_success_handler``, and a request that never logs produces **no**
    spend row at all. Both routes are served here; the streamed Responses counterpart is
    `_logged_stream`.

    Measured on a live gateway, two calls to the same model two seconds apart:

    ===============  ======================================================
    ``stream: true``  ``openai/gpt-5.5`` priced at ``0.00121``
    ``stream: false`` no row of any kind
    ===============  ======================================================

    The consequence is not a pricing bug but an accounting hole: ``x-litellm-key-spend``
    undercounts, and per-key budgets and rate limits never see these calls.

    Best effort by contract: a logging failure must not lose a response the upstream
    already produced and the user has already been charged for by the subscription.
    """
    # Marked before the await, not after: the two timestamps are what the row's duration
    # is computed from, and passing the same instant twice makes every call look
    # instantaneous. Measured on the live gateway's Logs tab: our rows read 0 ms while the
    # natively-served `anthropic/claude-opus-5` ones read 3.9 s to 14 s.
    started = datetime.datetime.now()
    response = await turn
    logging_obj = kwargs.get("litellm_logging_obj")
    handler = getattr(logging_obj, "async_success_handler", None)
    if handler is None:
        return response
    _stamp_cost(logging_obj, response, kwargs)
    with contextlib.suppress(Exception):
        await handler(
            result=response, start_time=started, end_time=datetime.datetime.now()
        )
    return response


def _stamp_cost(logging_obj: Any, response: Any, kwargs: dict[str, Any]) -> None:
    """Prices the turn and records it where the spend row reads it from.

    `get_standard_logging_object_payload` takes the number from
    ``kwargs["response_cost"]``, which the `@client` wrapper in ``litellm.utils`` fills in
    — and that wrapper is precisely what a request served here never reaches. Measured on
    the live gateway's Logs tab: 49 of 50 rows at zero, `anthropic/claude-opus-5` turns of
    123k tokens among them, whose rate is in the map.

    So the identity fix was necessary and not sufficient: the row named the right model
    and still billed nothing. `completion_cost` is asked under the **wire** identity, which
    is what has a rate — the response keeps the public name the client asked for, and
    `_select_model_name_for_cost_calc` prefers that name and prefixes the provider to it,
    producing a key absent from the map. Pricing a copy sidesteps that.

    A model with no rate leaves the cost unset rather than zero: `None` means "not priced"
    to every consumer downstream, while a literal 0.0 asserts the turn was free.
    """
    details = getattr(logging_obj, "model_call_details", None)
    if not isinstance(details, dict):
        return
    model = str(kwargs.get("model") or "")
    cost_model, cost_provider = _cost_identity(model, kwargs)
    if cost_model == model:
        return
    with contextlib.suppress(Exception):
        priceable = response.model_copy()
        priceable.model = cost_model
        cost = float(
            litellm.completion_cost(
                completion_response=priceable,
                model=cost_model,
                custom_llm_provider=cost_provider,
            )
        )
        if cost:
            details["response_cost"] = cost
            response._hidden_params["response_cost"] = cost


class _LoggedResponsesStream(BaseResponsesAPIStreamingIterator):
    """Relays a Responses stream, dispatches its spend row, and carries the turn.

    Two separate things in LiteLLM have to recognise this object, and a bare
    ``async_generator`` satisfies neither.

    **The Router checks the type.** `_aresponses_with_streaming_fallbacks` ends with::

        if kwargs.get("stream") and isinstance(response, BaseResponsesAPIStreamingIterator):
            return await self._aresponses_streaming_iterator(...)
        return response

    Anything else is handed back raw and never reaches the path that logs the turn. That
    is why subclassing is not cosmetic: the `isinstance` is the gate. ``__init__`` is not
    called — the base wants an `httpx.Response` this object does not have and does not
    need, since it relays events that are already parsed.

    **The proxy reads the finished turn off the object**, not off anything the iteration
    yields: `_extract_completed_responses_response` does ``attribute_of(stream_response,
    "completed_response")``. The gateway said so about our own generator::

        15:18:35 WARNING common_request_processing.py:2814 - Container ownership
        recording skipped on streaming /v1/responses: no completed_response on
        stream iterator async_generator

    Measured across those builds, same model and key: the streamed chat turn logged
    ``dur=3958 ttft=3794``, the non-streamed Responses turn logged ``dur=1719``, and the
    streamed Responses turn left no row at all.

    The row is dispatched **when the terminal event is seen**, not after iteration ends. A
    consumer that stops reading at ``response.completed`` closes the underlying generator,
    and ``aclose()`` raises `GeneratorExit` at the ``yield``, so anything after the loop
    never runs. `aclose` covers a stream that ends without a terminal event.

    Best effort by contract: a logging failure must not truncate a stream the subscription
    has already been charged for.
    """

    def __init__(self, events: AsyncIterator[Any], kwargs: dict[str, Any]) -> None:
        # `super().__init__` is deliberately not called — it wants an `httpx.Response`
        # this object does not have, relaying events that are already parsed. But every
        # attribute it would have set is mirrored here, because the inherited methods read
        # them: `_check_max_streaming_duration` reads `start_time`, `_handle_failure`
        # reads `_failure_handled`, `_log_completed_response` reads
        # `_completed_response_logged`, and so on for fourteen names across fourteen
        # methods. Subclassing for the `isinstance` gate while leaving them unset trades a
        # missing row for an AttributeError mid-stream.
        self._events = events
        self._kwargs = kwargs
        self._started = datetime.datetime.now()
        self._first_token_at: datetime.datetime | None = None
        self._emitted = False

        #: The terminal event's response, where the proxy looks for the finished turn.
        self.completed_response: Any = None
        #: The terminal event itself, which is the shape the success handler requires.
        self._terminal_event: Any = None
        self.response: Any = None
        self.model = str(kwargs.get("model") or "")
        self.logging_obj: Any = kwargs.get("litellm_logging_obj")
        self.responses_api_provider_config = None
        self.litellm_metadata = kwargs.get("litellm_metadata")
        self.custom_llm_provider = kwargs.get("custom_llm_provider")
        self.start_time = self._started
        self.finished = False
        self._failure_handled = False
        self._yielded_first_chunk = False
        self._generated_content = ""
        self._completed_response_cached = False
        self._completed_response_logged = False
        self._completed_response_cache_hit: bool | None = None
        self._persist_completed_response_before_logging = True
        # Annotated assignments in the base constructor, which an earlier version missed
        # because the test derived its list by regexing `self.<name> =` out of the base
        # source and that pattern cannot match `self.<name>: T = ...`. Two of them bite
        # today rather than hypothetically: `_stream_created_time` is read by
        # `_check_max_streaming_duration` on every `__anext__` (latent only because
        # `LITELLM_MAX_STREAMING_DURATION_SECONDS` defaults to None), and `_hidden_params`
        # is read by the proxy's `get_hidden_params_dict` to build the response headers —
        # empty here meant plugin-served streaming turns answered without
        # `x-litellm-model-id`, `x-litellm-api-base` and `x-litellm-response-cost` while
        # natively-served ones carried them.
        self._stream_created_time = time.time()
        self.request_data: dict[str, Any] = {}
        self.call_type: str | None = None
        self._hidden_params: dict[str, Any] = {
            "custom_llm_provider": self.custom_llm_provider,
            "additional_headers": {},
        }
        self._raw_response_headers: Mapping[str, Any] = MappingProxyType({})

    def __aiter__(self) -> _LoggedResponsesStream:
        return self

    async def __anext__(self) -> Any:
        try:
            event = await self._events.__anext__()
        except StopAsyncIteration:
            await self._emit()
            raise
        # The first event out is the time-to-first-token the Logs tab shows. Nothing else
        # can measure it: by the time the stream ends the moment has passed, and the
        # upstream does not report it.
        if self._first_token_at is None:
            self._first_token_at = datetime.datetime.now()
        if getattr(event, "type", None) in ("response.completed", "response.incomplete"):
            # Two consumers, two shapes, and they are not the same object.
            # `completed_response` is read by the proxy, which wants the **response**.
            # `async_success_handler(result=...)` is read by
            # `Logging._get_assembled_streaming_response`, whose streaming branch is
            # `isinstance(result, (ResponseCompletedEvent, ResponseIncompleteEvent,
            # ResponseFailedEvent))` and returns `None` for anything else — no assembled
            # response, no row. Passing the response to both is why the turn logged
            # nothing while reporting `terminal=True` and raising no error.
            self.completed_response = getattr(event, "response", None) or event
            self._terminal_event = event
            await self._emit()
        return event

    async def aclose(self) -> None:
        await self._emit()
        aclose = getattr(self._events, "aclose", None)
        if aclose is not None:
            await aclose()

    async def _emit(self) -> None:
        if self._emitted:
            return
        self._emitted = True
        logging_obj = self._kwargs.get("litellm_logging_obj")
        handler = getattr(logging_obj, "async_success_handler", None)
        if handler is None:
            _LOG.warning("mysubs: no async_success_handler on the responses stream turn")
            return
        try:
            # Inside the `try`, not before it. Both stamps start by reading
            # `logging_obj.model_call_details`, and that read is not itself protected: a
            # logging object whose attribute raises truncated the stream with zero events
            # delivered, on a turn the subscription had already paid for — while
            # `_emitted` was already set, so `aclose()` would not retry and the `except`
            # below never ran. Measured against a raising logging object: 0 of 3 events.
            if self.completed_response is not None:
                _stamp_cost(logging_obj, self.completed_response, self._kwargs)
            _stamp_first_token(logging_obj, self._first_token_at)
            # The **event**, not the response: the handler's streaming branch keys off
            # `isinstance(result, ResponseCompletedEvent)` and drops anything else.
            await handler(
                result=self._terminal_event,
                start_time=self._started,
                end_time=datetime.datetime.now(),
            )
        except Exception:
            # Still swallowed — the subscription has been charged and the answer must not
            # be truncated over a logging failure. But a silent `suppress` is what made
            # this defect cost two deploys to find: the row was missing with nothing
            # anywhere saying why. An operator gets the traceback; the client gets the
            # stream.
            _LOG.exception("mysubs: the responses stream turn produced no spend row")


def _logged_stream(events: AsyncIterator[Any], kwargs: dict[str, Any]) -> _LoggedResponsesStream:
    """Wraps a Responses event stream so it logs its spend row and carries the turn.

    The chat route gets this for free: `_wrap_stream` hands the turn to
    `CustomStreamWrapper`, which fires the handler itself at end of stream. The Responses
    route emits its own event sequence and never touches that wrapper, so it produced
    **no** spend row at all — the same accounting hole `_logged` closes for the
    non-streaming path, on the path that matters most in practice: a client that discovers
    models through LiteLLM routes every OpenAI-backed model here.

    Measured on the live gateway: an omp run exercising all five Codex models end to end
    left no Codex row, while Anthropic and Gemini — which go through chat — logged 33 and
    50 rows over the same minutes.
    """
    return _LoggedResponsesStream(events, kwargs)




def _stamp_first_token(logging_obj: Any, moment: datetime.datetime | None) -> None:
    """Records when the first chunk left, which is what the row's TTFT is read from.

    A stream that produced nothing has no first token, and leaving the field unset says
    exactly that — a fabricated instant would read as a fast answer that never came.

    Writing the dict entry alone is not enough, and looked like it was: LiteLLM's
    `_success_handler_helper_fn` does ``if self.completion_start_time is None`` and then
    overwrites both the attribute and the dict entry with `end_time`. The attribute is
    what it tests, so a row stamped only through `model_call_details` came out with
    TTFT equal to the whole duration. `_update_completion_start_time` sets both, and is
    the method LiteLLM's own `_process_chunk` calls for this.
    """
    if moment is None:
        return
    updater = getattr(logging_obj, "_update_completion_start_time", None)
    if callable(updater):
        with contextlib.suppress(Exception):
            updater(completion_start_time=moment)
            return
    details = getattr(logging_obj, "model_call_details", None)
    if isinstance(details, dict):
        details["completion_start_time"] = moment


def _cost_identity(model: str, kwargs: dict[str, Any]) -> tuple[str, str]:
    """The ``(model, provider)`` pair the cost calculation needs.

    ``model`` arrives here as the **public** name (``mysubs/codex/gpt-5.5``) — what the
    client asked for and what the Router resolved. No rate exists under that name, and
    ``custom_openai`` has no price table either, so every streamed call was logged at
    ``0.0``. Measured on ``litellm[proxy]`` 1.101.0, identical usage:

    ===================================== ============
    ``(model, custom_llm_provider)``       cost
    ===================================== ============
    ``("mysubs/codex/gpt-5.5", "custom_openai")``  ``0.0``
    ``("openai/gpt-5.5", "openai")``               ``0.0202325``
    ===================================== ============

    The wire name is already on the deployment, in ``litellm_params.model``, carrying the
    family prefix ``to_deployment`` picked from ``modelProvider``. It is the same pair the
    non-streaming path gets from the Router — which is why only streaming lost the cost,
    and why 83% of real calls were logged as free.

    Without a deployment there is nothing to look up: a direct ``litellm.acompletion``
    call has no Router. The old pair is kept for that case rather than guessing a family
    from the name, because a wrong guess prices the call against another model's rate.

    A third case is Antigravity, whose catalog names carry the **effort** as part of the
    model id. Those names are not in the price map and the base name is::

        gemini/gemini-3.6-flash-low   -> no rate
        gemini/gemini-3.6-flash       -> in=7.5e-07 out=3.75e-06

    Measured on the live gateway, every Gemini row across all six routes was logged at
    ``0.0`` while usage was recorded correctly (62 prompt + 266 completion on one of
    them). Effort changes the thinking budget, not the per-token rate, so the base name
    is the right thing to price against. It recovers 21 of the 32 served models; the
    other 11 (``chat_20706``, ``gemini-pro-agent``, ``gpt-oss-120b``, the ``tab_*``
    previews, Claude-over-Antigravity) have no rate under any name and stay unpriced —
    `_stamp_cost` leaves the field unset, which reads as "not priced" rather than free.
    """
    wire = str(kwargs.get(_WIRE_MODEL_KEY) or "")
    if not wire:
        return model, "custom_openai"
    family = wire.split("/", 1)[0] if "/" in wire else "openai"
    return _priceable_name(wire), family


#: Effort and routing suffixes Antigravity appends to a model id. Longest first: a plain
#: ``-low`` must not eat the ``-extra-low`` that shares its ending.
_EFFORT_SUFFIXES: Final = (
    "-extra-low",
    "-thinking",
    "-minimal",
    "-tiered",
    "-medium",
    "-agent",
    "-xhigh",
    "-high",
    "-low",
    "-max",
)


def _priceable_name(wire: str) -> str:
    """``wire`` itself when the map knows it, else the name without its effort suffix.

    Only falls back when the trimmed name actually has a rate: trimming blindly would
    turn ``gemini-3-flash-agent`` into ``gemini-3-flash``, which has no rate either, and
    would silently reprice a model that merely ends in a familiar word.
    """
    if wire in litellm.model_cost:
        return wire
    for suffix in _EFFORT_SUFFIXES:
        if wire.endswith(suffix):
            base = wire[: -len(suffix)]
            return base if base in litellm.model_cost else wire
    return wire


def _stamp_logging_identity(model: str, kwargs: dict[str, Any]) -> None:
    """Record the **wire** ``(model, provider)`` pair on the caller's logging object.

    The spend log reads `custom_llm_provider` and `model` from
    `logging_obj.model_call_details`, not from the deployment. A request this plugin serves
    never reaches the provider client that would normally fill them in, so without this the
    row lands with the public name and **no provider at all** — measured on a live gateway:
    rows served natively read `anthropic/claude-opus-5` + `anthropic`, while rows served
    here read `mysubs/antigravity/...` with an empty provider.

    That empty provider is what leaves the Logs tab without an icon: the UI maps a provider
    id to a logo, and there is nothing to map. Declaring `custom_llm_provider` on the
    deployment fixes the Models tab only, because that one is built from the Router while
    this one is built per request.

    `model_group` keeps the public name, so the client still sees what it asked for.
    """
    logging_obj = kwargs.get("litellm_logging_obj")
    details = getattr(logging_obj, "model_call_details", None)
    if not isinstance(details, dict):
        return
    wire_model, provider = _cost_identity(model, kwargs)
    if not wire_model or wire_model == model:
        # No deployment to read the wire name from: leave what the proxy already set
        # rather than stamping a guessed family, which would bill against another rate.
        return
    details["model"] = wire_model
    details["custom_llm_provider"] = provider
    details.setdefault("model_group", model)
    with contextlib.suppress(Exception):
        logging_obj.model = wire_model  # type: ignore[union-attr]
        logging_obj.custom_llm_provider = provider  # type: ignore[union-attr]


def _wrap_stream(
    chunks: AsyncIterator[ModelResponseStream], model: str, kwargs: dict[str, Any]
) -> litellm.CustomStreamWrapper:
    """Wraps it in LiteLLM's iterator: that is what the proxy knows how to consume.

    Returning the raw generator gave the client objects without the protocol the
    ``/v1/chat/completions`` route expects — and no spend log callback would fire.

    The model and provider handed to the wrapper are the **wire** ones, not the public
    name: see ``_cost_identity``. The client still sees the name it asked for, because the
    spend log records ``model_group`` for that, but the cost calculation now gets a pair
    it can price.
    """
    cost_model, cost_provider = _cost_identity(model, kwargs)
    return litellm.CustomStreamWrapper(
        completion_stream=chunks,
        model=cost_model,
        custom_llm_provider=cost_provider,
        logging_obj=kwargs.get("litellm_logging_obj") or _logging_obj(cost_model, kwargs),
    )


def _logging_obj(model: str, kwargs: dict[str, Any]) -> Logging:
    """Logging object for when the caller does not bring its own.

    `CustomStreamWrapper` dereferences it in the constructor — passing ``None`` blows up
    before the first chunk. The proxy always injects its own; a direct library call does
    not.
    """
    return Logging(
        model=model,
        messages=kwargs.get("messages") or [],
        stream=True,
        call_type="acompletion",
        start_time=datetime.datetime.now(),
        litellm_call_id=str(kwargs.get("litellm_call_id") or uuid.uuid4()),
        function_id=str(kwargs.get("id") or uuid.uuid4()),
    )


