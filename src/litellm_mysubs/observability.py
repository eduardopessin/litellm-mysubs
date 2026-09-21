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
import uuid
from collections.abc import AsyncIterator, Coroutine
from typing import Any, Final

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.types.utils import ModelResponseStream

from .transport.client import RedeemRequired, RemapRequired, UpstreamError

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


class _LoggedResponsesStream:
    """Relays a Responses stream, dispatches its spend row, and carries the turn.

    A bare ``async_generator`` is not enough on this route, even though it streams
    correctly. The proxy reads the finished turn off the **object** it was handed —
    ``_extract_completed_responses_response`` does ``attribute_of(stream_response,
    "completed_response")`` — rather than from anything the iteration yields. Measured on
    the live gateway against the generator this class replaces::

        15:18:35 WARNING common_request_processing.py:2814 - Container ownership
        recording skipped on streaming /v1/responses: no completed_response on
        stream iterator async_generator

    ``async_generator`` in that line is ours. Same minute, same model, same key: the
    streamed chat turn logged ``dur=3958 ttft=3794`` and the streamed Responses turn left
    no row at all. So the attribute is the contract, and this class holds it while
    remaining an async iterator — which is all `_is_streaming_response` requires.

    The row is dispatched **when the terminal event is seen**, not after iteration ends. A
    consumer that stops reading at ``response.completed`` closes the underlying generator,
    and ``aclose()`` raises `GeneratorExit` at the ``yield``, so anything after the loop
    never runs. `aclose` covers a stream that ends without a terminal event.

    Best effort by contract: a logging failure must not truncate a stream the subscription
    has already been charged for.
    """

    def __init__(self, events: AsyncIterator[Any], kwargs: dict[str, Any]) -> None:
        self._events = events
        self._kwargs = kwargs
        self._started = datetime.datetime.now()
        self._first_token_at: datetime.datetime | None = None
        self._emitted = False
        #: The terminal event's response, where the proxy looks for the finished turn.
        self.completed_response: Any = None

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
            self.completed_response = getattr(event, "response", None) or event
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
            return
        terminal = self.completed_response
        if terminal is not None:
            _stamp_cost(logging_obj, terminal, self._kwargs)
        _stamp_first_token(logging_obj, self._first_token_at)
        with contextlib.suppress(Exception):
            await handler(
                result=terminal, start_time=self._started, end_time=datetime.datetime.now()
            )


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
    """
    if moment is None:
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
    """
    wire = str(kwargs.get(_WIRE_MODEL_KEY) or "")
    if not wire:
        return model, "custom_openai"
    family = wire.split("/", 1)[0] if "/" in wire else "openai"
    return wire, family


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


