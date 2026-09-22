"""Entry point: intercepts LiteLLM and serves the subscription models.

This module is the only part of the package that knows about LiteLLM. It translates
OpenAI ⇄ provider and delegates the HTTP to `transport/client.py`; body construction lives
entirely in `wire/*`.

Why monkey-patching and not `custom_provider_map`: the official map requires the model name
to carry a provider prefix (``mysubs/gpt-5.5``). Clients ask for ``gpt-5.5``, and rewriting
the name along the way would make the spend log record a model nobody asked for. The patch
catches the name exactly as it arrives.

Decisions taken where the contract leaves room
----------------------------------------------

``RemapRequired`` **propagates**. The transport raises it when the account refuses the
model name and signals that an alias may exist. But `codex.resolve_model` has already
applied the alias table *before* sending: if the upstream refused the result, there is no
name left to try, and retrying would send exactly the same request. Substituting another
model is what the README forbids — the response would come back with the ``model`` field
echoing the request and the billing would start lying. Since `RemapRequired` derives from
`UpstreamError`, the real status and body reach the client.

``RedeemRequired`` **propagates** for the same order of reasons: redeeming a reset credit
spends the user's balance, and the plugin has no mandate to do that unasked. An honest 429
is better than a silent charge.

The synchronous path (`litellm.main.completion`) runs the same asynchronous routine in a
private loop: the transport is natively async by contract, and duplicating the logic in a
synchronous version is exactly what made the two drift apart in the original.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any, Final

import litellm
import litellm.main

from .credentials.store import ProviderId
from .observability import (
    _WIRE_MODEL_KEY,
)
from .turns import (
    StreamError,
)
from .wire import anthropic

if TYPE_CHECKING:
    # Imported lazily at the call sites: `litellm.types.llms.openai` pulls the OpenAI SDK
    # response models, and this module has to stay importable without the proxy extras.
    pass

from .routes import (
    dispatch,
    dispatch_messages,
    dispatch_responses,
)
from .specs import (
    _PROVIDER_IDS,
    ANTIGRAVITY_USER_AGENT,
    CODEX_URL,
    _access_token,
    _normalize,
    _state,
    configure,
)

# -- request specs, process state: see `specs.py` ------------------------------


def is_gemini_model(model: str) -> bool:
    """Models served by the Google Antigravity subscription.

    No OMP anchor on purpose: there the distinction is made by ``model.provider`` in a
    typed catalog (`google-gemini-cli.ts`), not by a predicate over the name. Here the name
    is all that arrives from the client. The shape comes from the original,
    `sitecustomize.py:1549`.
    """
    lowered = str(model).lower()
    return "gemini" in lowered or "antigravity" in lowered


#: Provider declared on the deployment, when the request comes from the Router.
#:
#: Guessing the provider from the name fails on Antigravity, which serves models from
#: **three** families. Measured on the account's real catalog: of the 32 models served,
#: seven have no "gemini" in the name — `claude-opus-4-6-thinking`, `claude-sonnet-4-6`,
#: `chat_23310`, `chat_20706`, `tab_flash_lite_preview`, `tab_jump_flash_lite_preview` fell
#: through to native LiteLLM (which has no credential and blows up), and
#: `gpt-oss-120b-medium` was dispatched to **Codex** — another subscription, another wire,
#: another account being charged.
#:
#: `ModelRegistry` already writes `model_info.mysubs_provider` on every entry it injects.
#: Reading that mark is the difference between knowing and assuming.
_PROVIDER_KEY: Final = "mysubs_provider"


def provider_of_deployment(router: Any, model: str) -> ProviderId | None:
    """The provider declared for `model`, or `None` if it is not one of our entries.

    Looks up `model_name` in the Router's list. A name that is not there — or that is there
    without the mark — returns `None`, and dispatch falls back to the name heuristic, which
    is what serves callers of `litellm.acompletion` directly, without a Router.
    """
    for deployment in getattr(router, "model_list", None) or []:
        if not isinstance(deployment, dict):
            continue
        if deployment.get("model_name") != model:
            continue
        info = deployment.get("model_info") or {}
        declared = info.get(_PROVIDER_KEY)
        if declared in _PROVIDER_IDS.values():
            return declared
    return None


def wire_model_of_deployment(router: Any, model: str) -> str | None:
    """`litellm_params.model` of our deployment for `model`, or `None`.

    Only entries carrying our mark are read: taking the wire name off someone else's
    deployment would price a call this plugin never served.
    """
    for deployment in getattr(router, "model_list", None) or []:
        if not isinstance(deployment, dict):
            continue
        if deployment.get("model_name") != model:
            continue
        info = deployment.get("model_info") or {}
        if info.get(_PROVIDER_KEY) not in _PROVIDER_IDS.values():
            continue
        wire = (deployment.get("litellm_params") or {}).get("model")
        return str(wire) if wire else None
    return None



# -- the three routes: see `routes.py` -----------------------------------------


def _adapt_native_response(response: Any, aliases: dict[str, str]) -> Any:
    """Map tool names back on whatever LiteLLM's own client answered.

    Anthropic has no branch in `dispatch`: the turn is served natively, so the response
    never passes through this package's wire code and the renaming has to be undone from
    here. The model answers with the name it was given, and a client that declared
    ``skills_list`` cannot dispatch a call to ``mcp__skills_list``.

    A streamed turn keeps its ``CustomStreamWrapper`` -- the proxy reads the finished
    turn off that object, so a generator in its place loses the interface it needs.
    ``chunk_creator`` is wrapped instead, being the one place every chunk passes through.
    """
    if not aliases:
        return response
    if not hasattr(response, "chunk_creator"):
        return anthropic.restore_tool_names(response, aliases)

    original = response.chunk_creator

    def chunk_creator(chunk: Any) -> Any:
        return anthropic.restore_tool_names(original(chunk), aliases)

    response.chunk_creator = chunk_creator
    return response


def _pop_aliases(kwargs: dict[str, Any]) -> dict[str, str]:
    """Take the alias map out of the kwargs so it never reaches the upstream."""
    aliases = kwargs.pop(anthropic.TOOL_ALIAS_KEY, None)
    return aliases if isinstance(aliases, dict) else {}


async def _wrapped_acompletion(*args: Any, **kwargs: Any) -> Any:
    kwargs = _normalize(args, kwargs)
    served = await dispatch(**kwargs)
    if served is not None:
        return served
    original = _state.original_acompletion
    assert original is not None
    kwargs.pop(_WIRE_MODEL_KEY, None)
    delegated = await _delegate_kwargs(kwargs)
    aliases = _pop_aliases(delegated)
    return _adapt_native_response(await original(**delegated), aliases)


def _wrapped_completion(*args: Any, **kwargs: Any) -> Any:
    """Synchronous path: runs the same `dispatch` in a private loop.

    Duplicating the logic in a synchronous version is what made the two drift apart in the
    original. The loop is private because `asyncio.run` refuses to run inside an already
    active loop, and the proxy calls this from threads with no loop at all.
    """
    kwargs = _normalize(args, kwargs)
    served = _run_sync(dispatch(**kwargs))
    if served is not None:
        return served
    original = _state.original_completion
    assert original is not None
    kwargs.pop(_WIRE_MODEL_KEY, None)
    delegated = _run_sync(_delegate_kwargs(kwargs))
    aliases = _pop_aliases(delegated)
    return _adapt_native_response(original(**delegated), aliases)


def _run_sync(coroutine: Coroutine[Any, Any, Any]) -> Any:
    """Runs a coroutine from synchronous code, whether or not a loop is active."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    # Synchronous call from inside a loop: running on a thread with its own loop is the
    # only way out that does not deadlock the caller's loop against itself.
    result: list[Any] = []
    error: list[BaseException] = []

    def run() -> None:
        try:
            result.append(asyncio.run(coroutine))
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result[0]


#: Places that refer to the entry functions. `litellm/__init__.py` does
#: `from .main import acompletion`, which **copies** the reference: rebinding only
#: `litellm.main` leaves `litellm.acompletion` pointing at the original function, and a
#: client calling `litellm.acompletion(...)` — the documented form — never goes through
#: dispatch. The original `sitecustomize.py` patches both (lines 2916-2917) and the port
#: started out patching only one: the request blew up with "LLM Provider NOT provided",
#: because it reached the native path with a name no provider knows.
_ASYNC_TARGETS: Final = ((litellm, "acompletion"), (litellm.main, "acompletion"))
_SYNC_TARGETS: Final = ((litellm, "completion"), (litellm.main, "completion"))


def _router_class() -> Any:
    """LiteLLM's `Router` class.

    Imported late: `litellm.router` drags in the whole proxy, and this module has to be
    importable without it.
    """
    from litellm.router import Router

    return Router


async def _delegate_kwargs(
    kwargs: dict[str, Any],
    *,
    provider: ProviderId | None = None,
    native_system: bool = False,
) -> dict[str, Any]:
    """Kwargs for the original, with the Claude prompt applied when it is a Claude model.

    This is what the original's ``_inject_claude_prompt`` does: the Anthropic subscription
    only validates the Claude Code identity as a system message, and without it the request
    is refused.

    A declared ``provider`` that is **not** `anthropic` blocks the injection. The
    Antigravity catalog serves `claude-sonnet-4-6` and `claude-opus-4-6-thinking`: without
    this guard, those requests carried the Anthropic subscription token to a Google
    endpoint — one account's credential sent to another.

    ``native_system`` is passed through for ``/v1/messages``, where the identity belongs in
    the top-level ``system`` rather than in ``messages[0]``.
    """
    model = str(kwargs.get("model") or "")
    if provider is not None and provider != "anthropic":
        return kwargs
    return anthropic.build_request(
        kwargs, model, await _access_token("anthropic"), native_system=native_system
    )


async def _wrapped_router_acompletion(
    self: Any, model: str, messages: list[Any], stream: bool = False, **kwargs: Any
) -> Any:
    """The path the **proxy** actually uses.

    The proxy does not call `litellm.acompletion`: it calls `Router.acompletion`, which
    resolves the deployment and builds the provider client **before** any module function is
    touched. Without this patch, a subscription model reached the native client and blew up
    with `Illegal header value b'Bearer '` — the deployment carries no `api_key` because the
    credential is OAuth and lives in the store, not in `config.yaml`.

    The original `sitecustomize.py` says so in the comment on line 2812: *"Proxy routes
    through Router.acompletion, not necessarily the module functions above."* The port
    patched only the module functions, and the symptom showed up only in the proxy — never
    on a library call.
    """
    # The provider comes from the deployment, not from the name: this is where the Router
    # has the information, and it is the only way to tell Anthropic's `claude-sonnet-4-6`
    # from the namesake served by Antigravity.
    declared = provider_of_deployment(self, model)
    # Read here because this is where the Router still has the deployment: the streaming
    # path needs the wire name to price the call, and by then the deployment is gone.
    wire = wire_model_of_deployment(self, model)
    served = await dispatch(
        provider=declared,
        model=model,
        messages=messages,
        stream=stream,
        **{_WIRE_MODEL_KEY: wire},
        **kwargs,
    )
    if served is not None:
        return served
    original = _state.original_router_acompletion
    assert original is not None
    delegated = await _delegate_kwargs(
        {"model": model, "messages": messages, **kwargs}, provider=declared
    )
    delegated.pop("model", None)
    delegated.pop("messages", None)
    delegated.pop(_WIRE_MODEL_KEY, None)
    aliases = _pop_aliases(delegated)
    served = await original(
        self, model=model, messages=messages, stream=stream, **delegated
    )
    return _adapt_native_response(served, aliases)


def bind_responses_route(router: Any) -> bool:
    """Routes ``/v1/responses`` through the plugin for `router`. ``True`` if it bound.

    `Router.aresponses` is **not** a class method. `Router.__init__` builds it per instance
    with ``self.aresponses = self.factory_function(litellm.aresponses, ...)``, capturing
    `litellm.aresponses` by value at construction. Two consequences, both measured:

    - patching the class does nothing — there is no class attribute to override;
    - patching `litellm.aresponses` after the Router exists does nothing either, because
      the factory already holds the old reference.

    And the Router **does** already exist by the time this package loads: the proxy builds
    it before constructing the `CustomLogger` that brings us in (see `bootstrap`). So the
    only thing that works is replacing the bound attribute on the live instance, which is
    what this does.

    Idempotent per router: rebinding twice would save our own wrapper as the original and
    leave `unbind_responses_route` unable to restore anything.
    """
    if router is None:
        return False
    key = id(router)
    if key in _state.rebound_routers:
        return False
    original = getattr(router, "aresponses", None)
    if original is None:
        return False

    async def _wrapped_router_aresponses(**kwargs: Any) -> Any:
        model = str(kwargs.get("model") or "")
        declared = provider_of_deployment(router, model)
        # On the Router the mark is authoritative and the name heuristic is not consulted:
        # `codex.is_codex_model` matches any name containing "gpt-", so an operator's own
        # `gpt-4o` deployment would be answered from our subscription — measured, and the
        # reason this guard exists rather than deferring to `dispatch_responses` alone.
        if declared is None:
            return await original(**kwargs)
        # Read here for the same reason the chat wrapper does: this is where the Router
        # still has the deployment, and without the wire name the spend log records the
        # public one with no provider — no icon, and no rate to price it against.
        wire = wire_model_of_deployment(router, model)
        served = await dispatch_responses(
            provider=declared, **{_WIRE_MODEL_KEY: wire}, **kwargs
        )
        if served is not None:
            return served
        return await original(**kwargs)

    _state.rebound_routers[key] = (router, original)
    router.aresponses = _wrapped_router_aresponses
    return True


def unbind_responses_route() -> None:
    """Restores every `aresponses` this module replaced."""
    for router, original in _state.rebound_routers.values():
        try:
            router.aresponses = original
        except Exception:  # teardown is best-effort; a dead router is fine
            continue
    _state.rebound_routers.clear()


#: The Router exposes the Messages call under two names, both built by the same factory.
#: The proxy picks one depending on the route, so both are replaced and both restored.
_MESSAGES_ATTRS: Final = ("aanthropic_messages", "anthropic_messages")


def bind_messages_route(router: Any) -> bool:
    """Routes ``/v1/messages`` through the plugin for `router`. ``True`` if it bound.

    Anthropic-native clients speak Messages, and a proxy that answers only
    chat-completions and Responses makes each of them adapt. Measured on the live gateway
    before this existed::

        /v1/messages  mysubs/claudecode/*  401 Missing Anthropic API Key
        /v1/messages  mysubs/codex/*       401 AuthenticationError

    The 401 is the failure mode the Responses route had before 0.1.3, for the same reason:
    with no interception the request reaches LiteLLM's native client and the credential is
    OAuth, living in the store rather than in ``config.yaml``.

    Same binding strategy as `bind_responses_route`, because the Router builds these the
    same way — ``self.aanthropic_messages = self.factory_function(litellm.anthropic_messages,
    ...)`` — and for the same reason there is no class attribute to patch.
    """
    if router is None:
        return False
    key = id(router)
    if key in _state.rebound_messages_routers:
        return False
    originals = {name: getattr(router, name, None) for name in _MESSAGES_ATTRS}
    if all(value is None for value in originals.values()):
        return False

    def _wrap(original: Any) -> Any:
        async def _wrapped(**kwargs: Any) -> Any:
            model = str(kwargs.get("model") or "")
            declared = provider_of_deployment(router, model)
            # The deployment mark is authoritative, as on the other two routes: a name
            # heuristic would answer an operator's own `claude-*` deployment from our
            # subscription.
            if declared is None:
                return await original(**kwargs)
            # Same as the other two routes: the wire name is only available here, and the
            # spend log needs it to price the call and draw a provider icon.
            wire = wire_model_of_deployment(router, model)
            served = await dispatch_messages(
                provider=declared, **{_WIRE_MODEL_KEY: wire}, **kwargs
            )
            if served is not None:
                return served
            # Claude Max declines translation because Messages is already its wire — but
            # declining is not the same as needing no credential. The token still has to
            # be injected, exactly as the chat route does before delegating, or the native
            # client answers `Missing Anthropic API Key`.
            delegated = await _delegate_kwargs(
                kwargs, provider=declared, native_system=True
            )
            aliases = _pop_aliases(delegated)
            return _adapt_native_response(await original(**delegated), aliases)

        return _wrapped

    _state.rebound_messages_routers[key] = (router, originals)
    for name, original in originals.items():
        if original is not None:
            setattr(router, name, _wrap(original))
    return True


def unbind_messages_route() -> None:
    """Restores every Messages attribute this module replaced."""
    for router, originals in _state.rebound_messages_routers.values():
        for name, original in originals.items():
            if original is None:
                continue
            try:
                setattr(router, name, original)
            except Exception:  # teardown is best-effort; a dead router is fine
                continue
    _state.rebound_messages_routers.clear()


def install() -> None:
    """Applies the patch. Idempotent.

    The guard is not cosmetic: installing twice chained two wrappers, and the second saved
    the first as the "original" — `uninstall` then left the patch half applied and every
    request went through dispatch twice.
    """
    if _state.original_acompletion is not None:
        return
    _state.original_acompletion = litellm.main.acompletion
    _state.original_completion = litellm.main.completion
    router_class = _router_class()
    _state.original_router_acompletion = router_class.acompletion
    for module, name in _ASYNC_TARGETS:
        setattr(module, name, _wrapped_acompletion)
    for module, name in _SYNC_TARGETS:
        setattr(module, name, _wrapped_completion)
    router_class.acompletion = _wrapped_router_acompletion


def uninstall() -> None:
    """Restores the originals everywhere. With no patch applied, does nothing."""
    if _state.original_acompletion is None:
        return
    for module, name in _ASYNC_TARGETS:
        setattr(module, name, _state.original_acompletion)
    if _state.original_completion is not None:
        for module, name in _SYNC_TARGETS:
            setattr(module, name, _state.original_completion)
    if _state.original_router_acompletion is not None:
        _router_class().acompletion = _state.original_router_acompletion
        _state.original_router_acompletion = None
    unbind_responses_route()
    unbind_messages_route()
    _state.original_acompletion = None
    _state.original_completion = None


__all__ = [
    "ANTIGRAVITY_USER_AGENT",
    "CODEX_URL",
    "StreamError",
    "bind_messages_route",
    "bind_responses_route",
    "configure",
    "dispatch",
    "dispatch_responses",
    "install",
    "is_gemini_model",
    "unbind_messages_route",
    "unbind_responses_route",
    "uninstall",
]
