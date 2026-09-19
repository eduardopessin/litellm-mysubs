"""The `CustomLogger` that `config.yaml` loads.

    litellm_settings:
      callbacks: ["litellm_mysubs.MySubs"]

It is the package's only entry point in a normal installation. Everything else — the patch,
the UI, the registry — is triggered from here.

## The rule that is not broken

**This callback never alters the request.** LiteLLM's `async_pre_call_hook` may return a
modified `data`, and that is how guardrails and prompt injectors work. Here `None` is always
returned: the hook serves only as a startup trigger, and a plugin whose purpose is to *add*
models has no business touching the requests of the ones that were already there.

For the same reason `async_filter_deployments` is not implemented: filtering deployments is
meddling with the routing of models that are not ours.
"""

from __future__ import annotations

import contextlib
from typing import Any, cast

from .bootstrap import Bootstrap, disabled
from .credentials.store import ProviderId


def _base() -> type:
    """The base class: `CustomLogger` if LiteLLM is present, `object` if not.

    The proxy **requires** `isinstance(loaded, CustomLogger)` — measured: an object with the
    right hooks but without the inheritance makes `load_config` raise and startup fails
    outright. It is not optional.

    Resolution is late so that `litellm_mysubs` stays importable without LiteLLM:
    `mysubs-setup` runs before there is a configured proxy, and this layer's tests must not
    drag in the whole package.
    """
    try:
        # Without LiteLLM's stubs `CustomLogger` resolves to `Any`, and returning `Any`
        # from a `-> type` function is an error under `--strict`. The cast states what the
        # symbol is: a class, whatever the type checker can see of it.
        from litellm.integrations.custom_logger import CustomLogger

        return cast(type, CustomLogger)
    except ImportError:
        return object


class MySubs(_base()):  # type: ignore[misc]
    """Connects the subscriptions to the proxy."""

    def __init__(self, store: Any = None) -> None:
        self._bootstrap = Bootstrap()
        self._store = store
        # Mount here, not on the first request. The proxy instantiates the `config.yaml`
        # `callbacks` **inside** `proxy_startup_event`, before the `yield` that opens the
        # server to traffic — measured: by the end of the lifespan the callback already
        # exists and `llm_router` is already ready. Mounting on `async_pre_call_hook` left
        # `/mysubs` answering 404 until someone made an inference request, and a 404
        # teaches nothing to whoever restarted the proxy and opened the page first.
        #
        # The UI only. The patch still depends on a connected credential, and that decision
        # belongs to `setup`: mounting a page adds a prefix, patching means entering the
        # path of every request in the installation.
        with contextlib.suppress(Exception):
            self._mount_ui()

    def _mount_ui(self) -> None:
        """Mounts `/mysubs` if the proxy already has an app. Silent if it does not.

        With no app — imported by a test, or by `mysubs-setup` — there is nowhere to mount,
        and that is not an error: `setup` tries again once there is one.
        """
        if disabled():
            return
        app = _proxy_app()
        if app is not None:
            self._bootstrap.mount(app, self.store)

    # -- state ---------------------------------------------------------------------

    @property
    def store(self) -> Any:
        """The credential store, resolved on first use.

        Late because the choice depends on `litellm.secret_manager_client`, which the proxy
        only fills in after processing `key_management_system` — and this object is built
        before that.
        """
        if self._store is None:
            from .ui.install import default_store

            self._store = default_store()
        return self._store

    @property
    def status(self) -> dict[str, Any]:
        """For diagnostics: `mysubs-setup --status` shows this."""
        return {
            "disabled": disabled(),
            "patched": self._bootstrap.patched,
            "mounted": self._bootstrap.mounted,
            "menu": self._bootstrap.menu,
            "error": self._bootstrap.error,
        }

    # -- startup -------------------------------------------------------------------

    def setup(self, app: Any = None) -> dict[str, Any]:
        """Mounts the UI and applies the patch if a subscription is connected.

        Called by the hooks and by `mysubs-setup`. Idempotent.
        """
        if disabled():
            return self.status
        target = app if app is not None else _proxy_app()
        if target is not None:
            self._bootstrap.mount(target, self.store)
        self._bootstrap.patch(self.store)
        return self.status

    # -- LiteLLM hooks -------------------------------------------------------------

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any = None,
        cache: Any = None,
        data: dict[str, Any] | None = None,
        call_type: str = "",
    ) -> None:
        """Startup trigger. **Always returns `None`.**

        `None` means "I modified nothing" and is what guarantees the request goes on exactly
        as it arrived. Returning `data` here — even untouched — would put this plugin on the
        write path of every request in the installation, including those of the models that
        were already there.
        """
        self.setup()
        return None

    async def async_post_call_success_hook(
        self,
        data: dict[str, Any] | None = None,
        user_api_key_dict: Any = None,
        response: Any = None,
    ) -> None:
        """Absorbs whatever usage comes in the response headers.

        It is how the cards know the quota: a subscription has no usage endpoint, the state
        only travels on the responses. An error here must not affect the response the client
        receives.
        """
        with contextlib.suppress(Exception):
            self._observe(data, response)
        return None

    def _observe(self, data: dict[str, Any] | None, response: Any) -> None:
        from .ui.install import shared_service

        service = shared_service()
        if service is None:
            return
        headers = getattr(response, "_hidden_params", {}) or {}
        headers = headers.get("additional_headers") or {}
        if not headers:
            return
        model = str((data or {}).get("model", ""))
        provider = _provider_of(model)
        if provider:
            service.observe(provider, headers)


def _provider_of(model: str) -> ProviderId | None:
    from .wire import codex
    from .wire.anthropic import is_anthropic_model

    lowered = model.lower()
    if "gemini" in lowered or "antigravity" in lowered:
        return "google-antigravity"
    if codex.is_codex_model(model):
        return "openai-codex"
    if is_anthropic_model(model):
        return "anthropic"
    return None


def _proxy_app() -> Any:
    try:
        from litellm.proxy import proxy_server

        return getattr(proxy_server, "app", None)
    except Exception:
        return None
