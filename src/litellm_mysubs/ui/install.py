"""Mounting the UI on the LiteLLM proxy."""

from __future__ import annotations

from typing import Any

from litellm._logging import verbose_proxy_logger

from ..credentials.file_store import FileCredentialStore
from ..credentials.secret_store import SecretManagerCredentialStore, is_available
from ..credentials.store import CredentialStore
from .app import MOUNT_PATH, mount
from .service import MySubsService


def _live_router() -> Any:
    """The proxy's Router, read **at the moment** instead of captured.

    The proxy only creates `llm_router` at startup, after the sub-app is mounted. Storing
    the reference at mount time pinned `None` forever and the apply button would never do
    anything — the failure mode this package exists to prevent.
    """
    from litellm.proxy import proxy_server

    return getattr(proxy_server, "llm_router", None)


def default_store() -> CredentialStore:
    """Where to store the credentials, in order of preference.

    The LiteLLM vault first: whoever configured `key_management_system` has already decided
    where the installation's secrets live, and writing the subscriptions' somewhere else
    would leave tokens on disk outside that policy.

    With no vault, the file in `~/.litellm/mysubs/` with `0600` permissions — which is what
    works on any installation, with no infrastructure.

    The choice is made at startup and does not change: a store that moved halfway would
    scatter half the credentials in each place.
    """
    if is_available():
        return SecretManagerCredentialStore()
    return FileCredentialStore()


#: How long to wait for `llm_router` at startup. The proxy creates it moments after
#: instantiating the callbacks; a 30s ceiling covers a slow startup without leaving a task
#: alive forever on a proxy that never creates it.
_ROUTER_WAIT_S = 0.25
_ROUTER_WAIT_TRIES = 120

#: The mounted service, for whoever needs to talk to it later — the callback uses it to
#: hand over the usage headers from responses. One per process: two services would have
#: different discovery states and the page would show that of whoever mounted first.
_SERVICE: MySubsService | None = None


def shared_service() -> MySubsService | None:
    """The mounted UI's service, or `None` if it has not been mounted yet."""
    return _SERVICE


def install(app: Any | None = None, *, store: CredentialStore | None = None) -> str:
    """Mounts `/mysubs`. Returns the mounted path.

    Without `app`, it uses the proxy's. Without `store`, the best available one is chosen.
    """
    if app is None:
        from litellm.proxy import proxy_server

        app = proxy_server.app

    global _SERVICE
    service = MySubsService(
        store=store or default_store(),
        router_source=_live_router,
    )
    mount(app, service)
    _SERVICE = service
    _start_refresher_with(app, service)
    return MOUNT_PATH


def _start_refresher_with(app: Any, service: MySubsService) -> None:
    """Starts the refresher in the **host** app's life cycle.

    Not in the sub-app: measured that Starlette **does not propagate the lifespan to mounted
    sub-apps** — a `lifespan=` in `build_app` never runs when the app is mounted with
    `app.mount()`, and the refresher stayed `None` forever. The symptom would be the worst
    possible: everything green, and the tokens expiring all the same.

    It registers in `app.router.on_startup` and not through `app.add_event_handler`: that
    method **does not exist** in FastAPI 0.141 (measured: `AttributeError`). The router's
    lists are what the app actually consumes at startup.

    If a loop is already running, the proxy's startup event has already passed and never
    comes back — so it starts immediately.
    """
    import asyncio
    import contextlib

    async def _stop() -> None:
        await service.stop_refresher()

    def _boot() -> None:
        """What has to run when the proxy opens: bind the extra routes, reapply, refresh.

        The binds happen here, and not in `plugin.install()`, because `Router.aresponses`
        and `Router.aanthropic_messages` are per-instance attributes built in
        `Router.__init__` — there is no class attribute to patch, and by the time this
        package loads the Router already exists. See `plugin.bind_responses_route`.
        """
        from ..plugin import bind_messages_route, bind_responses_route

        router = service.router_source()
        # Reported, not suppressed. A bind that fails here costs the route its spend
        # logging and says nothing: the turn still answers, from LiteLLM's native path,
        # so the only visible symptom is a row that never appears. That symptom took
        # several deploys to attribute.
        for name, bind in (
            ("aresponses", bind_responses_route),
            ("aanthropic_messages", bind_messages_route),
        ):
            try:
                bound = bind(router)
            except Exception:
                verbose_proxy_logger.exception(
                    "mysubs: binding %s failed; that route will not log", name
                )
            else:
                verbose_proxy_logger.warning(
                    "mysubs: %s bound=%s router=%s", name, bound, router is not None
                )
        try:
            service.reapply()
        except Exception:
            verbose_proxy_logger.exception("mysubs: reapply failed; stored models may be missing")
        service.start_refresher()

    async def _boot_when_ready() -> None:
        """Waits for `llm_router` before reapplying.

        Measured: `install()` runs **inside** `proxy_startup_event`, with a loop already
        running but **before** `llm_router` is built — instrumented, the immediate
        `reapply()` returned 0 and the stored models never came back. Registering in
        `on_startup` does not work either: at that point the lifespan has already begun and
        the list is no longer consumed.

        So the Router is waited for in a task, with a ceiling: an unbounded wait would leave
        a task alive forever on a proxy that never creates it.
        """
        for _ in range(_ROUTER_WAIT_TRIES):
            if service.router_source() is not None:
                break
            await asyncio.sleep(_ROUTER_WAIT_S)
        _boot()

    router = getattr(app, "router", None)
    shutdown = getattr(router, "on_shutdown", None)
    if isinstance(shutdown, list):
        shutdown.append(_stop)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop — imported by a test or by `mysubs-setup`. The startup event is still to
        # come and it is what runs this.
        startup = getattr(router, "on_startup", None)
        if isinstance(startup, list):
            startup.append(_boot)
        return
    # With a loop: we are in the proxy's startup. The Router does not exist yet, so the
    # reapplication waits for it instead of running now and finding nothing.
    with contextlib.suppress(RuntimeError):
        asyncio.get_running_loop().create_task(_boot_when_ready())
