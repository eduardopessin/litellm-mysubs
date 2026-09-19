"""Plugin startup, through a `CustomLogger` that `config.yaml` loads.

## Why not a `.pth` with an import hook

The first version of this module registered a finder in `sys.meta_path` from a `.pth` file,
so as to react to the proxy being imported. It was measured and abandoned:

    finder that raises at meta_path[0]  ->  `import secrets` blows up with RuntimeError

A `.pth` runs in **every** Python process in the environment, and a finder at
`meta_path[0]` sees **every** import. A defect in it does not break the plugin — it breaks
the interpreter, for `pip`, `pytest` and any script in the same environment, with the error
showing up before any log exists. For a package whose purpose is to add models, that risk
is out of proportion.

## What is done instead

`MySubs` is a `litellm.integrations.custom_logger.CustomLogger`, the extension LiteLLM
documents. It goes in as one line in `config.yaml`:

    litellm_settings:
      callbacks: ["litellm_mysubs.MySubs"]

The proxy instantiates it during startup, inside its own flow. Consequences:

- **Existing routing is untouched.** The `config.yaml` deployments are not read, reordered
  or replaced. `ModelRegistry.apply` preserves everything without
  `model_info.managed_by == "mysubs"`, and that is the only mark the plugin writes.
- **The patch is applied only if there is something to serve.** With no connected
  credentials and no applied models, `install()` does not run: the package is present and
  inert.
- **Uninstalling is deleting the line.** No files in `site-packages` to hunt down.

## And the UI

Mounted at `/mysubs` at startup, by `app.mount()` — the same route the proxy uses for `/ui`
and `/swagger`. Mounting adds a new prefix; no existing route changes destination.

The moment is the construction of the `CustomLogger`, which the proxy does **inside**
`proxy_startup_event`, before the `yield` that opens the server to traffic. Measured: by
the end of the lifespan the callback already exists and `llm_router` is already ready. The
previous version mounted on the first `async_pre_call_hook`, and that left `/mysubs`
answering 404 until someone made an inference request — a 404 that teaches nothing to
whoever restarted the proxy and opened the page first.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Final

#: Turns everything off without editing `config.yaml`. It exists for when the plugin is the
#: suspect in a problem: an environment variable is faster and more reversible than
#: uninstalling.
DISABLE_ENV: Final = "MYSUBS_DISABLE"


#: Prefix where the UI is mounted. Resolved late, not at import: it is `ui/app.py` that
#: decides it from the environment, and importing it here at the top would drag FastAPI
#: into a module that has to be importable without it.
def ui_path() -> str:
    from .ui.app import mount_path

    return mount_path()


def disabled(environ: dict[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return str(env.get(DISABLE_ENV, "")).strip().lower() in ("1", "true", "yes")


class Bootstrap:
    """Applies the plugin exactly once, and records what happened.

    Separate from the `CustomLogger` so it is testable without LiteLLM: what lives here is
    the decision of *whether* and *what*, not the fit into the proxy.
    """

    def __init__(self) -> None:
        self.patched = False
        self.mounted = False
        self.menu = ""
        self.error = ""
        self._lock = threading.Lock()

    # -- decision ----------------------------------------------------------------

    def should_patch(self, store: Any) -> bool:
        """Whether it is worth touching the request path.

        With no subscription connected there is no subscription model to serve, and the
        patch would only add a wrapper that always delegates to the original. An installed
        plugin with no credentials has to be indistinguishable from an absent one.
        """
        try:
            return any(store.get(provider) is not None for provider in _providers())
        except Exception:
            # An unreadable store is no proof that credentials exist.
            return False

    # -- application -------------------------------------------------------------

    def _inject_menu(self, app: Any) -> None:
        """Adds the item to LiteLLM's menu. Failing here is not a failure.

        The page works by direct URL; the button is a convenience. The UI chunk has a name
        that is a build hash, and a new LiteLLM version may go unrecognized — in that case
        the reason is recorded and the run continues.
        """
        try:
            self.menu = _inject_menu_impl(app)
        except Exception as error:
            self.menu = f"not injected: {type(error).__name__}: {error}"

    def patch(self, store: Any) -> bool:
        """Applies the monkey-patch if a subscription is connected. Idempotent."""
        with self._lock:
            if self.patched or disabled() or not self.should_patch(store):
                return False
            try:
                from . import plugin

                # Without this the plugin has nowhere to get the credential from:
                # `_access_token` returns "" and the request goes out with
                # `Authorization: Bearer `, which httpx refuses with
                # `Illegal header value b'Bearer '`. `install()` on its own applies the
                # patch and leaves it useless — and the symptom shows up far from the
                # cause, in the provider's client.
                plugin.configure(store=store)
                plugin.install()
                self.patched = True
                return True
            except Exception as error:
                self.error = f"patch: {type(error).__name__}: {error}"
                return False

    def mount(self, app: Any, store: Any) -> bool:
        """Mounts the UI. Idempotent, and never over an already occupied prefix."""
        with self._lock:
            if self.mounted or disabled():
                return False
            try:
                if _already_mounted(app, ui_path()):
                    # Another process or a manual mount got there first. Mounting on top
                    # would create two sub-apps on the same prefix, with the second
                    # catching the requests and the first left unreachable.
                    self.mounted = True
                    return False
                from .ui.install import install as install_ui

                install_ui(app, store=store)
                self.mounted = True
                self._inject_menu(app)
                return True
            except Exception as error:
                self.error = f"mount: {type(error).__name__}: {error}"
                return False


def _inject_menu_impl(app: Any) -> str:
    from .ui.menu import install_menu

    return install_menu(app).reason


def _providers() -> tuple[str, ...]:
    from .credentials.store import PROVIDER_IDS

    return PROVIDER_IDS


def _already_mounted(app: Any, path: str) -> bool:
    return any(getattr(route, "path", None) == path for route in getattr(app, "routes", []))
