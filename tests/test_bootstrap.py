"""Plugin startup: do not disturb what was already there.

The risk of this layer is not failing — it is working too much. A plugin that installs
itself in the request path of an installation that already serves models has to be
invisible to them.
"""

from __future__ import annotations

from typing import Any

from litellm_mysubs.bootstrap import Bootstrap, disabled
from litellm_mysubs.credentials.store import Credential
from litellm_mysubs.registry import ModelRegistry

CONFIG_DEPLOYMENTS = [
    {"model_name": "gpt-4o", "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-x"}},
    {
        "model_name": "my-llama",
        "litellm_params": {"model": "openai/llama-3", "api_base": "http://gpu.lan:8000"},
    },
]


class EmptyStore:
    owns_refresh = True

    def get(self, provider: str) -> Credential | None:
        return None

    def set(self, provider: str, credential: Credential) -> None: ...

    def delete(self, provider: str) -> None: ...

    def reload(self) -> bool:
        return False


class ConnectedStore(EmptyStore):
    def get(self, provider: str) -> Credential | None:
        if provider == "anthropic":
            return Credential(provider="anthropic", access_token="AT")
        return None


class UnreadableStore(EmptyStore):
    def get(self, provider: str) -> Credential | None:
        raise PermissionError("0644, should be 0600")


class FakeRouter:
    def __init__(self) -> None:
        self.model_list: list[dict[str, Any]] = [dict(d) for d in CONFIG_DEPLOYMENTS]

    def set_model_list(self, model_list: list[dict[str, Any]]) -> None:
        self.model_list = model_list


class FakeApp:
    def __init__(self, existing: list[str] | None = None) -> None:
        self.routes = [type("R", (), {"path": p})() for p in (existing or [])]
        self.mounted: list[str] = []

    def mount(self, path: str, app: Any) -> None:
        self.mounted.append(path)
        self.routes.append(type("R", (), {"path": path})())


class TestInertWithoutSubscriptions:
    def test_no_credentials_means_no_patch(self) -> None:
        """A plugin that is installed and has no credentials has to be indistinguishable
        from an absent plugin. With no connected subscription the patch would only add a
        wrapper that always delegates to the original — cost without benefit, in the path
        of every request."""
        boot = Bootstrap()
        assert boot.should_patch(EmptyStore()) is False
        assert boot.patch(EmptyStore()) is False
        assert boot.patched is False

    def test_one_connected_subscription_is_enough(self) -> None:
        assert Bootstrap().should_patch(ConnectedStore()) is True

    def test_an_unreadable_store_is_not_proof_of_credentials(self) -> None:
        """A file with loose permissions — which the store refuses, and rightly so —
        cannot be read as "there are connected subscriptions"."""
        assert Bootstrap().should_patch(UnreadableStore()) is False

    def test_the_kill_switch_works_without_editing_yaml(self) -> None:
        """When the plugin is the suspect of a problem, an environment variable is faster
        and more reversible than uninstalling."""
        assert disabled({"MYSUBS_DISABLE": "1"}) is True
        assert disabled({"MYSUBS_DISABLE": "true"}) is True
        assert disabled({}) is False
        assert disabled({"MYSUBS_DISABLE": "0"}) is False


class TestExistingRoutingSurvives:
    def test_applying_subscription_models_keeps_the_config_ones(self) -> None:
        """`apply` replaces the plugin's entries. Preserving the `config.yaml` ones is what
        stops an "apply" from erasing the routing of whoever installed us."""
        router = FakeRouter()
        ModelRegistry(router=router).apply(
            [
                {
                    "model_name": "claude-opus-5",
                    "litellm_params": {"model": "anthropic/claude-opus-5"},
                }
            ]
        )
        names = [d["model_name"] for d in router.model_list]
        assert "gpt-4o" in names
        assert "my-llama" in names
        assert "claude-opus-5" in names

    def test_config_deployments_are_returned_untouched(self) -> None:
        """Surviving is not enough: they have to survive **identical**. A key added to
        somebody else's deployment can change its authentication mode."""
        router = FakeRouter()
        ModelRegistry(router=router).apply([{"model_name": "x", "litellm_params": {"model": "y"}}])
        kept = [d for d in router.model_list if d["model_name"] in ("gpt-4o", "my-llama")]
        assert kept == CONFIG_DEPLOYMENTS

    def test_only_our_entries_are_replaced_on_reapply(self) -> None:
        router = FakeRouter()
        registry = ModelRegistry(router=router)
        registry.apply([{"model_name": "a", "litellm_params": {"model": "anthropic/a"}}])
        registry.apply([{"model_name": "b", "litellm_params": {"model": "anthropic/b"}}])
        names = [d["model_name"] for d in router.model_list]
        assert names.count("gpt-4o") == 1
        assert "a" not in names
        assert "b" in names


class TestMounting:
    def test_mounting_adds_a_prefix_and_touches_nothing_else(self) -> None:
        app = FakeApp(["/ui", "/swagger", "/v1/models"])
        Bootstrap().mount(app, EmptyStore())
        assert app.mounted == ["/mysubs"]
        assert [r.path for r in app.routes][:3] == ["/ui", "/swagger", "/v1/models"]

    def test_an_occupied_prefix_is_not_mounted_over(self) -> None:
        """Two sub-apps on the same prefix: the second one catches the requests and the
        first becomes unreachable, with no error saying so."""
        app = FakeApp(["/mysubs"])
        boot = Bootstrap()
        assert boot.mount(app, EmptyStore()) is False
        assert app.mounted == []

    def test_mounting_twice_is_a_noop(self) -> None:
        app = FakeApp()
        boot = Bootstrap()
        boot.mount(app, EmptyStore())
        boot.mount(app, EmptyStore())
        assert app.mounted == ["/mysubs"]


class TestProxyContract:
    """What the proxy requires of a callback. Verified against the installed LiteLLM."""

    def test_the_exported_handler_is_an_instance_not_a_class(self) -> None:
        """The proxy refuses a class with `ValueError` and **does not start**.

        Measured: `callbacks: ["litellm_mysubs.MySubs"]` made `load_config` raise and the
        process exit with code 3. The LiteLLM message names the fix — export an instance —
        because a non-dispatchable callback would load without complaint and be ignored on
        every request.
        """
        import litellm_mysubs

        assert not isinstance(litellm_mysubs.proxy_handler_instance, type)

    def test_the_handler_is_a_custom_logger(self) -> None:
        """`isinstance(loaded, CustomLogger)` is the proxy's acceptance condition. An
        object with the right hooks but without the inheritance is refused at startup —
        the inheritance is not style, it is contract."""
        from litellm.integrations.custom_logger import CustomLogger

        import litellm_mysubs

        assert isinstance(litellm_mysubs.proxy_handler_instance, CustomLogger)

    def test_the_installer_writes_the_instance_path(self) -> None:
        """The path in `config.yaml` has to be the one the proxy accepts, otherwise the
        installer leaves the installation in a state that does not start."""
        from litellm_mysubs.setup_cli import CALLBACK_PATH

        assert CALLBACK_PATH.endswith("proxy_handler_instance")


class TestStoreIsWired:
    """A patch without a store is a useless patch — and the symptom shows up far from the
    cause."""

    def test_patching_configures_the_plugin_store(self) -> None:
        """Measured on the real proxy: `install()` without `configure()` leaves
        `_access_token` returning "", the request goes out with `Authorization: Bearer `
        and httpx refuses it with `Illegal header value b'Bearer '`. The error comes from
        the provider's client, not from the plugin, and says nothing about the missing
        credential.
        """
        from litellm_mysubs import plugin

        store = ConnectedStore()
        boot = Bootstrap()
        try:
            assert boot.patch(store) is True
            assert plugin._state.store is store
        finally:
            plugin.uninstall()

    def test_the_router_entrypoint_is_patched(self) -> None:
        """The proxy routes through `Router.acompletion`, not through the module functions.

        `route_llm_request.py:487` does `getattr(llm_router, route_type)(**data)`. Without
        this patch, a subscription model resolved the deployment and went to the native
        client before any module function was touched. The original `sitecustomize.py` says
        so in the comment on line 2812; the port only patched the module functions.
        """
        from litellm.router import Router

        from litellm_mysubs import plugin

        original = Router.acompletion
        try:
            plugin.configure(store=ConnectedStore())
            plugin.install()
            assert Router.acompletion is not original
            assert Router.acompletion.__name__ == "_wrapped_router_acompletion"
        finally:
            plugin.uninstall()
        assert Router.acompletion is original
