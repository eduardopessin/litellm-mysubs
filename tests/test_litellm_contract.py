"""LiteLLM symbols the plugin depends on.

The plugin runs inside somebody else's image and touches internals that are not part of
any public API. When LiteLLM renames or moves one of these symbols, the plugin stops
applying the patch — and, without this file, that is only discovered in production, with
requests going to the wrong upstream.

Every assertion here answers "what breaks if this changes".

``pytest.importorskip`` keeps the file usable in the development environment, where
LiteLLM is not a mandatory dependency. In CI, the ``litellm-contract`` job installs it
explicitly, so the skip does not happen there.
"""

from __future__ import annotations

import inspect

import pytest

litellm = pytest.importorskip("litellm", reason="litellm not installed in this environment")


class TestRouterSurface:
    """The registry injects deployments through here."""

    def test_router_has_set_model_list(self) -> None:
        assert callable(getattr(litellm.Router, "set_model_list", None))

    def test_router_instances_expose_model_list(self) -> None:
        router = litellm.Router(model_list=[])
        assert isinstance(router.model_list, list)

    def test_router_acompletion_is_async(self) -> None:
        """The name guard wraps this method; if it stops being async, it breaks."""
        assert inspect.iscoroutinefunction(litellm.Router.acompletion)

    def test_router_acompletion_signature(self) -> None:
        """We wrap it with (self, model, messages, stream=False, **kwargs)."""
        params = inspect.signature(litellm.Router.acompletion).parameters
        assert {"model", "messages"} <= set(params)


class TestModuleSurface:
    def test_acompletion_entrypoints_exist(self) -> None:
        assert inspect.iscoroutinefunction(litellm.acompletion)
        assert inspect.iscoroutinefunction(litellm.main.acompletion)

    def test_route_request_module_exists(self) -> None:
        """The /v1/responses route is only caught at this boundary."""
        from litellm.proxy import route_llm_request

        assert inspect.iscoroutinefunction(route_llm_request.route_request)


class TestOfficialExtensionPoints:
    """Public surfaces — preferable to monkey-patching wherever they reach."""

    def test_custom_provider_map_is_supported(self) -> None:
        from litellm.utils import custom_llm_setup

        assert callable(custom_llm_setup)
        assert isinstance(litellm.custom_provider_map, list)

    def test_custom_llm_base_class_shape(self) -> None:
        from litellm import CustomLLM

        for method in ("completion", "acompletion", "streaming", "astreaming"):
            assert hasattr(CustomLLM, method), method

    def test_exceptions_used_by_the_guard(self) -> None:
        """The guard raises NotFoundError for a name the upstream has already refused."""
        assert issubclass(litellm.NotFoundError, Exception)
