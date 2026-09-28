"""LiteLLM symbols the plugin depends on.

The plugin runs inside somebody else's image and touches internals that are not part of
any public API. When LiteLLM renames or moves one of these symbols, the plugin stops
applying the patch — and, without this file, that is only discovered in production, with
requests going to the wrong upstream.

Every assertion here answers "what breaks if this changes".

``pytest.importorskip`` keeps the file usable in the development environment, where
LiteLLM is not a mandatory dependency. In CI, every ``test`` job installs it explicitly
(1.101.0 and the current release), so the skip does not happen there.
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


class TestProxyModelInfo:
    """`catalog/deployments.py` states a declared output ceiling in ``model_info`` and relies
    on it surviving the Router and the proxy filling only the keys a deployment lacks. If
    either stops holding, `/model/info` goes back to LiteLLM's price map — 128000 on
    `claude-opus-4-5`, whose ceiling is 64000.

    Through the real Router, not a hand-built dict: the Router stores ``model_info`` with
    ``exclude_none``, which is what made a stated ``None`` vanish in 0.1.13 while a test that
    skipped the Router passed."""

    def test_a_declared_ceiling_survives_the_router_and_enrichment(self) -> None:
        proxy_server = pytest.importorskip("litellm.proxy.proxy_server")
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "m",
                    "litellm_params": {"model": "anthropic/claude-opus-5-5"},
                    "model_info": {"id": "m", "max_output_tokens": 64000},
                }
            ]
        )
        stored = dict(router.model_list[0])
        out = proxy_server._enrich_model_info_with_litellm_data(stored)
        assert out["model_info"]["max_output_tokens"] == 64000


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
