"""Símbolos do LiteLLM de que o plugin depende.

O plugin corre dentro da imagem de outra pessoa e toca em internals que não fazem parte
de nenhuma API pública. Quando o LiteLLM renomeia ou move um destes símbolos, o plugin
deixa de aplicar o patch — e, sem este ficheiro, isso só se descobre em produção, com
pedidos a ir para o upstream errado.

Cada asserção aqui responde a "o que é que parte se isto mudar".

``pytest.importorskip`` mantém o ficheiro utilizável no ambiente de desenvolvimento, onde
o LiteLLM não é dependência obrigatória. No CI, o job ``litellm-contract`` instala-o
explicitamente, por isso lá o skip não acontece.
"""

from __future__ import annotations

import inspect

import pytest

litellm = pytest.importorskip("litellm", reason="litellm não instalado neste ambiente")


class TestRouterSurface:
    """O registry injecta deployments por aqui."""

    def test_router_has_set_model_list(self) -> None:
        assert callable(getattr(litellm.Router, "set_model_list", None))

    def test_router_instances_expose_model_list(self) -> None:
        router = litellm.Router(model_list=[])
        assert isinstance(router.model_list, list)

    def test_router_acompletion_is_async(self) -> None:
        """A guarda de nomes embrulha este método; se deixar de ser async, parte."""
        assert inspect.iscoroutinefunction(litellm.Router.acompletion)

    def test_router_acompletion_signature(self) -> None:
        """Embrulhamos com (self, model, messages, stream=False, **kwargs)."""
        params = inspect.signature(litellm.Router.acompletion).parameters
        assert {"model", "messages"} <= set(params)


class TestModuleSurface:
    def test_acompletion_entrypoints_exist(self) -> None:
        assert inspect.iscoroutinefunction(litellm.acompletion)
        assert inspect.iscoroutinefunction(litellm.main.acompletion)

    def test_route_request_module_exists(self) -> None:
        """A rota /v1/responses só é apanhada neste boundary."""
        from litellm.proxy import route_llm_request

        assert inspect.iscoroutinefunction(route_llm_request.route_request)


class TestOfficialExtensionPoints:
    """Superfícies públicas — preferíveis ao monkey-patch onde cheguem."""

    def test_custom_provider_map_is_supported(self) -> None:
        from litellm.utils import custom_llm_setup

        assert callable(custom_llm_setup)
        assert isinstance(litellm.custom_provider_map, list)

    def test_custom_llm_base_class_shape(self) -> None:
        from litellm import CustomLLM

        for method in ("completion", "acompletion", "streaming", "astreaming"):
            assert hasattr(CustomLLM, method), method

    def test_exceptions_used_by_the_guard(self) -> None:
        """A guarda levanta NotFoundError para um nome já recusado pelo upstream."""
        assert issubclass(litellm.NotFoundError, Exception)
