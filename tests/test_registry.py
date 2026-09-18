"""Guardas do registry.

Cada teste aqui corresponde a um defeito que existiu em produção. A ordem em que
aparecem é a ordem em que foram descobertos: cada um só ficou visível depois de o
anterior estar corrigido.
"""

from __future__ import annotations

from typing import Any

import pytest

from litellm_mysubs.registry import MANAGED_BY, ModelRegistry, is_declared


class FakeRouter:
    def __init__(self, model_list: list[dict[str, Any]]) -> None:
        self.model_list = model_list
        self.set_calls = 0

    def set_model_list(self, model_list: list[dict[str, Any]]) -> None:
        self.model_list = model_list
        self.set_calls += 1


def declared(name: str, upstream: str) -> dict[str, Any]:
    """Entrada como o LiteLLM a carrega do config.yaml: id igual ao model_name."""
    return {"model_name": name, "model_info": {"id": name}, "litellm_params": {"model": upstream}}


def shadow(name: str, upstream: str, digest: str) -> dict[str, Any]:
    """Cópia materializada pela resolução de wildcard: id em hash."""
    return {"model_name": name, "model_info": {"id": digest}, "litellm_params": {"model": upstream}}


@pytest.fixture
def router() -> FakeRouter:
    return FakeRouter(
        [
            declared("claude-opus-5", "anthropic/claude-opus-5"),
            shadow("claude-opus-5", "anthropic/claude-opus-5", "80c06503"),
            # Alias: o model_name difere do modelo a que aponta.
            declared("claude-opus", "anthropic/claude-opus-4-8"),
            shadow("claude-opus", "anthropic/claude-opus-4-8", "d63bb200"),
            # Nome sem gémeo no config — pode ser um modelo novo da família.
            shadow("claude-opus-6", "anthropic/claude-opus-6", "aa11bb22"),
            declared("gpt-5", "openai/gpt-5.5"),
        ]
    )


class TestEvictShadow:
    def test_removes_shadow_of_declared_model(self, router: FakeRouter) -> None:
        registry = ModelRegistry(router)
        assert registry.evict_shadow("claude-opus-5", only_if_declared=True) == 1
        survivors = [d for d in router.model_list if d["model_name"] == "claude-opus-5"]
        assert len(survivors) == 1
        assert is_declared(survivors[0])

    def test_removes_shadow_of_alias(self, router: FakeRouter) -> None:
        """A sombra de um alias herda o upstream do alias, não o nome pedido.

        Uma implementação que exija ``upstream == nome_pedido`` deixa os seis aliases de
        fora e o leak continua neles. Foi exactamente o que aconteceu.
        """
        registry = ModelRegistry(router)
        assert registry.evict_shadow("claude-opus", only_if_declared=True) == 1
        survivors = [d for d in router.model_list if d["model_name"] == "claude-opus"]
        assert len(survivors) == 1
        assert survivors[0]["model_info"]["id"] == "claude-opus"

    def test_success_path_spares_undeclared_name(self, router: FakeRouter) -> None:
        """Um modelo novo da família tem de sobreviver ao caminho de sucesso.

        É o que permite servir um modelo lançado hoje sem editar o config; removê-lo aqui
        anulava o valor do wildcard.
        """
        registry = ModelRegistry(router)
        assert registry.evict_shadow("claude-opus-6", only_if_declared=True) == 0
        assert any(d["model_name"] == "claude-opus-6" for d in router.model_list)

    def test_error_path_removes_undeclared_name(self, router: FakeRouter) -> None:
        """Depois de o upstream recusar o nome, a entrada sai."""
        registry = ModelRegistry(router)
        assert registry.evict_shadow("claude-opus-6", only_if_declared=False) == 1
        assert not any(d["model_name"] == "claude-opus-6" for d in router.model_list)

    def test_never_removes_declared_entry(self, router: FakeRouter) -> None:
        """Nenhum caminho pode apagar uma entrada do config."""
        registry = ModelRegistry(router)
        for only_if_declared in (True, False):
            registry.evict_shadow("gpt-5", only_if_declared=only_if_declared)
        assert sum(d["model_name"] == "gpt-5" for d in router.model_list) == 1

    def test_no_shadow_leaves_router_untouched(self, router: FakeRouter) -> None:
        """Sem nada a remover não se reescreve a lista: um set_model_list gratuito é uma
        janela para pedidos concorrentes verem uma lista a meio."""
        registry = ModelRegistry(router)
        assert registry.evict_shadow("nao-existe") == 0
        assert router.set_calls == 0


class TestNotFoundMemory:
    def test_remembers_once(self) -> None:
        registry = ModelRegistry(FakeRouter([]))
        assert registry.remember_not_found("Claude-Ghost") is True
        assert registry.remember_not_found("claude-ghost") is False
        assert registry.is_known_bad("CLAUDE-GHOST") is True

    def test_unknown_name_is_not_bad(self) -> None:
        assert ModelRegistry(FakeRouter([])).is_known_bad("claude-opus-5") is False


class TestApply:
    def test_replaces_only_managed_entries(self, router: FakeRouter) -> None:
        registry = ModelRegistry(router)
        before = len(router.model_list)

        registry.apply([{"model_name": "claude-sonnet-5", "litellm_params": {"model": "x"}}])
        assert len(registry.managed()) == 1
        assert len(router.model_list) == before + 1

        # Reaplicar substitui, não acumula.
        registry.apply([{"model_name": "claude-haiku-4-5", "litellm_params": {"model": "y"}}])
        assert [d["model_name"] for d in registry.managed()] == ["claude-haiku-4-5"]
        assert len(router.model_list) == before + 1

    def test_marks_entries_so_they_survive_eviction(self, router: FakeRouter) -> None:
        """Uma entrada do plugin não pode ser confundida com uma sombra de wildcard."""
        registry = ModelRegistry(router)
        registry.apply([{"model_name": "claude-opus-5", "litellm_params": {"model": "z"}}])
        registry.evict_shadow("claude-opus-5", only_if_declared=True)
        managed = [d for d in router.model_list if d.get("model_info", {}).get("managed_by")]
        assert [d["model_info"]["managed_by"] for d in managed] == [MANAGED_BY]
