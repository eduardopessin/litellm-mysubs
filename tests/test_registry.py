"""Registry guards.

Each test here corresponds to a defect that existed in production. The order in which
they appear is the order in which they were found: each one only became visible after
the previous one was fixed.
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
    """Entry as LiteLLM loads it from config.yaml: id equal to model_name."""
    return {"model_name": name, "model_info": {"id": name}, "litellm_params": {"model": upstream}}


def shadow(name: str, upstream: str, digest: str) -> dict[str, Any]:
    """Copy materialised by wildcard resolution: id is a hash."""
    return {"model_name": name, "model_info": {"id": digest}, "litellm_params": {"model": upstream}}


@pytest.fixture
def router() -> FakeRouter:
    return FakeRouter(
        [
            declared("claude-opus-5", "anthropic/claude-opus-5"),
            shadow("claude-opus-5", "anthropic/claude-opus-5", "80c06503"),
            # Alias: the model_name differs from the model it points at.
            declared("claude-opus", "anthropic/claude-opus-4-8"),
            shadow("claude-opus", "anthropic/claude-opus-4-8", "d63bb200"),
            # Name with no twin in the config — may be a new model of the family.
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
        """An alias's shadow inherits the alias's upstream, not the requested name.

        An implementation that requires ``upstream == requested_name`` leaves the six
        aliases out and the leak persists in them. That is exactly what happened.
        """
        registry = ModelRegistry(router)
        assert registry.evict_shadow("claude-opus", only_if_declared=True) == 1
        survivors = [d for d in router.model_list if d["model_name"] == "claude-opus"]
        assert len(survivors) == 1
        assert survivors[0]["model_info"]["id"] == "claude-opus"

    def test_success_path_spares_undeclared_name(self, router: FakeRouter) -> None:
        """A new model of the family has to survive the success path.

        This is what allows serving a model released today without editing the config;
        removing it here would void the value of the wildcard.
        """
        registry = ModelRegistry(router)
        assert registry.evict_shadow("claude-opus-6", only_if_declared=True) == 0
        assert any(d["model_name"] == "claude-opus-6" for d in router.model_list)

    def test_error_path_removes_undeclared_name(self, router: FakeRouter) -> None:
        """Once upstream refuses the name, the entry goes."""
        registry = ModelRegistry(router)
        assert registry.evict_shadow("claude-opus-6", only_if_declared=False) == 1
        assert not any(d["model_name"] == "claude-opus-6" for d in router.model_list)

    def test_never_removes_declared_entry(self, router: FakeRouter) -> None:
        """No path may delete an entry from the config."""
        registry = ModelRegistry(router)
        for only_if_declared in (True, False):
            registry.evict_shadow("gpt-5", only_if_declared=only_if_declared)
        assert sum(d["model_name"] == "gpt-5" for d in router.model_list) == 1

    def test_no_shadow_leaves_router_untouched(self, router: FakeRouter) -> None:
        """With nothing to remove the list is not rewritten: a gratuitous set_model_list
        is a window for concurrent requests to see a half-built list."""
        registry = ModelRegistry(router)
        assert registry.evict_shadow("does-not-exist") == 0
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

        # Reapplying replaces, it does not accumulate.
        registry.apply([{"model_name": "claude-haiku-4-5", "litellm_params": {"model": "y"}}])
        assert [d["model_name"] for d in registry.managed()] == ["claude-haiku-4-5"]
        assert len(router.model_list) == before + 1

    def test_marks_entries_so_they_survive_eviction(self, router: FakeRouter) -> None:
        """A plugin entry must not be mistaken for a wildcard shadow."""
        registry = ModelRegistry(router)
        registry.apply([{"model_name": "claude-opus-5", "litellm_params": {"model": "z"}}])
        registry.evict_shadow("claude-opus-5", only_if_declared=True)
        managed = [d for d in router.model_list if d.get("model_info", {}).get("managed_by")]
        assert [d["model_info"]["managed_by"] for d in managed] == [MANAGED_BY]
