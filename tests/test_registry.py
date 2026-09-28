"""Registry: `apply` replaces the plugin's entries and leaves the config's alone."""

from __future__ import annotations

from typing import Any

import pytest

from litellm_mysubs.registry import ModelRegistry


class FakeRouter:
    def __init__(self, model_list: list[dict[str, Any]]) -> None:
        self.model_list = model_list

    def set_model_list(self, model_list: list[dict[str, Any]]) -> None:
        self.model_list = model_list


def declared(name: str, upstream: str) -> dict[str, Any]:
    """Entry as LiteLLM loads it from config.yaml."""
    return {"model_name": name, "model_info": {"id": name}, "litellm_params": {"model": upstream}}


@pytest.fixture
def router() -> FakeRouter:
    return FakeRouter(
        [
            declared("claude-opus-5", "anthropic/claude-opus-5"),
            declared("claude-opus", "anthropic/claude-opus-4-8"),
            declared("gpt-5", "openai/gpt-5.5"),
        ]
    )


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
