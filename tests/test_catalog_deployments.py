"""Conversão de modelos descobertos em deployments do Router."""

from __future__ import annotations

from litellm_mysubs import registry
from litellm_mysubs.catalog.deployments import to_deployment, to_deployments
from litellm_mysubs.catalog.discovery import DiscoveredModel


class FakeRouter:
    def __init__(self, model_list: list[dict[str, object]] | None = None) -> None:
        self.model_list: list[dict[str, object]] = model_list or []

    def set_model_list(self, model_list: list[dict[str, object]]) -> None:
        self.model_list = model_list


class TestDeploymentShape:
    def test_public_name_is_bare_and_wire_name_is_prefixed(self) -> None:
        """Sem prefixo, o nome cai na resolução por wildcard do provider nativo, que
        materializa um deployment para qualquer nome antes de falar com o upstream."""
        out = to_deployment(
            DiscoveredModel(wire_name="gpt-6-astra", suggested_name="gpt-6", verified=True),
            "openai-codex",
        )
        assert out["model_name"] == "gpt-6"
        assert out["litellm_params"]["model"] == "openai/gpt-6-astra"

    def test_wire_name_may_differ_from_the_public_one(self) -> None:
        """`gpt-6` resolve para `gpt-6-astra` upstream. É o nome nu que o cliente pede e
        que o `/spend/logs` regista — se fosse o do fio, a facturação nomearia um modelo
        que ninguém pediu."""
        out = to_deployment(
            DiscoveredModel(wire_name="gpt-6-astra", suggested_name="gpt-6", verified=True),
            "openai-codex",
        )
        assert out["model_name"] != out["litellm_params"]["model"].split("/", 1)[1]

    def test_each_provider_gets_its_own_prefix(self) -> None:
        model = DiscoveredModel(wire_name="x", suggested_name="x", verified=True)
        assert to_deployment(model, "anthropic")["litellm_params"]["model"] == "anthropic/x"
        assert to_deployment(model, "google-antigravity")["litellm_params"]["model"] == "openai/x"


class TestSelection:
    def test_unverified_models_are_kept_by_default(self) -> None:
        """Um modelo pode estar por verificar só porque a rede falhou na sonda. Descartá-lo
        em silêncio faria a lista mentir sobre o que a subscrição serve."""
        models = [
            DiscoveredModel(wire_name="a", suggested_name="a", verified=True),
            DiscoveredModel(wire_name="b", suggested_name="b", verified=False, note="rede"),
        ]
        assert len(to_deployments(models, "anthropic")) == 2

    def test_only_verified_drops_the_unproven(self) -> None:
        models = [
            DiscoveredModel(wire_name="a", suggested_name="a", verified=True),
            DiscoveredModel(wire_name="b", suggested_name="b", verified=False, note="rede"),
        ]
        chosen = to_deployments(models, "anthropic", only_verified=True)
        assert [d["model_name"] for d in chosen] == ["a"]


class TestRegistryIntegration:
    def test_applied_deployments_are_marked_and_survive_a_roundtrip(self) -> None:
        """O registry é que declara a posse: sem `managed_by`, `apply` apagaria as próprias
        entradas na chamada seguinte por não as reconhecer como suas."""
        router = FakeRouter()
        reg = registry.ModelRegistry(router=router)  # type: ignore[arg-type]
        models = [DiscoveredModel(wire_name="gpt-6-astra", suggested_name="gpt-6", verified=True)]

        reg.apply(to_deployments(models, "openai-codex"))
        assert [d["model_name"] for d in reg.managed()] == ["gpt-6"]

        reg.apply(to_deployments(models, "openai-codex"))
        assert len(reg.managed()) == 1

    def test_config_entries_are_never_replaced(self) -> None:
        """Um deployment do `config.yaml` com o mesmo nome não pode ser comido pelo
        plugin."""
        declared = {"model_name": "gpt-6", "litellm_params": {"model": "openai/outro"}}
        router = FakeRouter([declared])
        reg = registry.ModelRegistry(router=router)  # type: ignore[arg-type]

        reg.apply(
            to_deployments(
                [DiscoveredModel(wire_name="gpt-6-astra", suggested_name="gpt-6", verified=True)],
                "openai-codex",
            )
        )
        assert declared in router.model_list
