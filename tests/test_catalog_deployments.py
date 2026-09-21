"""Conversion of discovered models into Router deployments."""

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
    def test_public_name_carries_the_subscription_and_the_wire_name_the_family(self) -> None:
        """Without a prefix on the wire, the name falls into the native provider's wildcard
        resolution, which materialises a deployment for any name before talking to the
        upstream. The public name carries the subscription because two subscriptions serve
        the same model."""
        out = to_deployment(
            DiscoveredModel(
                wire_name="gpt-6-astra", suggested_name="gpt-6", verified=True, family="openai"
            ),
            "openai-codex",
        )
        assert out["model_name"] == "mysubs/codex/gpt-6"
        assert out["litellm_params"]["model"] == "openai/gpt-6-astra"

    def test_wire_name_may_differ_from_the_public_one(self) -> None:
        """`gpt-6` resolves to `gpt-6-astra` upstream. It is the bare name that the client
        asks for and that `/spend/logs` records — if it were the wire one, the billing
        would name a model nobody asked for."""
        out = to_deployment(
            DiscoveredModel(wire_name="gpt-6-astra", suggested_name="gpt-6", verified=True),
            "openai-codex",
        )
        assert out["model_name"] != out["litellm_params"]["model"].split("/", 1)[1]

    def test_each_provider_gets_its_own_prefix(self) -> None:
        model = DiscoveredModel(wire_name="x", suggested_name="x", verified=True)
        assert to_deployment(model, "anthropic")["litellm_params"]["model"] == "anthropic/x"
        assert to_deployment(model, "google-antigravity")["litellm_params"]["model"] == "openai/x"

    def test_the_provider_is_declared_not_left_to_be_inferred(self) -> None:
        """The UI reads `custom_llm_provider` to pick the model's icon.

        Without it LiteLLM infers from `model_name`, and `get_llm_provider` **raises** on
        the public forms — measured, `BadRequestError` for `mysubs/codex/gpt-5.5` and
        `mysubs/antigravity/gemini-3-flash`. The Logs tab then showed the generic icon for
        every subscription model, with OpenAI and Google indistinguishable.
        """
        cases = [
            ("openai", "openai-codex", "openai"),
            ("anthropic", "anthropic", "anthropic"),
            ("google", "google-antigravity", "gemini"),
        ]
        for family, provider, expected in cases:
            out = to_deployment(
                DiscoveredModel(
                    wire_name="m", suggested_name="m", verified=True, family=family
                ),
                provider,
            )
            assert out["litellm_params"]["custom_llm_provider"] == expected

    def test_the_declared_provider_matches_the_wire_prefix(self) -> None:
        """Icon and price must not be two independent guesses.

        The wire prefix is the one with a price table behind it (`gemini/` prices,
        `openai/` does not for a Gemini model), so declaring anything else would make the
        row show one provider and bill against another.
        """
        families = (
            ("google", "google-antigravity"),
            ("anthropic", "anthropic"),
            ("openai", "openai-codex"),
        )
        for family, provider in families:
            out = to_deployment(
                DiscoveredModel(
                    wire_name="m", suggested_name="m", verified=True, family=family
                ),
                provider,
            )
            params = out["litellm_params"]
            assert params["model"].split("/", 1)[0] == params["custom_llm_provider"]


class TestSelection:
    def test_unverified_models_are_kept_by_default(self) -> None:
        """A model may be unverified just because the network failed during the probe.
        Discarding it silently would make the list lie about what the subscription
        serves."""
        models = [
            DiscoveredModel(wire_name="a", suggested_name="a", verified=True),
            DiscoveredModel(wire_name="b", suggested_name="b", verified=False, note="network"),
        ]
        assert len(to_deployments(models, "anthropic")) == 2

    def test_only_verified_drops_the_unproven(self) -> None:
        models = [
            DiscoveredModel(wire_name="a", suggested_name="a", verified=True),
            DiscoveredModel(wire_name="b", suggested_name="b", verified=False, note="network"),
        ]
        chosen = to_deployments(models, "anthropic", only_verified=True)
        assert [d["model_name"] for d in chosen] == ["mysubs/claudecode/a"]


class TestRegistryIntegration:
    def test_applied_deployments_are_marked_and_survive_a_roundtrip(self) -> None:
        """The registry is what declares ownership: without `managed_by`, `apply` would
        erase its own entries on the next call for not recognising them as its own."""
        router = FakeRouter()
        reg = registry.ModelRegistry(router=router)  # type: ignore[arg-type]
        models = [DiscoveredModel(wire_name="gpt-6-astra", suggested_name="gpt-6", verified=True)]

        reg.apply(to_deployments(models, "openai-codex"))
        assert [d["model_name"] for d in reg.managed()] == ["mysubs/codex/gpt-6"]

        reg.apply(to_deployments(models, "openai-codex"))
        assert len(reg.managed()) == 1

    def test_config_entries_are_never_replaced(self) -> None:
        """A `config.yaml` deployment with the same name cannot be eaten by the plugin."""
        declared = {"model_name": "mysubs/codex/gpt-6", "litellm_params": {"model": "openai/other"}}
        router = FakeRouter([declared])
        reg = registry.ModelRegistry(router=router)  # type: ignore[arg-type]

        reg.apply(
            to_deployments(
                [DiscoveredModel(wire_name="gpt-6-astra", suggested_name="gpt-6", verified=True)],
                "openai-codex",
            )
        )
        assert declared in router.model_list
