"""The `litellm_params.model` prefix: present **and** correct.

There are two silent ways of getting it wrong, and this file covers both:

* without a prefix, the name falls into the native provider's wildcard resolution, which
  materialises a deployment for any name before talking to the upstream — 25 shadows
  measured on this installation (see the top of `registry.py`);
* with the *provider* prefix instead of the *family* one, the model drops out of the
  LiteLLM price table: the Models tab shows `input_cost_per_token=0` and `/spend/logs`
  records zero cost for everything the subscription served.

Neither of them raises an error. The last class in this file calls `litellm.completion_cost`
itself, because an assertion on the string only proved that `gemini/` had been written;
what matters to prove is that this prefix has a price and the previous one had none.

No network: the Antigravity catalog comes through `httpx.MockTransport`.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from litellm_mysubs.catalog.deployments import PUBLIC_PREFIX, WIRE_PREFIX, to_deployment
from litellm_mysubs.catalog.discovery import DiscoveredModel, discover
from litellm_mysubs.credentials.store import Credential, ProviderId
from litellm_mysubs.transport.hosts import MODELS_PATH

Handler = Callable[[httpx.Request], httpx.Response]

ANTHROPIC = Credential(provider="anthropic", access_token="tok-a")
CODEX = Credential(
    provider="openai-codex",
    access_token=(
        "eyJhbGciOiJub25lIn0."
        "eyJodHRwczovL2FwaS5vcGVuYWkuY29tL2F1dGgiOnsiY2hhdGdwdF9hY2NvdW50X2lkIjoiYWNjLTEifX0."
    ),
)
GOOGLE = Credential(provider="google-antigravity", access_token="tok-g")


def client(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def wire_of(model: DiscoveredModel, provider: ProviderId) -> str:
    return str(to_deployment(model, provider)["litellm_params"]["model"])


def discovered(wire: str, family: str) -> DiscoveredModel:
    return DiscoveredModel(
        wire_name=wire, suggested_name=wire, verified=True, family=family
    )


class TestFamilyDecidesThePrefix:
    """Who serves is not what is served: Antigravity resells all three families."""

    def test_google_family_gets_the_gemini_prefix(self) -> None:
        """`gemini/` and not `vertex_ai/`: they give the same measured price, but the
        second one requires a GCP project and region, which an Antigravity subscription
        does not have."""
        model = discovered("gemini-2.5-pro", "google")
        assert wire_of(model, "google-antigravity") == "gemini/gemini-2.5-pro"

    def test_anthropic_family_gets_the_anthropic_prefix(self) -> None:
        """The same provider, another family — and therefore another prefix. While both
        carried `openai/`, the two cost zero."""
        model = discovered("claude-sonnet-4-6", "anthropic")
        assert wire_of(model, "google-antigravity") == "anthropic/claude-sonnet-4-6"

    def test_openai_family_gets_the_openai_prefix(self) -> None:
        model = discovered("gpt-oss-120b-medium", "openai")
        assert wire_of(model, "google-antigravity") == "openai/gpt-oss-120b-medium"

    def test_family_beats_the_provider_map(self) -> None:
        """The central assertion: the provider prefix and the family prefix diverge, and it
        is the family one that comes out. Without this, the tests above would pass with the
        old map."""
        model = discovered("gemini-2.5-flash", "google")
        assert WIRE_PREFIX["google-antigravity"] == "openai"
        assert wire_of(model, "google-antigravity").startswith("gemini/")


class TestUnknownFamilyFallsBack:
    """An empty family means "I do not know", and a guess here would trade zero cost for
    wrong cost."""

    def test_empty_family_uses_the_provider_prefix(self) -> None:
        model = discovered("chat_23310", "")
        assert wire_of(model, "google-antigravity") == "openai/chat_23310"
        assert wire_of(model, "anthropic") == "anthropic/chat_23310"

    def test_unknown_family_string_still_uses_the_provider_prefix(self) -> None:
        """A family outside the table cannot become a prefix: that would invent a provider
        LiteLLM does not know, and then not even the native routing would resolve it."""
        model = discovered("x", "cohere")
        assert wire_of(model, "openai-codex") == "openai/x"

    @pytest.mark.parametrize(
        "family", ["", "google", "anthropic", "openai", "unknown", "GOOGLE"]
    )
    @pytest.mark.parametrize("provider", ["anthropic", "openai-codex", "google-antigravity"])
    def test_prefix_is_never_empty(self, family: str, provider: ProviderId) -> None:
        """A `litellm_params.model` without a `/` reopens the wildcard shadows — 25
        measured. Whatever the input, there is always a prefix in front of the wire name."""
        wire = wire_of(discovered("m", family), provider)
        prefix, _, bare = wire.partition("/")
        assert prefix
        assert bare == "m"


class TestFamilyComesFromDiscovery:
    """The family is not guessed from the name: `chat_23310` and `tab_flash_lite_preview`
    are `MODEL_PROVIDER_GOOGLE` in the real catalog and would not tell anyone so."""

    def catalog(self, entries: dict[str, object]) -> Handler:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path.endswith(MODELS_PATH.lstrip("/"))
            return httpx.Response(200, json={"models": entries, "deprecatedModelIds": []})

        return handler

    async def test_model_provider_enum_maps_to_the_family(self) -> None:
        entries: dict[str, object] = {
            "gemini-2.5-pro": {"modelProvider": "MODEL_PROVIDER_GOOGLE"},
            "claude-sonnet-4-6": {"modelProvider": "MODEL_PROVIDER_ANTHROPIC"},
            "gpt-oss-120b-medium": {"modelProvider": "MODEL_PROVIDER_OPENAI"},
        }
        async with client(self.catalog(entries)) as http:
            models = await discover(GOOGLE, client=http)
        assert {m.wire_name: m.family for m in models} == {
            "gemini-2.5-pro": "google",
            "claude-sonnet-4-6": "anthropic",
            "gpt-oss-120b-medium": "openai",
        }

    async def test_opaque_name_still_gets_its_family_from_the_catalog(self) -> None:
        """`chat_23310` is `MODEL_PROVIDER_GOOGLE` on the measured account. This is why the
        family is read from the payload and not from the name."""
        opaque: dict[str, object] = {"chat_23310": {"modelProvider": "MODEL_PROVIDER_GOOGLE"}}
        async with client(self.catalog(opaque)) as http:
            models = await discover(GOOGLE, client=http)
        assert models[0].family == "google"
        assert wire_of(models[0], "google-antigravity") == "gemini/chat_23310"

    async def test_unknown_enum_yields_no_family(self) -> None:
        """A new enum from the upstream cannot become a guessed prefix."""
        entries: dict[str, object] = {
            "new": {"modelProvider": "MODEL_PROVIDER_XAI"},
            "no-field": {"displayName": "no-field"},
        }
        async with client(self.catalog(entries)) as http:
            models = await discover(GOOGLE, client=http)
        assert [m.family for m in models] == ["", ""]

    async def test_anthropic_subscription_needs_no_catalog(self) -> None:
        """The Max subscription only serves `claude-*`: the family is constant and known,
        and a probe that never reaches the upstream cannot leave it unfilled."""

        def unreachable(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no network", request=request)

        async with client(unreachable) as http:
            models = await discover(ANTHROPIC, client=http)
        assert models
        assert {m.family for m in models} == {"anthropic"}

    async def test_codex_subscription_needs_no_catalog(self) -> None:
        def unreachable(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no network", request=request)

        async with client(unreachable) as http:
            models = await discover(CODEX, client=http)
        assert models
        assert {m.family for m in models} == {"openai"}


class TestPublicNameCarriesTheSubscription:
    """Two subscriptions serve `claude-sonnet-4-6`. Without separation, one eats the
    other."""

    def name_of(self, model: DiscoveredModel, provider: ProviderId) -> str:
        return str(to_deployment(model, provider)["model_name"])

    def test_each_subscription_has_its_own_public_prefix(self) -> None:
        model = discovered("claude-sonnet-4-6", "anthropic")
        assert self.name_of(model, "anthropic") == "mysubs/claudecode/claude-sonnet-4-6"
        assert (
            self.name_of(model, "google-antigravity")
            == "mysubs/antigravity/claude-sonnet-4-6"
        )
        assert self.name_of(discovered("gpt-6", "openai"), "openai-codex") == "mysubs/codex/gpt-6"

    def test_same_model_from_two_subscriptions_gets_two_names(self) -> None:
        """The whole point of the prefix: the second `apply` would stop overwriting the
        first, and a `claude-sonnet-4-6` from `config.yaml` stops being contested."""
        model = discovered("claude-sonnet-4-6", "anthropic")
        assert self.name_of(model, "anthropic") != self.name_of(model, "google-antigravity")
        assert model.suggested_name not in {
            self.name_of(model, "anthropic"),
            self.name_of(model, "google-antigravity"),
        }

    def test_public_prefix_never_reaches_the_wire(self) -> None:
        """On the wire it is poison: it kills the tariff the family prefix has just
        recovered."""
        model = discovered("gemini-2.5-pro", "google")
        out = to_deployment(model, "google-antigravity")
        assert out["litellm_params"]["model"] == "gemini/gemini-2.5-pro"
        for prefix in PUBLIC_PREFIX.values():
            assert prefix not in out["litellm_params"]["model"]

    def test_prefixing_an_already_prefixed_name_is_a_no_op(self) -> None:
        """`apply` runs over what has already been applied — the UI rewrites the list on
        every selection change. Without this, the spend log recorded
        `mysubs/codex/mysubs/codex/gpt-6`."""
        once = to_deployment(discovered("gpt-6-astra", "openai"), "openai-codex")
        again = DiscoveredModel(
            wire_name="gpt-6-astra",
            suggested_name=str(once["model_name"]),
            verified=True,
            family="openai",
        )
        assert self.name_of(again, "openai-codex") == once["model_name"]


class TestPrefixActuallyPrices:
    """The defect this fixes, measured in LiteLLM itself.

    An assertion on the string only proved that `gemini/` had been written. What failed in
    production was the next step: `completion_cost` raising "This model isn't mapped yet"
    and `/spend/logs` recording 0.
    """

    def cost(self, wire: str) -> float:
        from litellm import completion_cost
        from litellm.types.utils import ModelResponse, Usage

        response = ModelResponse(
            model=wire,
            usage=Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
        )
        return float(completion_cost(completion_response=response, model=wire))

    def test_google_family_prefix_has_a_price_and_the_provider_one_has_none(self) -> None:
        model = discovered("gemini-2.5-pro", "google")
        assert self.cost(wire_of(model, "google-antigravity")) > 0
        with pytest.raises(Exception, match="isn't mapped yet"):
            self.cost(f"{WIRE_PREFIX['google-antigravity']}/{model.wire_name}")

    def test_anthropic_family_prefix_has_a_price_under_the_google_provider(self) -> None:
        """The case that only the family solves: served by Antigravity, priced by
        Anthropic."""
        model = discovered("claude-sonnet-4-6", "anthropic")
        assert self.cost(wire_of(model, "google-antigravity")) > 0
        with pytest.raises(Exception, match="isn't mapped yet"):
            self.cost(f"{WIRE_PREFIX['google-antigravity']}/{model.wire_name}")

    def test_price_survives_the_public_prefix(self) -> None:
        """The whole point of separating the two names. The public name carries the
        subscription, the wire one does not — and it is the wire one that
        `completion_cost` reads. Measured: the same name with the public prefix in front
        has no tariff at all."""
        out = to_deployment(discovered("gemini-2.5-pro", "google"), "google-antigravity")
        assert str(out["model_name"]).startswith("mysubs/antigravity/")
        assert self.cost(str(out["litellm_params"]["model"])) > 0
        assert self.cost("gemini/mysubs/antigravity/gemini-2.5-pro") == 0.0
