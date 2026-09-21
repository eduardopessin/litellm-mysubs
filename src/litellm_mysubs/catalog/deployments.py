"""Conversion of discovered models into Router deployments.

The piece between `catalog/discovery.py`, which says what the subscription serves, and
`registry.py`, which injects into the Router. It is kept apart from both on purpose:
discovery does not have to know the LiteLLM format, and the registry does not have to know
where the names come from.
"""

from __future__ import annotations

from typing import Any, Final

from ..credentials.store import ProviderId
from .discovery import DiscoveredModel

#: Wire prefix by provider. A fallback: it only decides when the family is unknown.
#:
#: The prefix is not decorative. A `litellm_params.model` without one falls into the native
#: provider's wildcard resolution, which materialises a deployment for any name before
#: talking to the upstream — exactly the phantom-deployment defect described at the top of
#: `registry.py`, and measured on this installation (25 accumulated shadows).
WIRE_PREFIX: Final[dict[ProviderId, str]] = {
    "anthropic": "anthropic",
    "openai-codex": "openai",
    "google-antigravity": "openai",
}

#: Wire prefix by model family. Wins over `WIRE_PREFIX` whenever a family is known.
#:
#: Being present is not enough: it has to be right. A wrong prefix does not fall into the
#: wildcard, but it takes the model out of the LiteLLM pricing table, and the cost becomes
#: zero everywhere. Measured with `litellm.completion_cost` (1000 in / 500 out) over the
#: names Antigravity serves:
#:
#:   gemini-2.5-pro     openai/ → no table   gemini/ → $0.00625   vertex_ai/ → $0.00625
#:   gemini-2.5-flash   openai/ → no table   gemini/ → $0.00155   vertex_ai/ → $0.00155
#:   claude-sonnet-4-6  openai/ → no table   gemini/ → no table    anthropic/ → $0.0105
#:
#: While everything carried `openai/` (the *provider* prefix), the Models tab showed
#: `input_cost_per_token=0` and `/spend/logs` recorded zero cost. Both failures — the
#: wildcard shadow and the zero price — are silent, and that is why the prefix is chosen by
#: the model family and not by whoever resells it.
#:
#: `google` takes `gemini/` and not `vertex_ai/`: they price the same, but the second one
#: requires a GCP project and region, which an Antigravity subscription does not have.
FAMILY_PREFIX: Final[dict[str, str]] = {
    "google": "gemini",
    "anthropic": "anthropic",
    "openai": "openai",
}


def wire_prefix(model: DiscoveredModel, provider: ProviderId) -> str:
    """Prefix to put in ``litellm_params.model``. Never empty.

    The family wins because Antigravity resells three (measured: 32 models in the catalog,
    of which two are `MODEL_PROVIDER_ANTHROPIC` and one `MODEL_PROVIDER_OPENAI`). With no
    family — or with one this table does not know — the provider prefix is returned instead
    of guessing: a wrong prefix means zero cost, but no prefix reopens the wildcard
    shadows, and of the two defects only the second corrupts the Router.
    """
    return FAMILY_PREFIX.get(model.family) or WIRE_PREFIX[provider]


#: **Public** name prefix by subscription. It goes in `model_name`, never on the wire.
#:
#: It exists because the same name is served by two subscriptions: `claude-sonnet-4-6` is
#: in the curated Anthropic list and in the Antigravity catalog. Without separation, the
#: second `ModelRegistry.apply` overwrites the first, and a `claude-sonnet-4-6` the user
#: declared in `config.yaml` ends up contested by entries the plugin created.
#:
#: A slash and not an underscore: that is the LiteLLM convention for a namespace, and it
#: was **measured** that the proxy accepts it in a `model_name` without mistaking it for a
#: provider prefix. The Router resolves `mysubs/antigravity/gemini-2.5-pro` all the way to
#: the transport, just as it resolves the bare name — resolution is by equality against the
#: list, not by interpreting the name.
#:
#: (`get_llm_provider` fails on both forms, with and without the slash. That is irrelevant:
#: the Router resolves the deployment before that path is touched, and the prefix it reads
#: is the one in `litellm_params.model`.)
#:
#: And it stays in the public name **only**. Measured with `litellm.completion_cost`
#: (1000 in / 500 out):
#:
#:   gemini/gemini-2.5-pro                    -> $0.00625
#:   gemini/mysubs_antigravity_gemini-2.5-pro -> 0.0, no rate
#:
#: Carrying it into `litellm_params.model` killed exactly the price `FAMILY_PREFIX` had
#: just recovered.
PUBLIC_PREFIX: Final[dict[ProviderId, str]] = {
    "anthropic": "mysubs/claudecode/",
    "openai-codex": "mysubs/codex/",
    "google-antigravity": "mysubs/antigravity/",
}


def public_name(model: DiscoveredModel, provider: ProviderId) -> str:
    """Public deployment name: the suggested one, with the subscription in front.

    Idempotent on purpose. `apply` runs over what has already been applied — the UI
    rewrites the list on every selection change — and prefixing twice produced
    `mysubs_codex_mysubs_codex_gpt-6`, a name no client asked for and that the spend log
    would start recording.
    """
    prefix = PUBLIC_PREFIX[provider]
    name = model.suggested_name
    return name if name.startswith(prefix) else f"{prefix}{name}"


def to_deployment(model: DiscoveredModel, provider: ProviderId) -> dict[str, Any]:
    """A Router deployment from a discovered model.

    The two names are independent and neither is the other:

    * `model_name` is what the client asks for and what `/spend/logs` records. It carries
      the **subscription** prefix (`PUBLIC_PREFIX`), because two subscriptions serve the
      same name.
    * `litellm_params.model` carries the **family** prefix (`FAMILY_PREFIX`) and the wire
      name, which may differ from the public one — `gpt-6` resolves to `gpt-6-astra`
      upstream.

    The wire name never carries the public prefix: measured, `gemini/gemini-2.5-pro` costs
    $0.00625 and `gemini/mysubs_antigravity_gemini-2.5-pro` has no rate at all.

    Neither prefix interferes with routing: `plugin.py :: dispatch` picks the provider from
    `model_info.mysubs_provider`, never from what sits before the slash.

    The `managed_by` mark is set by `ModelRegistry`, not here: whoever injects is the one
    who declares ownership.

    `custom_llm_provider` is declared rather than left for LiteLLM to infer. The inference
    runs `get_llm_provider` over `model_name`, and that **raises** on both public forms —
    measured, `BadRequestError` for `mysubs/codex/gpt-5.5` and
    `mysubs/antigravity/gemini-3-flash`. Nothing consumes the exception, so the UI is left
    with no provider for the row: the Logs tab showed the generic icon for every
    subscription model, and OpenAI and Google rows were indistinguishable from each other.

    The value is the same family prefix that already goes on the wire, which is the one
    that has a price table behind it — so the icon and the cost agree by construction
    instead of being two independent guesses.
    """
    family = wire_prefix(model, provider)
    return {
        "model_name": public_name(model, provider),
        "litellm_params": {
            "model": f"{family}/{model.wire_name}",
            "custom_llm_provider": family,
        },
        "model_info": {
            "mysubs_provider": provider,
            "mysubs_verified": model.verified,
        },
    }


def to_deployments(
    models: list[DiscoveredModel], provider: ProviderId, *, only_verified: bool = False
) -> list[dict[str, Any]]:
    """Convert a list, optionally only what was verified.

    ``only_verified`` exists so the user can say "only what actually answered". It is not
    the default: an unverified model may be unverified only because the network failed
    during the probe, and dropping it silently would make the list lie about the
    subscription. The distinction between "the upstream refused" and "I could not ask" is
    already made during discovery — whatever was refused never gets here.
    """
    chosen = [m for m in models if m.verified] if only_verified else models
    return [to_deployment(model, provider) for model in chosen]
