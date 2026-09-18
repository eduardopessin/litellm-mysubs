"""Conversão de modelos descobertos em deployments do Router.

A peça entre `catalog/discovery.py`, que diz o que a subscrição serve, e `registry.py`,
que injecta no Router. Fica separada das duas de propósito: a descoberta não tem de saber
o formato do LiteLLM, e o registry não tem de saber de onde vêm os nomes.
"""

from __future__ import annotations

from typing import Any, Final

from ..credentials.store import ProviderId
from .discovery import DiscoveredModel

#: Prefixo do fio por provedor.
#:
#: O prefixo não é decorativo. Um `litellm_params.model` sem ele cai na resolução por
#: wildcard do provider nativo, que materializa um deployment para qualquer nome antes de
#: falar com o upstream — é exactamente o defeito dos deployments fantasma descrito no topo
#: de `registry.py`, e foi medido nesta instalação (25 sombras acumuladas).
WIRE_PREFIX: Final[dict[ProviderId, str]] = {
    "anthropic": "anthropic",
    "openai-codex": "openai",
    "google-antigravity": "openai",
}


def to_deployment(model: DiscoveredModel, provider: ProviderId) -> dict[str, Any]:
    """Um deployment do Router a partir de um modelo descoberto.

    O `model_name` é o nome **nu**: é o que o cliente pede e o que o `/spend/logs` regista.
    O `litellm_params.model` leva o prefixo do provedor e o nome do fio, que pode diferir
    do público — `gpt-6` resolve para `gpt-6-astra` upstream, e é o nome nu que tem de
    voltar na resposta para a facturação não mentir.

    A marca `managed_by` é posta pelo `ModelRegistry`, não aqui: quem injecta é que declara
    a posse.
    """
    return {
        "model_name": model.suggested_name,
        "litellm_params": {"model": f"{WIRE_PREFIX[provider]}/{model.wire_name}"},
        "model_info": {
            "mysubs_provider": provider,
            "mysubs_verified": model.verified,
        },
    }


def to_deployments(
    models: list[DiscoveredModel], provider: ProviderId, *, only_verified: bool = False
) -> list[dict[str, Any]]:
    """Converte uma lista, opcionalmente só o que foi verificado.

    ``only_verified`` existe para o utilizador poder dizer "só o que respondeu mesmo". Não
    é o default: um modelo por verificar pode sê-lo apenas porque a rede falhou durante a
    sonda, e descartá-lo em silêncio faria a lista mentir sobre a subscrição. A distinção
    entre "o upstream recusou" e "não consegui perguntar" já é feita na descoberta — quem
    recusou nem chega aqui.
    """
    chosen = [m for m in models if m.verified] if only_verified else models
    return [to_deployment(model, provider) for model in chosen]
