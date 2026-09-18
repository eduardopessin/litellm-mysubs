"""Resolução de nomes de modelo do Google Antigravity (Cloud Code API).

Este é o único provedor com catálogo consultável: ``:fetchAvailableModels`` devolve o que
a conta serve, incluindo ``deprecatedModelIds``. Descontar a lista do próprio catálogo é
melhor que manter uma estática, porque o catálogo anuncia variantes que o
``streamGenerateContent`` recusa.

O mapa estático é o plano B, usado apenas quando o catálogo não respondeu. Serve para não
deixar a conta inutilizável por uma falha de rede, mas nunca para adivinhar: um nome que
não corresponde levanta, em vez de ser servido por outro modelo em silêncio.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Final

#: O catálogo anuncia estas variantes, mas o streamGenerateContent devolve
#: 400 INVALID_ARGUMENT para elas. O omp documenta o mesmo e encaminha para gemini-pro-agent.
BROKEN_WIRE: Final[tuple[str, ...]] = ("gemini-3.1-pro-high", "gemini-3-pro-high")

#: Effort -> sufixo de variante, por ordem de preferência.
EFFORT_SUFFIXES: Final[dict[str, tuple[str, ...]]] = {
    "none": ("-extra-low", "-low", "", "-tiered"),
    "minimal": ("-extra-low", "-low", "", "-tiered"),
    "low": ("-low", "-extra-low", "", "-tiered"),
    "medium": ("-medium", "-low", "", "-tiered"),
    "high": ("-high", "-medium", "-low", ""),
    "xhigh": ("-high", "-medium", "-low", ""),
    "max": ("-high", "-medium", "-low", ""),
}

#: Famílias sem variante de sufixo utilizável no topo da escala.
EFFORT_OVERRIDES: Final[dict[tuple[str, str], str]] = {
    ("gemini-3.1-pro", "high"): "gemini-pro-agent",
    ("gemini-3.1-pro", "xhigh"): "gemini-pro-agent",
    ("gemini-3.1-pro", "max"): "gemini-pro-agent",
    ("gemini-3-pro", "high"): "gemini-pro-agent",
    ("gemini-3.5-flash", "high"): "gemini-3-flash-agent",
    ("gemini-3.5-flash", "xhigh"): "gemini-3-flash-agent",
    ("gemini-3.5-flash", "max"): "gemini-3-flash-agent",
}

# Sufixos de variante que se descascam para chegar à família.
#
# `-thinking` saiu de propósito: `gemini-2.5-flash-thinking` existe no catálogo e casa
# pelo nome exacto, enquanto `gemini-3.8-flash-thinking` não existe — descascá-lo fazia um
# nome inventado ser servido por `-low` em silêncio, exactamente o que tirar essas
# entradas do config pretendia evitar.
SUFFIXES: Final[tuple[str, ...]] = (
    "-tiered",
    "-extra-low",
    "-low",
    "-medium",
    "-high",
    "-agent",
)

# Plano B para quando o catálogo não responde. As entradas `-thinking` saíram: não existem
# no upstream. As `-tiered` existem e apontam para si mesmas, porque descascar um pedido
# explícito é mentir sobre o que se serviu.
STATIC_MAP: Final[dict[str, str]] = {
    "gemini-3.8-flash-tiered": "gemini-3.8-flash-tiered",
    "gemini-3.8-flash": "gemini-3.8-flash-low",
    "gemini-3.7-flash-tiered": "gemini-3.7-flash-tiered",
    "gemini-3.7-flash": "gemini-3.7-flash-low",
    "gemini-3.6-flash": "gemini-3.6-flash-low",
    "gemini-3.5-flash": "gemini-3.5-flash-extra-low",
    "gemini-3.1-flash-lite": "gemini-3.1-flash-lite",
    "gemini-3.1-pro": "gemini-3.1-pro-low",
    "gemini-3-flash": "gemini-3-flash",
    "gemini-3-pro": "gemini-3-pro-low",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
    "gemini-2.5-flash": "gemini-2.5-flash",
}

CATALOG_TTL_S: Final = 600.0


class ModelNotServedError(Exception):
    """O nome pedido não corresponde a nada que a conta sirva.

    Levantar é deliberado: o wildcard ``gemini-*`` faria qualquer nome inventado responder
    como ``gemini-2.5-flash``, com o campo ``model`` a ecoar o nome pedido — e a
    facturação, as comparações e a reprodutibilidade passariam a mentir.
    """


@dataclass(slots=True)
class ModelCatalog:
    """Catálogo da conta, com TTL.

    Instância em vez de global: dois proxies no mesmo processo teriam contas diferentes, e
    um cache partilhado serviria o catálogo de um ao outro.
    """

    ids: tuple[str, ...] = ()
    info: dict[str, Any] = field(default_factory=dict)
    fetched_at: float = 0.0

    def is_fresh(self, *, now: float | None = None) -> bool:
        if not self.ids:
            return False
        return ((now if now is not None else time.time()) - self.fetched_at) < CATALOG_TTL_S

    def update(self, payload: dict[str, Any], *, now: float | None = None) -> tuple[str, ...]:
        """Absorve uma resposta de ``:fetchAvailableModels``.

        O catálogo lista variantes que já não respondem e marca-as em
        ``deprecatedModelIds`` — é assim que ``gemini-3.1-pro-high`` aparece servido e
        devolve 400.
        """
        models = payload.get("models") or {}
        deprecated = {str(x).lower() for x in (payload.get("deprecatedModelIds") or [])}
        ids = tuple(key for key in models if str(key).lower() not in deprecated)
        if ids:
            self.ids = ids
            self.info = models
            self.fetched_at = now if now is not None else time.time()
        return self.ids


def base_family(model: str) -> str:
    """Nome sem o sufixo de variante: ``gemini-3.8-flash-low`` -> ``gemini-3.8-flash``."""
    base = str(model).split("/")[-1].lower()
    for suffix in SUFFIXES:
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def supports_function_ids(model: str) -> bool:
    return str(model).split("/")[-1].lower().startswith("gemini-3")


def _from_catalog(raw: str, effort: str, available: tuple[str, ...]) -> str | None:
    # Se o nome pedido *for* uma variante servida, respeita-se: pedir
    # `gemini-3.8-flash-tiered` (real no catálogo) não pode acabar em `-low` porque o
    # effort assim decidiu. Antes o sufixo era sempre descascado e o pedido perdia-se.
    if raw in available and raw not in BROKEN_WIRE:
        return raw

    base = base_family(raw)
    candidates: list[str] = []
    if override := EFFORT_OVERRIDES.get((base, effort)):
        candidates.append(override)
    candidates.extend(base + suffix for suffix in EFFORT_SUFFIXES.get(effort, ("-low", "")))

    for candidate in candidates:
        if candidate in BROKEN_WIRE:
            continue
        if candidate in available:
            return candidate
    return None


def _from_static_map(raw: str) -> str | None:
    if raw in STATIC_MAP:
        return STATIC_MAP[raw]
    # Correspondência parcial só quando o que sobra é um sufixo de variante conhecido. Com
    # `if k in raw` cru, `gemini-3.8-flash` casava dentro de `gemini-3.8-flash-thinking` e
    # servia `-low` para um nome que não existe.
    for known, wire in STATIC_MAP.items():
        if not raw.startswith(known):
            continue
        rest = raw[len(known) :]
        if not rest or rest in SUFFIXES:
            return wire
    return None


# omp: providers/google-gemini-cli.ts :: lastGoodEndpoint
def map_model(model: str, effort: str | None = None, catalog: ModelCatalog | None = None) -> str:
    """Nome que vai no fio. Levanta ``ModelNotServedError`` se nada corresponder."""
    raw = str(model).split("/")[-1].lower()
    normalized_effort = str(effort or "medium").strip().lower() or "medium"

    if catalog is not None and catalog.ids:
        resolved = _from_catalog(raw, normalized_effort, catalog.ids)
        if resolved is not None:
            return resolved

    if resolved := _from_static_map(raw):
        return resolved

    raise ModelNotServedError(
        f"Google Antigravity: modelo '{raw}' não é servido por esta conta "
        f"(nenhuma variante corresponde no catálogo nem no mapa estático)"
    )
