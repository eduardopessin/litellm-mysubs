"""Gestão dos deployments que o plugin acrescenta ao Router do LiteLLM.

Porque não se usa ``POST /model/new``: o endpoint devolve 200 e grava em Postgres, mas o
modelo nunca chega ao Router quando ``general_settings.supported_db_objects`` não inclui
``"models"`` — e não incluir é a configuração correcta em instalações que servem agentes
A2A. Medido: ``/model/new`` → 200, e a seguir ``/v1/models`` sem o modelo e a chamada com
400 ``no healthy deployments``. Um "aplicar" que devolve sucesso e não faz nada é o pior
modo de falha possível, por isso o plugin injecta directamente e persiste por sua conta.

A guarda contra deployments fantasma vem de um defeito medido em produção: um wildcard
``claude-*`` faz o provider nativo materializar um deployment para **qualquer** nome
pedido, antes sequer de falar com o upstream. O 404 que se segue está certo, mas a
entrada fica colada ao Router para sempre — 25 acumuladas, e com ``simple-shuffle`` as
cópias entram no sorteio do modelo legítimo do mesmo nome.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

#: Marca as entradas criadas por este plugin. Nunca se toca no que veio do config.yaml.
MANAGED_BY = "mysubs"


class RouterLike(Protocol):
    """A parte do ``litellm.Router`` de que dependemos."""

    model_list: list[dict[str, Any]]

    def set_model_list(self, model_list: list[dict[str, Any]]) -> None: ...


def is_declared(deployment: dict[str, Any]) -> bool:
    """Se o deployment veio do ``config.yaml``.

    O LiteLLM dá às entradas de config o ``model_info.id`` que elas declaram — e o nosso
    config declara ``id`` igual ao ``model_name``. As criadas por resolução de wildcard
    recebem um id em hash. É essa a diferença que distingue uma entrada real de uma
    sombra, e é o que impede a limpeza de apagar um modelo declarado.
    """
    info = deployment.get("model_info") or {}
    return str(info.get("id") or "") == deployment.get("model_name")


def is_managed(deployment: dict[str, Any]) -> bool:
    """Se o deployment foi acrescentado por este plugin."""
    info = deployment.get("model_info") or {}
    return info.get("managed_by") == MANAGED_BY


@dataclass(slots=True)
class ModelRegistry:
    """Injecta, remove e reaplica os deployments do plugin."""

    router: RouterLike
    #: Nomes que o upstream recusou com "não existe". Ver ``remember_not_found``.
    not_found: set[str] = field(default_factory=set)

    # -- guardas ---------------------------------------------------------------

    def remember_not_found(self, wire_name: str) -> bool:
        """Memoriza um nome que o upstream disse não servir.

        Só chega aqui um erro de "não existe". Um 429 ou um 500 são transitórios: tratá-los
        como inexistência desligaria um modelo bom até ao próximo restart.
        """
        wire = wire_name.strip().lower()
        if not wire or wire in self.not_found:
            return False
        self.not_found.add(wire)
        return True

    def is_known_bad(self, wire_name: str) -> bool:
        return wire_name.strip().lower() in self.not_found

    def evict_shadow(self, wire_name: str, *, only_if_declared: bool = False) -> int:
        """Remove deployments que a resolução de wildcard materializou para ``wire_name``.

        ``only_if_declared=True`` é o caminho de sucesso: um pedido a um modelo declarado
        casa a entrada do config **e** o wildcard, e a cópia do wildcard fica a concorrer
        no sorteio sem o ``model_info`` da entrada real. Só se remove quando existe um
        gémeo declarado, ou seja quando a cópia é redundante por construção.

        ``only_if_declared=False`` é o caminho de erro: o nome já foi recusado pelo
        upstream, logo nunca é um modelo servido.

        Um nome sem gémeo no config sobrevive ao caminho de sucesso de propósito — é o
        modelo novo de uma família, servido no dia um sem editar configuração.
        """
        wire = wire_name.strip().lower()
        model_list = list(self.router.model_list or [])
        declared = {d.get("model_name") for d in model_list if is_declared(d)}

        keep: list[dict[str, Any]] = []
        dropped = 0
        for deployment in model_list:
            name = str(deployment.get("model_name") or "").lower()
            shadow = name == wire and not is_declared(deployment) and not is_managed(deployment)
            if shadow and (not only_if_declared or deployment.get("model_name") in declared):
                dropped += 1
                continue
            keep.append(deployment)

        if dropped:
            self.router.set_model_list(keep)
        return dropped

    # -- injecção --------------------------------------------------------------

    def apply(self, deployments: list[dict[str, Any]]) -> int:
        """Substitui as entradas do plugin pelas indicadas, preservando as do config."""
        kept = [d for d in (self.router.model_list or []) if not is_managed(d)]
        marked = [self._mark(d) for d in deployments]
        self.router.set_model_list(kept + marked)
        return len(marked)

    def managed(self) -> list[dict[str, Any]]:
        return [d for d in (self.router.model_list or []) if is_managed(d)]

    @staticmethod
    def _mark(deployment: dict[str, Any]) -> dict[str, Any]:
        out = dict(deployment)
        info = dict(out.get("model_info") or {})
        info["managed_by"] = MANAGED_BY
        info.setdefault("id", out.get("model_name"))
        out["model_info"] = info
        return out
