"""Política de reabertura de ligação, separada do transporte.

No original, a decisão do que fazer com um 401, um 400 ou um 429 estava embutida dentro dos
laços de ``httpx``, duplicada entre a versão síncrona e a assíncrona — e as duas tinham
derivado formas diferentes da mesma regra. Aqui a decisão é uma função pura sobre
``(status, corpo)``, e o transporte limita-se a executá-la.

Isso torna testável o que interessa — *quando* se retenta e porquê — sem abrir ligações.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final


class Action(Enum):
    """O que fazer a seguir."""

    RETURN = "return"
    """Resposta boa: entregar."""

    REFRESH_TOKEN = "refresh_token"
    """Credencial rejeitada: reler e tentar de novo com a nova."""

    REMAP_MODEL = "remap_model"
    """A conta não serve este nome; se for um alias conhecido, reencaminhar."""

    REDEEM_CREDIT = "redeem_credit"
    """Quota esgotada e há crédito de reset por usar."""

    FAIL = "fail"
    """Nada a fazer: propagar o erro do upstream."""


@dataclass(frozen=True, slots=True)
class Decision:
    action: Action
    reason: str = ""

    @property
    def should_retry(self) -> bool:
        return self.action in (Action.REFRESH_TOKEN, Action.REMAP_MODEL, Action.REDEEM_CREDIT)


#: Tentativas de abertura. Três chega para renovar o token e remapear o modelo uma vez cada.
MAX_ATTEMPTS: Final = 3

#: Marca da recusa de modelo pela conta ChatGPT.
UNSUPPORTED_MARKER: Final = "is not supported when using Codex"


def is_unsupported_model(body: str) -> bool:
    return UNSUPPORTED_MARKER in str(body)


def decide_codex(
    status: int,
    body: str = "",
    *,
    can_remap: bool = False,
    can_redeem: bool = False,
) -> Decision:
    """O que fazer com a resposta de abertura do Codex.

    ``can_remap`` e ``can_redeem`` são capacidades do chamador, não do estado global: se
    não houver alias conhecido nem crédito, a decisão tem de ser falhar — retentar o mesmo
    pedido daria o mesmo erro três vezes e triplicava a latência antes de o dizer.
    """
    if status == 200:
        return Decision(Action.RETURN)
    if status == 401:
        return Decision(Action.REFRESH_TOKEN, "credencial rejeitada")
    if status == 400 and is_unsupported_model(body):
        if can_remap:
            return Decision(Action.REMAP_MODEL, "alias conhecido de um modelo servido")
        # Um nome arbitrário recusado é a resposta correcta: substituí-lo por outro modelo
        # devolvia 200 com o campo `model` a ecoar o pedido, e a facturação passava a mentir.
        return Decision(Action.FAIL, "a conta não serve este modelo")
    if status == 429 and can_redeem:
        return Decision(Action.REDEEM_CREDIT, "quota esgotada, crédito de reset disponível")
    return Decision(Action.FAIL, f"HTTP {status}")


def decide_antigravity(status: int) -> Decision:
    """O que fazer com a resposta de abertura do Antigravity.

    O failover é só no *endpoint*: um 404 ou 503 não autoriza responder com outro modelo.
    Um 404 é "esta conta não serve este modelo" e um 503 é capacidade; em qualquer dos
    casos tenta-se o host seguinte e, esgotados, propaga-se.
    """
    if status == 200:
        return Decision(Action.RETURN)
    if status == 401:
        return Decision(Action.REFRESH_TOKEN, "credencial rejeitada")
    return Decision(Action.FAIL, f"HTTP {status}")
