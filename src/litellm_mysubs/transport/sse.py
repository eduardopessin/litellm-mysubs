"""Leitura de Server-Sent Events.

Separado do transporte porque é a parte que engana: um evento partido a meio, um `[DONE]`
com espaços, um comentário de keep-alive. Cada um destes já custou um stream a alguém, e
nenhum precisa de socket para ser testado.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any, Final

DATA_PREFIX: Final = "data: "
DONE: Final = "[DONE]"


def parse_line(line: str) -> tuple[bool, Any]:
    """Interpreta uma linha de SSE.

    Devolve ``(terminou, payload)``. ``payload`` é ``None`` quando a linha não traz dados
    utilizáveis — comentário, linha em branco, ou JSON que não abre.

    Um evento malformado é ignorado, não levanta: o upstream intercala keep-alives e
    fragmentos, e matar o stream por causa de um deles perdia a resposta inteira.
    """
    stripped = line.strip()
    if not stripped.startswith(DATA_PREFIX):
        return False, None

    payload = stripped[len(DATA_PREFIX) :].strip()
    if payload == DONE:
        return True, None
    if not payload:
        return False, None

    try:
        return False, json.loads(payload)
    except (ValueError, TypeError):
        return False, None


def iter_events(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Eventos decodificados, parando no ``[DONE]``."""
    for line in lines:
        done, event = parse_line(line)
        if done:
            return
        if isinstance(event, dict):
            yield event
