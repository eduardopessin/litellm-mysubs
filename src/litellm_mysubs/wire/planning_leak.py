"""Filtro do leak de planeamento dos modelos flash.

Porte de ``providers/google-gemini-cli.ts``. Os modelos flash do Antigravity vertem por
vezes o objecto de planeamento interno para o texto visível — um JSON com ``thought``,
``call``, ``paths`` ou ``path``+``content`` que nunca deveria chegar ao cliente.

Duas coisas tornam isto mais difícil do que parece:

1. **O objecto atravessa chunks.** Um delta pode trazer `{"thou` e o seguinte `ght": …}`.
   Decidir por chunk deixava passar tudo o que não coubesse num só; daí o buffering.
2. **Nem todo o JSON é leak.** Um modelo que responda legitimamente `{"command": "ls"}`
   não pode ver a resposta apagada — por isso o filtro só se aplica à família que de facto
   verte, e exige assinatura de leak.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Final, Literal

#: Chaves cuja presença marca o objecto como planeamento interno.
LEAK_MARKERS: Final[tuple[str, ...]] = ("thought", "_i", "call", "paths", "command")

#: Comprimento máximo de um prefixo ainda incompleto que se aceita como possível leak.
#: Acima disto é texto legítimo que por acaso começa por chaveta.
MAX_PREFIX_CHARS: Final = 100


# omp: providers/google-gemini-cli.ts :: isFlashLeakModel
def is_flash_leak_model(model: str) -> bool:
    """Só a família flash verte planeamento para o texto visível.

    Aplicar o filtro a todos os modelos faria um `pro` que responda legitimamente
    ``{"command": "ls"}`` ver a resposta apagada.
    """
    return "flash" in str(model).split("/")[-1].lower()


# omp: providers/google-gemini-cli.ts :: isPlanningLeakPrefix
def is_leak_prefix(text: str) -> bool:
    """Se o texto **pode** vir a ser um objecto de planeamento.

    Reconhece um prefixo incompleto: `{`, `{"tho`, `{"thought"`. É o que permite reter o
    buffer em vez de emitir metade de um leak.
    """
    trimmed = text.lstrip()
    if not trimmed.startswith("{"):
        return False

    after_brace = trimmed[1:].lstrip()
    if not after_brace:
        return len(trimmed) <= MAX_PREFIX_CHARS
    if after_brace[0] != '"':
        return False

    next_quote = after_brace.find('"', 1)
    if next_quote == -1:
        key_prefix = after_brace[1:]
        return "thought".startswith(key_prefix) and len(trimmed) <= MAX_PREFIX_CHARS

    if after_brace[1:next_quote] != "thought":
        return False

    after_key = after_brace[next_quote + 1 :].lstrip()
    return len(trimmed) <= MAX_PREFIX_CHARS if not after_key else after_key[0] == ":"


# omp: providers/google-gemini-cli.ts :: isPlanningLeakObject
def is_leak_object(parsed: object, tool_names: frozenset[str] = frozenset()) -> bool:
    """Se o objecto já decodificado tem assinatura de planeamento."""
    if not isinstance(parsed, dict):
        return False
    if isinstance(parsed.get("thought"), str):
        return True
    call = parsed.get("call")
    if isinstance(call, str) and call in tool_names:
        return True
    if any(key in parsed for key in ("_i", "paths", "command")):
        return True
    return "path" in parsed and "content" in parsed


def _split_leading_object(text: str, *, honour_strings: bool = True) -> tuple[str, str] | None:
    """Primeiro objecto JSON equilibrado em chavetas, e o que sobra.

    ``honour_strings=False`` é o plano B: um leak com aspas desequilibradas nunca fecharia
    o objecto pela via normal, e deixá-lo passar era o pior resultado.
    """
    prefix = len(text) - len(text.lstrip())
    trimmed = text[prefix:]
    if not trimmed.startswith("{"):
        return None

    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(trimmed):
        if honour_strings and in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if honour_strings and char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return trimmed[: index + 1], trimmed[index + 1 :]
    return None


Outcome = Literal["incomplete", "plain", "leak"]


@dataclass(slots=True)
class PlanningLeakFilter:
    """Filtra o texto visível de um stream, retendo o que pode ser um leak.

    Uso: ``feed`` por cada delta, ``flush`` no fim. ``stripped`` diz se alguma coisa foi
    descartada — o OMP usa esse sinal para não aceitar como "silêncio válido" uma resposta
    cujo conteúdo foi todo deitado fora.
    """

    tool_names: frozenset[str] = frozenset()
    buffer: str = ""
    buffering: bool = False
    stripped: bool = False
    _emitted: list[str] = field(default_factory=list, repr=False)

    def feed(self, text: str) -> str:
        """Texto a entregar ao cliente por causa deste delta. Pode ser vazio."""
        if not text:
            return ""
        if not self.buffering and not self.buffer:
            if not is_leak_prefix(text):
                return text
            self.buffering = True
        self.buffer += text
        return self._drain()

    def flush(self) -> str:
        """O que sobra no fim do stream.

        Um buffer com assinatura de leak mas sem chaveta a fechar é descartado por
        inteiro: entregá-lo seria mostrar metade do planeamento interno.
        """
        if not self.buffer:
            return ""
        pending, self.buffer = self.buffer, ""
        self.buffering = False
        emitted = self._consume(pending, final=True)
        return emitted

    def _drain(self) -> str:
        emitted = self._consume(self.buffer, final=False)
        return emitted

    def _consume(self, text: str, *, final: bool) -> str:
        split = _split_leading_object(text) or _split_leading_object(text, honour_strings=False)
        if split is None:
            if not final:
                # Objecto ainda por fechar: retém-se à espera do próximo delta.
                self.buffer = text
                return ""
            # No fim do stream, um prefixo com assinatura de leak nunca chega a ser texto.
            if is_leak_prefix(text):
                self.stripped = True
                return ""
            return text

        json_text, rest = split
        try:
            parsed: Any = json.loads(json_text)
        except ValueError:
            parsed = None

        if parsed is not None:
            leaked = is_leak_object(parsed, self.tool_names)
        else:
            # JSON malformado — tipicamente aspas desequilibradas dentro do leak. Sem
            # objecto para inspeccionar, decide-se pela assinatura do prefixo; deixá-lo
            # passar por não decodificar era o pior resultado possível.
            leaked = is_leak_prefix(json_text)

        if leaked:
            self.stripped = True
            visible = rest
        else:
            visible = json_text + rest

        self.buffer = ""
        self.buffering = False
        return visible
