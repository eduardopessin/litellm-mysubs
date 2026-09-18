"""Detecção de raciocínio em fuga.

Porte directo de ``utils/thinking-loop.ts``. Os limiares não são escolhas nossas: o
comentário da fonte diz que foram calibrados contra **536 mil blocos reais de raciocínio**,
e que a corrida legítima mais longa de segmentos de baixa informação observada foi 7 — daí
o gatilho em 8. Mexer neles sem um corpus equivalente é adivinhar.

Três formas de fuga, com propósitos distintos:

1. **Ciclo exacto no sufixo** — repetição literal. Dois regimes: ciclos curtos (≤60 chars)
   exigem 4 repetições e ≥180 chars; ciclos longos exigem 3 e ≥1024. Aplica-se sempre.
2. **Aglomerado de quase-duplicados** — o mesmo parágrafo reescrito com deriva cosmética,
   por sobreposição de trigramas de palavras.
3. **Estagnação de léxico** — parágrafos que reciclam vocabulário recente e não introduzem
   nenhuma referência concreta nova.

As duas últimas são heurísticas semânticas e podem ser desligadas; a primeira não.
"""

from __future__ import annotations

import re
from typing import Final

#: Cauda retida para detecção de ciclo exacto.
EXACT_TAIL_WINDOW: Final = 4096
#: Maior ciclo considerado.
EXACT_MAX_UNIT: Final = 1024
#: Caracteres novos entre varrimentos. Evita trabalho quadrático por cada delta.
EXACT_CHECK_STRIDE: Final = 128
#: Fronteira entre o regime curto e o longo.
EXACT_SHORT_MAX_UNIT: Final = 60
EXACT_SHORT_MIN_REPEATED_CHARS: Final = 180
EXACT_LONG_MIN_REPEATED_CHARS: Final = 1024

#: Tecto de um segmento sem terminador; força um flush para que um muro de texto sem
#: linhas em branco continue a ser segmentado.
SEGMENT_CHAR_CAP: Final = 700
#: Abaixo deste comprimento normalizado o segmento é ignorado — demasiado curto para ser
#: um parágrafo com significado, e um título sozinho não pode disparar a detecção.
SEGMENT_MIN_NORM_CHARS: Final = 60
SEGMENT_WINDOW: Final = 16
SEGMENT_SIMILARITY: Final = 0.8
#: Aquecimento: segmentos substanciais necessários antes de a detecção poder disparar.
SEGMENT_MIN_COUNT: Final = 8
#: Tamanho do aglomerado de quase-duplicados que dispara.
SEGMENT_MIN_CLUSTER: Final = 4

#: Janela cujo vocabulário é a linha de base da novidade.
LEX_NOVELTY_WINDOW: Final = 8
LEX_STALL_NOVELTY_FLOOR: Final = 0.2
LEX_STALL_MIN_RUN: Final = 8

# Uma referência concreta sobre a qual o modelo está de facto a raciocinar: um trecho de
# código, uma extensão ou membro com ponto, um caminho de vários segmentos, ou um
# identificador snake/camel/Pascal. Exclui dígitos nus, abreviaturas e decimais ("Passo 2",
# "i.e.", "1.2") para que enchimento numerado não seja auto-âncora.
CONCRETE_ANCHOR: Final = re.compile(
    r"`[^`]+`"
    r"|\b\w{2,}\.[a-zA-Z]\w{0,4}\b"
    r"|[\w-]+(?:/[\w-]+){2,}"
    r"|\b\w+_\w+\b"
    r"|\b[a-z]+[A-Z]\w*\b"
    r"|\b[A-Z][a-z]+[A-Z]\w*\b"
)

# Títulos de sumário ("**Mantendo o Ritmo**", "## Secção") são formatação por pensamento,
# não raciocínio. A sua redacção sempre diferente inflacionaria a novidade e mascararia um
# loop, por isso são removidos antes da análise.
_HEADING = re.compile(r"^[ \t]*#{1,6}[ \t].*$", re.M)
_BOLD_TITLE = re.compile(r"^[ \t]*\*{2,3}.+?\*{2,3}[ \t]*$", re.M)

_PARAGRAPH_BOUNDARY = re.compile(r"\n\s*\n")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_BACKTICKED = re.compile(r"`([^`]*)`")
_HAS_LETTER = re.compile(r"[a-z]")
_LETTER_OR_EMOJI = re.compile(r"[^\W\d_]", re.UNICODE)


class ThinkingLoopError(Exception):
    """Raciocínio em fuga.

    Distinta de ``Exception`` para atravessar os handlers que toleram chunks malformados:
    um loop detectado não é um chunk malformado.
    """


def normalize_segment(segment: str) -> str:
    """Minúsculas, sem pontuação, só palavras que contenham letras."""
    lowered = _BACKTICKED.sub(r" \1 ", segment.lower())
    tokens = _NON_ALNUM.sub(" ", lowered).split()
    return " ".join(token for token in tokens if _HAS_LETTER.search(token))


def trigram_shingles(normalized: str) -> set[str]:
    """Trigramas de **palavras** — não de caracteres.

    Trigramas de caracteres davam semelhança alta a textos sem relação nenhuma, o que faz
    a detecção disparar em raciocínio legítimo.
    """
    words = [word for word in normalized.split(" ") if word]
    if len(words) < 3:
        return {" ".join(words)} if words else set()
    return {" ".join(words[index : index + 3]) for index in range(len(words) - 2)}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    small, large = (left, right) if len(left) < len(right) else (right, left)
    intersection = sum(1 for item in small if item in large)
    union = len(left) + len(right) - intersection
    return intersection / union if union else 0.0


def detect_exact_suffix_cycle(text: str) -> tuple[str, int] | None:
    """Ciclo literal repetido no fim do texto, por algoritmo Z sobre o inverso.

    Dois regimes: um ciclo curto tem de aparecer 4 vezes e cobrir 180 chars; um longo, 3
    vezes e 1024. Um ciclo sem letras nenhumas (só pontuação ou espaços) não conta.
    """
    if len(text) < EXACT_SHORT_MIN_REPEATED_CHARS:
        return None

    reversed_text = text[::-1]
    length = len(reversed_text)
    z = [0] * length
    left = right = 0
    for index in range(1, length):
        if index <= right:
            z[index] = min(right - index + 1, z[index - left])
        while (
            index + z[index] < length and reversed_text[z[index]] == reversed_text[index + z[index]]
        ):
            z[index] += 1
        if index + z[index] - 1 > right:
            left, right = index, index + z[index] - 1

    max_unit = min(EXACT_MAX_UNIT, length // 3)
    for unit_length in range(2, max_unit + 1):
        count = 1 + z[unit_length] // unit_length
        short = unit_length <= EXACT_SHORT_MAX_UNIT
        min_count = 4 if short else 3
        min_chars = EXACT_SHORT_MIN_REPEATED_CHARS if short else EXACT_LONG_MIN_REPEATED_CHARS
        if count < min_count or unit_length * count < min_chars:
            continue
        unit = text[-unit_length:]
        if _LETTER_OR_EMOJI.search(unit):
            return unit, count
    return None


# omp: utils/thinking-loop.ts :: ThinkingLoopDetector
class ThinkingLoopDetector:
    """Alimentado com os deltas de raciocínio; devolve a razão na primeira fuga.

    Desvio deliberado do OMP: lá o gatilho é um erro *retryable* e a camada de retry volta
    a pedir. Aqui levanta-se. O gerador já despejou ao cliente tudo o que for
    ``reasoning_content`` antes de a detecção acontecer, e retentar duplicaria raciocínio
    no mesmo stream — o OMP evita isso com uma janela replay-safe que aqui não existe.
    """

    __slots__ = (
        "_anchor_window",
        "_count",
        "_exact_scanned_at",
        "_lex_stall_run",
        "_pending",
        "_semantic",
        "_tail",
        "_window",
        "_word_window",
        "chars",
    )

    def __init__(self, *, semantic_heuristics: bool = True) -> None:
        self._semantic = semantic_heuristics
        self._tail = ""
        self._exact_scanned_at = 0
        self._pending = ""
        self._window: list[set[str]] = []
        self._word_window: list[set[str]] = []
        self._anchor_window: list[set[str]] = []
        self._count = 0
        self._lex_stall_run = 0
        self.chars = 0

    def feed(self, delta: str) -> str | None:
        """Razão do loop, ou ``None``. Nunca levanta."""
        if not delta:
            return None
        self.chars += len(delta)

        # 1. Ciclos exactos, varridos a cadência limitada em vez de trabalho quadrático
        # por cada delta do tamanho de um token.
        self._tail = (self._tail + delta)[-EXACT_TAIL_WINDOW:]
        self._exact_scanned_at += len(delta)
        if self._exact_scanned_at >= EXACT_CHECK_STRIDE or len(delta) >= EXACT_CHECK_STRIDE:
            self._exact_scanned_at = 0
            if reason := self._exact_reason():
                return reason

        if not self._semantic:
            return None

        # 2. Segmentos: acumula e drena os que fecharam.
        self._pending += delta
        while True:
            boundary = _PARAGRAPH_BOUNDARY.search(self._pending)
            if boundary:
                raw = self._pending[: boundary.start()]
                self._pending = self._pending[boundary.end() :]
            elif len(self._pending) > SEGMENT_CHAR_CAP:
                # Sem fronteira mas longo de mais: força um flush para que um muro de
                # texto sem parágrafos continue a ser analisado.
                raw = self._pending[:SEGMENT_CHAR_CAP]
                self._pending = self._pending[SEGMENT_CHAR_CAP:]
            else:
                return None
            if reason := self._consume_chunks(raw):
                return reason

    def flush(self) -> str | None:
        """Processa o parágrafo final, que pode ser o que completa um aglomerado.

        Um stream pode terminar antes da próxima cadência, por isso força-se também uma
        verificação exacta final.
        """
        if reason := self._exact_reason():
            return reason
        if not self._semantic or not self._pending:
            return None
        pending, self._pending = self._pending, ""
        return self._consume_chunks(pending)

    # -- internos --------------------------------------------------------------

    def _exact_reason(self) -> str | None:
        found = detect_exact_suffix_cycle(self._tail)
        if found is None:
            return None
        unit, times = found
        return f"ciclo exacto de {len(unit)} caracteres repetido {times}x seguidas"

    def _consume_chunks(self, raw: str) -> str | None:
        """Parte um segmento longo de mais para que cada pedaço fique comparável."""
        rest = raw
        while rest:
            chunk, rest = rest[:SEGMENT_CHAR_CAP], rest[SEGMENT_CHAR_CAP:]
            if reason := self._consume_segment(chunk):
                return reason
        return None

    def _consume_segment(self, raw: str) -> str | None:
        segment = _BOLD_TITLE.sub("", _HEADING.sub("", raw))
        normalized = normalize_segment(segment)
        if len(normalized) < SEGMENT_MIN_NORM_CHARS:
            return None

        # (a) Aglomerado de quase-duplicados.
        fingerprint = trigram_shingles(normalized)
        cluster = 1 + sum(
            1 for previous in self._window if jaccard(fingerprint, previous) >= SEGMENT_SIMILARITY
        )

        # (b) Estagnação de léxico: parágrafos que reciclam o vocabulário recente e não
        # acrescentam nenhuma referência concreta *nova*. Exigir uma âncora nova — e não
        # apenas uma qualquer — apanha o enchimento que repete o mesmo caminho a cada
        # parágrafo, e poupa o trabalho genuíno que nomeia um ficheiro diferente de cada vez.
        words = {word for word in normalized.split(" ") if word}
        prior_vocabulary: set[str] = set()
        for seen in self._word_window:
            prior_vocabulary |= seen
        unseen = sum(1 for word in words if word not in prior_vocabulary)
        novelty = 1.0 if not prior_vocabulary else unseen / len(words)

        # Canonicaliza para que a mesma referência escrita como `Foo`, Foo ou FOO seja uma
        # só âncora e não se possa fazer passar por nova.
        anchors = {
            match.group(0).replace("`", "").lower() for match in CONCRETE_ANCHOR.finditer(segment)
        }
        new_anchor = any(
            all(anchor not in seen for seen in self._anchor_window) for anchor in anchors
        )

        if novelty <= LEX_STALL_NOVELTY_FLOOR and not new_anchor:
            self._lex_stall_run += 1
        else:
            self._lex_stall_run = 0

        self._window.append(fingerprint)
        del self._window[:-SEGMENT_WINDOW]
        self._word_window.append(words)
        del self._word_window[:-LEX_NOVELTY_WINDOW]
        self._anchor_window.append(anchors)
        del self._anchor_window[:-LEX_NOVELTY_WINDOW]
        self._count += 1

        if self._count < SEGMENT_MIN_COUNT:
            return None
        if cluster >= SEGMENT_MIN_CLUSTER:
            return f"{cluster} segmentos quase idênticos nos últimos {SEGMENT_WINDOW}"
        if self._lex_stall_run >= LEX_STALL_MIN_RUN:
            return f"{self._lex_stall_run} segmentos de baixa informação a reciclar o texto recente"
        return None


# omp: utils/thinking-loop.ts :: isLoopGuardedModel
def guard_for(model: str) -> ThinkingLoopDetector | None:
    """Vigia as famílias que de facto fogem.

    O OMP guarda Gemini, DeepSeek e xAI. Aqui só o Gemini é servido, mas a lista fica
    alinhada com a fonte para que um provedor novo não passe despercebido.
    """
    lowered = str(model).lower()
    guarded = ("gemini", "deepseek", "grok", "xai")
    return ThinkingLoopDetector() if any(name in lowered for name in guarded) else None
