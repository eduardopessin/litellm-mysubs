"""Detecção de raciocínio em fuga.

Porte do ``ThinkingLoopDetector`` do omp (``packages/ai/src/utils/thinking-loop.ts``,
descrito em ``provider-quirks.md:192``), com os limiares dele. O omp vigia as famílias
Gemini, DeepSeek e Grok antes de cada tool call e mata o stream com um erro retryable.

Quatro formas de fuga:

1. repetição verbatim na cauda (janela 4096, >= 180 chars repetidos)
2. segmentos quase duplicados (Jaccard de trigramas >= 0.8 nos últimos 16)
3. estagnação de progresso (novidade <= 0.2 em 8 segmentos sem âncoras novas)
4. títulos de sumário em excesso (24)

Módulo próprio porque não depende de nada do Antigravity: é análise de texto, e um
provedor novo que sofra do mesmo problema reutiliza-o sem arrastar o bridge.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Final

TAIL_WINDOW: Final = 4096
TAIL_REPEAT: Final = 180
SEGMENT_WINDOW: Final = 16
TRIGRAM_JACCARD: Final = 0.8
MIN_SEGMENT_CHARS: Final = 40
STALL_SEGMENTS: Final = 8
STALL_NOVELTY: Final = 0.2
HEADER_RUNAWAY: Final = 24

HEADER_RE: Final = re.compile(r"^\s*(?:\*\*[^*\n]{3,}\*\*|#{1,6}\s+\S.*)\s*$", re.M)

# Âncoras concretas: caminhos, identificadores com chamada, e números. Um segmento que
# traga uma destas nova está a progredir, mesmo com pouco léxico novo.
ANCHOR_RE: Final = re.compile(r"[\w./-]+\.[A-Za-z0-9]{1,8}\b|\b[A-Za-z_][\w]*\(|\b\d+\b")
WORD_RE: Final = re.compile(r"[^\W\d_]{3,}", re.UNICODE)


class ThinkingLoopError(Exception):
    """Raciocínio em fuga.

    Distinta de ``Exception`` para atravessar os handlers que toleram chunks malformados:
    um loop detectado não é um chunk malformado.
    """


def trigrams(text: str) -> set[str]:
    squashed = re.sub(r"\s+", " ", text.strip().lower())
    return {squashed[i : i + 3] for i in range(max(0, len(squashed) - 2))}


# omp: utils/thinking-loop.ts :: ThinkingLoopDetector
class ThinkingLoopDetector:
    """Detecta raciocínio em fuga a partir do texto que vai passando.

    Desvio deliberado do omp: lá o gatilho é um erro *retryable* e a camada de retry volta
    a pedir. Aqui não se retenta, levanta-se. O gerador já despejou ao cliente tudo o que
    for ``reasoning_content``, e a detecção acontece depois de 8 segmentos ou 180 chars
    repetidos — muito depois do primeiro despejo. Retentar duplicaria raciocínio no mesmo
    stream, o que o omp evita com uma janela replay-safe que aqui não existe.
    """

    __slots__ = (
        "buffer", "chars", "headers", "seen_anchors",
        "seen_words", "segments", "stalled", "tail",
    )

    def __init__(self) -> None:
        self.tail = ""
        self.buffer = ""
        self.segments: deque[set[str]] = deque(maxlen=SEGMENT_WINDOW)
        self.seen_words: set[str] = set()
        self.seen_anchors: set[str] = set()
        self.stalled = 0
        self.headers = 0
        self.chars = 0

    def feed(self, text: str) -> str | None:
        """Devolve a razão do loop, ou ``None``. Nunca levanta."""
        if not text:
            return None
        self.chars += len(text)
        self.tail = (self.tail + text)[-TAIL_WINDOW:]
        if reason := self._verbatim_tail():
            return reason

        self.buffer += text
        # Um segmento fecha num parágrafo; assim um delta a meio de uma frase não conta
        # como progresso nem como estagnação.
        while "\n\n" in self.buffer:
            segment, _, self.buffer = self.buffer.partition("\n\n")
            if reason := self._close_segment(segment):
                return reason
        return None

    def _verbatim_tail(self) -> str | None:
        if len(self.tail) < TAIL_REPEAT * 2:
            return None
        probe = self.tail[-TAIL_REPEAT:]
        if probe.strip() and probe in self.tail[:-TAIL_REPEAT]:
            return f"repeticao verbatim de {TAIL_REPEAT} chars na cauda"
        return None

    def _close_segment(self, segment: str) -> str | None:
        text = segment.strip()
        if not text:
            return None

        if HEADER_RE.match(text):
            self.headers += 1
            if self.headers >= HEADER_RUNAWAY:
                return f"{self.headers} titulos de sumario sem accao"

        if len(text) < MIN_SEGMENT_CHARS:
            return None

        grams = trigrams(text)
        if grams:
            for previous in self.segments:
                union = grams | previous
                if union and len(grams & previous) / len(union) >= TRIGRAM_JACCARD:
                    return "segmentos de raciocinio quase duplicados"
        self.segments.append(grams)

        words = set(WORD_RE.findall(text.lower()))
        anchors = set(ANCHOR_RE.findall(text))
        novelty = len(words - self.seen_words) / len(words) if words else 0.0
        fresh_anchor = bool(anchors - self.seen_anchors)
        self.seen_words |= words
        self.seen_anchors |= anchors

        if novelty <= STALL_NOVELTY and not fresh_anchor:
            self.stalled += 1
            if self.stalled >= STALL_SEGMENTS:
                return (
                    f"{self.stalled} segmentos sem novidade "
                    f"(novidade {novelty:.2f} <= {STALL_NOVELTY})"
                )
        else:
            self.stalled = 0
        return None


def guard_for(model: str) -> ThinkingLoopDetector | None:
    """Vigia só as famílias que de facto fogem."""
    return ThinkingLoopDetector() if "gemini" in str(model).lower() else None
