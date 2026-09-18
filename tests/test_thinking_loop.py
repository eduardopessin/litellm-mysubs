"""Detecção de raciocínio em fuga.

Um falso negativo custa a quota inteira de um turno; um falso positivo mata uma resposta
legítima. Os testes cobrem as quatro formas de fuga e, para cada uma, um caso vizinho que
**não** deve disparar.
"""

from __future__ import annotations

import random
from typing import ClassVar

from litellm_mysubs.wire.thinking_loop import (
    HEADER_RUNAWAY,
    STALL_SEGMENTS,
    TAIL_REPEAT,
    ThinkingLoopDetector,
    guard_for,
    trigrams,
)


def segments(*texts: str) -> str:
    """Segmentos fecham em parágrafo."""
    return "".join(f"{text}\n\n" for text in texts)


class TestVerbatimRepetition:
    def test_detects_repeated_tail(self) -> None:
        detector = ThinkingLoopDetector()
        chunk = "vou verificar o ficheiro de configuração outra vez. " * 8
        assert detector.feed(chunk) is None or True
        reason = detector.feed(chunk)
        assert reason is not None and "verbatim" in reason

    def test_short_output_never_triggers(self) -> None:
        """Abaixo de 2x a janela não há material para comparar."""
        detector = ThinkingLoopDetector()
        assert detector.feed("a" * (TAIL_REPEAT - 1)) is None

    def test_long_varied_output_is_fine(self) -> None:
        detector = ThinkingLoopDetector()
        varied = "".join(
            f"passo {i}: analisar o modulo {i} e registar o resultado. " for i in range(60)
        )
        assert detector.feed(varied) is None


class TestNearDuplicateSegments:
    def test_detects_jaccard_similarity(self) -> None:
        detector = ThinkingLoopDetector()
        base = "Analisar o ficheiro de configuracao do servidor e verificar as entradas duplicadas"
        assert detector.feed(segments(base)) is None
        reason = detector.feed(segments(base + " agora"))
        assert reason is not None and "duplicados" in reason

    def test_distinct_segments_pass(self) -> None:
        detector = ThinkingLoopDetector()
        assert (
            detector.feed(
                segments(
                    "Primeiro passo: abrir o ficheiro de configuracao e ler as entradas todas",
                    "Segundo momento: comparar os totais com o que a base de dados devolveu",
                )
            )
            is None
        )

    def test_short_segments_are_not_compared(self) -> None:
        """Um segmento curto não tem trigramas suficientes para ser conclusivo."""
        detector = ThinkingLoopDetector()
        assert detector.feed(segments("ok.", "ok.", "ok.")) is None


class TestStallDetection:
    """Estagnação é léxico já visto sem âncoras novas.

    Construir o caso exige cuidado: segmentos parecidos disparam o Jaccard **antes** da
    estagnação, e segmentos totalmente novos nunca estagnam. A forma que isola a
    estagnação é vocabulário conhecido recombinado — cada segmento tem trigramas
    distintos do anterior, mas nenhuma palavra nova.
    """

    POOL: ClassVar[tuple[str, ...]] = (
        "alfa",
        "beta",
        "gama",
        "delta",
        "epsilon",
        "zeta",
        "eta",
        "teta",
        "iota",
        "kappa",
        "lambda",
        "miu",
        "niu",
        "xi",
        "omicron",
        "pi",
        "rho",
        "sigma",
        "tau",
        "upsilon",
        "fi",
        "chi",
        "psi",
        "omega",
        "aleph",
        "bet",
        "gimel",
        "dalet",
    )

    def segment(self, pool_index: int, anchor: int | None = None) -> str:
        """Combinação determinística do vocabulário, opcionalmente com âncora nova."""
        words = random.Random(500 + pool_index).sample(list(self.POOL), 14)
        text = " ".join(words)
        if anchor is not None:
            text += f" ficheiro_{anchor}.py"
        return segments(text)

    def test_detects_no_progress(self) -> None:
        detector = ThinkingLoopDetector()
        reason = None
        for index in range(STALL_SEGMENTS + 6):
            # Os primeiros introduzem o vocabulário; a partir daí recicla-se.
            reason = detector.feed(self.segment(index if index < 6 else 6 + (index % 6)))
            if reason:
                break
        assert reason is not None and "novidade" in reason

    def test_fresh_anchor_holds_off_the_stall(self) -> None:
        """Um caminho novo é progresso, mesmo sem léxico novo.

        Com âncora, a estagnação nunca dispara dentro da janela de segmentos — o que
        acaba por disparar é o Jaccard, e só bem mais tarde.
        """
        detector = ThinkingLoopDetector()
        for index in range(STALL_SEGMENTS + 2):
            reason = detector.feed(
                self.segment(index if index < 6 else 6 + (index % 6), anchor=index)
            )
            assert reason is None, f"disparou em {index}: {reason}"


class TestHeaderRunaway:
    def test_detects_endless_headings(self) -> None:
        detector = ThinkingLoopDetector()
        reason = None
        for index in range(HEADER_RUNAWAY + 2):
            reason = detector.feed(segments(f"## Seccao {index}"))
            if reason:
                break
        assert reason is not None and "titulos" in reason

    def test_a_few_headings_are_fine(self) -> None:
        detector = ThinkingLoopDetector()
        for index in range(5):
            assert detector.feed(segments(f"## Seccao {index}")) is None


class TestMechanics:
    def test_empty_feed_is_a_noop(self) -> None:
        detector = ThinkingLoopDetector()
        assert detector.feed("") is None
        assert detector.chars == 0

    def test_counts_characters_seen(self) -> None:
        detector = ThinkingLoopDetector()
        detector.feed("doze chars!!")
        assert detector.chars == 12

    def test_partial_delta_does_not_close_a_segment(self) -> None:
        """Um delta a meio de uma frase não conta como progresso nem estagnação."""
        detector = ThinkingLoopDetector()
        detector.feed("frase sem fim")
        assert len(detector.segments) == 0

    def test_trigrams_normalise_whitespace(self) -> None:
        assert trigrams("a  b\n c") == trigrams("a b c")

    def test_guard_only_for_gemini(self) -> None:
        """Vigia-se a família que de facto foge."""
        assert isinstance(guard_for("gemini-3-pro"), ThinkingLoopDetector)
        assert guard_for("claude-opus-5") is None
        assert guard_for("gpt-5.5") is None
