"""Detecção de raciocínio em fuga.

Os limiares vêm do OMP, que os calibrou contra 536 mil blocos reais de raciocínio. Um
falso negativo custa a quota de um turno; um falso positivo mata uma resposta legítima.
Para cada forma de fuga há um caso vizinho que **não** deve disparar.
"""

from __future__ import annotations

import random
from typing import ClassVar

from litellm_mysubs.wire.thinking_loop import (
    EXACT_SHORT_MIN_REPEATED_CHARS,
    LEX_STALL_MIN_RUN,
    SEGMENT_CHAR_CAP,
    SEGMENT_MIN_CLUSTER,
    SEGMENT_MIN_COUNT,
    ThinkingLoopDetector,
    detect_exact_suffix_cycle,
    guard_for,
    jaccard,
    normalize_segment,
    trigram_shingles,
)


def paragraph(text: str) -> str:
    """Um segmento fecha num parágrafo."""
    return f"{text}\n\n"


#: Frase longa o suficiente para passar o piso de 60 chars normalizados.
FILLER = (
    "portanto convem rever aquilo que estava previsto para este momento da analise "
    "em curso e ponderar as alternativas disponiveis antes de avancar"
)


class TestExactCycle:
    def test_short_cycle_needs_four_repeats(self) -> None:
        """Ciclos curtos (<=60 chars) exigem 4 repetições e 180 caracteres."""
        unit = "vou verificar o ficheiro outra vez. "
        assert detect_exact_suffix_cycle(unit * 3) is None
        found = detect_exact_suffix_cycle(unit * 8)
        assert found is not None and found[1] >= 4

    def test_too_short_to_judge(self) -> None:
        assert detect_exact_suffix_cycle("a" * (EXACT_SHORT_MIN_REPEATED_CHARS - 1)) is None

    def test_punctuation_only_cycle_ignored(self) -> None:
        """Um ciclo sem letras é formatação, não raciocínio em fuga."""
        assert detect_exact_suffix_cycle("-=" * 200) is None

    def test_varied_text_is_fine(self) -> None:
        varied = "".join(
            f"passo {i}: analisar o modulo e registar o que se observou. " for i in range(40)
        )
        assert detect_exact_suffix_cycle(varied) is None

    def test_detector_reports_the_cycle(self) -> None:
        detector = ThinkingLoopDetector()
        reason = detector.feed("vou verificar o ficheiro outra vez. " * 10)
        assert reason is not None and "ciclo exacto" in reason

    def test_exact_detection_survives_disabled_semantics(self) -> None:
        """A detecção exacta aplica-se sempre, mesmo sem heurísticas semânticas."""
        detector = ThinkingLoopDetector(semantic_heuristics=False)
        assert detector.feed("repete isto sem parar por favor. " * 12) is not None


class TestNearDuplicateCluster:
    """Exige SEGMENT_MIN_CLUSTER segmentos parecidos, não dois."""

    def test_cluster_fires_after_warmup(self) -> None:
        detector = ThinkingLoopDetector()
        base = (
            "analisar o ficheiro de configuracao do servidor e verificar todas as entradas "
            "duplicadas que aparecem na listagem actual do sistema"
        )
        reason = None
        for index in range(SEGMENT_MIN_COUNT + SEGMENT_MIN_CLUSTER + 2):
            reason = detector.feed(paragraph(f"{base} variante {index}"))
            if reason:
                break
        assert reason is not None and "quase idênticos" in reason

    def test_two_duplicates_are_not_enough(self) -> None:
        """O OMP exige um aglomerado de 4; disparar a 2 matava raciocínio legítimo."""
        detector = ThinkingLoopDetector()
        base = (
            "analisar o ficheiro de configuracao do servidor e verificar as entradas "
            "duplicadas na listagem actual"
        )
        assert detector.feed(paragraph(base)) is None
        assert detector.feed(paragraph(base + " novamente")) is None

    def test_warmup_gate_blocks_early_detection(self) -> None:
        """Abaixo de SEGMENT_MIN_COUNT segmentos nada dispara, por muito repetido."""
        detector = ThinkingLoopDetector()
        base = (
            "exactamente o mesmo paragrafo repetido vezes sem conta para testar o "
            "aquecimento do detector de ciclos semanticos"
        )
        for _ in range(SEGMENT_MIN_COUNT - 1):
            assert detector.feed(paragraph(base)) is None

    def test_distinct_segments_pass(self) -> None:
        """Parágrafos com conteúdo realmente diferente nunca disparam.

        Variar só um número não os torna distintos: os trigramas de palavras ficam
        idênticos e o Jaccard dá 1.0 — que é precisamente o que a heurística deve apanhar.
        """
        topics = [
            "abrir o ficheiro de configuracao e confirmar que as entradas estao ordenadas",
            "comparar os totais devolvidos pela base de dados com os valores em memoria",
            "rever o historico de alteracoes para perceber quando surgiu a divergencia",
            "escrever um caso de teste que reproduza a falha observada em producao",
            "medir o tempo gasto em cada etapa do pipeline de importacao dos dados",
            "documentar a decisao tomada e as alternativas que foram descartadas",
            "verificar se o certificado usado pelo servico continua dentro da validade",
            "limpar os registos antigos que ja nao sao consultados por ninguem",
            "confirmar que o alerta dispara quando o limite configurado e ultrapassado",
            "actualizar a dependencia e correr a bateria de testes de integracao",
            "isolar a consulta lenta e avaliar se um indice resolve o problema",
            "preparar o plano de reversao caso a migracao corra mal a meio",
        ]
        detector = ThinkingLoopDetector()
        for topic in topics:
            assert detector.feed(paragraph(topic)) is None, topic


class TestLexicalStall:
    """Vocabulário reciclado sem âncoras novas."""

    POOL: ClassVar[tuple[str, ...]] = (
        "portanto", "convem", "rever", "aquilo", "previsto", "momento",
        "analise", "curso", "ponderar", "alternativas", "disponiveis",
        "antes", "avancar", "situacao", "presente", "decisao", "tomada",
        "resultado", "esperado", "processo", "seguinte", "etapa",
        "consideracao", "relevante",
    )

    def segment(self, seed: int, anchor: str | None = None) -> str:
        """Recombinação determinística do mesmo vocabulário."""
        words = random.Random(seed).sample(list(self.POOL), 16)
        text = " ".join(words)
        if anchor:
            text += f" {anchor}"
        return paragraph(text)

    def test_recycled_vocabulary_stalls(self) -> None:
        detector = ThinkingLoopDetector()
        reason = None
        for index in range(SEGMENT_MIN_COUNT + LEX_STALL_MIN_RUN + 4):
            reason = detector.feed(self.segment(500 + (index % 6)))
            if reason:
                break
        assert reason is not None and "baixa informação" in reason

    def test_fresh_anchor_resets_the_run(self) -> None:
        """Um ficheiro novo a cada parágrafo é trabalho genuíno, não enchimento."""
        detector = ThinkingLoopDetector()
        for index in range(SEGMENT_MIN_COUNT + LEX_STALL_MIN_RUN + 4):
            reason = detector.feed(self.segment(500 + (index % 6), anchor=f"modulo_{index}.py"))
            assert reason is None, f"disparou em {index}: {reason}"

    def test_same_anchor_every_paragraph_still_stalls(self) -> None:
        """Repetir uma referência fixa não é progresso: a âncora tem de ser nova."""
        detector = ThinkingLoopDetector()
        reason = None
        for index in range(SEGMENT_MIN_COUNT + LEX_STALL_MIN_RUN + 4):
            reason = detector.feed(self.segment(500 + (index % 6), anchor="config_geral.py"))
            if reason:
                break
        assert reason is not None


class TestSegmentation:
    def test_headings_are_stripped_before_analysis(self) -> None:
        """A redacção sempre diferente dos títulos inflacionaria a novidade."""
        assert normalize_segment("## Uma Seccao\ntexto real") == "uma seccao texto real"

    def test_runaway_paragraph_is_force_flushed(self) -> None:
        """Um muro de texto sem linhas em branco ainda tem de ser segmentado.

        O texto tem de ser variado: repetir a mesma palavra seria apanhado primeiro pelo
        detector de ciclo exacto, e o flush nunca chegaria a correr.
        """
        detector = ThinkingLoopDetector()
        wall = " ".join(f"termo{index}" for index in range(SEGMENT_CHAR_CAP // 4))
        detector.feed(wall)
        assert detector._count > 0

    def test_partial_delta_does_not_close_a_segment(self) -> None:
        detector = ThinkingLoopDetector()
        detector.feed("frase sem fim")
        assert detector._count == 0

    def test_flush_processes_the_final_paragraph(self) -> None:
        """O último segmento pode ser o que completa o aglomerado."""
        detector = ThinkingLoopDetector()
        detector.feed(paragraph(FILLER))
        detector.feed(FILLER)
        before = detector._count
        detector.flush()
        assert detector._count == before + 1

    def test_short_segment_ignored(self) -> None:
        detector = ThinkingLoopDetector()
        detector.feed(paragraph("ok."))
        assert detector._count == 0


class TestTextHelpers:
    def test_trigrams_are_word_based(self) -> None:
        """Trigramas de caracteres davam semelhança alta a textos sem relação."""
        assert trigram_shingles("um dois tres quatro") == {"um dois tres", "dois tres quatro"}

    def test_short_text_is_one_shingle(self) -> None:
        assert trigram_shingles("um dois") == {"um dois"}

    def test_empty_text_has_no_shingles(self) -> None:
        assert trigram_shingles("") == set()

    def test_backticks_become_words(self) -> None:
        """O conteúdo entre crases vira texto; a pontuação (incluindo `_`) é separador."""
        assert normalize_segment("usa `foo_bar` aqui") == "usa foo bar aqui"

    def test_digits_alone_are_dropped(self) -> None:
        assert normalize_segment("passo 2 de 3") == "passo de"

    def test_jaccard_bounds(self) -> None:
        assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
        assert jaccard({"a"}, {"b"}) == 0.0
        assert jaccard(set(), {"a"}) == 0.0


class TestGuardSelection:
    def test_guards_the_families_that_run_away(self) -> None:
        for model in ("gemini-3-pro", "deepseek-v3", "grok-4"):
            assert isinstance(guard_for(model), ThinkingLoopDetector), model

    def test_other_families_unguarded(self) -> None:
        for model in ("claude-opus-5", "gpt-5.5", "qwen-agent-coder"):
            assert guard_for(model) is None, model
