"""Filtro do leak de planeamento dos modelos flash.

Um falso negativo entrega o planeamento interno ao cliente; um falso positivo apaga uma
resposta legítima. Cada teste de detecção tem um vizinho que **não** deve disparar.
"""

from __future__ import annotations

import pytest

from litellm_mysubs.wire.planning_leak import (
    PlanningLeakFilter,
    is_flash_leak_model,
    is_leak_object,
    is_leak_prefix,
)


class TestModelGate:
    @pytest.mark.parametrize("model", ["gemini-3.8-flash", "gemini-2.5-flash-lite"])
    def test_flash_family_is_filtered(self, model: str) -> None:
        assert is_flash_leak_model(model) is True

    @pytest.mark.parametrize("model", ["gemini-3-pro", "gemini-3.1-pro", "gemini-pro-agent"])
    def test_other_families_are_not(self, model: str) -> None:
        """Um `pro` que responda `{"command": "ls"}` não pode ver a resposta apagada."""
        assert is_flash_leak_model(model) is False


class TestPrefixDetection:
    @pytest.mark.parametrize(
        "text",
        ['{"thought": "vou ler o ficheiro"}', '{"tho', '{"thought"', "{", '{  "thought"  :'],
    )
    def test_recognises_partial_leaks(self, text: str) -> None:
        """Um objecto atravessa chunks: `{"thou` tem de reter, não emitir."""
        assert is_leak_prefix(text) is True

    @pytest.mark.parametrize(
        "text",
        ["texto normal", '{"outra": 1}', '{"thoughts": 1}', "  sem chaveta"],
    )
    def test_ignores_other_text(self, text: str) -> None:
        assert is_leak_prefix(text) is False

    def test_long_unclosed_text_is_not_a_prefix(self) -> None:
        """Acima de 100 chars é texto legítimo que por acaso começa por chaveta."""
        assert is_leak_prefix("{" + "x" * 200) is False


class TestObjectSignature:
    @pytest.mark.parametrize(
        "payload",
        [
            {"thought": "planeando"},
            {"_i": 1},
            {"paths": ["a"]},
            {"command": "ls"},
            {"path": "a.py", "content": "x"},
        ],
    )
    def test_leak_signatures(self, payload: dict[str, object]) -> None:
        assert is_leak_object(payload) is True

    def test_known_tool_call_is_a_leak(self) -> None:
        assert is_leak_object({"call": "read"}, frozenset({"read"})) is True

    def test_unknown_call_is_not(self) -> None:
        """Só conta como leak quando o nome é de uma tool que declarámos."""
        assert is_leak_object({"call": "outra_coisa"}, frozenset({"read"})) is False

    @pytest.mark.parametrize("payload", [{"resultado": 42}, {"path": "a.py"}, [], "texto", None])
    def test_plain_objects_pass(self, payload: object) -> None:
        assert is_leak_object(payload) is False


class TestStreamFiltering:
    def test_plain_text_passes_through(self) -> None:
        assert PlanningLeakFilter().feed("olá mundo") == "olá mundo"

    def test_whole_leak_in_one_chunk_is_dropped(self) -> None:
        leak = PlanningLeakFilter()
        assert leak.feed('{"thought": "vou ler"}') == ""
        assert leak.stripped is True

    def test_leak_split_across_chunks_is_dropped(self) -> None:
        """Decidir por chunk deixava passar tudo o que não coubesse num só."""
        leak = PlanningLeakFilter()
        assert leak.feed('{"thou') == ""
        assert leak.feed('ght": "vou ler o fich') == ""
        assert leak.feed('eiro"}') == ""
        assert leak.stripped is True

    def test_text_after_the_leak_survives(self) -> None:
        leak = PlanningLeakFilter()
        assert leak.feed('{"thought": "x"}resposta visível') == "resposta visível"

    def test_legitimate_json_is_not_dropped(self) -> None:
        """Um objecto sem assinatura de planeamento é resposta, não leak."""
        leak = PlanningLeakFilter()
        assert leak.feed('{"resultado": 42}') == '{"resultado": 42}'
        assert leak.stripped is False

    def test_unbalanced_quotes_still_close_the_object(self) -> None:
        """Um leak com aspas desequilibradas nunca fecharia pela via normal."""
        leak = PlanningLeakFilter()
        assert leak.feed('{"thought": "aspa " a mais"}') == ""
        assert leak.stripped is True

    def test_unterminated_leak_is_discarded_at_eof(self) -> None:
        """Entregá-lo seria mostrar metade do planeamento interno."""
        leak = PlanningLeakFilter()
        assert leak.feed('{"thought": "nunca fecha') == ""
        assert leak.flush() == ""
        assert leak.stripped is True

    def test_lone_brace_is_discarded_at_eof(self) -> None:
        """Uma chaveta solitária tem assinatura de leak incompleto, não é texto."""
        leak = PlanningLeakFilter()
        assert leak.feed("{") == ""
        assert leak.flush() == ""
        assert leak.stripped is True

    def test_non_leak_json_is_emitted_immediately(self) -> None:
        """Só o que tem assinatura de leak entra no buffer; o resto passa logo.

        Reter JSON legítimo atrasaria o stream sem razão nenhuma.
        """
        leak = PlanningLeakFilter()
        assert leak.feed('{"resultado": ') == '{"resultado": '
        assert leak.stripped is False

    def test_empty_feed_is_a_noop(self) -> None:
        leak = PlanningLeakFilter()
        assert leak.feed("") == ""
        assert leak.flush() == ""
