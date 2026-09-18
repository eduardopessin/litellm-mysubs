"""Ramos de defesa do wire da Anthropic.

Entradas malformadas, limites e caminhos de desistência. Estão separados do contrato
principal porque respondem a outra pergunta: não "que forma vai no fio", mas "o que
acontece quando a entrada não é o que se espera". Um pedido chega ao proxy vindo de
qualquer cliente, e uma excepção aqui é um 500 em vez de um pedido servido.
"""

from __future__ import annotations

from typing import Any

import pytest

from litellm_mysubs.wire import anthropic as ant


class TestMalformedInput:
    """O proxy recebe pedidos de clientes que não controlamos."""

    @pytest.mark.parametrize("message", [None, "texto", 42, []])
    def test_non_dict_message_is_not_markable(self, message: object) -> None:
        assert ant.is_markable(message) is False

    def test_non_dict_entries_ignored_when_counting(self) -> None:
        assert ant.count_breakpoints([None, "x", 42]) == 0

    def test_non_list_tool_calls_has_no_anchor(self) -> None:
        assert ant.tool_call_anchor({"tool_calls": "nao-e-lista"}) is None

    def test_non_dict_tool_call_skipped(self) -> None:
        assert ant.tool_call_anchor({"tool_calls": [None, "x"]}) is None

    def test_non_function_tool_call_skipped(self) -> None:
        """convert_to_anthropic_tool_invoke salta o que não é type: function."""
        assert ant.tool_call_anchor({"tool_calls": [{"id": "c", "type": "custom"}]}) is None

    def test_last_valid_tool_call_wins(self) -> None:
        """A âncora é a última chamada marcável, porque cobre mais prefixo."""
        message = {
            "tool_calls": [
                {"id": "a", "type": "function"},
                {"id": "srvtoolu_x", "type": "function"},
                {"id": "b", "type": "function"},
            ]
        }
        assert ant.tool_call_anchor(message) == 2

    def test_unknown_role_is_not_markable(self) -> None:
        assert ant.is_markable({"role": "funcao-estranha", "content": "x"}) is False

    def test_non_string_content_is_not_markable(self) -> None:
        assert ant.is_markable({"role": "user", "content": {"a": 1}}) is False

    def test_non_list_messages_returns_unchanged(self) -> None:
        """Um cliente pode mandar messages como string; não deve rebentar."""
        out = ant.build_request({"messages": "nao-e-lista"}, "claude-opus-5")
        assert out["messages"] == "nao-e-lista"

    def test_missing_messages_key(self) -> None:
        out = ant.build_request({}, "claude-opus-5")
        assert "messages" not in out

    def test_non_dict_extra_headers_left_alone(self) -> None:
        out = ant.build_request({"extra_headers": "x", "messages": []}, "claude-opus-5")
        assert out["extra_headers"] == "x"

    def test_system_content_list_extracts_text_blocks(self) -> None:
        client, rest = ant.split_system_messages(
            [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "instrução"},
                        {"type": "image", "source": {}},
                    ],
                }
            ]
        )
        assert client == "instrução"
        assert rest == []


class TestMarkBreakpointGivesUp:
    """Já marcado significa não voltar a marcar: dois marcadores na mesma âncora
    gastam orçamento sem cobrir mais prefixo."""

    def test_tool_message_already_marked(self) -> None:
        message: dict[str, Any] = {"role": "tool", "tool_call_id": "c", "cache_control": {}}
        assert ant.mark_breakpoint(message) is False

    def test_tool_call_already_marked(self) -> None:
        message: dict[str, Any] = {
            "role": "assistant",
            "tool_calls": [{"id": "c", "type": "function", "cache_control": {}}],
        }
        assert ant.mark_breakpoint(message) is False

    def test_text_block_already_marked(self) -> None:
        message: dict[str, Any] = {
            "role": "user",
            "content": [{"type": "text", "text": "a", "cache_control": {}}],
        }
        assert ant.mark_breakpoint(message) is False

    def test_non_list_content_cannot_be_marked(self) -> None:
        assert ant.mark_breakpoint({"role": "user", "content": {"a": 1}}) is False

    def test_string_content_becomes_marked_block(self) -> None:
        message: dict[str, Any] = {"role": "user", "content": "olá"}
        assert ant.mark_breakpoint(message) is True
        assert message["content"] == [
            {"type": "text", "text": "olá", "cache_control": {"type": "ephemeral"}}
        ]

    def test_skips_blank_and_thinking_blocks(self) -> None:
        """A âncora recai no primeiro bloco de texto real, de trás para a frente."""
        message: dict[str, Any] = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "real"},
                {"type": "thinking", "text": "raciocínio"},
                {"type": "text", "text": "   "},
            ],
        }
        assert ant.mark_breakpoint(message) is True
        assert message["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_only_unmarkable_blocks_fails(self) -> None:
        message: dict[str, Any] = {"role": "user", "content": [{"type": "image"}]}
        assert ant.mark_breakpoint(message) is False


class TestCacheDeepCopy:
    """Marcar tem de produzir estruturas novas: mutar o que o cliente mandou faz o
    marcador aparecer no histórico dele."""

    def test_tool_calls_are_copied_not_mutated(self) -> None:
        call = {"id": "c1", "type": "function", "function": {"name": "f"}}
        original = {"role": "assistant", "tool_calls": [call]}
        messages: list[Any] = [original]
        ant.apply_conversation_cache(messages)
        assert "cache_control" in messages[0]["tool_calls"][0]
        assert "cache_control" not in call

    def test_content_blocks_are_copied_not_mutated(self) -> None:
        block = {"type": "text", "text": "a"}
        messages: list[Any] = [{"role": "user", "content": [block]}]
        ant.apply_conversation_cache(messages)
        assert "cache_control" not in block

    def test_counts_marks_inside_tool_calls(self) -> None:
        messages: list[Any] = [
            {
                "role": "assistant",
                "tool_calls": [{"id": "c", "cache_control": {"type": "ephemeral"}}],
            }
        ]
        assert ant.count_breakpoints(messages) == 1

    def test_counts_message_level_marks(self) -> None:
        marked = [{"role": "tool", "cache_control": {"type": "ephemeral"}}]
        assert ant.count_breakpoints(marked) == 1

    def test_empty_marker_does_not_consume_budget(self) -> None:
        """Um ``cache_control`` vazio não é um marcador: a Anthropic conta os blocos que
        o trazem preenchido, e descontá-lo gastaria orçamento sem cobrir prefixo."""
        assert ant.count_breakpoints([{"role": "tool", "cache_control": {}}]) == 0


class TestToolChoiceShapes:
    @pytest.mark.parametrize(
        "choice",
        [{"type": "any"}, {"type": "tool"}, {"type": "function"}, "required", "any"],
    )
    def test_forced_shapes(self, choice: object) -> None:
        assert ant._forced_tool_choice(choice) is True

    @pytest.mark.parametrize("choice", ["auto", "none", {"type": "auto"}, None, 42])
    def test_free_shapes(self, choice: object) -> None:
        assert ant._forced_tool_choice(choice) is False


class TestThinkingEdges:
    def test_explicit_budget_is_capped(self) -> None:
        """Tecto de 8192 pela janela TPM curta da subscrição."""
        out = ant.apply_thinking_params(
            {"thinking": {"type": "enabled", "budget_tokens": 99999}}, "claude-haiku-4-5"
        )
        assert out["thinking"]["budget_tokens"] == 8192

    def test_adaptive_object_keeps_its_shape(self) -> None:
        out = ant.apply_thinking_params(
            {"thinking": {"type": "adaptive", "display": "summarized"}}, "claude-opus-5"
        )
        assert out["thinking"]["type"] == "adaptive"

    def test_temperature_one_is_left_alone(self) -> None:
        out = ant.apply_thinking_params(
            {"reasoning_effort": "low", "temperature": 1.0}, "claude-haiku-4-5"
        )
        assert out["temperature"] == 1.0

    def test_temperature_without_thinking_drops_reasoning(self) -> None:
        """Sem thinking activo, uma temperatura custom é do cliente e manda ela."""
        out = ant.apply_thinking_params({"temperature": 0.3}, "claude-opus-5")
        assert out["temperature"] == 0.3
        assert "thinking" not in out

    def test_default_max_tokens_when_absent(self) -> None:
        out = ant.apply_thinking_params({"reasoning_effort": "low"}, "claude-haiku-4-5")
        assert out["max_tokens"] == 16384

    def test_unknown_effort_uses_medium_default(self) -> None:
        out = ant.apply_thinking_params({"reasoning_effort": "turbo"}, "claude-opus-5")
        assert "thinking" not in out

    def test_no_thinking_returns_early(self) -> None:
        kwargs: dict[str, Any] = {"messages": []}
        assert ant.apply_thinking_params(kwargs, "claude-opus-5") is kwargs
