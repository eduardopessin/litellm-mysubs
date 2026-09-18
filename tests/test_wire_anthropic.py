"""Contrato de wire da Anthropic.

Cada teste corresponde a uma forma que foi medida contra o serviço real. O comentário diz
o que o upstream devolve quando a forma está errada — é isso que torna o teste uma
defesa, e não uma descrição da implementação.
"""

from __future__ import annotations

from typing import Any

import pytest

from litellm_mysubs.wire import anthropic as ant


class TestModelDetection:
    @pytest.mark.parametrize(
        "model",
        ["claude-opus-5", "anthropic/claude-haiku-4-5", "CLAUDE-SONNET-5"],
    )
    def test_recognises_claude(self, model: str) -> None:
        assert ant.is_anthropic_model(model) is True

    @pytest.mark.parametrize("model", ["gpt-5.5", "gemini-3-pro", "openai/qwen35b"])
    def test_ignores_other_providers(self, model: str) -> None:
        assert ant.is_anthropic_model(model) is False


class TestAdaptiveDetection:
    """Errar para adaptive dá 400; errar para budget dá 200 com 0 chars de raciocínio.

    A assimetria é o que decide o default: um modelo desconhecido tem de cair no lado
    que se detecta.
    """

    @pytest.mark.parametrize("model", ["claude-opus-5", "claude-fable-5", "claude-sonnet-5"])
    def test_adaptive_models(self, model: str) -> None:
        assert ant.is_adaptive(model) is True

    @pytest.mark.parametrize(
        "model",
        ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5", "claude-3-5-sonnet"],
    )
    def test_budget_only_models(self, model: str) -> None:
        assert ant.is_adaptive(model) is False

    def test_unknown_model_defaults_to_adaptive(self) -> None:
        """Um modelo novo tem de falhar de forma detectável, não silenciosa."""
        assert ant.is_adaptive("claude-opus-6") is True


class TestNormalizeEffort:
    def test_plain_string(self) -> None:
        assert ant.normalize_effort("HIGH") == ("high", None)

    def test_responses_route_object(self) -> None:
        """/v1/responses entrega um objecto; tratá-lo como string põe o repr no fio."""
        assert ant.normalize_effort({"effort": "medium", "summary": "auto"}) == (
            "medium",
            "auto",
        )

    def test_empty_is_none(self) -> None:
        assert ant.normalize_effort(None) == (None, None)
        assert ant.normalize_effort("") == (None, None)


class TestSystemBlocks:
    def test_identity_is_first_block(self) -> None:
        """system=[cliente] devolve 429; a identidade tem de vir primeiro."""
        blocks = ant.build_system_blocks("sê breve")
        assert blocks[0]["text"] == ant.CLAUDE_CODE_PROMPT
        assert blocks[1]["text"] == "sê breve"

    def test_identity_alone_when_no_client_prompt(self) -> None:
        assert ant.build_system_blocks("") == [{"type": "text", "text": ant.CLAUDE_CODE_PROMPT}]

    def test_client_prompt_keeps_system_authority(self) -> None:
        """Enfiar o prompt do cliente num turno user tirava-lhe autoridade de system."""
        kwargs: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": "regra do cliente"},
                {"role": "user", "content": "olá"},
            ]
        }
        out = ant.build_request(kwargs, "claude-opus-5")
        assert out["messages"][0]["role"] == "system"
        assert len(out["messages"][0]["content"]) == 2
        assert out["messages"][1]["role"] == "user"

    def test_merges_multiple_system_messages(self) -> None:
        client, rest = ant.split_system_messages(
            [
                {"role": "system", "content": "um"},
                {"role": "user", "content": "x"},
                {"role": "system", "content": "dois"},
            ]
        )
        assert client == "um\n\ndois"
        assert len(rest) == 1

    def test_strips_duplicated_identity(self) -> None:
        """Um cliente que já mande a identidade não a deve duplicar no bloco dele."""
        client, _ = ant.split_system_messages(
            [{"role": "system", "content": ant.CLAUDE_CODE_PROMPT}]
        )
        assert client == ""


class TestCacheAnchors:
    def test_tool_result_is_markable(self) -> None:
        """Recusá-lo prendia a janela à cabeça: 67% do prompt relido a preço cheio."""
        assert ant.is_markable({"role": "tool", "tool_call_id": "c1", "content": "r"}) is True

    def test_thinking_block_is_not_an_anchor(self) -> None:
        assert (
            ant.is_markable({"role": "assistant", "content": [{"type": "thinking", "text": "x"}]})
            is False
        )

    def test_hosted_tool_call_is_not_an_anchor(self) -> None:
        """server_tool_use é emitido sem cache_control pelo LiteLLM."""
        message = {
            "role": "assistant",
            "tool_calls": [{"id": "srvtoolu_1", "type": "function", "function": {"name": "s"}}],
        }
        assert ant.tool_call_anchor(message) is None

    def test_empty_content_is_not_markable(self) -> None:
        assert ant.is_markable({"role": "user", "content": "   "}) is False

    def test_marks_two_tail_messages(self) -> None:
        messages: list[Any] = [{"role": "user", "content": f"m{i}"} for i in range(5)]
        assert ant.apply_conversation_cache(messages) == 2
        assert ant.count_breakpoints(messages) == 2
        # Os marcadores ficam na cauda, que é o prefixo que interessa manter.
        assert ant.count_breakpoints(messages[-2:]) == 2

    def test_respects_client_breakpoints_within_ceiling(self) -> None:
        """5 marcadores dão 400 "A maximum of 4 blocks with cache_control"."""
        messages: list[Any] = [
            {"role": "user", "content": [{"type": "text", "text": "a", "cache_control": {}}]},
            {"role": "user", "content": [{"type": "text", "text": "b", "cache_control": {}}]},
            {"role": "user", "content": [{"type": "text", "text": "c", "cache_control": {}}]},
            {"role": "user", "content": "d"},
            {"role": "user", "content": "e"},
        ]
        ant.apply_conversation_cache(messages)
        assert ant.count_breakpoints(messages) <= ant.CACHE_BREAKPOINT_CEILING

    def test_gives_up_when_client_exhausted_budget(self) -> None:
        messages: list[Any] = [
            {"role": "user", "content": [{"type": "text", "text": str(i), "cache_control": {}}]}
            for i in range(4)
        ]
        assert ant.apply_conversation_cache(messages) == 0

    def test_synthetic_continue_is_skipped(self) -> None:
        """Um "Continue." sintético no fim não é uma âncora útil."""
        messages: list[Any] = [
            {"role": "user", "content": "pergunta real"},
            {"role": "user", "content": "Continue."},
        ]
        ant.apply_conversation_cache(messages)
        assert "cache_control" not in str(messages[1])

    def test_no_anchors_is_a_noop(self) -> None:
        messages: list[Any] = [{"role": "system", "content": "x"}]
        assert ant.apply_conversation_cache(messages) == 0


class TestThinkingParams:
    def test_adaptive_model_gets_output_config(self) -> None:
        """budget_tokens é ignorado nestes modelos; adaptive + effort é a única forma."""
        out = ant.apply_thinking_params({"reasoning_effort": "high"}, "claude-opus-5")
        assert out["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert out["output_config"] == {"effort": "high"}
        assert "reasoning_effort" not in out

    def test_budget_model_gets_budget_tokens(self) -> None:
        out = ant.apply_thinking_params({"reasoning_effort": "low"}, "claude-haiku-4-5")
        assert out["thinking"] == {"type": "enabled", "budget_tokens": 2048}
        assert "output_config" not in out

    def test_xhigh_and_max_are_distinct_steps(self) -> None:
        """Medido: out=164 em high, 273 em xhigh, 275 em max — colapsá-los perde degraus."""
        assert ant.ADAPTIVE_EFFORT["xhigh"] == "xhigh"
        assert ant.ADAPTIVE_EFFORT["max"] == "max"

    def test_temperature_forced_to_one_when_thinking(self) -> None:
        """400 "may only be set to 1 when thinking is enabled"."""
        out = ant.apply_thinking_params(
            {"reasoning_effort": "medium", "temperature": 0.7}, "claude-opus-5"
        )
        assert out["temperature"] == 1.0

    def test_low_top_p_dropped_when_thinking(self) -> None:
        """400 "`top_p` must be greater than or equal to 0.95 or unset"."""
        out = ant.apply_thinking_params(
            {"reasoning_effort": "medium", "top_p": 0.5}, "claude-opus-5"
        )
        assert "top_p" not in out

    def test_high_top_p_survives(self) -> None:
        out = ant.apply_thinking_params(
            {"reasoning_effort": "medium", "top_p": 0.99}, "claude-opus-5"
        )
        assert out["top_p"] == 0.99

    def test_effort_none_disables_thinking(self) -> None:
        out = ant.apply_thinking_params({"reasoning_effort": "none"}, "claude-opus-5")
        assert "thinking" not in out
        assert "reasoning_effort" not in out

    def test_forced_tool_choice_disables_budget_thinking(self) -> None:
        """400 "Thinking may not be enabled when tool_choice forces tool use"."""
        out = ant.apply_thinking_params(
            {"reasoning_effort": "high", "tool_choice": "required"}, "claude-haiku-4-5"
        )
        assert "thinking" not in out

    def test_forced_tool_choice_survives_on_adaptive(self) -> None:
        """Nos modelos adaptive o par é aceite (200); desligar era perder raciocínio."""
        out = ant.apply_thinking_params(
            {"reasoning_effort": "high", "tool_choice": "required"}, "claude-opus-5"
        )
        assert out["thinking"]["type"] == "adaptive"

    def test_max_tokens_raised_above_budget_never_lowered(self) -> None:
        """max_tokens=64000 é aceite; um tecto fixo truncava o que o cliente pediu."""
        out = ant.apply_thinking_params(
            {"reasoning_effort": "high", "max_tokens": 64000}, "claude-haiku-4-5"
        )
        assert out["max_tokens"] == 64000

        raised = ant.apply_thinking_params(
            {"reasoning_effort": "high", "max_tokens": 100}, "claude-haiku-4-5"
        )
        assert raised["max_tokens"] == 8192 + 2048

    def test_max_completion_tokens_renamed_not_duplicated(self) -> None:
        """Preencher as duas chaves fazia o default sobrepor o valor do cliente."""
        out = ant.apply_thinking_params(
            {"reasoning_effort": "low", "max_completion_tokens": 30000}, "claude-haiku-4-5"
        )
        assert out["max_tokens"] == 30000
        assert "max_completion_tokens" not in out

    def test_disabled_thinking_object_is_removed(self) -> None:
        out = ant.apply_thinking_params({"thinking": {"type": "disabled"}}, "claude-opus-5")
        assert "thinking" not in out


class TestBuildRequest:
    def test_non_claude_model_untouched(self) -> None:
        kwargs = {"messages": [{"role": "user", "content": "x"}], "temperature": 0.2}
        assert ant.build_request(dict(kwargs), "gpt-5.5") == kwargs

    def test_injects_client_headers(self) -> None:
        out = ant.build_request({"messages": []}, "claude-opus-5")
        assert out["extra_headers"]["x-app"] == "cli"
        assert "claude-code-20250219" in out["extra_headers"]["anthropic-beta"]

    def test_redact_thinking_beta_absent(self) -> None:
        """Com essa beta a Anthropic devolve thinking assinado mas vazio: 74 -> 0 chars."""
        assert "redact-thinking" not in ant.ANTHROPIC_BETAS

    def test_context_1m_beta_absent(self) -> None:
        """context-1m-2025-08-07 dá 429 de crédito em tokens de subscrição."""
        assert "context-1m" not in ant.ANTHROPIC_BETAS

    def test_token_applied_when_given(self) -> None:
        out = ant.build_request({"messages": []}, "claude-opus-5", access_token="tok-1")
        assert out["api_key"] == "tok-1"

    def test_no_token_leaves_api_key_alone(self) -> None:
        """Sem token não se apaga o que o chamador tenha posto."""
        out = ant.build_request({"messages": [], "api_key": "existente"}, "claude-opus-5")
        assert out["api_key"] == "existente"
