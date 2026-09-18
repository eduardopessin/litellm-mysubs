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


class TestHeadCaching:
    """Sem âncora no head, o prefixo tools+system só é coberto pela âncora de cauda, que
    muda de posição a cada turno: o head grande e imutável é reescrito a preço cheio em
    todos os pedidos."""

    def test_last_non_deferred_tool_is_anchored(self) -> None:
        tools: list[Any] = [
            {"type": "function", "function": {"name": "a"}},
            {"type": "function", "function": {"name": "b"}},
        ]
        ant.apply_head_cache(None, tools)
        assert "cache_control" not in tools[0]
        assert tools[1]["cache_control"] == ant.cache_control()

    def test_deferred_tool_is_skipped(self) -> None:
        """Uma tool deferred não entra no prefixo verificado até ser referida, logo ancorar
        nela deixava de fora tudo o que vem antes."""
        tools: list[Any] = [
            {"type": "function", "function": {"name": "a"}},
            {"type": "function", "function": {"name": "b"}, "defer_loading": True},
        ]
        ant.apply_head_cache(None, tools)
        assert tools[1].get("cache_control") is None
        assert tools[0]["cache_control"] == ant.cache_control()

    def test_last_stable_system_block_is_anchored(self) -> None:
        """Com um sufixo volátil, ancorar na cauda do array fazia um refresh de memória
        re-facturar o head inteiro em vez de só o sufixo."""
        blocks: list[Any] = [
            {"type": "text", "text": "identidade"},
            {"type": "text", "text": "prompt estável"},
            {"type": "text", "text": "<memories>ontem comeste sopa</memories>"},
        ]
        ant.apply_head_cache(blocks, None)
        assert blocks[1]["cache_control"] == ant.cache_control()
        assert "cache_control" not in blocks[2]

    def test_all_volatile_system_falls_back_to_tail(self) -> None:
        blocks: list[Any] = [{"type": "text", "text": "<memories>x</memories>"}]
        ant.apply_head_cache(blocks, None)
        assert blocks[-1]["cache_control"] == ant.cache_control()

    def test_head_budget_is_deducted_from_messages(self) -> None:
        """O tecto de 4 é por pedido: 5 marcadores dão 400 "A maximum of 4 blocks"."""
        messages: list[Any] = [{"role": "user", "content": f"m{i}"} for i in range(5)]
        assert ant.apply_conversation_cache(messages, head_breakpoints=4) == 0
        assert ant.apply_conversation_cache(messages, head_breakpoints=3) == 1

    def test_build_request_never_exceeds_the_ceiling(self) -> None:
        kwargs: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": "regras"},
                *({"role": "user", "content": f"m{i}"} for i in range(40)),
            ],
            "tools": [{"type": "function", "function": {"name": "a"}}],
        }
        out = ant.build_request(kwargs, "claude-opus-5")
        system_blocks = out["messages"][0]["content"]
        total = ant.count_head_breakpoints(system_blocks, out["tools"]) + ant.count_breakpoints(
            out["messages"][1:]
        )
        assert total == ant.CACHE_BREAKPOINT_CEILING


class TestDecimation:
    """As duas âncoras de cauda movem-se a cada turno; quando a janela de 5 min expira não
    resta entrada viva a cobrir o prefixo antigo e ele é relido a preço cheio."""

    def test_checkpoints_at_the_fifteenth_and_thirtieth_turn(self) -> None:
        messages: list[Any] = [{"role": "user", "content": f"m{i}"} for i in range(35)]
        ant.apply_conversation_cache(messages)
        marked = {i for i, m in enumerate(messages) if ant.count_breakpoints([m])}
        # Ordinais 15 e 30 -> índices 14 e 29; as duas da cauda ficam no fim.
        assert {14, 29} <= marked
        assert {33, 34} & marked

    def test_short_conversation_has_no_checkpoint(self) -> None:
        messages: list[Any] = [{"role": "user", "content": f"m{i}"} for i in range(5)]
        assert ant.apply_conversation_cache(messages) == ant.CACHE_BREAKPOINT_MESSAGES

    def test_checkpoint_outranks_the_second_tail_anchor(self) -> None:
        """Com orçamento curto é o checkpoint estável que sobrevive: a segunda âncora de
        cauda é redundante com a primeira, o checkpoint não tem substituto."""
        messages: list[Any] = [{"role": "user", "content": f"m{i}"} for i in range(20)]
        assert ant.apply_conversation_cache(messages, head_breakpoints=2) == 2
        marked = {i for i, m in enumerate(messages) if ant.count_breakpoints([m])}
        assert marked == {14, 19}


class TestCacheRetention:
    def test_defaults_to_one_hour(self) -> None:
        """Sem ttl a entrada morre aos 5 min e a pausa entre turnos de agente passa disso,
        o que faz o prefixo ser reescrito a frio."""
        assert ant.cache_control() == {"type": "ephemeral", "ttl": "1h"}

    def test_short_retention_omits_ttl(self) -> None:
        assert ant.cache_control(None) == {"type": "ephemeral"}

    def test_extended_ttl_beta_absent_on_oauth(self) -> None:
        """No caminho OAuth o `ttl: "1h"` é honrado sem beta nenhuma.

        O OMP só a junta quando `!isOAuth`. O cabeçalho de `usage/claude.ts` traz-na e
        parece o contra-exemplo, mas é da rota de usage e traz também
        `redact-thinking-2026-02-12`, que medimos a esvaziar os blocos de raciocínio.
        """
        assert ant.EXTENDED_CACHE_TTL_BETA not in ant.build_betas(thinking=True)

    def test_adaptive_model_gets_output_config(self) -> None:
        """budget_tokens é ignorado nestes modelos; adaptive + effort é a única forma."""
        out = ant.apply_thinking_params({"reasoning_effort": "high"}, "claude-opus-5")
        assert out["thinking"] == {"type": "adaptive"}
        assert out["output_config"] == {"effort": "high"}
        assert "reasoning_effort" not in out

    def test_display_is_not_forced(self) -> None:
        """O OMP fecha `display` por suporte do modelo: 4.6+ rejeitam-no com 400.

        Medido contra esta subscrição: não dá 400, mas também não muda o raciocínio
        devolvido (opus-4-6 e sonnet-4-6 dão os mesmos chars com e sem). Não havendo
        ganho, segue-se a fonte em vez de arriscar o 400 que ela documenta.
        """
        out = ant.apply_thinking_params({"reasoning_effort": "high"}, "claude-opus-5")
        assert "display" not in out["thinking"]

    def test_forced_tool_choice_pins_effort_on_adaptive(self) -> None:
        """Omitir thinking num modelo adaptive não o desliga: a API volta a ligá-lo."""
        out = ant.apply_thinking_params(
            {"reasoning_effort": "high", "tool_choice": {"type": "any"}}, "claude-opus-5"
        )
        assert out["thinking"] == {"type": "adaptive"}
        assert out["output_config"] == {"effort": "low"}

    def test_budget_model_gets_budget_tokens(self) -> None:
        """Os degraus são os do OMP (low=4096), não metade deles."""
        out = ant.apply_thinking_params({"reasoning_effort": "low"}, "claude-haiku-4-5")
        assert out["thinking"] == {"type": "enabled", "budget_tokens": ant.EFFORT_BUDGET["low"]}
        assert "output_config" not in out

    def test_ceiling_caps_the_top_steps(self) -> None:
        """A escala do OMP é preservada; só o tecto da subscrição a corta."""
        out = ant.apply_thinking_params({"reasoning_effort": "max"}, "claude-haiku-4-5")
        assert out["thinking"]["budget_tokens"] == ant.THINKING_CEILING

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
        assert out["output_config"]["effort"] == "low"

    def test_max_tokens_preserved_up_to_the_claude_code_ceiling(self) -> None:
        out = ant.apply_thinking_params(
            {"reasoning_effort": "high", "max_tokens": 64000}, "claude-haiku-4-5"
        )
        assert out["max_tokens"] == ant.MAX_OUTPUT_TOKENS

    def test_output_gets_room_beyond_the_thinking_budget(self) -> None:
        """Sem a margem, a resposta sai truncada depois de o modelo pensar."""
        out = ant.apply_thinking_params(
            {"reasoning_effort": "high", "max_tokens": 100}, "claude-haiku-4-5"
        )
        assert out["max_tokens"] == ant.THINKING_CEILING + ant.OUTPUT_FALLBACK_BUFFER

    def test_narrow_margin_is_widened(self) -> None:
        """budget+500 não é margem: o OMP sobe sempre que falta OUTPUT_FALLBACK_BUFFER."""
        out = ant.apply_thinking_params(
            {"reasoning_effort": "high", "max_tokens": ant.THINKING_CEILING + 500},
            "claude-haiku-4-5",
        )
        assert out["max_tokens"] == ant.THINKING_CEILING + ant.OUTPUT_FALLBACK_BUFFER

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

    def test_oauth_beta_is_present(self) -> None:
        """Sem ela o servidor classifica o pedido como sendo de API key."""
        assert "oauth-2025-04-20" in ant.build_betas(thinking=False)

    def test_redact_thinking_beta_absent(self) -> None:
        """Com essa beta a Anthropic devolve thinking assinado mas vazio: 74 -> 0 chars."""
        assert "redact-thinking" not in ant.build_betas(thinking=True)

    def test_context_1m_beta_absent(self) -> None:
        """context-1m-2025-08-07 dá 429 de crédito em tokens de subscrição."""
        assert "context-1m" not in ant.build_betas(thinking=True)

    def test_effort_beta_only_when_thinking(self) -> None:
        """O OMP só a acrescenta quando o pedido pede raciocínio."""
        assert ant.EFFORT_BETA in ant.build_betas(thinking=True)
        assert ant.EFFORT_BETA not in ant.build_betas(thinking=False)

    def test_user_agent_matches_the_x_app_entrypoint(self) -> None:
        """`claude-desktop` no UA com `x-app: cli` era um fingerprint incoerente."""
        assert "(external, cli)" in ant.CLIENT_HEADERS["User-Agent"]
        assert ant.CLIENT_HEADERS["x-app"] == "cli"

    def test_token_applied_when_given(self) -> None:
        out = ant.build_request({"messages": []}, "claude-opus-5", access_token="tok-1")
        assert out["api_key"] == "tok-1"

    def test_no_token_leaves_api_key_alone(self) -> None:
        """Sem token não se apaga o que o chamador tenha posto."""
        out = ant.build_request({"messages": [], "api_key": "existente"}, "claude-opus-5")
        assert out["api_key"] == "existente"
