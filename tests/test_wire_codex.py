"""Contrato de wire do Codex (Responses API).

O corpo é construído de raiz, não ajustado: cada campo que falta é um comportamento que
desaparece em silêncio (reasoning sem eventos, cache sem hits, multimodal sem imagem).
"""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest

from litellm_mysubs.wire import codex


def jwt(payload: dict[str, Any]) -> str:
    """JWT sem assinatura: só o corpo importa, e não se verifica nada ao lê-lo."""
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"header.{body}.signature"


class TestModelDetection:
    @pytest.mark.parametrize("model", ["gpt-5.5", "codex", "gpt-6-astra", "openai/gpt-5.6-terra"])
    def test_recognises_codex(self, model: str) -> None:
        assert codex.is_codex_model(model) is True

    @pytest.mark.parametrize("model", ["claude-opus-5", "gemini-3-pro", "qwen-agent-coder"])
    def test_ignores_others(self, model: str) -> None:
        assert codex.is_codex_model(model) is False


class TestAliases:
    @pytest.mark.parametrize(
        ("requested", "wire"),
        [("gpt-5", "gpt-5.5"), ("codex", "gpt-5.5"), ("gpt-6", "gpt-6-astra")],
    )
    def test_family_aliases_resolve(self, requested: str, wire: str) -> None:
        """Nomes de família não prometem versão, logo resolvê-los é honesto."""
        assert codex.resolve_model(requested) == wire

    @pytest.mark.parametrize("name", ["gpt-5.4", "gpt-5.4-mini"])
    def test_version_names_are_not_remapped(self, name: str) -> None:
        """Nomeiam uma versão que a conta não serve: remapear facturava o cliente contra
        um modelo que nunca correu. A recusa do upstream é a resposta correcta."""
        assert codex.resolve_model(name) == name

    def test_provider_prefix_stripped(self) -> None:
        assert codex.resolve_model("openai/gpt-5.5") == "gpt-5.5"

    def test_learned_refusal_applies(self) -> None:
        assert codex.resolve_model("gpt-x", {"gpt-x": "gpt-5.5"}) == "gpt-5.5"


class TestWireGeneration:
    @pytest.mark.parametrize(
        ("model", "generation"),
        [("gpt-5.6-terra", 5.6), ("gpt-5.5", 5.5), ("gpt-6-astra", 6.0), ("codex", 0.0)],
    )
    def test_parses(self, model: str, generation: float) -> None:
        assert codex.wire_generation(model) == generation


class TestTokenClaims:
    def test_extracts_account_id(self) -> None:
        token = jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-1"}})
        assert codex.account_id(token) == "acct-1"

    @pytest.mark.parametrize("token", ["", "nao-e-jwt", "a.b", "a.!!!.c"])
    def test_malformed_token_is_empty(self, token: str) -> None:
        """Um token partido não pode rebentar a construção do pedido."""
        assert codex.token_claims(token) == {}
        assert codex.account_id(token) is None


class TestHeaders:
    def test_transport_identity(self) -> None:
        headers = codex.build_headers(jwt({}), window_id="w-1")
        assert headers["originator"] == "omp"
        assert headers["OpenAI-Beta"] == "responses=experimental"
        assert headers["version"] == codex.CLIENT_VERSION
        assert headers["session_id"] == "w-1"

    def test_installation_id_is_not_sent(self) -> None:
        """O OMP apaga-o explicitamente dos cabeçalhos; viaja só no envelope."""
        assert "x-codex-installation-id" not in codex.build_headers(jwt({}), window_id="w")

    def test_request_kind_is_from_the_vocabulary(self) -> None:
        """ "chat" não pertence ao conjunto "turn" | "prewarm" | "compaction"."""
        headers = codex.build_headers(jwt({}), window_id="w")
        assert json.loads(headers["x-codex-turn-metadata"])["request_kind"] == "turn"

    def test_routing_hint_carries_the_model(self) -> None:
        """O backend usa-a para escolher a rota; sem ela o encaminhamento é o default."""
        headers = codex.build_headers(jwt({}), window_id="w", model="gpt-5.5")
        assert headers["x-codex-routing-hint"] == "model=gpt-5.5"

    def test_routing_hint_includes_the_tier(self) -> None:
        headers = codex.build_headers(
            jwt({}), window_id="w", model="gpt-5.5", service_tier="priority"
        )
        assert headers["x-codex-routing-hint"] == "model=gpt-5.5;tier=priority"

    def test_session_id_from_token_wins(self) -> None:
        headers = codex.build_headers(jwt({"session_id": "s-token"}), window_id="w")
        assert headers["session_id"] == "s-token"

    def test_turn_state_echoed_when_present(self) -> None:
        """O backend devolve-o e espera-o de volta no turno seguinte."""
        headers = codex.build_headers(jwt({}), window_id="w", turn_state="st-1")
        assert headers["x-codex-turn-state"] == "st-1"

    def test_turn_state_absent_on_first_turn(self) -> None:
        headers = codex.build_headers(jwt({}), window_id="w")
        assert "x-codex-turn-state" not in headers

    def test_residency_header_only_for_constrained_workspaces(self) -> None:
        """401 "Workspace is not authorized in this region" quando falta; contas
        pessoais não têm a claim e o header não deve viajar."""
        constrained = codex.build_headers(
            jwt({"https://api.openai.com/auth": {"chatgpt_data_residency": "eu"}}),
            window_id="w",
        )
        assert constrained["x-openai-internal-codex-residency"] == "eu"

        personal = codex.build_headers(jwt({}), window_id="w")
        assert "x-openai-internal-codex-residency" not in personal

    def test_no_constraint_is_not_sent(self) -> None:
        headers = codex.build_headers(
            jwt({"https://api.openai.com/auth": {"chatgpt_data_residency": "no_constraint"}}),
            window_id="w",
        )
        assert "x-openai-internal-codex-residency" not in headers


class TestMultimodal:
    def test_image_url_becomes_input_image(self) -> None:
        part = codex.image_part(
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}}
        )
        assert part == {
            "type": "input_image",
            "image_url": "data:image/png;base64,x",
            "detail": "auto",
        }

    def test_detail_original_survives_when_host_supports_it(self) -> None:
        """`original` é o único nível que preserva a resolução nativa de uma screenshot;
        forçá-lo sempre a "auto" degradava-a contra hosts que o servem."""
        part = codex.image_part({"image_url": {"url": "u", "detail": "original"}})
        assert part is not None and part["detail"] == "original"

    def test_detail_original_degrades_when_host_rejects_it(self) -> None:
        """Hosts como o GitHub Copilot devolvem 400 a `original`; degradar salva o pedido
        em vez de o perder."""
        part = codex.image_part(
            {"image_url": {"url": "u", "detail": "original"}}, supports_detail_original=False
        )
        assert part is not None and part["detail"] == "auto"

    def test_unknown_detail_falls_back_to_auto(self) -> None:
        part = codex.image_part({"image_url": {"url": "u", "detail": "ultra"}})
        assert part is not None and part["detail"] == "auto"

    def test_image_by_file_id_is_not_discarded(self) -> None:
        """Uma imagem já carregada no backend não tem url; sem este ramo desaparecia em
        silêncio e o modelo respondia sobre algo que nunca viu."""
        part = codex.image_part({"type": "input_image", "image_url": {"file_id": "file-7"}})
        assert part == {"type": "input_image", "detail": "auto", "file_id": "file-7"}

    def test_valid_detail_preserved(self) -> None:
        part = codex.image_part({"image_url": {"url": "u", "detail": "high"}})
        assert part is not None and part["detail"] == "high"

    def test_image_without_url_dropped(self) -> None:
        assert codex.image_part({"image_url": {}}) is None

    def test_file_data_becomes_input_file(self) -> None:
        part = codex.file_part({"file": {"file_data": "b64", "filename": "a.pdf"}})
        assert part == {"type": "input_file", "filename": "a.pdf", "file_data": "b64"}

    def test_file_id_preferred_over_data(self) -> None:
        part = codex.file_part({"file": {"file_id": "f-1", "file_data": "b64"}})
        assert part is not None and part["file_id"] == "f-1" and "file_data" not in part

    def test_empty_file_dropped(self) -> None:
        assert codex.file_part({"file": {}}) is None

    def test_parts_preserve_image_alongside_text(self) -> None:
        """Antes disto o pedido chegava só com o texto e o modelo falava de uma imagem
        que nunca viu."""
        parts = codex.content_to_parts(
            [
                {"type": "text", "text": "que cor?"},
                {"type": "image_url", "image_url": {"url": "u"}},
            ]
        )
        assert [p["type"] for p in parts] == ["input_text", "input_image"]

    def test_assistant_parts_use_output_text(self) -> None:
        parts = codex.content_to_parts("resposta", assistant=True)
        assert parts == [{"type": "output_text", "text": "resposta"}]

    def test_empty_content_yields_no_parts(self) -> None:
        assert codex.content_to_parts("") == []
        assert codex.content_to_parts(None) == []


class TestCallIds:
    def test_composite_joins_pair(self) -> None:
        """Sem o par exacto, chamadas paralelas desalinham-se no replay."""
        assert codex.composite_call_id("call-1", "item-9") == "call-1|item-9"

    def test_identical_ids_not_doubled(self) -> None:
        assert codex.composite_call_id("x", "x") == "x"

    def test_split_recovers_call_id(self) -> None:
        assert codex.split_call_id("call-1|item-9") == "call-1"

    def test_split_tolerates_plain_id(self) -> None:
        assert codex.split_call_id("call-1") == "call-1"

    def test_generates_id_when_both_missing(self) -> None:
        assert codex.composite_call_id(None, None).startswith("call_")


class TestToolPairRepair:
    def test_orphan_output_becomes_message(self) -> None:
        """Um histórico truncado traz outputs sem a chamada; o Responses rejeita-os."""
        items = codex.repair_tool_pairs(
            [{"type": "function_call_output", "call_id": "orphan", "output": "perdido"}]
        )
        assert items[0]["type"] == "message"
        assert "orphan" in items[0]["content"]

    def test_call_without_output_gets_placeholder(self) -> None:
        items = codex.repair_tool_pairs(
            [{"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"}]
        )
        assert items[1]["type"] == "function_call_output"
        assert items[1]["call_id"] == "c1"

    def test_complete_pair_untouched(self) -> None:
        pair: list[dict[str, Any]] = [
            {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "ok"},
        ]
        assert codex.repair_tool_pairs(pair) == pair

    def test_orphan_custom_tool_call_gets_custom_output(self) -> None:
        """Um `custom_tool_call` órfão dava 400 por não ser indexado; e o output que o
        fecha tem de ser do mesmo tipo, senão o 400 volta."""
        items = codex.repair_tool_pairs(
            [{"type": "custom_tool_call", "call_id": "c2", "name": "f", "input": "x"}]
        )
        assert [i["type"] for i in items] == ["custom_tool_call", "custom_tool_call_output"]

    def test_orphan_computer_call_becomes_note(self) -> None:
        """A screenshot que faltou não se sintetiza: a chamada passa a nota, com o texto
        exacto que o OMP usa."""
        items = codex.repair_tool_pairs([{"type": "computer_call", "call_id": "c3"}])
        assert items == [
            {
                "type": "message",
                "role": "assistant",
                "content": (
                    "[Computer call interrupted before a screenshot was recorded; call_id=c3]"
                ),
            }
        ]

    def test_mismatched_kinds_do_not_pair(self) -> None:
        """Emparelhar por `call_id` só fazia um `custom` output "fechar" um `function`
        call; o backend recusa a troca e ambas as metades precisam de reparação."""
        items = codex.repair_tool_pairs(
            [
                {"type": "function_call", "call_id": "c4", "name": "f", "arguments": "{}"},
                {"type": "custom_tool_call_output", "call_id": "c4", "output": "ok"},
            ]
        )
        assert [i["type"] for i in items] == [
            "function_call",
            "function_call_output",
            "message",
        ]

    def test_complete_custom_pair_untouched(self) -> None:
        pair: list[dict[str, Any]] = [
            {"type": "custom_tool_call", "call_id": "c5", "name": "f", "input": "x"},
            {"type": "custom_tool_call_output", "call_id": "c5", "output": "ok"},
        ]
        assert codex.repair_tool_pairs(pair) == pair


class TestMessagesToInput:
    def test_first_system_prompt_goes_to_instructions(self) -> None:
        """`instructions` é o prompt base que o backend cacheia; mandá-lo como item
        developer perde o tratamento e o hit de cache."""
        instructions, items = codex.messages_to_input([{"role": "system", "content": "regra"}])
        assert instructions == "regra"
        assert not any(i.get("role") == "developer" for i in items)

    def test_extra_system_prompts_become_developer_items(self) -> None:
        """`instructions` é uma string: o segundo prompt não cabe lá e perdia-se."""
        instructions, items = codex.messages_to_input(
            [
                {"role": "system", "content": "base"},
                {"role": "system", "content": "extra"},
                {"role": "user", "content": "olá"},
            ]
        )
        assert instructions == "base"
        assert items[0] == {
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_text", "text": "extra"}],
        }
        assert items[1]["role"] == "user"

    def test_developer_only_input_promotes_last_instruction_to_user(self) -> None:
        """Sem um turno visível o backend devolve resposta vazia; promover a última
        instrução dá-lhe algo a que responder."""
        _, items = codex.messages_to_input(
            [
                {"role": "system", "content": "base"},
                {"role": "system", "content": "faz isto"},
            ]
        )
        assert items[-1] == {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "faz isto"}],
        }

    def test_single_system_prompt_promotes_instructions_to_user(self) -> None:
        """Só um system prompt: `instructions` é o único texto que existe, e o input
        ficaria vazio."""
        instructions, items = codex.messages_to_input([{"role": "system", "content": "regra"}])
        assert instructions == "regra"
        assert items == [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "regra"}],
            }
        ]

    def test_user_turn_suppresses_promotion(self) -> None:
        """Com turno de utilizador não se duplica a instrução no input."""
        _, items = codex.messages_to_input(
            [{"role": "system", "content": "base"}, {"role": "user", "content": "olá"}]
        )
        assert [i["content"][0]["text"] for i in items] == ["olá"]

    def test_unknown_role_falls_back_to_user(self) -> None:
        _, items = codex.messages_to_input([{"role": "bizarro", "content": "x"}])
        assert items[0]["role"] == "user"

    def test_tool_message_becomes_function_call_output(self) -> None:
        _, items = codex.messages_to_input(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "resultado"},
            ]
        )
        assert [i["type"] for i in items] == ["function_call", "function_call_output"]

    def test_dict_arguments_are_serialised(self) -> None:
        """O Responses exige arguments como string JSON."""
        _, items = codex.messages_to_input(
            [
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "c", "function": {"name": "f", "arguments": {"a": 1}}}],
                },
                {"role": "tool", "tool_call_id": "c", "content": "r"},
            ]
        )
        assert items[0]["arguments"] == '{"a": 1}'


class TestTools:
    def test_function_tool_flattened(self) -> None:
        tools = codex.tools_to_codex_tools(
            [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]
        )
        assert tools == [
            {
                "type": "function",
                "name": "read",
                "description": "",
                "parameters": {"type": "object"},
            }
        ]

    def test_hosted_tool_passes_with_own_spec(self) -> None:
        """Não têm `function` e eram descartadas antes disto."""
        tools = codex.tools_to_codex_tools([{"type": "web_search"}])
        assert tools == [{"type": "web_search"}]

    def test_nameless_tool_dropped(self) -> None:
        assert codex.tools_to_codex_tools([{"type": "function", "function": {}}]) is None

    def test_no_tools_is_none(self) -> None:
        assert codex.tools_to_codex_tools([]) is None
        assert codex.tools_to_codex_tools(None) is None

    def test_tool_choice_drops_function_level(self) -> None:
        assert codex.tool_choice({"type": "function", "function": {"name": "read"}}) == {
            "type": "function",
            "name": "read",
        }

    def test_string_tool_choice_passes(self) -> None:
        assert codex.tool_choice("auto") == "auto"


class TestRequestBody:
    def test_reasoning_object_always_present(self) -> None:
        """Sem ele, zero eventos response.reasoning_summary_text.delta."""
        body = codex.build_request_body("gpt-5.5", [{"role": "user", "content": "x"}])
        assert body["reasoning"] == {"effort": "medium", "summary": "auto"}

    def test_all_turns_context_is_not_forced(self) -> None:
        """O OMP só o força no transporte Lite e apaga-o nos modelos que não o suportam."""
        body = codex.build_request_body("gpt-5.5", [{"role": "user", "content": "x"}])
        assert "context" not in body["reasoning"]

    def test_encrypted_reasoning_is_requested(self) -> None:
        """Sem isto não há replay de raciocínio num histórico stateless."""
        body = codex.build_request_body("gpt-5.5", [{"role": "user", "content": "x"}])
        assert body["include"] == ["reasoning.encrypted_content"]

    def test_effort_none_adds_juice_item_on_new_generations(self) -> None:
        """GPT-5.6+ continua a reservar juice com o reasoning desligado.

        O valor é o do effort pedido, não zero: desligar o raciocínio não significa que o
        modelo deva ficar sem orçamento nenhum.
        """
        body = codex.build_request_body(
            "gpt-5.6-terra", [{"role": "user", "content": "x"}], extra={"reasoning_effort": "none"}
        )
        assert "reasoning" not in body
        # `none` é o pedido explícito de desligar; o juice segue esse valor.
        assert (
            body["input"][-1]["content"][0]["text"] == f"# Juice: {codex.JUICE['none']} !important"
        )

    def test_juice_follows_a_separate_effort_when_given(self) -> None:
        """No OMP o desligar é um flag à parte do effort: quem pede `high` e desliga o
        raciocínio continua a reservar o orçamento de `high`."""
        body = codex.build_request_body(
            "gpt-5.6-terra",
            [{"role": "user", "content": "x"}],
            extra={"reasoning_effort": "none", "juice_effort": "high"},
        )
        assert (
            body["input"][-1]["content"][0]["text"] == f"# Juice: {codex.JUICE['high']} !important"
        )

    def test_juice_defaults_to_medium(self) -> None:
        assert codex.juice_for(None) == codex.JUICE["medium"]
        assert codex.juice_for("inventado") == codex.JUICE["medium"]

    def test_juice_follows_the_requested_effort(self) -> None:
        assert codex.juice_for("high") == 48
        assert codex.juice_for("max") == 960

    def test_effort_none_skips_juice_on_older_generations(self) -> None:
        body = codex.build_request_body(
            "gpt-5.5", [{"role": "user", "content": "x"}], extra={"reasoning_effort": "none"}
        )
        assert "reasoning" not in body
        assert all(i.get("role") != "developer" for i in body["input"])

    def test_invalid_summary_falls_back_to_auto(self) -> None:
        body = codex.build_request_body(
            "gpt-5.5",
            [{"role": "user", "content": "x"}],
            extra={"reasoning_effort": {"effort": "high", "summary": "inventado"}},
        )
        assert body["reasoning"]["summary"] == "auto"

    def test_stream_and_store_are_fixed(self) -> None:
        body = codex.build_request_body("gpt-5.5", [{"role": "user", "content": "x"}])
        assert body["stream"] is True and body["store"] is False

    def test_cache_key_is_the_session_identity(self) -> None:
        """Derivar do conteúdo fazia duas conversas com o mesmo prompt de sistema
        partilharem chave — entre sessões e entre utilizadores."""
        body = codex.build_request_body(
            "gpt-5.5", [{"role": "user", "content": "x"}], session_id="sessao-1"
        )
        assert body["prompt_cache_key"] == "sessao-1"

    def test_cache_key_survives_history_edits(self) -> None:
        """A mesma sessão mantém o hit mesmo com a cabeça da conversa editada."""
        first = codex.build_request_body(
            "gpt-5.5", [{"role": "user", "content": "original"}], session_id="s"
        )
        later = codex.build_request_body(
            "gpt-5.5", [{"role": "user", "content": "editada"}], session_id="s"
        )
        assert first["prompt_cache_key"] == later["prompt_cache_key"]

    def test_cache_can_be_disabled(self) -> None:
        """Sem isto não havia forma de o chamador dispensar o cache."""
        body = codex.build_request_body(
            "gpt-5.5",
            [{"role": "user", "content": "x"}],
            extra={"cache_retention": "none"},
            session_id="s",
        )
        assert "prompt_cache_key" not in body

    def test_no_cache_key_without_session(self) -> None:
        body = codex.build_request_body("gpt-5.5", [{"role": "user", "content": "x"}])
        assert "prompt_cache_key" not in body

    def test_service_tier_forwarded(self) -> None:
        body = codex.build_request_body(
            "gpt-5.5", [{"role": "user", "content": "x"}], extra={"service_tier": "priority"}
        )
        assert body["service_tier"] == "priority"

    def test_alias_resolved_in_body(self) -> None:
        body = codex.build_request_body("codex", [{"role": "user", "content": "x"}])
        assert body["model"] == "gpt-5.5"
