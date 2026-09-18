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
        headers = codex.build_headers(jwt({}), installation_id="i-1", window_id="w-1")
        assert headers["originator"] == "pi"
        assert headers["OpenAI-Beta"] == "responses=experimental"
        assert headers["x-codex-installation-id"] == "i-1"
        assert headers["session_id"] == "w-1"

    def test_session_id_from_token_wins(self) -> None:
        headers = codex.build_headers(
            jwt({"session_id": "s-token"}), installation_id="i", window_id="w"
        )
        assert headers["session_id"] == "s-token"

    def test_turn_state_echoed_when_present(self) -> None:
        """O backend devolve-o e espera-o de volta no turno seguinte."""
        headers = codex.build_headers(
            jwt({}), installation_id="i", window_id="w", turn_state="st-1"
        )
        assert headers["x-codex-turn-state"] == "st-1"

    def test_turn_state_absent_on_first_turn(self) -> None:
        headers = codex.build_headers(jwt({}), installation_id="i", window_id="w")
        assert "x-codex-turn-state" not in headers

    def test_residency_header_only_for_constrained_workspaces(self) -> None:
        """401 "Workspace is not authorized in this region" quando falta; contas
        pessoais não têm a claim e o header não deve viajar."""
        constrained = codex.build_headers(
            jwt({"https://api.openai.com/auth": {"chatgpt_data_residency": "eu"}}),
            installation_id="i",
            window_id="w",
        )
        assert constrained["x-openai-internal-codex-residency"] == "eu"

        personal = codex.build_headers(jwt({}), installation_id="i", window_id="w")
        assert "x-openai-internal-codex-residency" not in personal

    def test_no_constraint_is_not_sent(self) -> None:
        headers = codex.build_headers(
            jwt({"https://api.openai.com/auth": {"chatgpt_data_residency": "no_constraint"}}),
            installation_id="i",
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

    def test_detail_original_rewritten(self) -> None:
        """O Codex recusa detail: "original"."""
        part = codex.image_part({"image_url": {"url": "u", "detail": "original"}})
        assert part is not None and part["detail"] == "auto"

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


class TestMessagesToInput:
    def test_system_becomes_developer(self) -> None:
        items = codex.messages_to_input([{"role": "system", "content": "regra"}])
        assert items[0]["role"] == "developer"

    def test_unknown_role_falls_back_to_user(self) -> None:
        items = codex.messages_to_input([{"role": "bizarro", "content": "x"}])
        assert items[0]["role"] == "user"

    def test_tool_message_becomes_function_call_output(self) -> None:
        items = codex.messages_to_input(
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
        items = codex.messages_to_input(
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
        assert body["reasoning"] == {"effort": "medium", "summary": "auto", "context": "all_turns"}

    def test_effort_none_adds_juice_item_on_new_generations(self) -> None:
        """GPT-5.6+ continua a reservar juice com o reasoning desligado."""
        body = codex.build_request_body(
            "gpt-5.6-terra", [{"role": "user", "content": "x"}], extra={"reasoning_effort": "none"}
        )
        assert "reasoning" not in body
        assert body["input"][-1]["content"][0]["text"] == "# Juice: 0 !important"

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

    def test_cache_key_stable_across_turns(self) -> None:
        """A chave vem da cabeça da conversa; mudá-la a cada turno anulava o cache."""
        head = [{"role": "system", "content": "regra"}, {"role": "user", "content": "primeira"}]
        first = codex.build_request_body("gpt-5.5", head)
        later = codex.build_request_body(
            "gpt-5.5",
            [*head, {"role": "assistant", "content": "r"}, {"role": "user", "content": "b"}],
        )
        assert first["prompt_cache_key"] == later["prompt_cache_key"]

    def test_cache_key_differs_per_conversation(self) -> None:
        a = codex.build_request_body("gpt-5.5", [{"role": "user", "content": "um"}])
        b = codex.build_request_body("gpt-5.5", [{"role": "user", "content": "dois"}])
        assert a["prompt_cache_key"] != b["prompt_cache_key"]

    def test_no_cache_key_without_head_messages(self) -> None:
        body = codex.build_request_body("gpt-5.5", [{"role": "assistant", "content": "x"}])
        assert "prompt_cache_key" not in body

    def test_service_tier_forwarded(self) -> None:
        body = codex.build_request_body(
            "gpt-5.5", [{"role": "user", "content": "x"}], extra={"service_tier": "priority"}
        )
        assert body["service_tier"] == "priority"

    def test_alias_resolved_in_body(self) -> None:
        body = codex.build_request_body("codex", [{"role": "user", "content": "x"}])
        assert body["model"] == "gpt-5.5"
