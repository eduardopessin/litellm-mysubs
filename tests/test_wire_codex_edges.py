"""Ramos de defesa do wire do Codex.

Entrada malformada e caminhos de descarte. O proxy recebe pedidos de clientes que não
controlamos: uma excepção aqui é um 500 em vez de um pedido servido, e um descarte
silencioso é pior — o modelo responde sobre conteúdo que nunca recebeu.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest

from litellm_mysubs.wire import codex


def jwt(payload: dict[str, Any]) -> str:
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"header.{body}.signature"


class TestMalformedInput:
    def test_non_numeric_generation_is_zero(self) -> None:
        """Um nome sem versão não pode rebentar a decisão do item de juice."""
        assert codex.wire_generation("gpt-terra-x") == 0.0

    def test_non_dict_claims_are_empty(self) -> None:
        """Um JWT cujo corpo não é um objecto não é uma identidade utilizável."""
        body = base64.urlsafe_b64encode(json.dumps(["lista"]).encode()).decode().rstrip("=")
        assert codex.token_claims(f"a.{body}.c") == {}

    def test_account_header_absent_without_claim(self) -> None:
        headers = codex.build_headers(jwt({}), installation_id="i", window_id="w")
        assert "chatgpt-account-id" not in headers

    def test_account_header_present_with_claim(self) -> None:
        headers = codex.build_headers(
            jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-9"}}),
            installation_id="i",
            window_id="w",
        )
        assert headers["chatgpt-account-id"] == "acct-9"

    def test_content_to_text_ignores_non_text_parts(self) -> None:
        """Só os blocos de texto contam para a chave de cache; uma imagem não é texto."""
        text = codex.content_to_text(
            [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {"url": "u"}}]
        )
        assert text == "a"

    def test_content_to_text_handles_scalars(self) -> None:
        assert codex.content_to_text(42) == "42"
        assert codex.content_to_text(None) == ""

    @pytest.mark.parametrize("part", [None, "texto", 42])
    def test_non_dict_parts_skipped(self, part: object) -> None:
        assert codex.content_to_parts(["ok", part]) == []

    def test_unknown_part_type_skipped(self) -> None:
        parts = codex.content_to_parts([{"type": "audio", "data": "x"}])
        assert parts == []

    def test_blank_text_part_skipped(self) -> None:
        assert codex.content_to_parts([{"type": "text", "text": ""}]) == []

    def test_broken_image_part_skipped_but_text_kept(self) -> None:
        """Um descarte não pode levar o resto do turno com ele."""
        parts = codex.content_to_parts(
            [{"type": "text", "text": "olá"}, {"type": "image_url", "image_url": {}}]
        )
        assert parts == [{"type": "input_text", "text": "olá"}]

    def test_broken_file_part_skipped(self) -> None:
        parts = codex.content_to_parts([{"type": "file", "file": {"filename": "sem-dados.pdf"}}])
        assert parts == []

    def test_file_part_accepted_inline(self) -> None:
        parts = codex.content_to_parts([{"type": "input_file", "file_data": "b64"}])
        assert parts == [{"type": "input_file", "file_data": "b64"}]

    def test_non_dict_tool_skipped(self) -> None:
        tools = codex.tools_to_codex_tools([None, {"type": "function", "function": {"name": "f"}}])
        assert tools is not None and len(tools) == 1

    def test_flat_tool_without_function_wrapper(self) -> None:
        """Alguns clientes mandam o spec achatado, sem o nível `function`."""
        tools = codex.tools_to_codex_tools([{"name": "read", "parameters": {"type": "object"}}])
        assert tools is not None and tools[0]["name"] == "read"

    def test_tool_without_parameters_gets_empty_object(self) -> None:
        """O Responses exige `parameters`; omiti-lo dá 400."""
        tools = codex.tools_to_codex_tools([{"type": "function", "function": {"name": "f"}}])
        assert tools is not None
        assert tools[0]["parameters"] == {"type": "object", "properties": {}}

    @pytest.mark.parametrize("choice", [None, "auto", 42, {"type": "auto"}])
    def test_tool_choice_passthrough(self, choice: object) -> None:
        assert codex.tool_choice(choice) == choice

    def test_tool_choice_without_name_passes_through(self) -> None:
        payload = {"type": "function", "function": {}}
        assert codex.tool_choice(payload) == payload


class TestCacheKeyEdges:
    def test_no_session_means_no_key(self) -> None:
        assert codex.prompt_cache_key(None) is None
        assert codex.prompt_cache_key("") is None

    def test_long_session_id_is_hashed(self) -> None:
        """O backend recusa chaves acima de 64 caracteres."""
        key = codex.prompt_cache_key("s" * 200)
        assert key is not None
        assert key.startswith("pc_") and len(key) <= 64

    def test_short_session_id_travels_verbatim(self) -> None:
        assert codex.prompt_cache_key("sessao-curta") == "sessao-curta"

    def test_retention_none_disables_it(self) -> None:
        assert codex.prompt_cache_key("s", cache_retention="none") is None


class TestCallIdSanitisation:
    def test_invalid_characters_are_replaced(self) -> None:
        """Um id com caracteres fora do conjunto dá 400 do backend."""
        assert codex.split_call_id("call id!") != "call id!"
        assert " " not in codex.split_call_id("call id!")

    def test_valid_id_passes_through(self) -> None:
        assert codex.split_call_id("call_abc-123") == "call_abc-123"

    def test_composite_id_keeps_only_the_call(self) -> None:
        assert codex.split_call_id("call-1|item-9") == "call-1"

    def test_newline_also_separates(self) -> None:
        """Ids reencaminhados de outro provedor trazem `\n`; cortar só em `|` deixava
        passar o segundo segmento."""
        assert codex.split_call_id("call-1\nlixo") == "call-1"

    def test_long_id_is_truncated_with_a_hash(self) -> None:
        sanitized = codex.split_call_id("c" * 200)
        assert len(sanitized) <= codex.CALL_ID_MAX_CHARS

    def test_different_long_ids_do_not_collide(self) -> None:
        """Truncar sem hash fazia dois ids distintos colapsarem no mesmo."""
        assert codex.split_call_id("a" * 100) != codex.split_call_id("b" * 100)

    def test_empty_id_gets_a_stable_placeholder(self) -> None:
        assert codex.split_call_id("").startswith("call_")


class TestOrphanRepair:
    def test_tool_name_is_preserved(self) -> None:
        """Sem o nome, o modelo não sabe o que produziu o resultado órfão."""
        items = codex.repair_tool_pairs(
            [{"type": "function_call_output", "call_id": "x", "name": "ler", "output": "r"}]
        )
        assert "[Previous ler result; call_id=x]" in items[0]["content"]

    def test_missing_name_falls_back(self) -> None:
        items = codex.repair_tool_pairs(
            [{"type": "function_call_output", "call_id": "x", "output": "r"}]
        )
        assert "[Previous tool result" in items[0]["content"]

    def test_huge_output_is_truncated(self) -> None:
        """Um ficheiro de 2 MB rebentava o limite do corpo em vez de ser cortado."""
        items = codex.repair_tool_pairs(
            [{"type": "function_call_output", "call_id": "x", "output": "y" * 40_000}]
        )
        assert "...[truncated]" in items[0]["content"]
        assert len(items[0]["content"]) < 20_000

    def test_structured_output_is_serialised_as_json(self) -> None:
        """`str()` de um dict dá aspas simples e `True`/`None` — repr Python no prompt."""
        items = codex.repair_tool_pairs(
            [{"type": "function_call_output", "call_id": "x", "output": {"ok": True}}]
        )
        assert '{"ok": true}' in items[0]["content"]


class TestBodyOptionalFields:
    def test_no_tools_key_when_absent(self) -> None:
        body = codex.build_request_body("gpt-5.5", [{"role": "user", "content": "x"}])
        assert "tools" not in body
        assert "tool_choice" not in body

    def test_tools_key_present_when_given(self) -> None:
        body = codex.build_request_body(
            "gpt-5.5",
            [{"role": "user", "content": "x"}],
            tools=[{"type": "function", "function": {"name": "f"}}],
        )
        assert body["tools"][0]["name"] == "f"

    def test_no_service_tier_key_when_absent(self) -> None:
        body = codex.build_request_body("gpt-5.5", [{"role": "user", "content": "x"}])
        assert "service_tier" not in body

    def test_empty_message_produces_no_item(self) -> None:
        """Uma mensagem sem conteúdo nem tool calls não tem nada para enviar."""
        body = codex.build_request_body("gpt-5.5", [{"role": "user", "content": ""}])
        assert body["input"] == []
