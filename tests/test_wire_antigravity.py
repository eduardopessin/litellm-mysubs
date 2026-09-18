"""Contrato de wire do Antigravity.

O envelope é o mais rico dos três: mapeamento de modelo por effort, thinking por budget ou
por nível, assinaturas de raciocínio entre turnos e media dentro de tool results. Cada
campo omitido é um comportamento que desaparece em silêncio.
"""

from __future__ import annotations

from typing import Any

import pytest

from litellm_mysubs.wire import antigravity as ag
from litellm_mysubs.wire.antigravity_models import ModelCatalog, ModelNotServedError, map_model

PNG = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
REQUEST_ID = "agent/abc/1000/def/1"


def payload(messages: list[Any], model: str = "gemini-3-pro", **kwargs: Any) -> dict[str, Any]:
    return ag.build_payload(model, messages, "proj-1", REQUEST_ID, **kwargs)


class TestEnvelope:
    def test_fixed_envelope_fields(self) -> None:
        """`labels` e `sessionId` não entram: o endpoint devolve 400 Unknown name."""
        body = payload([{"role": "user", "content": "x"}])
        assert body["userAgent"] == "antigravity"
        assert body["requestType"] == "agent"
        assert body["project"] == "proj-1"
        assert set(body) == {"project", "requestId", "model", "userAgent", "requestType", "request"}

    def test_system_becomes_native_instruction(self) -> None:
        """O campo nativo é aceite com role "user"; o splice no primeiro turno deixou de
        ser necessário."""
        body = payload([{"role": "system", "content": "regra"}, {"role": "user", "content": "x"}])
        assert body["request"]["systemInstruction"] == {
            "role": "user",
            "parts": [{"text": "regra"}],
        }
        assert body["request"]["contents"][0]["role"] == "user"

    def test_assistant_becomes_model_role(self) -> None:
        body = payload([{"role": "assistant", "content": "resposta"}])
        assert body["request"]["contents"][0]["role"] == "model"

    def test_default_output_ceiling(self) -> None:
        body = payload([{"role": "user", "content": "x"}])
        assert body["request"]["generationConfig"]["maxOutputTokens"] == 64000

    def test_accepts_openai_spelling_of_max_tokens(self) -> None:
        """O OMP envia max_completion_tokens; ignorá-lo substituía o tecto do cliente."""
        body = payload([{"role": "user", "content": "x"}], extra={"max_completion_tokens": 1234})
        assert body["request"]["generationConfig"]["maxOutputTokens"] == 1234


class TestThinkingConfig:
    def test_always_present(self) -> None:
        """Omitir faz o CCA reaplicar defaults e facturar thinking sem devolver texto."""
        body = payload([{"role": "user", "content": "x"}])
        assert "thinkingConfig" in body["request"]["generationConfig"]

    def test_effort_none_disables_thoughts(self) -> None:
        body = payload([{"role": "user", "content": "x"}], extra={"reasoning_effort": "none"})
        config = body["request"]["generationConfig"]["thinkingConfig"]
        assert config["includeThoughts"] is False

    def test_minimal_never_goes_on_the_wire(self) -> None:
        """400 "Thinking level MINIMAL is not supported for this model"."""
        body = payload(
            [{"role": "user", "content": "x"}],
            model="gemini-3.5-flash",
            extra={"reasoning_effort": "minimal"},
        )
        config = body["request"]["generationConfig"]["thinkingConfig"]
        assert config.get("thinkingLevel") == "LOW"

    def test_catalog_budget_wins_over_level(self) -> None:
        """Com catálogo usa-se o budget anunciado para a variante."""
        catalog = ModelCatalog(
            ids=("gemini-3-pro-low",),
            info={"gemini-3-pro-low": {"thinkingBudget": 1000}},
            fetched_at=1.0,
        )
        body = payload([{"role": "user", "content": "x"}], catalog=catalog)
        config = body["request"]["generationConfig"]["thinkingConfig"]
        assert config["thinkingBudget"] == 1000
        assert "thinkingLevel" not in config

    def test_min_budget_used_to_disable(self) -> None:
        catalog = ModelCatalog(
            ids=("gemini-3-pro-low",),
            info={"gemini-3-pro-low": {"minThinkingBudget": 128}},
            fetched_at=1.0,
        )
        body = payload(
            [{"role": "user", "content": "x"}],
            catalog=catalog,
            extra={"reasoning_effort": "none"},
        )
        assert body["request"]["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 128


class TestModelMapping:
    def test_explicit_variant_is_respected(self) -> None:
        """Pedir -tiered não pode acabar em -low porque o effort assim decidiu."""
        catalog = ModelCatalog(
            ids=("gemini-3.8-flash-tiered", "gemini-3.8-flash-low"), fetched_at=1.0
        )
        assert map_model("gemini-3.8-flash-tiered", "low", catalog) == "gemini-3.8-flash-tiered"

    def test_effort_picks_variant_from_catalog(self) -> None:
        catalog = ModelCatalog(
            ids=("gemini-3.8-flash-low", "gemini-3.8-flash-high"), fetched_at=1.0
        )
        assert map_model("gemini-3.8-flash", "high", catalog) == "gemini-3.8-flash-high"

    def test_broken_variant_never_chosen(self) -> None:
        """gemini-3.1-pro-high está no catálogo e devolve 400 INVALID_ARGUMENT."""
        catalog = ModelCatalog(ids=("gemini-3.1-pro-high", "gemini-pro-agent"), fetched_at=1.0)
        assert map_model("gemini-3.1-pro", "high", catalog) == "gemini-pro-agent"

    def test_unknown_name_raises_instead_of_substituting(self) -> None:
        """O wildcard gemini-* faria um nome inventado responder como 2.5-flash."""
        with pytest.raises(ModelNotServedError):
            map_model("gemini-inventado-9")

    def test_thinking_suffix_not_peeled(self) -> None:
        """gemini-3.8-flash-thinking não existe; descascá-lo servia -low em silêncio."""
        with pytest.raises(ModelNotServedError):
            map_model("gemini-3.8-flash-thinking")

    def test_static_map_used_without_catalog(self) -> None:
        assert map_model("gemini-3-pro") == "gemini-3-pro-low"

    def test_deprecated_ids_excluded_from_catalog(self) -> None:
        catalog = ModelCatalog()
        catalog.update(
            {
                "models": {"gemini-3-pro-low": {}, "gemini-3.1-pro-high": {}},
                "deprecatedModelIds": ["gemini-3.1-pro-high"],
            },
            now=1.0,
        )
        assert catalog.ids == ("gemini-3-pro-low",)


class TestMultimodal:
    def test_data_uri_becomes_bare_base64(self) -> None:
        """O prefixo data: dá 400 "Invalid value at ... inline_data.data"."""
        part = ag.media_from_url(PNG)
        assert part is not None
        assert part["inlineData"]["mimeType"] == "image/png"
        assert not part["inlineData"]["data"].startswith("data:")

    def test_gs_uri_becomes_file_data(self) -> None:
        part = ag.media_from_url("gs://bucket/x.pdf", "application/pdf")
        assert part == {"fileData": {"mimeType": "application/pdf", "fileUri": "gs://bucket/x.pdf"}}

    def test_web_url_needs_a_fetcher(self) -> None:
        """fileData com um URL da web dá 404 Requested entity was not found."""
        with pytest.raises(ag.MediaFetchError):
            ag.media_from_url("https://example.com/a.png")

    def test_web_url_is_inlined_when_fetchable(self) -> None:
        part = ag.media_from_url(
            "https://example.com/a.png",
            fetch=lambda _url: ag.FetchedMedia("image/png", b"bytes"),
        )
        assert part is not None and part["inlineData"]["mimeType"] == "image/png"

    def test_oversized_media_raises(self) -> None:
        with pytest.raises(ag.MediaTooLargeError):
            ag.inline_part("image/png", b"x" * (ag.INLINE_MAX_BYTES + 1))

    def test_text_and_media_keep_input_order(self) -> None:
        parts = ag.content_parts(
            [{"type": "text", "text": "cor?"}, {"type": "image_url", "image_url": {"url": PNG}}]
        )
        assert "text" in parts[0] and "inlineData" in parts[1]

    def test_pdf_mime_from_filename(self) -> None:
        part = ag.media_part({"type": "file", "file": {"file_data": PNG, "filename": "doc.pdf"}})
        assert part is not None


class TestToolCalls:
    def test_function_ids_only_on_gemini_3(self) -> None:
        messages: list[Any] = [
            {
                "role": "assistant",
                "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "r"},
        ]
        modern = payload(messages, model="gemini-3-pro")["request"]["contents"]
        assert modern[0]["parts"][0]["functionCall"]["id"] == "c1"

        legacy = payload(messages, model="gemini-2.5-flash")["request"]["contents"]
        assert "id" not in legacy[0]["parts"][0]["functionCall"]

    def test_sentinel_used_once_per_request(self) -> None:
        """O CCA valida a assinatura do primeiro functionCall do turno."""
        body = payload(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "f", "arguments": "{}"}},
                        {"id": "c2", "function": {"name": "g", "arguments": "{}"}},
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "r1"},
                {"role": "tool", "tool_call_id": "c2", "content": "r2"},
            ]
        )
        parts = body["request"]["contents"][0]["parts"]
        assert parts[0]["thoughtSignature"] == ag.SIGNATURE_SENTINEL
        assert "thoughtSignature" not in parts[1]

    def test_real_signature_beats_sentinel(self) -> None:
        body = payload(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "c1|sig-real", "function": {"name": "f", "arguments": "{}"}}
                    ],
                },
                {"role": "tool", "tool_call_id": "c1|sig-real", "content": "r"},
            ]
        )
        assert body["request"]["contents"][0]["parts"][0]["thoughtSignature"] == "sig-real"

    def test_remembered_signature_is_used(self) -> None:
        body = payload(
            [
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "r"},
            ],
            thought_signatures={"c1": "sig-lembrada"},
        )
        assert body["request"]["contents"][0]["parts"][0]["thoughtSignature"] == "sig-lembrada"

    def test_invalid_arguments_preserved_as_raw(self) -> None:
        """Deitar fora os argumentos perdia a intenção da chamada."""
        body = payload(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "f", "arguments": "nao-json"}}
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "r"},
            ]
        )
        assert body["request"]["contents"][0]["parts"][0]["functionCall"]["args"] == {
            "__raw": "nao-json"
        }

    def test_tool_results_are_grouped_in_one_turn(self) -> None:
        body = payload(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "f", "arguments": "{}"}},
                        {"id": "c2", "function": {"name": "g", "arguments": "{}"}},
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "r1"},
                {"role": "tool", "tool_call_id": "c2", "content": "r2"},
            ]
        )
        responses = body["request"]["contents"][1]
        assert responses["role"] == "user" and len(responses["parts"]) == 2

    def test_tool_name_recovered_from_the_call(self) -> None:
        """O formato OpenAI não traz o nome no resultado."""
        body = payload(
            [
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "c1", "function": {"name": "ler", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "r"},
            ]
        )
        assert body["request"]["contents"][1]["parts"][0]["functionResponse"]["name"] == "ler"

    def test_media_travels_inside_function_response(self) -> None:
        """Medido: a imagem em functionResponse.parts é vista em todas as gerações."""
        body = payload(
            [
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "c1", "function": {"name": "shot", "arguments": "{}"}}],
                },
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "content": [
                        {"type": "text", "text": "captura"},
                        {"type": "image_url", "image_url": {"url": PNG}},
                    ],
                },
            ]
        )
        response = body["request"]["contents"][1]["parts"][0]["functionResponse"]
        assert response["response"] == {"output": "captura"}
        assert "inlineData" in response["parts"][0]

    def test_error_result_uses_error_key(self) -> None:
        value, _ = ag.tool_result_value({"content": "falhou", "is_error": True})
        assert value == {"error": "falhou"}


class TestTools:
    def test_modern_schema_field(self) -> None:
        """parametersJsonSchema aceita OpenAPI 3.0 completo."""
        tools, _ = ag.tools_to_declarations(
            "gemini-3-pro",
            [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
        )
        assert tools is not None
        assert "parametersJsonSchema" in tools[0]["functionDeclarations"][0]

    def test_claude_uses_legacy_parameters(self) -> None:
        tools, _ = ag.tools_to_declarations(
            "claude-sonnet-4-6", [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        )
        assert tools is not None
        assert "parameters" in tools[0]["functionDeclarations"][0]

    def test_validated_is_the_default_mode(self) -> None:
        assert ag.tool_config(None, []) == {"functionCallingConfig": {"mode": "VALIDATED"}}

    @pytest.mark.parametrize(
        ("choice", "mode"), [("none", "NONE"), ("required", "ANY"), ("any", "ANY")]
    )
    def test_choice_modes(self, choice: str, mode: str) -> None:
        assert ag.tool_config(choice, [])["functionCallingConfig"]["mode"] == mode

    def test_named_choice_restricts_to_that_function(self) -> None:
        config = ag.tool_config(
            {"type": "function", "function": {"name": "read"}}, [{"name": "read"}]
        )
        assert config["functionCallingConfig"]["allowedFunctionNames"] == ["read"]

    def test_unknown_named_choice_falls_back(self) -> None:
        """Um nome que não foi declarado não pode restringir a nada."""
        config = ag.tool_config(
            {"type": "function", "function": {"name": "inexistente"}}, [{"name": "read"}]
        )
        assert config["functionCallingConfig"]["mode"] == "VALIDATED"

    def test_no_tools_means_no_config(self) -> None:
        body = payload([{"role": "user", "content": "x"}])
        assert "tools" not in body["request"] and "toolConfig" not in body["request"]
