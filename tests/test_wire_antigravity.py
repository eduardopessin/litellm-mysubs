"""Antigravity wire contract.

The envelope is the richest of the three: model mapping by effort, thinking by budget or
by level, reasoning signatures across turns, and media inside tool results. Every omitted
field is a behaviour that disappears silently.
"""

from __future__ import annotations

import json
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
        """`labels` and `sessionId` do not go in: the endpoint returns 400 Unknown name."""
        body = payload([{"role": "user", "content": "x"}])
        assert body["userAgent"] == "antigravity"
        assert body["requestType"] == "agent"
        assert body["project"] == "proj-1"
        assert set(body) == {"project", "requestId", "model", "userAgent", "requestType", "request"}

    def test_system_becomes_native_instruction(self) -> None:
        """The native field is accepted with role "user"; splicing into the first turn is
        no longer necessary."""
        body = payload([{"role": "system", "content": "rule"}, {"role": "user", "content": "x"}])
        assert body["request"]["systemInstruction"] == {
            "role": "user",
            "parts": [{"text": "rule"}],
        }
        assert body["request"]["contents"][0]["role"] == "user"

    def test_assistant_becomes_model_role(self) -> None:
        body = payload([{"role": "assistant", "content": "answer"}])
        assert body["request"]["contents"][0]["role"] == "model"

    def test_default_output_ceiling(self) -> None:
        body = payload([{"role": "user", "content": "x"}])
        assert body["request"]["generationConfig"]["maxOutputTokens"] == 64000

    def test_accepts_openai_spelling_of_max_tokens(self) -> None:
        """OMP sends max_completion_tokens; ignoring it overrode the client's ceiling."""
        body = payload([{"role": "user", "content": "x"}], extra={"max_completion_tokens": 1234})
        assert body["request"]["generationConfig"]["maxOutputTokens"] == 1234


class TestThinkingConfig:
    def test_always_present(self) -> None:
        """Omitting it makes the CCA reapply defaults and bill thinking without returning text."""
        body = payload([{"role": "user", "content": "x"}])
        assert "thinkingConfig" in body["request"]["generationConfig"]

    def test_effort_none_disables_thoughts(self) -> None:
        body = payload([{"role": "user", "content": "x"}], extra={"reasoning_effort": "none"})
        config = body["request"]["generationConfig"]["thinkingConfig"]
        assert config["includeThoughts"] is False

    def test_minimal_is_served_as_minimal(self) -> None:
        """OMP has MINIMAL in the type and sends it; there is no clamp to LOW.

        Serving `minimal` as `LOW` cost more latency and more tokens than the client asked
        for, on models that accept MINIMAL.
        """
        body = payload(
            [{"role": "user", "content": "x"}],
            model="gemini-3.5-flash",
            extra={"reasoning_effort": "minimal"},
        )
        assert body["request"]["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "MINIMAL"

    def test_catalog_budget_wins_over_level(self) -> None:
        """With a catalog, the budget announced for the variant is used."""
        catalog = ModelCatalog(
            ids=("gemini-3-pro-low",),
            info={"gemini-3-pro-low": {"thinkingBudget": 1000}},
            fetched_at=1.0,
        )
        body = payload([{"role": "user", "content": "x"}], catalog=catalog)
        config = body["request"]["generationConfig"]["thinkingConfig"]
        assert config["thinkingBudget"] == 1000
        assert "thinkingLevel" not in config

    def test_suppression_is_zero_budget_not_the_catalog_minimum(self) -> None:
        """With `includeThoughts: False`, a positive budget is billed without returning any
        text at all. The catalog's `minThinkingBudget` is not zero on several variants."""
        catalog = ModelCatalog(
            ids=("gemini-3-pro-low",),
            info={"gemini-3-pro-low": {"thinkingBudget": 1000, "minThinkingBudget": 128}},
            fetched_at=1.0,
        )
        config = payload(
            [{"role": "user", "content": "x"}],
            catalog=catalog,
            extra={"reasoning_effort": "none"},
        )["request"]["generationConfig"]["thinkingConfig"]
        assert config["includeThoughts"] is False
        assert config["thinkingBudget"] == 0

    def test_suppression_without_catalog_uses_minimal(self) -> None:
        config = payload([{"role": "user", "content": "x"}], extra={"reasoning_effort": "none"})[
            "request"
        ]["generationConfig"]["thinkingConfig"]
        assert config["thinkingLevel"] == "MINIMAL"


class TestModelMapping:
    def test_explicit_variant_is_respected(self) -> None:
        """Asking for -tiered cannot end up as -low because the effort decided so."""
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
        """gemini-3.1-pro-high is in the catalog and returns 400 INVALID_ARGUMENT."""
        catalog = ModelCatalog(ids=("gemini-3.1-pro-high", "gemini-pro-agent"), fetched_at=1.0)
        assert map_model("gemini-3.1-pro", "high", catalog) == "gemini-pro-agent"

    def test_unknown_name_raises_instead_of_substituting(self) -> None:
        """The gemini-* wildcard would make a made-up name answer as 2.5-flash."""
        with pytest.raises(ModelNotServedError):
            map_model("gemini-made-up-9")

    def test_thinking_suffix_not_peeled(self) -> None:
        """gemini-3.8-flash-thinking does not exist; peeling it served -low silently."""
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
        """The data: prefix gives 400 "Invalid value at ... inline_data.data"."""
        part = ag.media_from_url(PNG)
        assert part is not None
        assert part["inlineData"]["mimeType"] == "image/png"
        assert not part["inlineData"]["data"].startswith("data:")

    def test_gs_uri_becomes_file_data(self) -> None:
        part = ag.media_from_url("gs://bucket/x.pdf", "application/pdf")
        assert part == {"fileData": {"mimeType": "application/pdf", "fileUri": "gs://bucket/x.pdf"}}

    def test_web_url_needs_a_fetcher(self) -> None:
        """fileData with a web URL gives 404 Requested entity was not found."""
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
            [{"type": "text", "text": "colour?"}, {"type": "image_url", "image_url": {"url": PNG}}]
        )
        assert "text" in parts[0] and "inlineData" in parts[1]

    def test_pdf_mime_from_filename(self) -> None:
        part = ag.media_part({"type": "file", "file": {"file_data": PNG, "filename": "doc.pdf"}})
        assert part is not None

    def test_image_replaced_by_placeholder_without_vision(self) -> None:
        """A model without vision returns 400 for the image; dropping it silently made the
        model answer about content it never received."""
        parts = ag.content_parts(
            [{"type": "text", "text": "colour?"}, {"type": "image_url", "image_url": {"url": PNG}}],
            supports_images=False,
        )
        assert parts == [{"text": "colour?"}, {"text": ag.NON_VISION_IMAGE_PLACEHOLDER}]

    def test_non_image_media_survives_without_vision(self) -> None:
        """The guard is about vision, not attachments: a PDF does not go through the image
        path."""
        parts = ag.content_parts(
            [{"type": "file", "file": {"file_data": PNG, "filename": "doc.pdf"}}],
            supports_images=False,
        )
        assert len(parts) == 1 and "inlineData" in parts[0]

    def test_blank_text_block_produces_no_part(self) -> None:
        """A `{"text": "   "}` carries no information and breaks some models served by this
        API (Claude among them)."""
        assert ag.content_parts([{"type": "text", "text": "   "}]) == []
        assert ag.content_parts("   ") == []

    def test_lone_surrogate_never_reaches_the_wire(self) -> None:
        """A lone surrogate does not encode as UTF-8: the payload blew up serialisation
        instead of being sent."""
        parts = ag.content_parts([{"type": "text", "text": "a\ud800b"}])
        json.dumps(parts)
        assert parts == [{"text": "a\ufffdb"}]

    def test_real_emoji_survives(self) -> None:
        """Replacing by position destroyed legitimate emoji coming from the client.

        An emoji in a Python `str` is **one** code point, not a surrogate pair.
        """
        assert ag.well_formed("hello 😀") == "hello 😀"

    def test_surrogate_pair_is_also_replaced(self) -> None:
        """Where OMP preserves the pair, here it has to go.

        In JavaScript `\\ud83d\\ude00` is one character and `toWellFormed()` keeps it. In
        Python they are two code points that do not encode: preserving them blows up
        `str.encode("utf-8")` and the `json.dumps(..., ensure_ascii=False)` that many HTTP
        clients use to build the body.
        """
        cleaned = ag.well_formed("pair \ud83d\ude00 here")
        cleaned.encode("utf-8")
        assert "\ud83d" not in cleaned

    def test_every_text_on_the_wire_is_encodable(self) -> None:
        """The guarantee that matters is not "no lone surrogates", it is "it serialises"."""
        payload = ag.build_payload(
            "gemini-3-pro",
            [{"role": "user", "content": "a\ud800b \ud83d\ude00 c"}],
            "proj",
            REQUEST_ID,
        )
        json.dumps(payload, ensure_ascii=False).encode("utf-8")


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
        """The CCA validates the signature of the turn's first functionCall."""
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
                        {"id": "c1|c2lnLXJlYWw=", "function": {"name": "f", "arguments": "{}"}}
                    ],
                },
                {"role": "tool", "tool_call_id": "c1|c2lnLXJlYWw=", "content": "r"},
            ]
        )
        assert body["request"]["contents"][0]["parts"][0]["thoughtSignature"] == "c2lnLXJlYWw="

    def test_remembered_signature_is_used(self) -> None:
        body = payload(
            [
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "r"},
            ],
            thought_signatures={"c1": "c2lnLWxlbWJyYWRh"},
        )
        assert body["request"]["contents"][0]["parts"][0]["thoughtSignature"] == "c2lnLWxlbWJyYWRh"

    def test_invalid_arguments_preserved_as_raw(self) -> None:
        """Throwing the arguments away lost the intent of the call."""
        body = payload(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "f", "arguments": "not-json"}}
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "r"},
            ]
        )
        assert body["request"]["contents"][0]["parts"][0]["functionCall"]["args"] == {
            "__raw": "not-json"
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
        """The OpenAI format does not carry the name in the result."""
        body = payload(
            [
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "c1", "function": {"name": "read", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "r"},
            ]
        )
        assert body["request"]["contents"][1]["parts"][0]["functionResponse"]["name"] == "read"

    def test_media_travels_inside_function_response(self) -> None:
        """Measured: the image in functionResponse.parts is seen on every generation."""
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
                        {"type": "text", "text": "capture"},
                        {"type": "image_url", "image_url": {"url": PNG}},
                    ],
                },
            ]
        )
        response = body["request"]["contents"][1]["parts"][0]["functionResponse"]
        assert response["response"] == {"output": "capture"}
        assert "inlineData" in response["parts"][0]

    def test_claude_models_keep_the_call_id(self) -> None:
        """Antigravity serves Anthropic too, and Vertex requires `tool_use.id`.

        `supports_function_ids` gated the id on the name starting with `gemini-3`, so a
        Claude served by this account sent `functionCall` with no id. The first turn passes
        — the call is made — and the second one, carrying the result back, is refused::

            HTTP 400 messages.1.content.0.tool_use.id: Field required

        Measured on the live gateway: 97 failures on `claude-sonnet-4-6` and 88 on
        `claude-opus-4-6-thinking`, every one of them on the second turn, while the same
        models served natively by Anthropic were fine.
        """
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{"id": "c1", "function": {"name": "read", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "r"},
        ]
        catalog = ModelCatalog(ids=("claude-sonnet-4-6",), info={}, fetched_at=1.0)
        body = payload(messages, model="claude-sonnet-4-6", catalog=catalog)
        call = body["request"]["contents"][0]["parts"][0]["functionCall"]
        result = body["request"]["contents"][1]["parts"][0]["functionResponse"]
        assert call["id"] == "c1", "Vertex refuses a tool_use without an id"
        assert result["id"] == "c1", "the result has to name the call it answers"

    def test_sentinel_is_per_turn_not_per_request(self) -> None:
        """The CCA requires the sentinel on the first call of **every** assistant turn.

        Marking it once per request left the following turns with bare calls, and the
        backend answered 400 on signature validation.
        """
        body = payload(
            [
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "a1", "function": {"name": "f", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": "a1", "content": "r1"},
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "b1", "function": {"name": "g", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": "b1", "content": "r2"},
            ]
        )
        turns = [c for c in body["request"]["contents"] if c["role"] == "model"]
        assert len(turns) == 2
        for turn in turns:
            assert turn["parts"][0]["thoughtSignature"] == ag.SIGNATURE_SENTINEL

    def test_invalid_signature_is_replaced_by_the_sentinel(self) -> None:
        """A non-base64 signature gives 400 and, being truthy, blocked the sentinel."""
        body = payload(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "thoughtSignature": "this is not base64!",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "r"},
            ]
        )
        assert (
            body["request"]["contents"][0]["parts"][0]["thoughtSignature"] == ag.SIGNATURE_SENTINEL
        )

    def test_secondary_calls_stay_bare_when_first_is_signed(self) -> None:
        """Turn whose first call is signed: the following ones carry no sentinel."""
        body = payload(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "c1|c2lnLXJlYWw=", "function": {"name": "f", "arguments": "{}"}},
                        {"id": "c2", "function": {"name": "g", "arguments": "{}"}},
                    ],
                },
                {"role": "tool", "tool_call_id": "c1|c2lnLXJlYWw=", "content": "r1"},
                {"role": "tool", "tool_call_id": "c2", "content": "r2"},
            ]
        )
        parts = body["request"]["contents"][0]["parts"]
        assert parts[0]["thoughtSignature"] == "c2lnLXJlYWw="
        assert "thoughtSignature" not in parts[1]

    def test_multipart_text_keeps_a_separator(self) -> None:
        """Without a separator, the last word of one part sticks to the first of the next."""
        value, _ = ag.tool_result_value(
            {"content": [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}]}
        )
        assert value == {"output": "first\nsecond"}

    def test_image_only_result_is_announced(self) -> None:
        """`output: ""` is read as a tool with no result and the model repeats the call."""
        value, media = ag.tool_result_value(
            {"content": [{"type": "image_url", "image_url": {"url": PNG}}]}
        )
        assert value == {"output": ag.IMAGE_ONLY_RESULT}
        assert len(media) == 1

    def test_error_result_uses_error_key(self) -> None:
        value, _ = ag.tool_result_value({"content": "failed", "is_error": True})
        assert value == {"error": "failed"}

    def test_tool_result_image_replaced_by_placeholder_without_vision(self) -> None:
        """The image the tool returned cannot leave the result without a trace: the model
        read a text that silenced it."""
        value, media = ag.tool_result_value(
            {
                "content": [
                    {"type": "text", "text": "capture"},
                    {"type": "image_url", "image_url": {"url": PNG}},
                ]
            },
            supports_images=False,
        )
        assert media == []
        assert value == {"output": f"capture\n{ag.NON_VISION_IMAGE_PLACEHOLDER}"}

    def test_payload_carries_the_placeholder_not_the_image(self) -> None:
        """The capability has to cross the envelope: filtering only in `content_parts` did
        not save the request the proxy sends."""
        body = payload(
            [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": PNG}}]}],
            supports_images=False,
        )
        parts = body["request"]["contents"][0]["parts"]
        assert parts == [{"text": ag.NON_VISION_IMAGE_PLACEHOLDER}]


class TestTools:
    def test_schema_always_travels_as_parameters(self) -> None:
        """`parametersJsonSchema` never reaches this backend's wire.

        OMP converts **every** declaration on the Antigravity path; the full JSON Schema
        field was a misreading of the public Gemini API.
        """
        tools, _ = ag.tools_to_declarations(
            "gemini-3-pro",
            [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
        )
        assert tools is not None
        declaration = tools[0]["functionDeclarations"][0]
        assert "parameters" in declaration
        assert "parametersJsonSchema" not in declaration

    def test_unsupported_constructs_are_sanitised(self) -> None:
        """`anyOf`/`$ref`/`not` give 400 on the CCA; sending them raw made the request fail."""
        tools, _ = ag.tools_to_declarations(
            "gemini-3-pro",
            [
                {
                    "type": "function",
                    "function": {
                        "name": "f",
                        "parameters": {
                            "type": "object",
                            "properties": {"x": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
                        },
                    },
                }
            ],
        )
        assert tools is not None
        serialised = json.dumps(tools[0]["functionDeclarations"][0]["parameters"])
        assert "anyOf" not in serialised

    def test_every_model_uses_the_same_field(self) -> None:
        """Choosing by family was a local invention: OMP does not distinguish here."""
        for model in ("gemini-3-pro", "claude-sonnet-4-6", "gemini-2.5-flash"):
            tools, _ = ag.tools_to_declarations(
                model, [{"type": "function", "function": {"name": "f", "parameters": {}}}]
            )
            assert tools is not None
            assert "parameters" in tools[0]["functionDeclarations"][0], model

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
        """A name that was not declared cannot restrict to anything."""
        config = ag.tool_config(
            {"type": "function", "function": {"name": "nonexistent"}}, [{"name": "read"}]
        )
        assert config["functionCallingConfig"]["mode"] == "VALIDATED"

    def test_no_tools_means_no_config(self) -> None:
        body = payload([{"role": "user", "content": "x"}])
        assert "tools" not in body["request"] and "toolConfig" not in body["request"]
