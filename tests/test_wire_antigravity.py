"""Antigravity wire contract.

The envelope is the richest of the three: model mapping by effort, thinking by budget or
by level, reasoning signatures across turns, and media inside tool results. Every omitted
field is a behaviour that disappears silently.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Final

import pytest

from litellm_mysubs.wire import antigravity as ag
from litellm_mysubs.wire.antigravity_models import ModelCatalog, ModelNotServedError, map_model

PNG = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def payload(messages: list[Any], model: str = "gemini-3-pro", **kwargs: Any) -> dict[str, Any]:
    return ag.build_payload(model, messages, "proj-1", **kwargs)


class TestEnvelope:
    def test_session_fields_ride_inside_the_request(self) -> None:
        """``labels`` and ``sessionId`` go inside ``request``, where omp's
        ``buildAntigravityRequestEnvelope`` puts them; the top level keeps its six fields.

        Measured on the live backend on 2026-09-30: ``request.sessionId`` and
        ``request.labels`` answered HTTP 200 on gemini-3-flash and on claude-sonnet-4-6.
        The 400 "Unknown name" once recorded here was for the top-level placement.
        """
        body = payload([{"role": "user", "content": "x"}])
        assert body["userAgent"] == "antigravity"
        assert body["requestType"] == "agent"
        assert body["project"] == "proj-1"
        assert set(body) == {"project", "requestId", "model", "userAgent", "requestType", "request"}
        assert list(body["request"]) == [
            "contents",
            "labels",
            "generationConfig",
            "sessionId",
        ]

    def test_without_state_the_session_id_comes_from_the_first_user_text(self) -> None:
        """omp's ``deriveAntigravitySessionId``: the first 8 bytes of the text's SHA-256,
        masked to 63 bits, as a negative decimal; the step is 2."""
        body = payload([{"role": "system", "content": "s"}, {"role": "user", "content": "x"}])
        digest = hashlib.sha256(b"x").digest()
        expected = -(int.from_bytes(digest[:8], "big") & ((1 << 63) - 1))
        assert body["request"]["sessionId"] == str(expected)
        assert body["requestId"].startswith("agent/") and body["requestId"].endswith("/2")
        assert body["request"]["labels"]["last_step_index"] == "1"

    def test_without_user_text_the_session_id_is_random(self) -> None:
        first = payload([{"role": "assistant", "content": "x"}])["request"]["sessionId"]
        second = payload([{"role": "assistant", "content": "x"}])["request"]["sessionId"]
        assert first != second
        assert first.startswith("-") and 0 <= int(first[1:]) < 9_000_000_000_000_000_000

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

    def test_no_ceiling_is_invented_when_nothing_declares_one(self) -> None:
        """A flat 64000 filled this gap: below the 65536 the gemini 3.x variants accept, and
        above the 4096 of the `tab_*` models. With no caller value and no catalog entry the
        field is left out, as omp does, and the backend applies its own."""
        body = payload([{"role": "user", "content": "x"}])
        assert "maxOutputTokens" not in body["request"]["generationConfig"]

    def test_the_declared_ceiling_fills_the_gap(self) -> None:
        wire = payload([{"role": "user", "content": "x"}])["model"]
        catalog = ModelCatalog(
            ids=(wire,), info={wire: {"maxOutputTokens": 65536}}, fetched_at=1.0
        )
        body = payload([{"role": "user", "content": "x"}], catalog=catalog)
        assert body["request"]["generationConfig"]["maxOutputTokens"] == 65536

    def test_the_caller_is_held_to_the_declared_ceiling(self) -> None:
        """Claude on this backend answers `maxOutputTokens > 64000` with 400."""
        wire = payload([{"role": "user", "content": "x"}])["model"]
        catalog = ModelCatalog(
            ids=(wire,), info={wire: {"maxOutputTokens": 64000}}, fetched_at=1.0
        )
        above = payload(
            [{"role": "user", "content": "x"}], catalog=catalog, extra={"max_tokens": 100000}
        )
        below = payload(
            [{"role": "user", "content": "x"}], catalog=catalog, extra={"max_tokens": 1000}
        )
        assert above["request"]["generationConfig"]["maxOutputTokens"] == 64000
        assert below["request"]["generationConfig"]["maxOutputTokens"] == 1000

    def test_only_a_positive_integer_is_a_declaration(self) -> None:
        for entry in ({}, {"maxOutputTokens": 0}, {"maxOutputTokens": True}, None, "x"):
            assert ag.declared_output_tokens(entry) is None, entry

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
        text at all. A model whose catalog minimum is 32 accepts a budget of 0 (measured on
        the gemini-3 / 3.6-3.8 flash ids), so that is what goes."""
        catalog = ModelCatalog(
            ids=("gemini-3-pro-low",),
            info={"gemini-3-pro-low": {"thinkingBudget": 1000, "minThinkingBudget": 32}},
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
        )
        json.dumps(payload, ensure_ascii=False).encode("utf-8")


class TestToolCalls:
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

    def _config(
        self, model: str, ceiling: int | None, entry: dict[str, Any], effort: str | None = None
    ) -> dict[str, Any]:
        # The variant each name resolves to with a catalog listing only it.
        wire = {
            "gemini-3.1-pro": "gemini-3.1-pro-low",
            "gemini-3.8-flash": "gemini-3.8-flash-medium",
        }.get(model, model)
        catalog = ModelCatalog(ids=(wire,), info={wire: entry}, fetched_at=1.0)
        extra: dict[str, Any] = {} if ceiling is None else {"max_tokens": ceiling}
        if effort:
            extra["reasoning_effort"] = effort
        body = payload(
            [{"role": "user", "content": "x"}], model=model, catalog=catalog, extra=extra
        )
        return body["request"]["generationConfig"]

    CLAUDE: Final = {"thinkingBudget": 1024, "maxOutputTokens": 64000}
    PRO_LOW: Final = {"thinkingBudget": 1001, "minThinkingBudget": 128, "maxOutputTokens": 65535}
    FLASH: Final = {"thinkingBudget": 4000, "minThinkingBudget": 32, "maxOutputTokens": 65536}
    GPT_OSS: Final = {"thinkingBudget": 8192, "maxOutputTokens": 32768}

    @pytest.mark.parametrize("effort", [None, "low", "high"])
    def test_the_callers_ceiling_is_the_total(self, effort: str | None) -> None:
        """The budget is not added on top of ``max_tokens``: with 1024 on top of 64,
        claude-sonnet-4-6 thought 61 characters and answered 623 words (live,
        2026-09-30). Anthropic needs a budget of at least 1024 under the ceiling, so a small
        one turns thinking off — measured ``MAX_TOKENS`` within the 64 for every effort."""
        config = self._config("claude-sonnet-4-6", 64, self.CLAUDE, effort)
        assert config == {
            "maxOutputTokens": 64,
            "thinkingConfig": {"includeThoughts": False, "thinkingBudget": 0},
        }

    def test_a_roomy_ceiling_keeps_the_catalog_budget(self) -> None:
        config = self._config("claude-sonnet-4-6", 64000, self.CLAUDE)
        assert config["maxOutputTokens"] == 64000
        assert config["thinkingConfig"] == {"includeThoughts": True, "thinkingBudget": 1024}

    def test_a_tight_ceiling_leaves_omps_room_for_the_answer(self) -> None:
        """omp: the budget becomes ``ceiling - MIN_OUTPUT_TOKENS`` when it does not fit."""
        config = self._config("gpt-oss-120b-medium", 5000, self.GPT_OSS)
        assert config["maxOutputTokens"] == 5000
        assert config["thinkingConfig"]["thinkingBudget"] == 5000 - ag.MIN_OUTPUT_TOKENS

    def test_a_thinking_only_model_keeps_its_minimum_budget(self) -> None:
        """gemini-3.1-pro-low refuses a budget of 0 ("This model only works in thinking
        mode") and accepts a budget above the ceiling; with ``minThinkingBudget`` it
        answered 200 ``MAX_TOKENS`` within 64 (live, 2026-09-30)."""
        config = self._config("gemini-3.1-pro", 64, self.PRO_LOW)
        assert config == {
            "maxOutputTokens": 64,
            "thinkingConfig": {"includeThoughts": True, "thinkingBudget": 128},
        }

    def test_a_flash_model_thinks_at_its_minimum_under_a_small_ceiling(self) -> None:
        """Its own 4000 took all 64 tokens and left no answer; at 32 it answered ~50 words."""
        config = self._config("gemini-3.8-flash", 64, self.FLASH, "medium")
        assert config["thinkingConfig"] == {"includeThoughts": True, "thinkingBudget": 32}

    def test_without_a_minimum_the_budget_stays(self) -> None:
        """gpt-oss refuses a budget of 0 and declares no minimum; 8192 over a 64 ceiling
        answered 200 ``MAX_TOKENS`` within the 64."""
        config = self._config("gpt-oss-120b-medium", 64, self.GPT_OSS)
        assert config["thinkingConfig"] == {"includeThoughts": True, "thinkingBudget": 8192}

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
        tools = ag.tools_to_declarations(
            "gemini-3-pro",
            [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
        )
        assert tools is not None
        declaration = tools[0]["functionDeclarations"][0]
        assert "parameters" in declaration
        assert "parametersJsonSchema" not in declaration

    def test_unsupported_constructs_are_sanitised(self) -> None:
        """`anyOf`/`$ref`/`not` give 400 on the CCA; sending them raw made the request fail."""
        tools = ag.tools_to_declarations(
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
            tools = ag.tools_to_declarations(
                model, [{"type": "function", "function": {"name": "f", "parameters": {}}}]
            )
            assert tools is not None
            assert "parameters" in tools[0]["functionDeclarations"][0], model

    @pytest.mark.parametrize(
        ("choice", "config"),
        [
            pytest.param(None, None, id="absent"),
            pytest.param("auto", None, id="auto"),
            pytest.param("none", {"mode": "NONE"}, id="none"),
            pytest.param("required", {"mode": "ANY"}, id="required"),
            # Neither server parser accepts Anthropic's bare "any"; the Messages route
            # translates it to "required" before it gets here.
            pytest.param("any", None, id="bare-any"),
            pytest.param(
                {"type": "function", "function": {"name": "read"}},
                {"mode": "ANY", "allowedFunctionNames": ["read"]},
                id="chat-named",
            ),
            pytest.param(
                {"type": "function", "name": "read"},
                {"mode": "ANY", "allowedFunctionNames": ["read"]},
                id="responses-named",
            ),
            pytest.param(
                {"type": "tool", "name": "read"},
                {"mode": "ANY", "allowedFunctionNames": ["read"]},
                id="anthropic-named",
            ),
            pytest.param({"type": "function", "function": {"name": ""}}, None, id="nameless"),
            pytest.param({"type": "web_search_preview"}, None, id="hosted"),
            pytest.param({"type": "allowed_tools", "mode": "auto"}, None, id="allowed-tools"),
        ],
    )
    def test_choice_maps_as_omp_maps_it(self, choice: object, config: object) -> None:
        """``normalizeToolChoice`` / Responses ``mapToolChoice`` into ``mapGoogleToolChoice``:
        ``None`` leaves the request on the default mode."""
        assert ag.function_calling_config(choice) == config

    def test_no_tools_means_no_config(self) -> None:
        body = payload([{"role": "user", "content": "x"}])
        assert "tools" not in body["request"] and "toolConfig" not in body["request"]


class TestDemotedThinking:
    @pytest.mark.parametrize(
        ("text", "demoted"),
        [
            ("plain", "<thinking>\nplain\n</thinking>"),
            ("  <thinking>a</thinking>  ", "<thinking>\na\n</thinking>"),
            (
                "<thinking>\n a \n</thinking>\n<thinking>b</thinking>",
                "<thinking>\na\nb\n</thinking>",
            ),
            ("<thinking><thinking>n</thinking></thinking>", "<thinking>\nn\n</thinking>"),
            ("<thinking>open only", "<thinking>\n<thinking>open only\n</thinking>"),
            ("<thinking>a</thinking> tail", "<thinking>\n<thinking>a</thinking> tail\n</thinking>"),
        ],
    )
    def test_the_xml_fallback_does_not_wrap_twice(self, text: str, demoted: str) -> None:
        """omp 18.4.4 ``renderDelimitedThinking`` outputs for a model with no dialect of its
        own: reasoning already in the tags is unwrapped first, and only when every segment
        closes — otherwise it is wrapped as it came."""
        assert ag.demoted_thinking("tab_flash_lite_preview", text) == demoted


class TestToolPairing:
    def test_a_stray_result_inside_an_open_window_is_dropped(self) -> None:
        """A note there would sit between a call and its result, breaking the pair; omp
        drops the stray one and still closes the window."""
        call = {"id": "c1", "function": {"name": "f", "arguments": "{}"}}
        paired = ag.pair_tool_results(
            [
                {"role": "assistant", "tool_calls": [call]},
                {"role": "tool", "tool_call_id": "gone", "content": "stray"},
                {"role": "user", "content": "next"},
            ]
        )

        assert paired == [
            {"role": "assistant", "tool_calls": [call]},
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": ag.MISSING_TOOL_RESULT,
                "is_error": True,
            },
            {"role": "user", "content": "next"},
        ]
