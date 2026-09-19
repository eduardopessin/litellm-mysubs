"""Defensive branches of the Codex wire.

Malformed input and discard paths. The proxy receives requests from clients we do not
control: an exception here is a 500 instead of a served request, and a silent discard is
worse — the model answers about content it never received.
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
        """A name without a version cannot blow up the juice item decision."""
        assert codex.wire_generation("gpt-terra-x") == 0.0

    def test_non_dict_claims_are_empty(self) -> None:
        """A JWT whose body is not an object is not a usable identity."""
        body = base64.urlsafe_b64encode(json.dumps(["list"]).encode()).decode().rstrip("=")
        assert codex.token_claims(f"a.{body}.c") == {}

    def test_account_header_absent_without_claim(self) -> None:
        headers = codex.build_headers(jwt({}), window_id="w")
        assert "chatgpt-account-id" not in headers

    def test_account_header_present_with_claim(self) -> None:
        headers = codex.build_headers(
            jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-9"}}),
            window_id="w",
        )
        assert headers["chatgpt-account-id"] == "acct-9"

    def test_content_to_text_ignores_non_text_parts(self) -> None:
        """Only text blocks count towards the cache key; an image is not text."""
        text = codex.content_to_text(
            [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {"url": "u"}}]
        )
        assert text == "a"

    def test_content_to_text_handles_scalars(self) -> None:
        assert codex.content_to_text(42) == "42"
        assert codex.content_to_text(None) == ""

    @pytest.mark.parametrize("part", [None, "text", 42])
    def test_non_dict_parts_skipped(self, part: object) -> None:
        assert codex.content_to_parts(["ok", part]) == []

    def test_unknown_part_type_skipped(self) -> None:
        parts = codex.content_to_parts([{"type": "audio", "data": "x"}])
        assert parts == []

    def test_blank_text_part_skipped(self) -> None:
        assert codex.content_to_parts([{"type": "text", "text": ""}]) == []

    def test_broken_image_part_skipped_but_text_kept(self) -> None:
        """A discard cannot take the rest of the turn with it."""
        parts = codex.content_to_parts(
            [{"type": "text", "text": "hello"}, {"type": "image_url", "image_url": {}}]
        )
        assert parts == [{"type": "input_text", "text": "hello"}]

    def test_broken_file_part_skipped(self) -> None:
        parts = codex.content_to_parts([{"type": "file", "file": {"filename": "no-data.pdf"}}])
        assert parts == []

    def test_file_part_accepted_inline(self) -> None:
        parts = codex.content_to_parts([{"type": "input_file", "file_data": "b64"}])
        assert parts == [{"type": "input_file", "file_data": "b64"}]

    def test_non_dict_tool_skipped(self) -> None:
        tools = codex.tools_to_codex_tools([None, {"type": "function", "function": {"name": "f"}}])
        assert tools is not None and len(tools) == 1

    def test_flat_tool_without_function_wrapper(self) -> None:
        """Some clients send the spec flattened, without the `function` level."""
        tools = codex.tools_to_codex_tools([{"name": "read", "parameters": {"type": "object"}}])
        assert tools is not None and tools[0]["name"] == "read"

    def test_tool_without_parameters_gets_empty_object(self) -> None:
        """Responses requires `parameters`; omitting it gives 400."""
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
        """The backend refuses keys longer than 64 characters."""
        key = codex.prompt_cache_key("s" * 200)
        assert key is not None
        assert key.startswith("pc_") and len(key) <= 64

    def test_short_session_id_travels_verbatim(self) -> None:
        assert codex.prompt_cache_key("short-session") == "short-session"

    def test_retention_none_disables_it(self) -> None:
        assert codex.prompt_cache_key("s", cache_retention="none") is None


class TestCallIdSanitisation:
    def test_invalid_characters_are_replaced(self) -> None:
        """An id with characters outside the allowed set gives a 400 from the backend."""
        assert codex.split_call_id("call id!") != "call id!"
        assert " " not in codex.split_call_id("call id!")

    def test_valid_id_passes_through(self) -> None:
        assert codex.split_call_id("call_abc-123") == "call_abc-123"

    def test_composite_id_keeps_only_the_call(self) -> None:
        assert codex.split_call_id("call-1|item-9") == "call-1"

    def test_newline_also_separates(self) -> None:
        """Ids forwarded from another provider carry `\n`; cutting only on `|` let the
        second segment through."""
        assert codex.split_call_id("call-1\ngarbage") == "call-1"

    def test_long_id_is_truncated_with_a_hash(self) -> None:
        sanitized = codex.split_call_id("c" * 200)
        assert len(sanitized) <= codex.CALL_ID_MAX_CHARS

    def test_different_long_ids_do_not_collide(self) -> None:
        """Truncating without a hash made two distinct ids collapse into the same one."""
        assert codex.split_call_id("a" * 100) != codex.split_call_id("b" * 100)

    def test_empty_id_gets_a_stable_placeholder(self) -> None:
        assert codex.split_call_id("").startswith("call_")


class TestOrphanRepair:
    def test_tool_name_is_preserved(self) -> None:
        """Without the name, the model does not know what produced the orphan result."""
        items = codex.repair_tool_pairs(
            [{"type": "function_call_output", "call_id": "x", "name": "read", "output": "r"}]
        )
        assert "[Previous read result; call_id=x]" in items[0]["content"]

    def test_missing_name_falls_back(self) -> None:
        items = codex.repair_tool_pairs(
            [{"type": "function_call_output", "call_id": "x", "output": "r"}]
        )
        assert "[Previous tool result" in items[0]["content"]

    def test_huge_output_is_truncated(self) -> None:
        """A 2 MB file blew the body limit instead of being cut."""
        items = codex.repair_tool_pairs(
            [{"type": "function_call_output", "call_id": "x", "output": "y" * 40_000}]
        )
        assert "...[truncated]" in items[0]["content"]
        assert len(items[0]["content"]) < 20_000

    def test_structured_output_is_serialised_as_json(self) -> None:
        """`str()` of a dict gives single quotes and `True`/`None` — Python repr in the
        prompt."""
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
        """A message with no content and no tool calls has nothing to send."""
        body = codex.build_request_body("gpt-5.5", [{"role": "user", "content": ""}])
        assert body["input"] == []

    def test_no_instructions_key_without_a_system_prompt(self) -> None:
        """An `instructions: ""` is an empty base prompt occupying the cache entry."""
        body = codex.build_request_body("gpt-5.5", [{"role": "user", "content": "x"}])
        assert "instructions" not in body

    def test_blank_system_prompt_is_not_promoted_to_instructions(self) -> None:
        body = codex.build_request_body(
            "gpt-5.5",
            [{"role": "system", "content": "   "}, {"role": "user", "content": "x"}],
        )
        assert "instructions" not in body

    def test_system_prompt_leaves_the_input(self) -> None:
        """Duplicating it in the input wasted context and misaligned it from the cached
        prompt."""
        body = codex.build_request_body(
            "gpt-5.5",
            [{"role": "system", "content": "rule"}, {"role": "user", "content": "x"}],
        )
        assert body["instructions"] == "rule"
        assert [i["content"][0]["text"] for i in body["input"]] == ["x"]

    def test_unsupported_detail_original_degrades_in_the_body(self) -> None:
        """The capability has to cross the whole body: degrading only in `image_part` did
        not save the request the proxy actually sends."""
        body = codex.build_request_body(
            "gpt-5.5",
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "u", "detail": "original"}}
                    ],
                }
            ],
            supports_detail_original=False,
        )
        assert body["input"][0]["content"][0]["detail"] == "auto"
