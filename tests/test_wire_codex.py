"""Codex wire contract (Responses API).

The body is built from scratch, not adjusted: every missing field is a behaviour that
disappears silently (reasoning with no events, cache with no hits, multimodal with no
image).
"""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest

from litellm_mysubs.wire import codex


def jwt(payload: dict[str, Any]) -> str:
    """Unsigned JWT: only the body matters, and nothing is verified when reading it."""
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
        """Family names promise no version, so resolving them is honest."""
        assert codex.resolve_model(requested) == wire

    @pytest.mark.parametrize("name", ["gpt-5.4", "gpt-5.4-mini"])
    def test_version_names_are_not_remapped(self, name: str) -> None:
        """They name a version the account does not serve: remapping would bill the client
        against a model that never ran. The upstream refusal is the correct answer."""
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

    @pytest.mark.parametrize("token", ["", "not-a-jwt", "a.b", "a.!!!.c"])
    def test_malformed_token_is_empty(self, token: str) -> None:
        """A broken token must not blow up building the request."""
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
        """OMP deletes it from the headers explicitly; it travels only in the envelope."""
        assert "x-codex-installation-id" not in codex.build_headers(jwt({}), window_id="w")

    def test_request_kind_is_from_the_vocabulary(self) -> None:
        """ "chat" does not belong to the set "turn" | "prewarm" | "compaction"."""
        headers = codex.build_headers(jwt({}), window_id="w")
        assert json.loads(headers["x-codex-turn-metadata"])["request_kind"] == "turn"

    def test_routing_hint_carries_the_model(self) -> None:
        """The backend uses it to pick the route; without it routing is the default."""
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
        """The backend returns it and expects it back on the next turn."""
        headers = codex.build_headers(jwt({}), window_id="w", turn_state="st-1")
        assert headers["x-codex-turn-state"] == "st-1"

    def test_turn_state_absent_on_first_turn(self) -> None:
        headers = codex.build_headers(jwt({}), window_id="w")
        assert "x-codex-turn-state" not in headers

    def test_residency_header_only_for_constrained_workspaces(self) -> None:
        """401 "Workspace is not authorized in this region" when it is missing; personal
        accounts do not have the claim and the header must not travel."""
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
        """`original` is the only level that preserves a screenshot's native resolution;
        always forcing it to "auto" degraded it against hosts that do serve it."""
        part = codex.image_part({"image_url": {"url": "u", "detail": "original"}})
        assert part is not None and part["detail"] == "original"

    def test_detail_original_degrades_when_host_rejects_it(self) -> None:
        """Hosts such as GitHub Copilot return 400 for `original`; degrading saves the
        request instead of losing it."""
        part = codex.image_part(
            {"image_url": {"url": "u", "detail": "original"}}, supports_detail_original=False
        )
        assert part is not None and part["detail"] == "auto"

    def test_unknown_detail_falls_back_to_auto(self) -> None:
        part = codex.image_part({"image_url": {"url": "u", "detail": "ultra"}})
        assert part is not None and part["detail"] == "auto"

    def test_image_by_file_id_is_not_discarded(self) -> None:
        """An image already uploaded to the backend has no url; without this branch it
        disappeared silently and the model answered about something it never saw."""
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
        """Before this the request arrived with the text only and the model talked about
        an image it never saw."""
        parts = codex.content_to_parts(
            [
                {"type": "text", "text": "what colour?"},
                {"type": "image_url", "image_url": {"url": "u"}},
            ]
        )
        assert [p["type"] for p in parts] == ["input_text", "input_image"]

    def test_assistant_parts_use_output_text(self) -> None:
        parts = codex.content_to_parts("answer", assistant=True)
        assert parts == [{"type": "output_text", "text": "answer"}]

    def test_empty_content_yields_no_parts(self) -> None:
        assert codex.content_to_parts("") == []
        assert codex.content_to_parts(None) == []


class TestCallIds:
    def test_composite_joins_pair(self) -> None:
        """Without the exact pair, parallel calls go out of alignment on replay."""
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
        """A truncated history brings outputs without the call; Responses rejects them."""
        items = codex.repair_tool_pairs(
            [{"type": "function_call_output", "call_id": "orphan", "output": "lost"}]
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
        """An orphan `custom_tool_call` gave 400 for not being indexed; and the output
        that closes it has to be of the same kind, otherwise the 400 comes back."""
        items = codex.repair_tool_pairs(
            [{"type": "custom_tool_call", "call_id": "c2", "name": "f", "input": "x"}]
        )
        assert [i["type"] for i in items] == ["custom_tool_call", "custom_tool_call_output"]

    def test_orphan_computer_call_becomes_note(self) -> None:
        """The missing screenshot is not synthesised: the call becomes a note, with the
        exact text OMP uses."""
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
        """Pairing by `call_id` alone made a `custom` output "close" a `function` call; the
        backend refuses the swap and both halves need repair."""
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
        """`instructions` is the base prompt the backend caches; sending it as a developer
        item loses that treatment and the cache hit."""
        instructions, items = codex.messages_to_input([{"role": "system", "content": "rule"}])
        assert instructions == "rule"
        assert not any(i.get("role") == "developer" for i in items)

    def test_extra_system_prompts_become_developer_items(self) -> None:
        """`instructions` is a string: the second prompt does not fit there and was lost."""
        instructions, items = codex.messages_to_input(
            [
                {"role": "system", "content": "base"},
                {"role": "system", "content": "extra"},
                {"role": "user", "content": "hello"},
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
        """Without a visible turn the backend returns an empty response; promoting the
        last instruction gives it something to answer."""
        _, items = codex.messages_to_input(
            [
                {"role": "system", "content": "base"},
                {"role": "system", "content": "do this"},
            ]
        )
        assert items[-1] == {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "do this"}],
        }

    def test_single_system_prompt_promotes_instructions_to_user(self) -> None:
        """Only one system prompt: `instructions` is the only text there is, and the input
        would be left empty."""
        instructions, items = codex.messages_to_input([{"role": "system", "content": "rule"}])
        assert instructions == "rule"
        assert items == [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "rule"}],
            }
        ]

    def test_user_turn_suppresses_promotion(self) -> None:
        """With a user turn the instruction is not duplicated into the input."""
        _, items = codex.messages_to_input(
            [{"role": "system", "content": "base"}, {"role": "user", "content": "hello"}]
        )
        assert [i["content"][0]["text"] for i in items] == ["hello"]

    def test_unknown_role_falls_back_to_user(self) -> None:
        _, items = codex.messages_to_input([{"role": "weird", "content": "x"}])
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
                {"role": "tool", "tool_call_id": "c1", "content": "result"},
            ]
        )
        assert [i["type"] for i in items] == ["function_call", "function_call_output"]

    def test_dict_arguments_are_serialised(self) -> None:
        """Responses requires arguments as a JSON string."""
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
        """They have no `function` and were discarded before this."""
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
        """Without it, zero response.reasoning_summary_text.delta events."""
        body = codex.build_request_body("gpt-5.5", [{"role": "user", "content": "x"}])
        assert body["reasoning"] == {"effort": "medium", "summary": "auto"}

    def test_all_turns_context_is_not_forced(self) -> None:
        """OMP only forces it on the Lite transport and deletes it on models that do not
        support it."""
        body = codex.build_request_body("gpt-5.5", [{"role": "user", "content": "x"}])
        assert "context" not in body["reasoning"]

    def test_encrypted_reasoning_is_requested(self) -> None:
        """Without this there is no reasoning replay in a stateless history."""
        body = codex.build_request_body("gpt-5.5", [{"role": "user", "content": "x"}])
        assert body["include"] == ["reasoning.encrypted_content"]

    def test_effort_none_adds_juice_item_on_new_generations(self) -> None:
        """GPT-5.6+ still reserves juice with reasoning turned off.

        The value is the one of the requested effort, not zero: turning reasoning off does
        not mean the model should be left with no budget at all.
        """
        body = codex.build_request_body(
            "gpt-5.6-terra", [{"role": "user", "content": "x"}], extra={"reasoning_effort": "none"}
        )
        assert "reasoning" not in body
        # `none` is the explicit request to turn it off; the juice follows that value.
        assert (
            body["input"][-1]["content"][0]["text"] == f"# Juice: {codex.JUICE['none']} !important"
        )

    def test_juice_follows_a_separate_effort_when_given(self) -> None:
        """In OMP turning it off is a flag separate from the effort: whoever asks for
        `high` and turns reasoning off still reserves the `high` budget."""
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
        assert codex.juice_for("made-up") == codex.JUICE["medium"]

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
            extra={"reasoning_effort": {"effort": "high", "summary": "made-up"}},
        )
        assert body["reasoning"]["summary"] == "auto"

    def test_stream_and_store_are_fixed(self) -> None:
        body = codex.build_request_body("gpt-5.5", [{"role": "user", "content": "x"}])
        assert body["stream"] is True and body["store"] is False

    def test_cache_key_is_the_session_identity(self) -> None:
        """Deriving it from the content made two conversations with the same system prompt
        share a key — across sessions and across users."""
        body = codex.build_request_body(
            "gpt-5.5", [{"role": "user", "content": "x"}], session_id="session-1"
        )
        assert body["prompt_cache_key"] == "session-1"

    def test_cache_key_survives_history_edits(self) -> None:
        """The same session keeps the hit even with the head of the conversation edited."""
        first = codex.build_request_body(
            "gpt-5.5", [{"role": "user", "content": "original"}], session_id="s"
        )
        later = codex.build_request_body(
            "gpt-5.5", [{"role": "user", "content": "edited"}], session_id="s"
        )
        assert first["prompt_cache_key"] == later["prompt_cache_key"]

    def test_cache_can_be_disabled(self) -> None:
        """Without this there was no way for the caller to opt out of the cache."""
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
