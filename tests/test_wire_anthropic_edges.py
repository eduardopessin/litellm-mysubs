"""Defensive branches of the Anthropic wire.

Malformed inputs, limits and give-up paths. They are separated from the main contract
because they answer a different question: not "what shape goes on the wire", but "what
happens when the input is not what is expected". A request reaches the proxy from any
client, and an exception here is a 500 instead of a served request.
"""

from __future__ import annotations

from typing import Any

import pytest

from litellm_mysubs.wire import anthropic as ant


class TestMalformedInput:
    """The proxy receives requests from clients we do not control."""

    @pytest.mark.parametrize("message", [None, "text", 42, []])
    def test_non_dict_message_is_not_markable(self, message: object) -> None:
        assert ant.is_markable(message) is False

    def test_non_dict_entries_ignored_when_counting(self) -> None:
        assert ant.count_breakpoints([None, "x", 42]) == 0

    def test_non_list_tool_calls_has_no_anchor(self) -> None:
        assert ant.tool_call_anchor({"tool_calls": "not-a-list"}) is None

    def test_non_dict_tool_call_skipped(self) -> None:
        assert ant.tool_call_anchor({"tool_calls": [None, "x"]}) is None

    def test_non_function_tool_call_skipped(self) -> None:
        """convert_to_anthropic_tool_invoke skips whatever is not type: function."""
        assert ant.tool_call_anchor({"tool_calls": [{"id": "c", "type": "custom"}]}) is None

    def test_last_valid_tool_call_wins(self) -> None:
        """The anchor is the last markable call, because it covers more prefix."""
        message = {
            "tool_calls": [
                {"id": "a", "type": "function"},
                {"id": "srvtoolu_x", "type": "function"},
                {"id": "b", "type": "function"},
            ]
        }
        assert ant.tool_call_anchor(message) == 2

    def test_unknown_role_is_not_markable(self) -> None:
        assert ant.is_markable({"role": "strange-role", "content": "x"}) is False

    def test_non_string_content_is_not_markable(self) -> None:
        assert ant.is_markable({"role": "user", "content": {"a": 1}}) is False

    def test_non_list_messages_returns_unchanged(self) -> None:
        """A client may send messages as a string; it must not blow up."""
        out = ant.build_request({"messages": "not-a-list"}, "claude-opus-5")
        assert out["messages"] == "not-a-list"

    def test_missing_messages_key(self) -> None:
        out = ant.build_request({}, "claude-opus-5")
        assert "messages" not in out

    def test_non_dict_extra_headers_left_alone(self) -> None:
        out = ant.build_request({"extra_headers": "x", "messages": []}, "claude-opus-5")
        assert out["extra_headers"] == "x"

    def test_system_content_list_extracts_text_blocks(self) -> None:
        client, rest = ant.split_system_messages(
            [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "instruction"},
                        {"type": "image", "source": {}},
                    ],
                }
            ]
        )
        assert client == "instruction"
        assert rest == []


class TestMarkBreakpointGivesUp:
    """Already marked means do not mark again: two markers on the same anchor spend
    budget without covering more prefix."""

    def test_tool_message_already_marked(self) -> None:
        message: dict[str, Any] = {"role": "tool", "tool_call_id": "c", "cache_control": {}}
        assert ant.mark_breakpoint(message) is False

    def test_tool_call_already_marked(self) -> None:
        message: dict[str, Any] = {
            "role": "assistant",
            "tool_calls": [{"id": "c", "type": "function", "cache_control": {}}],
        }
        assert ant.mark_breakpoint(message) is False

    def test_text_block_already_marked(self) -> None:
        message: dict[str, Any] = {
            "role": "user",
            "content": [{"type": "text", "text": "a", "cache_control": {}}],
        }
        assert ant.mark_breakpoint(message) is False

    def test_non_list_content_cannot_be_marked(self) -> None:
        assert ant.mark_breakpoint({"role": "user", "content": {"a": 1}}) is False

    def test_string_content_becomes_marked_block(self) -> None:
        message: dict[str, Any] = {"role": "user", "content": "hello"}
        assert ant.mark_breakpoint(message) is True
        assert message["content"] == [
            {"type": "text", "text": "hello", "cache_control": ant.cache_control()}
        ]

    def test_skips_blank_and_thinking_blocks(self) -> None:
        """The anchor falls back to the first real text block, scanning backwards."""
        message: dict[str, Any] = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "real"},
                {"type": "thinking", "text": "reasoning"},
                {"type": "text", "text": "   "},
            ],
        }
        assert ant.mark_breakpoint(message) is True
        assert message["content"][0]["cache_control"] == ant.cache_control()

    def test_only_unmarkable_blocks_fails(self) -> None:
        message: dict[str, Any] = {"role": "user", "content": [{"type": "image"}]}
        assert ant.mark_breakpoint(message) is False


class TestCacheDeepCopy:
    """Marking has to produce new structures: mutating what the client sent makes the
    marker show up in their history."""

    def test_tool_calls_are_copied_not_mutated(self) -> None:
        call = {"id": "c1", "type": "function", "function": {"name": "f"}}
        original = {"role": "assistant", "tool_calls": [call]}
        messages: list[Any] = [original]
        ant.apply_conversation_cache(messages)
        assert "cache_control" in messages[0]["tool_calls"][0]
        assert "cache_control" not in call

    def test_content_blocks_are_copied_not_mutated(self) -> None:
        block = {"type": "text", "text": "a"}
        messages: list[Any] = [{"role": "user", "content": [block]}]
        ant.apply_conversation_cache(messages)
        assert "cache_control" not in block

    def test_counts_marks_inside_tool_calls(self) -> None:
        messages: list[Any] = [
            {
                "role": "assistant",
                "tool_calls": [{"id": "c", "cache_control": {"type": "ephemeral"}}],
            }
        ]
        assert ant.count_breakpoints(messages) == 1

    def test_counts_message_level_marks(self) -> None:
        marked = [{"role": "tool", "cache_control": {"type": "ephemeral"}}]
        assert ant.count_breakpoints(marked) == 1

    def test_empty_marker_does_not_consume_budget(self) -> None:
        """An empty ``cache_control`` is not a marker: Anthropic counts the blocks that
        carry it filled in, and discounting it would spend budget without covering
        prefix."""
        assert ant.count_breakpoints([{"role": "tool", "cache_control": {}}]) == 0


class TestToolChoiceShapes:
    @pytest.mark.parametrize("choice", [{"type": "any"}, {"type": "tool"}, "required", "any"])
    def test_forced_shapes(self, choice: object) -> None:
        assert ant._forced_tool_choice(choice) is True

    @pytest.mark.parametrize("choice", ["auto", "none", {"type": "auto"}, None, 42])
    def test_free_shapes(self, choice: object) -> None:
        assert ant._forced_tool_choice(choice) is False

    def test_openai_function_selection_is_not_forcing(self) -> None:
        """`{"type": "function"}` selects a tool; it does not force calling it.

        The Anthropic wire only knows `any`/`tool`/`auto`/`none`. Treating the OpenAI
        shape as forcing turned reasoning off for no reason at all on the server side.
        """
        assert ant._forced_tool_choice({"type": "function", "function": {"name": "f"}}) is False
        out = ant.apply_thinking_params(
            {"reasoning_effort": "high", "tool_choice": {"type": "function"}},
            "claude-haiku-4-5",
        )
        assert out["thinking"]["type"] == "enabled"


class TestThinkingEdges:
    def test_explicit_budget_is_capped(self) -> None:
        """Ceiling of 8192 because of the subscription's short TPM window."""
        out = ant.apply_thinking_params(
            {"thinking": {"type": "enabled", "budget_tokens": 99999}}, "claude-haiku-4-5"
        )
        assert out["thinking"]["budget_tokens"] == 8192

    def test_adaptive_object_keeps_its_shape(self) -> None:
        out = ant.apply_thinking_params(
            {"thinking": {"type": "adaptive", "display": "summarized"}}, "claude-opus-5"
        )
        assert out["thinking"]["type"] == "adaptive"

    def test_temperature_one_is_left_alone(self) -> None:
        out = ant.apply_thinking_params(
            {"reasoning_effort": "low", "temperature": 1.0}, "claude-haiku-4-5"
        )
        assert out["temperature"] == 1.0

    def test_temperature_without_thinking_drops_reasoning(self) -> None:
        """Without thinking active, a custom temperature belongs to the client and wins."""
        out = ant.apply_thinking_params({"temperature": 0.3}, "claude-opus-5")
        assert out["temperature"] == 0.3
        assert "thinking" not in out

    def test_default_max_tokens_when_absent(self) -> None:
        out = ant.apply_thinking_params({"reasoning_effort": "low"}, "claude-haiku-4-5")
        assert out["max_tokens"] == ant.EFFORT_BUDGET["low"] + ant.OUTPUT_FALLBACK_BUFFER

    def test_unknown_effort_uses_medium_default(self) -> None:
        out = ant.apply_thinking_params({"reasoning_effort": "turbo"}, "claude-opus-5")
        assert "thinking" not in out

    def test_no_thinking_returns_early(self) -> None:
        kwargs: dict[str, Any] = {"messages": []}
        assert ant.apply_thinking_params(kwargs, "claude-opus-5") is kwargs
