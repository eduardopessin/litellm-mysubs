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
        blocks, rest = ant.split_system_messages(
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
        assert blocks == [{"type": "text", "text": "instruction"}]
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

    def test_openai_function_shapes_follow_the_wire(self) -> None:
        """What forces is decided on what LiteLLM sends. A bare `{"type": "function"}` names
        nothing and LiteLLM drops it, so budget thinking stays. A named function becomes
        `{"type": "tool", "name": ...}` on the wire — forcing — and budget thinking beside it
        is the 400 "Thinking may not be enabled when tool_choice forces tool use"."""
        bare = ant.apply_thinking_params(
            {"reasoning_effort": "high", "tool_choice": {"type": "function"}},
            "claude-haiku-4-5",
        )
        named = ant.apply_thinking_params(
            {
                "reasoning_effort": "high",
                "tool_choice": {"type": "function", "function": {"name": "f"}},
            },
            "claude-haiku-4-5",
        )
        assert bare["thinking"]["type"] == "enabled"
        assert "thinking" not in named


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
        out = ant.apply_thinking_params({"temperature": 0.3}, "claude-opus-4-6")
        assert out["temperature"] == 0.3
        assert "thinking" not in out

    def test_nothing_asked_takes_the_models_ceiling(self) -> None:
        """Claude Code sends 128000 on `claude-opus-5-5`, and omp 18.4.1 the full model
        ceiling. Budget + margin used to fill the gap on the thinking path: a limit below the
        model's that nobody asked for."""
        plain = ant.apply_thinking_params({}, "claude-opus-5-5", ceiling=128000)
        thinking = ant.apply_thinking_params(
            {"reasoning_effort": "low"}, "claude-haiku-4-5", ceiling=64000
        )
        assert plain["max_tokens"] == 128000
        assert thinking["max_tokens"] == 64000

    def test_a_request_above_the_ceiling_is_lowered_to_it(self) -> None:
        """Measured: the 4.5 family answers `max_tokens: 64001 > 64000` with 400."""
        out = ant.apply_thinking_params({"max_tokens": 100000}, "claude-opus-4-5", ceiling=64000)
        assert out["max_tokens"] == 64000

    def test_an_unknown_ceiling_falls_back_to_omps_default(self) -> None:
        """The Messages API requires `max_tokens`; without a declared ceiling it is omp's
        value for the same case, not LiteLLM's price map."""
        out = ant.apply_thinking_params({}, "claude-opus-5-5")
        assert out["max_tokens"] == ant.UNKNOWN_MODEL_MAX_OUTPUT_TOKENS

    def test_a_lower_request_is_kept(self) -> None:
        out = ant.apply_thinking_params(
            {"max_completion_tokens": 1000}, "claude-opus-5-5", ceiling=128000
        )
        assert out["max_tokens"] == 1000
        assert "max_completion_tokens" not in out

    def test_a_tight_ceiling_shrinks_the_budget_under_it(self) -> None:
        """omp 18.4.4: the budget leaves the output buffer under the model's ceiling. The old
        request (budget 8192, `max_tokens` 8000) is the 400 "`max_tokens` must be greater
        than `thinking.budget_tokens`"."""
        out = ant.apply_thinking_params(
            {"reasoning_effort": "high", "max_tokens": 100}, "claude-haiku-4-5", ceiling=8000
        )
        assert out["thinking"]["budget_tokens"] == 8000 - ant.OUTPUT_FALLBACK_BUFFER
        assert out["max_tokens"] == 8000

    def test_a_ceiling_with_no_room_for_a_budget_turns_thinking_off(self) -> None:
        """Below Anthropic's 1024 minimum no budget is valid (measured: "budget_tokens: Input
        should be greater than or equal to 1024"), so omp 18.4.4 sends the turn without
        thinking; the caller's `max_tokens` stands. This request used to go out with budget
        8192 and `max_tokens` 4096."""
        out = ant.apply_thinking_params(
            {"reasoning_effort": "high", "max_tokens": 100}, "claude-haiku-4-5", ceiling=4096
        )
        assert out["thinking"] == {"type": "disabled"}
        assert out["max_tokens"] == 100

    def test_unknown_effort_uses_medium_default(self) -> None:
        out = ant.apply_thinking_params({"reasoning_effort": "turbo"}, "claude-opus-5")
        assert "thinking" not in out

    def test_no_thinking_returns_early(self) -> None:
        kwargs: dict[str, Any] = {"messages": []}
        assert ant.apply_thinking_params(kwargs, "claude-opus-5") is kwargs


class TestThinkingPrefixBinding:
    """Fable 5.1+, Sonnet 5.5 and (omp 18.8.6) Opus 5.5 and Haiku 5.5 bind signed thinking
    to the exact preceding conversation; omp asks them to drop a stale block (`drop_block`)
    instead of failing the turn."""

    @pytest.mark.parametrize(
        "model", ["claude-sonnet-5-5", "claude-fable-5-1", "claude-opus-5-5", "claude-haiku-5-5"]
    )
    def test_a_bound_model_drops_a_stale_block_with_its_beta(self, model: str) -> None:
        out = ant.build_request({"messages": [], "reasoning_effort": "high"}, model, "tok")

        assert out["thinking"]["block_binding"] == {"prefix_mismatch_behavior": "drop_block"}
        assert ant.THINKING_BINDING_BETA in out["extra_headers"]["anthropic-beta"]

    def test_the_clients_own_binding_is_kept(self) -> None:
        """A Messages client may ask for `error` instead; rebuilding its adaptive block used
        to drop the field."""
        binding = {"prefix_mismatch_behavior": "error"}
        out = ant.apply_thinking_params(
            {"thinking": {"type": "adaptive", "block_binding": binding}}, "claude-sonnet-5-5"
        )

        assert out["thinking"]["block_binding"] == binding

    @pytest.mark.parametrize("model", ["claude-opus-5", "claude-fable-5", "claude-sonnet-5"])
    def test_an_unbound_model_gets_neither(self, model: str) -> None:
        out = ant.build_request({"messages": [], "reasoning_effort": "high"}, model, "tok")

        assert "block_binding" not in out["thinking"]
        assert ant.THINKING_BINDING_BETA not in out["extra_headers"]["anthropic-beta"]

    @pytest.mark.parametrize("params", [{}, {"reasoning_effort": "none"}], ids=["plain", "none"])
    @pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-haiku-5-5"])
    def test_an_off_turn_names_adaptive_to_carry_the_binding(
        self, model: str, params: dict[str, Any]
    ) -> None:
        """omp 18.8.6 sends this for an Opus 5.5 / Haiku 5.5 turn that does not reason (read
        off its `onPayload`): adaptive thinking runs anyway on these models, so it is named
        to carry the binding — no `display`, the lowest effort, and the caller's
        `max_tokens` untouched."""
        out = ant.apply_thinking_params({"max_tokens": 100, **params}, model, ceiling=128000)

        assert out["thinking"] == {
            "type": "adaptive",
            "block_binding": {"prefix_mismatch_behavior": "drop_block"},
        }
        assert out["output_config"] == {"effort": "low"}
        assert out["max_tokens"] == 100

    def test_the_off_turn_survives_litellms_second_pass(self) -> None:
        """LiteLLM runs a chat request through the wire twice. Read as reasoning, the off
        turn's adaptive block gained `display` and a thinking budget's `max_tokens`."""
        first = ant.apply_thinking_params({"max_tokens": 100}, "claude-opus-5-5", ceiling=128000)
        second = ant.apply_thinking_params(
            {**first, "thinking": dict(first["thinking"])}, "claude-opus-5-5", ceiling=128000
        )

        assert second == first

    def test_sonnet_5_5_keeps_between_tools(self) -> None:
        """Its off turn is `between_tools`, which takes no binding."""
        out = ant.apply_thinking_params({}, "claude-sonnet-5-5")

        assert out["thinking"] == {"type": "between_tools"}


class TestSamplingParams:
    """omp 18.8.6 (`anthropic.kdl`, `supports-sampling-params #false`): adaptive Claude —
    Opus 4.7+, Sonnet/Fable/Mythos 5+, Haiku 5.5+ — answers `temperature`, `top_p` and
    `top_k` with 400, and omp never sends them there."""

    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-4-8",
            "claude-opus-5",
            "claude-opus-5-5",
            "claude-sonnet-5",
            "claude-fable-5",
            "claude-haiku-5-5",
        ],
    )
    @pytest.mark.parametrize("params", [{}, {"reasoning_effort": "high"}], ids=["off", "thinking"])
    def test_a_model_that_refuses_them_never_gets_them(
        self, model: str, params: dict[str, Any]
    ) -> None:
        out = ant.apply_thinking_params(
            {"temperature": 0.3, "top_p": 0.99, "top_k": 5, **params}, model
        )

        assert not {"temperature", "top_p", "top_k"} & out.keys()

    def test_a_dropped_temperature_does_not_turn_reasoning_off(self) -> None:
        """A custom temperature turns reasoning off only where it is sent."""
        out = ant.apply_thinking_params(
            {"temperature": 0.3, "reasoning_effort": "high"}, "claude-opus-5"
        )

        assert out["thinking"]["type"] == "adaptive"
        assert out["output_config"] == {"effort": "high"}

    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
            "claude-opus-45",
            "claude-mythos-preview",
        ],
    )
    def test_the_rest_keep_them(self, model: str) -> None:
        """Below the floors, and a separator-collapsed (`claude-opus-45` is Opus 4.5) or
        revision-less id, which the `<10` bound keeps out."""
        out = ant.apply_thinking_params({"temperature": 0.3, "top_k": 5}, model)

        assert out["temperature"] == 0.3
        assert out["top_k"] == 5


class TestHaiku55:
    def test_it_is_adaptive_and_shows_its_thinking(self) -> None:
        """omp 18.8.6: the first adaptive Haiku (`budget_tokens` is a 400), and it takes
        `display` like the rest of its generation."""
        out = ant.apply_thinking_params({"reasoning_effort": "low"}, "claude-haiku-5-5")

        assert out["thinking"]["type"] == "adaptive"
        assert out["thinking"]["display"] == "summarized"
        assert out["output_config"] == {"effort": "low"}

    def test_it_keeps_a_forced_tool_choice(self) -> None:
        """Unlike Opus/Sonnet 5.5, omp keeps forced tool selection on Haiku 5.5."""
        out = ant.apply_thinking_params({"tool_choice": "required"}, "claude-haiku-5-5")

        assert out["tool_choice"] == "required"
        assert "thinking" not in out


class TestModelRevisions:
    def test_a_dated_build_is_its_family_revision(self) -> None:
        """`claude-sonnet-5-5-20261001` is Sonnet 5.5, and a dated Opus 5 is not 5.5."""
        dated = ant.apply_thinking_params({}, "claude-sonnet-5-5-20261001")
        opus = ant.apply_thinking_params(
            {"tool_choice": "required"}, "claude-opus-5-20260101"
        )

        assert dated["thinking"] == {"type": "between_tools"}
        assert opus["tool_choice"] == "required"

    def test_a_separator_collapsed_id_is_not_read_as_5_5(self) -> None:
        """omp bounds the 5.5 rules below 6 because `claude-opus-45` (Opus 4.5) parses as
        revision 45: an open bound would take forced tool use away from it."""
        out = ant.apply_thinking_params({"tool_choice": "required"}, "claude-opus-45")

        assert out["tool_choice"] == "required"
