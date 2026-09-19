"""Anthropic wire contract.

Every test matches a shape that was measured against the real service. The comment states
what the upstream returns when the shape is wrong — that is what makes the test a defence
rather than a description of the implementation.
"""

from __future__ import annotations

from typing import Any

import pytest

from litellm_mysubs.wire import anthropic as ant


class TestModelDetection:
    @pytest.mark.parametrize(
        "model",
        ["claude-opus-5", "anthropic/claude-haiku-4-5", "CLAUDE-SONNET-5"],
    )
    def test_recognises_claude(self, model: str) -> None:
        assert ant.is_anthropic_model(model) is True

    @pytest.mark.parametrize("model", ["gpt-5.5", "gemini-3-pro", "openai/qwen35b"])
    def test_ignores_other_providers(self, model: str) -> None:
        assert ant.is_anthropic_model(model) is False


class TestAdaptiveDetection:
    """Guessing adaptive wrongly gives 400; guessing budget wrongly gives 200 with 0 chars
    of reasoning.

    The asymmetry is what decides the default: an unknown model has to land on the side
    that is detectable.
    """

    @pytest.mark.parametrize("model", ["claude-opus-5", "claude-fable-5", "claude-sonnet-5"])
    def test_adaptive_models(self, model: str) -> None:
        assert ant.is_adaptive(model) is True

    @pytest.mark.parametrize(
        "model",
        ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5", "claude-3-5-sonnet"],
    )
    def test_budget_only_models(self, model: str) -> None:
        assert ant.is_adaptive(model) is False

    def test_unknown_model_defaults_to_adaptive(self) -> None:
        """A new model has to fail detectably, not silently."""
        assert ant.is_adaptive("claude-opus-6") is True


class TestNormalizeEffort:
    def test_plain_string(self) -> None:
        assert ant.normalize_effort("HIGH") == ("high", None)

    def test_responses_route_object(self) -> None:
        """/v1/responses hands over an object; treating it as a string puts the repr on the
        wire."""
        assert ant.normalize_effort({"effort": "medium", "summary": "auto"}) == (
            "medium",
            "auto",
        )

    def test_empty_is_none(self) -> None:
        assert ant.normalize_effort(None) == (None, None)
        assert ant.normalize_effort("") == (None, None)


class TestSystemBlocks:
    def test_identity_is_first_block(self) -> None:
        """system=[client] returns 429; the identity has to come first."""
        blocks = ant.build_system_blocks("be brief")
        assert blocks[0]["text"] == ant.CLAUDE_CODE_PROMPT
        assert blocks[1]["text"] == "be brief"

    def test_identity_alone_when_no_client_prompt(self) -> None:
        assert ant.build_system_blocks("") == [{"type": "text", "text": ant.CLAUDE_CODE_PROMPT}]

    def test_client_prompt_keeps_system_authority(self) -> None:
        """Stuffing the client prompt into a user turn stripped its system authority."""
        kwargs: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": "client rule"},
                {"role": "user", "content": "hello"},
            ]
        }
        out = ant.build_request(kwargs, "claude-opus-5")
        assert out["messages"][0]["role"] == "system"
        assert len(out["messages"][0]["content"]) == 2
        assert out["messages"][1]["role"] == "user"

    def test_merges_multiple_system_messages(self) -> None:
        client, rest = ant.split_system_messages(
            [
                {"role": "system", "content": "one"},
                {"role": "user", "content": "x"},
                {"role": "system", "content": "two"},
            ]
        )
        assert client == "one\n\ntwo"
        assert len(rest) == 1

    def test_strips_duplicated_identity(self) -> None:
        """A client that already sends the identity must not duplicate it in its own block."""
        client, _ = ant.split_system_messages(
            [{"role": "system", "content": ant.CLAUDE_CODE_PROMPT}]
        )
        assert client == ""


class TestCacheAnchors:
    def test_tool_result_is_markable(self) -> None:
        """Refusing it pinned the window to the head: 67% of the prompt re-read at full price."""
        assert ant.is_markable({"role": "tool", "tool_call_id": "c1", "content": "r"}) is True

    def test_thinking_block_is_not_an_anchor(self) -> None:
        assert (
            ant.is_markable({"role": "assistant", "content": [{"type": "thinking", "text": "x"}]})
            is False
        )

    def test_hosted_tool_call_is_not_an_anchor(self) -> None:
        """server_tool_use is emitted without cache_control by LiteLLM."""
        message = {
            "role": "assistant",
            "tool_calls": [{"id": "srvtoolu_1", "type": "function", "function": {"name": "s"}}],
        }
        assert ant.tool_call_anchor(message) is None

    def test_empty_content_is_not_markable(self) -> None:
        assert ant.is_markable({"role": "user", "content": "   "}) is False

    def test_marks_two_tail_messages(self) -> None:
        messages: list[Any] = [{"role": "user", "content": f"m{i}"} for i in range(5)]
        assert ant.apply_conversation_cache(messages) == 2
        assert ant.count_breakpoints(messages) == 2
        # The breakpoints stay in the tail, which is the prefix worth keeping.
        assert ant.count_breakpoints(messages[-2:]) == 2

    def test_respects_client_breakpoints_within_ceiling(self) -> None:
        """5 breakpoints give 400 "A maximum of 4 blocks with cache_control"."""
        messages: list[Any] = [
            {"role": "user", "content": [{"type": "text", "text": "a", "cache_control": {}}]},
            {"role": "user", "content": [{"type": "text", "text": "b", "cache_control": {}}]},
            {"role": "user", "content": [{"type": "text", "text": "c", "cache_control": {}}]},
            {"role": "user", "content": "d"},
            {"role": "user", "content": "e"},
        ]
        ant.apply_conversation_cache(messages)
        assert ant.count_breakpoints(messages) <= ant.CACHE_BREAKPOINT_CEILING

    def test_gives_up_when_client_exhausted_budget(self) -> None:
        messages: list[Any] = [
            {"role": "user", "content": [{"type": "text", "text": str(i), "cache_control": {}}]}
            for i in range(4)
        ]
        assert ant.apply_conversation_cache(messages) == 0

    def test_synthetic_continue_is_skipped(self) -> None:
        """A synthetic "Continue." at the end is not a useful anchor."""
        messages: list[Any] = [
            {"role": "user", "content": "real question"},
            {"role": "user", "content": "Continue."},
        ]
        ant.apply_conversation_cache(messages)
        assert "cache_control" not in str(messages[1])

    def test_no_anchors_is_a_noop(self) -> None:
        messages: list[Any] = [{"role": "system", "content": "x"}]
        assert ant.apply_conversation_cache(messages) == 0


class TestHeadCaching:
    """Without a head anchor, the tools+system prefix is only covered by the tail anchor,
    which moves position on every turn: the large, immutable head is rewritten at full
    price on every request."""

    def test_last_non_deferred_tool_is_anchored(self) -> None:
        tools: list[Any] = [
            {"type": "function", "function": {"name": "a"}},
            {"type": "function", "function": {"name": "b"}},
        ]
        ant.apply_head_cache(None, tools)
        assert "cache_control" not in tools[0]
        assert tools[1]["cache_control"] == ant.cache_control()

    def test_deferred_tool_is_skipped(self) -> None:
        """A deferred tool does not enter the verified prefix until it is referenced, so
        anchoring on it left out everything that comes before."""
        tools: list[Any] = [
            {"type": "function", "function": {"name": "a"}},
            {"type": "function", "function": {"name": "b"}, "defer_loading": True},
        ]
        ant.apply_head_cache(None, tools)
        assert tools[1].get("cache_control") is None
        assert tools[0]["cache_control"] == ant.cache_control()

    def test_last_stable_system_block_is_anchored(self) -> None:
        """With a volatile suffix, anchoring on the tail of the array made a memory refresh
        re-bill the whole head instead of just the suffix."""
        blocks: list[Any] = [
            {"type": "text", "text": "identity"},
            {"type": "text", "text": "stable prompt"},
            {"type": "text", "text": "<memories>yesterday you had soup</memories>"},
        ]
        ant.apply_head_cache(blocks, None)
        assert blocks[1]["cache_control"] == ant.cache_control()
        assert "cache_control" not in blocks[2]

    def test_all_volatile_system_falls_back_to_tail(self) -> None:
        blocks: list[Any] = [{"type": "text", "text": "<memories>x</memories>"}]
        ant.apply_head_cache(blocks, None)
        assert blocks[-1]["cache_control"] == ant.cache_control()

    def test_head_budget_is_deducted_from_messages(self) -> None:
        """The ceiling of 4 is per request: 5 breakpoints give 400 "A maximum of 4 blocks"."""
        messages: list[Any] = [{"role": "user", "content": f"m{i}"} for i in range(5)]
        assert ant.apply_conversation_cache(messages, head_breakpoints=4) == 0
        assert ant.apply_conversation_cache(messages, head_breakpoints=3) == 1

    def test_build_request_never_exceeds_the_ceiling(self) -> None:
        kwargs: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": "rules"},
                *({"role": "user", "content": f"m{i}"} for i in range(40)),
            ],
            "tools": [{"type": "function", "function": {"name": "a"}}],
        }
        out = ant.build_request(kwargs, "claude-opus-5")
        system_blocks = out["messages"][0]["content"]
        total = ant.count_head_breakpoints(system_blocks, out["tools"]) + ant.count_breakpoints(
            out["messages"][1:]
        )
        assert total == ant.CACHE_BREAKPOINT_CEILING


class TestDecimation:
    """The two tail anchors move on every turn; when the 5 min window expires no live entry
    is left covering the old prefix and it is re-read at full price."""

    def test_checkpoints_at_the_fifteenth_and_thirtieth_turn(self) -> None:
        messages: list[Any] = [{"role": "user", "content": f"m{i}"} for i in range(35)]
        ant.apply_conversation_cache(messages)
        marked = {i for i, m in enumerate(messages) if ant.count_breakpoints([m])}
        # Ordinals 15 and 30 -> indices 14 and 29; the two tail ones stay at the end.
        assert {14, 29} <= marked
        assert {33, 34} & marked

    def test_short_conversation_has_no_checkpoint(self) -> None:
        messages: list[Any] = [{"role": "user", "content": f"m{i}"} for i in range(5)]
        assert ant.apply_conversation_cache(messages) == ant.CACHE_BREAKPOINT_MESSAGES

    def test_checkpoint_outranks_the_second_tail_anchor(self) -> None:
        """On a short budget it is the stable checkpoint that survives: the second tail
        anchor is redundant with the first, the checkpoint has no substitute."""
        messages: list[Any] = [{"role": "user", "content": f"m{i}"} for i in range(20)]
        assert ant.apply_conversation_cache(messages, head_breakpoints=2) == 2
        marked = {i for i, m in enumerate(messages) if ant.count_breakpoints([m])}
        assert marked == {14, 19}


class TestCacheRetention:
    def test_defaults_to_one_hour(self) -> None:
        """Without a ttl the entry dies at 5 min, and the pause between agent turns exceeds
        that, which makes the prefix be rewritten cold."""
        assert ant.cache_control() == {"type": "ephemeral", "ttl": "1h"}

    def test_short_retention_omits_ttl(self) -> None:
        assert ant.cache_control(None) == {"type": "ephemeral"}

    def test_extended_ttl_beta_absent_on_oauth(self) -> None:
        """On the OAuth path the `ttl: "1h"` is honoured with no beta at all.

        The OMP only adds it when `!isOAuth`. The header in `usage/claude.ts` carries it and
        looks like the counter-example, but it belongs to the usage route and also carries
        `redact-thinking-2026-02-12`, which we measured emptying the reasoning blocks.
        """
        assert ant.EXTENDED_CACHE_TTL_BETA not in ant.build_betas(thinking=True)

    def test_adaptive_model_gets_output_config(self) -> None:
        """budget_tokens is ignored on these models; adaptive + effort is the only way."""
        out = ant.apply_thinking_params({"reasoning_effort": "high"}, "claude-opus-5")
        assert out["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert out["output_config"] == {"effort": "high"}
        assert "reasoning_effort" not in out

    def test_display_is_gated_by_generation_not_by_adaptive(self) -> None:
        """Being adaptive is not enough: the field is gated by generation.

        The gate in the source is opus >= 4.7 and sonnet/fable/mythos >= 5. Opus 4.6 and
        Sonnet 4.6 are adaptive and still refuse `display` with 400, so the condition cannot
        be `is_adaptive`.
        """
        for model in ("claude-opus-5", "claude-opus-4-7", "claude-sonnet-5", "claude-fable-5"):
            out = ant.apply_thinking_params({"reasoning_effort": "high"}, model)
            assert out["thinking"]["display"] == "summarized", model

        for model in ("claude-opus-4-6", "claude-sonnet-4-6"):
            out = ant.apply_thinking_params({"reasoning_effort": "high"}, model)
            assert ant.is_adaptive(model), model
            assert "display" not in out["thinking"], model

    def test_client_display_choice_is_respected(self) -> None:
        """`omitted` is a legitimate choice for whoever does not want the reasoning text."""
        out = ant.apply_thinking_params(
            {"thinking": {"type": "adaptive", "display": "omitted"}}, "claude-opus-5"
        )
        assert out["thinking"]["display"] == "omitted"

    def test_forced_tool_choice_pins_effort_on_adaptive(self) -> None:
        """Omitting thinking on an adaptive model does not turn it off: the API turns it back
        on."""
        out = ant.apply_thinking_params(
            {"reasoning_effort": "high", "tool_choice": {"type": "any"}}, "claude-opus-5"
        )
        assert out["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert out["output_config"] == {"effort": "low"}

    def test_budget_model_gets_budget_tokens(self) -> None:
        """The steps are the OMP ones (low=4096), not half of them."""
        out = ant.apply_thinking_params({"reasoning_effort": "low"}, "claude-haiku-4-5")
        assert out["thinking"] == {"type": "enabled", "budget_tokens": ant.EFFORT_BUDGET["low"]}
        assert "output_config" not in out

    def test_ceiling_caps_the_top_steps(self) -> None:
        """The OMP scale is preserved; only the subscription ceiling cuts it."""
        out = ant.apply_thinking_params({"reasoning_effort": "max"}, "claude-haiku-4-5")
        assert out["thinking"]["budget_tokens"] == ant.THINKING_CEILING

    def test_xhigh_and_max_are_distinct_steps(self) -> None:
        """Measured: out=164 at high, 273 at xhigh, 275 at max — collapsing them loses steps."""
        assert ant.ADAPTIVE_EFFORT["xhigh"] == "xhigh"
        assert ant.ADAPTIVE_EFFORT["max"] == "max"

    def test_temperature_forced_to_one_when_thinking(self) -> None:
        """400 "may only be set to 1 when thinking is enabled"."""
        out = ant.apply_thinking_params(
            {"reasoning_effort": "medium", "temperature": 0.7}, "claude-opus-5"
        )
        assert out["temperature"] == 1.0

    def test_low_top_p_dropped_when_thinking(self) -> None:
        """400 "`top_p` must be greater than or equal to 0.95 or unset"."""
        out = ant.apply_thinking_params(
            {"reasoning_effort": "medium", "top_p": 0.5}, "claude-opus-5"
        )
        assert "top_p" not in out

    def test_high_top_p_survives(self) -> None:
        out = ant.apply_thinking_params(
            {"reasoning_effort": "medium", "top_p": 0.99}, "claude-opus-5"
        )
        assert out["top_p"] == 0.99

    def test_effort_none_disables_thinking(self) -> None:
        out = ant.apply_thinking_params({"reasoning_effort": "none"}, "claude-opus-5")
        assert "thinking" not in out
        assert "reasoning_effort" not in out

    def test_forced_tool_choice_disables_budget_thinking(self) -> None:
        """400 "Thinking may not be enabled when tool_choice forces tool use"."""
        out = ant.apply_thinking_params(
            {"reasoning_effort": "high", "tool_choice": "required"}, "claude-haiku-4-5"
        )
        assert "thinking" not in out

    def test_forced_tool_choice_survives_on_adaptive(self) -> None:
        """On adaptive models the pair is accepted (200); disabling it lost reasoning."""
        out = ant.apply_thinking_params(
            {"reasoning_effort": "high", "tool_choice": "required"}, "claude-opus-5"
        )
        assert out["thinking"]["type"] == "adaptive"
        assert out["output_config"]["effort"] == "low"

    def test_max_tokens_preserved_up_to_the_claude_code_ceiling(self) -> None:
        out = ant.apply_thinking_params(
            {"reasoning_effort": "high", "max_tokens": 64000}, "claude-haiku-4-5"
        )
        assert out["max_tokens"] == ant.MAX_OUTPUT_TOKENS

    def test_output_gets_room_beyond_the_thinking_budget(self) -> None:
        """Without the margin, the answer comes out truncated after the model thinks."""
        out = ant.apply_thinking_params(
            {"reasoning_effort": "high", "max_tokens": 100}, "claude-haiku-4-5"
        )
        assert out["max_tokens"] == ant.THINKING_CEILING + ant.OUTPUT_FALLBACK_BUFFER

    def test_narrow_margin_is_widened(self) -> None:
        """budget+500 is not a margin: the OMP raises it whenever OUTPUT_FALLBACK_BUFFER is
        missing."""
        out = ant.apply_thinking_params(
            {"reasoning_effort": "high", "max_tokens": ant.THINKING_CEILING + 500},
            "claude-haiku-4-5",
        )
        assert out["max_tokens"] == ant.THINKING_CEILING + ant.OUTPUT_FALLBACK_BUFFER

    def test_max_completion_tokens_renamed_not_duplicated(self) -> None:
        """Filling both keys made the default override the client's value."""
        out = ant.apply_thinking_params(
            {"reasoning_effort": "low", "max_completion_tokens": 30000}, "claude-haiku-4-5"
        )
        assert out["max_tokens"] == 30000
        assert "max_completion_tokens" not in out

    def test_disabled_thinking_object_is_removed(self) -> None:
        out = ant.apply_thinking_params({"thinking": {"type": "disabled"}}, "claude-opus-5")
        assert "thinking" not in out


class TestBuildRequest:
    def test_non_claude_model_untouched(self) -> None:
        kwargs = {"messages": [{"role": "user", "content": "x"}], "temperature": 0.2}
        assert ant.build_request(dict(kwargs), "gpt-5.5") == kwargs

    def test_injects_client_headers(self) -> None:
        out = ant.build_request({"messages": []}, "claude-opus-5")
        assert out["extra_headers"]["x-app"] == "cli"
        assert "claude-code-20250219" in out["extra_headers"]["anthropic-beta"]

    def test_oauth_beta_is_present(self) -> None:
        """Without it the server classifies the request as coming from an API key."""
        assert "oauth-2025-04-20" in ant.build_betas(thinking=False)

    def test_redact_thinking_beta_absent(self) -> None:
        """With that beta Anthropic returns thinking that is signed but empty: 74 -> 0 chars."""
        assert "redact-thinking" not in ant.build_betas(thinking=True)

    def test_context_1m_beta_absent(self) -> None:
        """context-1m-2025-08-07 gives a credit 429 on subscription tokens."""
        assert "context-1m" not in ant.build_betas(thinking=True)

    def test_effort_beta_only_when_thinking(self) -> None:
        """The OMP only adds it when the request asks for reasoning."""
        assert ant.EFFORT_BETA in ant.build_betas(thinking=True)
        assert ant.EFFORT_BETA not in ant.build_betas(thinking=False)

    def test_user_agent_matches_the_x_app_entrypoint(self) -> None:
        """`claude-desktop` in the UA with `x-app: cli` was an incoherent fingerprint."""
        assert "(external, cli)" in ant.CLIENT_HEADERS["User-Agent"]
        assert ant.CLIENT_HEADERS["x-app"] == "cli"

    def test_token_applied_when_given(self) -> None:
        out = ant.build_request({"messages": []}, "claude-opus-5", access_token="tok-1")
        assert out["api_key"] == "tok-1"

    def test_no_token_leaves_api_key_alone(self) -> None:
        """With no token, whatever the caller set is not erased."""
        out = ant.build_request({"messages": [], "api_key": "existente"}, "claude-opus-5")
        assert out["api_key"] == "existente"
