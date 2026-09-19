"""Token accounting and turn outcome.

Whatever is not reported here is lost: LiteLLM falls back to estimates and the cache hits
disappear from `/spend/logs`. On a subscription account, invisible cache is the difference
between knowing and not knowing why the quota ran out.
"""

from __future__ import annotations

import pytest

from litellm_mysubs.wire.usage import (
    codex_finish_reason,
    codex_usage,
    google_finish_reason,
    google_usage,
    make_usage,
)


class TestGoogleUsage:
    def test_cached_tokens_are_not_double_counted(self) -> None:
        """promptTokenCount includes the cached ones; adding them again inflates the bill."""
        usage = google_usage(
            {
                "promptTokenCount": 1000,
                "cachedContentTokenCount": 400,
                "candidatesTokenCount": 50,
                "totalTokenCount": 1050,
            }
        )
        assert usage.prompt_tokens == 600
        assert usage.cached_tokens == 400
        assert usage.total_tokens == 1050

    def test_thoughts_count_as_output(self) -> None:
        usage = google_usage({"candidatesTokenCount": 50, "thoughtsTokenCount": 120})
        assert usage.completion_tokens == 170
        assert usage.reasoning_tokens == 120

    def test_empty_metadata_is_zeroed(self) -> None:
        usage = google_usage({})
        assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (0, 0, 0)


class TestCodexUsage:
    def test_input_tokens_are_not_reduced_by_cache(self) -> None:
        """Unlike Google: subtracting here would undercount the prompt."""
        usage = codex_usage({"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 400}})
        assert usage.prompt_tokens == 1000
        assert usage.cached_tokens == 400

    def test_legacy_cache_field_is_accepted(self) -> None:
        usage = codex_usage({"input_tokens": 10, "prompt_cache_hit_tokens": 7})
        assert usage.cached_tokens == 7

    def test_reasoning_tokens_extracted(self) -> None:
        usage = codex_usage(
            {"output_tokens": 80, "output_tokens_details": {"reasoning_tokens": 60}}
        )
        assert usage.reasoning_tokens == 60


class TestMakeUsage:
    def test_total_defaults_to_the_sum(self) -> None:
        assert make_usage(prompt_tokens=10, completion_tokens=5).total_tokens == 15

    def test_explicit_total_wins(self) -> None:
        """Upstream knows better: it may include tokens we did not break down."""
        assert make_usage(prompt_tokens=10, completion_tokens=5, total_tokens=99).total_tokens == 99

    @pytest.mark.parametrize("value", [None, "", "not-a-number", -5])
    def test_junk_becomes_zero(self, value: object) -> None:
        """A broken counter must not blow up the turn accounting."""
        assert make_usage(prompt_tokens=value).prompt_tokens == 0


class TestGoogleFinishReason:
    def test_tool_calls_win_over_stop(self) -> None:
        assert google_finish_reason("STOP", has_tool_calls=True) == "tool_calls"

    def test_max_tokens_is_truncation(self) -> None:
        """Without this, a cut-off by limit arrived as a normal, short answer."""
        assert google_finish_reason("MAX_TOKENS", has_tool_calls=False) == "length"

    def test_max_tokens_with_pending_tool_call(self) -> None:
        """The call is still the turn outcome, even with the limit reached."""
        assert google_finish_reason("MAX_TOKENS", has_tool_calls=True) == "tool_calls"

    @pytest.mark.parametrize(
        "reason",
        [
            "SAFETY",
            "RECITATION",
            "PROHIBITED_CONTENT",
            "MALFORMED_FUNCTION_CALL",
            # The five that were missing when the error was enumerated instead of success.
            "FINISH_REASON_UNSPECIFIED",
            "LANGUAGE",
            "IMAGE_OTHER",
            "IMAGE_PROHIBITED_CONTENT",
            "IMAGE_RECITATION",
        ],
    )
    def test_server_side_blocks_surface_as_content_filter(self, reason: str) -> None:
        """The only OpenAI value that does not lie about a server-imposed cut-off."""
        assert google_finish_reason(reason, has_tool_calls=False) == "content_filter"

    def test_unknown_reason_is_an_error_not_a_stop(self) -> None:
        """OMP enumerates what is normal and treats the rest as an error.

        A new upstream reason treated as `stop` delivers a truncated answer as if it were
        complete. Treated as an error, at worst it is too noisy.
        """
        assert google_finish_reason("REASON_THAT_DOES_NOT_EXIST_YET", has_tool_calls=False) == (
            "content_filter"
        )

    def test_blocked_reason_is_not_masked_by_tool_calls(self) -> None:
        """A safety block must not pass as tool_calls."""
        assert google_finish_reason("SAFETY", has_tool_calls=True) == "content_filter"

    @pytest.mark.parametrize("reason", [None, "", "STOP"])
    def test_normal_completion(self, reason: object) -> None:
        assert google_finish_reason(reason, has_tool_calls=False) == "stop"

    def test_case_and_whitespace_tolerated(self) -> None:
        assert google_finish_reason("  max_tokens  ", has_tool_calls=False) == "length"


class TestCodexFinishReason:
    def test_incomplete_is_truncation(self) -> None:
        assert codex_finish_reason("incomplete", has_tool_calls=False) == "length"

    def test_tool_calls_win(self) -> None:
        assert codex_finish_reason("incomplete", has_tool_calls=True) == "tool_calls"

    def test_default_is_stop(self) -> None:
        assert codex_finish_reason(None, has_tool_calls=False) == "stop"
        assert codex_finish_reason("completed", has_tool_calls=False) == "stop"
