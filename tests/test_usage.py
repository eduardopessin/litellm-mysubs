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
    google_stop_reason,
    google_usage,
)


class TestGoogleUsage:
    def test_cached_tokens_stay_inside_the_prompt(self) -> None:
        """promptTokenCount includes the cached ones, and so does the chat `prompt_tokens`
        LiteLLM prices from: subtracting them here had LiteLLM subtract them twice."""
        usage = google_usage(
            {
                "promptTokenCount": 1000,
                "cachedContentTokenCount": 400,
                "candidatesTokenCount": 50,
                "totalTokenCount": 1050,
            }
        )
        assert usage.prompt_tokens == 1000
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

    def test_separate_orchestration_tokens_are_billed(self) -> None:
        """A reported total that matches primary + orchestration means the orchestration
        counters sit beside the primary ones; omp bills them at the model's rates."""
        usage = codex_usage(
            {
                "input_tokens": 1000,
                "output_tokens": 100,
                "total_tokens": 1400,
                "input_tokens_details": {
                    "cached_tokens": 400,
                    "orchestration_input_tokens": 250,
                    "orchestration_input_cached_tokens": 50,
                },
                "output_tokens_details": {"orchestration_output_tokens": 50},
            }
        )
        assert (usage.prompt_tokens, usage.cached_tokens, usage.completion_tokens) == (
            1250,
            450,
            150,
        )

    def test_included_orchestration_tokens_are_not_billed_twice(self) -> None:
        """A reported total that matches the primary counters alone means they already
        contain the orchestration tokens."""
        usage = codex_usage(
            {
                "input_tokens": 1000,
                "output_tokens": 100,
                "total_tokens": 1100,
                "input_tokens_details": {
                    "cached_tokens": 400,
                    "orchestration_input_tokens": 250,
                    "orchestration_input_cached_tokens": 50,
                },
                "output_tokens_details": {"orchestration_output_tokens": 50},
            }
        )
        assert (usage.prompt_tokens, usage.cached_tokens, usage.completion_tokens) == (
            1000,
            400,
            100,
        )

    @pytest.mark.parametrize("value", [None, "", "not-a-number", -5, True])
    def test_junk_becomes_zero(self, value: object) -> None:
        """A broken counter must not blow up the turn accounting."""
        assert codex_usage({"input_tokens": value}).prompt_tokens == 0


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
            # A new upstream reason treated as `stop` delivers a truncated answer as if it
            # were complete. Treated as an error, at worst it is too noisy.
            "REASON_THAT_DOES_NOT_EXIST_YET",
        ],
    )
    def test_everything_but_stop_and_max_tokens_is_an_error(self, reason: str) -> None:
        assert google_stop_reason(reason) == "error"

    @pytest.mark.parametrize("reason", [None, "", "STOP"])
    def test_normal_completion(self, reason: object) -> None:
        assert google_finish_reason(reason, has_tool_calls=False) == "stop"


class TestCodexFinishReason:
    def test_incomplete_is_truncation(self) -> None:
        assert codex_finish_reason("incomplete", has_tool_calls=False) == "length"

    def test_tool_calls_win(self) -> None:
        assert codex_finish_reason("incomplete", has_tool_calls=True) == "tool_calls"

    def test_default_is_stop(self) -> None:
        assert codex_finish_reason(None, has_tool_calls=False) == "stop"
        assert codex_finish_reason("completed", has_tool_calls=False) == "stop"
