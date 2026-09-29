"""Retry policy, endpoint rotation, retry hints and SSE decoding.

In the original these decisions lived inside the ``httpx`` loops, duplicated between the
synchronous and the asynchronous version — and the two had drifted into different shapes of
the same rule. Testing them in isolation is what stops that divergence from coming back.
"""

from __future__ import annotations

from datetime import UTC, datetime
from email.utils import format_datetime

import pytest

from litellm_mysubs.transport import sse
from litellm_mysubs.transport.hosts import HOSTS, STREAM_PATH, HostRotation
from litellm_mysubs.transport.replay import is_retryable_codex_failure
from litellm_mysubs.transport.retry import (
    Action,
    decide_antigravity,
    decide_codex,
    is_unsupported_model,
    retry_hint,
)

#: A fixed "now" for the hints that name an absolute time: 2026-09-01T00:00:00Z.
NOW = datetime(2026, 9, 1, tzinfo=UTC).timestamp()


class TestCodexDecisions:
    def test_401_refreshes_the_token(self) -> None:
        assert decide_codex(401).action is Action.REFRESH_TOKEN

    def test_unsupported_alias_is_remapped(self) -> None:
        """Family names only: resolving them to the served version is honest."""
        decision = decide_codex(
            400, "The 'codex' model is not supported when using Codex", can_remap=True
        )
        assert decision.action is Action.REMAP_MODEL

    def test_unsupported_arbitrary_name_fails(self) -> None:
        """Substituting an arbitrary name returned 200 with the `model` field echoing the
        request, and billing started to lie."""
        decision = decide_codex(
            400, "The 'gpt-4.1' model is not supported when using Codex", can_remap=False
        )
        assert decision.action is Action.FAIL

    def test_other_400_is_not_a_model_problem(self) -> None:
        """An invalid payload is not fixed by switching models."""
        assert decide_codex(400, "Invalid value at 'input'").action is Action.FAIL

    def test_429_redeems_when_credit_exists(self) -> None:
        assert decide_codex(429, can_redeem=True).action is Action.REDEEM_CREDIT

    def test_429_without_credit_fails(self) -> None:
        """Without credit, retrying gave the same error three times and tripled latency."""
        assert decide_codex(429, can_redeem=False).action is Action.FAIL

    @pytest.mark.parametrize("status", [403, 404, 500, 502, 503])
    def test_other_statuses_propagate(self, status: int) -> None:
        decision = decide_codex(status)
        assert decision.action is Action.FAIL
        assert str(status) in decision.reason

    def test_marker_detection(self) -> None:
        assert is_unsupported_model("The 'x' model is not supported when using Codex") is True
        assert is_unsupported_model("rate limit exceeded") is False


class TestAntigravityDecisions:
    def test_401_refreshes(self) -> None:
        assert decide_antigravity(401).action is Action.REFRESH_TOKEN

    @pytest.mark.parametrize("status", [408, 500, 503])
    def test_a_transient_status_moves_to_the_other_host(self, status: int) -> None:
        assert decide_antigravity(status).action is Action.FAIL

    @pytest.mark.parametrize("status", [400, 403, 404, 429])
    def test_the_rest_propagates_from_the_host_that_answered(self, status: int) -> None:
        """A 4xx is the request's fault or the account's verdict: the other host answers
        the same. None of them authorises answering with a different model either."""
        assert decide_antigravity(status).action is Action.ABORT


class TestHostRotation:
    def test_starts_on_the_primary(self) -> None:
        rotation = HostRotation()
        assert rotation.urls()[0].startswith(HOSTS[0])

    def test_always_offers_every_host(self) -> None:
        """Both are always tried: none is excluded because of an earlier failure."""
        assert len(HostRotation().urls()) == len(HOSTS)

    def test_commit_remembers_the_last_good_host(self) -> None:
        rotation = HostRotation()
        rotation.commit(HOSTS[1] + STREAM_PATH)
        assert rotation.urls()[0].startswith(HOSTS[1])
        assert rotation.current == HOSTS[1]

    def test_fallback_host_still_offered_after_switching(self) -> None:
        """The primary is not abandoned: it may start answering again."""
        rotation = HostRotation()
        rotation.commit(HOSTS[1] + STREAM_PATH)
        assert any(url.startswith(HOSTS[0]) for url in rotation.urls())

    def test_unknown_url_does_not_move_the_pointer(self) -> None:
        rotation = HostRotation()
        rotation.commit("https://example.invalid/x")
        assert rotation.current == HOSTS[0]

    def test_path_is_configurable(self) -> None:
        rotation = HostRotation()
        urls = rotation.urls("/v1internal:fetchAvailableModels")
        assert all(url.endswith("/v1internal:fetchAvailableModels") for url in urls)

    def test_instances_do_not_share_memory(self) -> None:
        """Two clients in the same process may sit on different hosts."""
        first, second = HostRotation(), HostRotation()
        first.commit(HOSTS[1] + STREAM_PATH)
        assert second.current == HOSTS[0]


class TestRetryHint:
    """omp's `extractRetryHint`: the wait the server asked for, in seconds."""

    @pytest.mark.parametrize(
        ("headers", "expected"),
        [
            ({"Retry-After": "7"}, 7.0),
            ({"retry-after-ms": "250"}, 0.25),
            ({"retry-after": format_datetime(datetime.fromtimestamp(NOW + 90, UTC))}, 90.0),
            ({"x-ratelimit-reset-ms": "1500"}, 1.5),
            ({"x-ratelimit-reset-ms": str(int((NOW + 30) * 1000))}, 30.0),
            ({"x-ratelimit-reset": str(int(NOW + 12))}, 12.0),
            ({"x-ratelimit-reset-after": "3"}, 3.0),
        ],
    )
    def test_headers(self, headers: dict[str, str], expected: float) -> None:
        assert retry_hint(headers, now=NOW) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            ("Your quota will reset after 1h2m3s.", 3723.0),
            ("Please retry in 250ms", 0.25),
            ('{"retryDelay": "34.074824224s"}', 34.074824224),
            ("Rate limited, try again in ~158 min.", 158 * 60.0),
            ("Resets in 2hr 15min", 2 * 3600 + 15 * 60.0),
            ("retry-after-ms=98497000", 98497.0),
            ("Your limit will reset at 2026-09-01T00:10:00Z", 600.0),
        ],
    )
    def test_body(self, body: str, expected: float) -> None:
        assert retry_hint({}, body, now=NOW) == pytest.approx(expected)

    def test_the_longest_signal_in_the_body_wins(self) -> None:
        """Retrying before every window clears re-hits a credential still blocked."""
        body = "Please retry in 5s. Your quota will reset after 2m0s."
        assert retry_hint({}, body, now=NOW) == 120.0

    def test_headers_win_over_the_body(self) -> None:
        assert retry_hint({"retry-after": "3"}, "reset after 1h0m0s", now=NOW) == 3.0

    def test_an_explicit_zero_means_now_not_absent(self) -> None:
        """``None`` would make the caller sleep its own backoff on a "retry now"."""
        assert retry_hint({"retry-after": "0"}, now=NOW) == 0.0
        assert retry_hint({}, "retry-after-ms: 0", now=NOW) == 0.0

    def test_a_naive_reset_time_only_counts_alone(self) -> None:
        """A wall clock with no zone is a guess; any unambiguous signal beats it."""
        naive = "limit will reset at 2026-09-01 01:00:00"
        assert retry_hint({}, naive, now=NOW) == 3600.0
        assert retry_hint({}, f"{naive}; try again in 20s", now=NOW) == 20.0

    def test_no_signal_is_none(self) -> None:
        assert retry_hint({"content-type": "text/plain"}, "quota exhausted", now=NOW) is None


class TestCodexFailureClassification:
    """omp's `isRetryableCodexFailureEvent`: which in-band failures it sends again."""

    @pytest.mark.parametrize(
        "event",
        [
            {"type": "response.failed", "response": {"error": {"code": "server_error"}}},
            {"type": "error", "code": "Internal_Error"},
            {"type": "error", "error": {"type": "model_error"}},
            {"type": "error", "message": "The service is temporarily unavailable"},
            {
                "type": "response.failed",
                "response": {
                    "error": {"message": "An error occurred while processing your request."}
                },
            },
            {
                "type": "error",
                "message": "peer closed connection without sending complete message body "
                "(incomplete chunked read)",
            },
        ],
    )
    def test_transient(self, event: dict[str, object]) -> None:
        assert is_retryable_codex_failure(event) is True

    @pytest.mark.parametrize(
        "event",
        [
            {"type": "response.failed", "response": {"error": {"code": "context_length_exceeded"}}},
            {"type": "error", "error": {"code": "invalid_prompt", "message": "bad input"}},
            {"type": "response.failed"},
        ],
    )
    def test_final(self, event: dict[str, object]) -> None:
        assert is_retryable_codex_failure(event) is False


class TestSseDecoder:
    """omp's `readSseEvents` framing and `readSseFrames` JSON rules."""

    def decode(self, *lines: str) -> list[object]:
        decoder = sse.SseDecoder()
        frames = [frame for line in lines for frame in decoder.feed(line)]
        return frames + decoder.close()

    def test_data_lines_join_into_one_event(self) -> None:
        assert self.decode('data: {"a":', "data: 1}", "") == [{"a": 1}]

    def test_the_space_after_the_colon_is_optional(self) -> None:
        assert self.decode('data:{"a":1}', "") == [{"a": 1}]

    def test_comments_and_other_fields_are_ignored(self) -> None:
        assert self.decode(": ping", "event: delta", "id: 7", 'data: {"a":1}', "") == [{"a": 1}]

    def test_a_leading_bom_is_dropped(self) -> None:
        assert self.decode('\ufeffdata: {"a":1}', "") == [{"a": 1}]

    def test_done_ends_the_stream(self) -> None:
        assert self.decode('data: {"a":1}', "", "data: [DONE]", "", 'data: {"b":2}', "") == [
            {"a": 1}
        ]

    def test_an_event_without_the_final_blank_line_still_counts(self) -> None:
        assert self.decode('data: {"a":1}') == [{"a": 1}]

    def test_a_cut_off_object_at_the_end_ends_quietly(self) -> None:
        assert self.decode('data: {"a":1}', "", 'data: {"b": [1, ') == [{"a": 1}]

    def test_a_malformed_event_is_reported_not_skipped(self) -> None:
        frames = self.decode("data: <html>", "", 'data: {"a":1}', "")
        assert isinstance(frames[0], sse.Malformed)
        assert frames[1:] == [{"a": 1}]

    def test_the_lenient_reader_skips_what_does_not_parse(self) -> None:
        lines = ["data: oops", "", 'data: {"a":1}', "", "data: [DONE]", "", 'data: {"b":2}']
        assert list(sse.iter_events(lines)) == [{"a": 1}]
