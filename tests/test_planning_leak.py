"""Planning-leak filter for the flash models.

A false negative hands the internal planning to the client; a false positive erases a
legitimate answer. Every detection test has a neighbour that must **not** fire.
"""

from __future__ import annotations

import pytest

from litellm_mysubs.wire.planning_leak import (
    PlanningLeakFilter,
    is_flash_leak_model,
    is_leak_object,
    is_leak_prefix,
)


class TestModelGate:
    @pytest.mark.parametrize("model", ["gemini-3.8-flash", "gemini-2.5-flash-lite"])
    def test_flash_family_is_filtered(self, model: str) -> None:
        assert is_flash_leak_model(model) is True

    @pytest.mark.parametrize("model", ["gemini-3-pro", "gemini-3.1-pro", "gemini-pro-agent"])
    def test_other_families_are_not(self, model: str) -> None:
        """A `pro` answering `{"command": "ls"}` must not have its answer erased."""
        assert is_flash_leak_model(model) is False


class TestPrefixDetection:
    @pytest.mark.parametrize(
        "text",
        ['{"thought": "let me read the file"}', '{"tho', '{"thought"', "{", '{  "thought"  :'],
    )
    def test_recognises_partial_leaks(self, text: str) -> None:
        """An object spans chunks: `{"thou` has to be held back, not emitted."""
        assert is_leak_prefix(text) is True

    @pytest.mark.parametrize(
        "text",
        ["plain text", '{"other": 1}', '{"thoughts": 1}', "  no brace"],
    )
    def test_ignores_other_text(self, text: str) -> None:
        assert is_leak_prefix(text) is False

    def test_long_unclosed_text_is_not_a_prefix(self) -> None:
        """Past 100 chars it is legitimate text that happens to start with a brace."""
        assert is_leak_prefix("{" + "x" * 200) is False


class TestObjectSignature:
    @pytest.mark.parametrize(
        "payload",
        [
            {"thought": "planning"},
            {"_i": 1},
            {"paths": ["a"]},
            {"command": "ls"},
            {"path": "a.py", "content": "x"},
        ],
    )
    def test_leak_signatures(self, payload: dict[str, object]) -> None:
        assert is_leak_object(payload) is True

    def test_known_tool_call_is_a_leak(self) -> None:
        assert is_leak_object({"call": "read"}, frozenset({"read"})) is True

    def test_unknown_call_is_not(self) -> None:
        """It only counts as a leak when the name is one of the tools we declared."""
        assert is_leak_object({"call": "something_else"}, frozenset({"read"})) is False

    @pytest.mark.parametrize("payload", [{"result": 42}, {"path": "a.py"}, [], "text", None])
    def test_plain_objects_pass(self, payload: object) -> None:
        assert is_leak_object(payload) is False


class TestStreamFiltering:
    def test_plain_text_passes_through(self) -> None:
        assert PlanningLeakFilter().feed("hello world") == "hello world"

    def test_whole_leak_in_one_chunk_is_dropped(self) -> None:
        leak = PlanningLeakFilter()
        assert leak.feed('{"thought": "let me read"}') == ""
        assert leak.stripped is True

    def test_leak_split_across_chunks_is_dropped(self) -> None:
        """Deciding per chunk let through everything that did not fit in a single one."""
        leak = PlanningLeakFilter()
        assert leak.feed('{"thou') == ""
        assert leak.feed('ght": "let me read the fi') == ""
        assert leak.feed('le"}') == ""
        assert leak.stripped is True

    def test_text_after_the_leak_survives(self) -> None:
        leak = PlanningLeakFilter()
        assert leak.feed('{"thought": "x"}visible answer') == "visible answer"

    def test_legitimate_json_is_not_dropped(self) -> None:
        """An object without a planning signature is an answer, not a leak."""
        leak = PlanningLeakFilter()
        assert leak.feed('{"result": 42}') == '{"result": 42}'
        assert leak.stripped is False

    def test_unbalanced_quotes_still_close_the_object(self) -> None:
        """A leak with unbalanced quotes would never close through the normal path."""
        leak = PlanningLeakFilter()
        assert leak.feed('{"thought": "quote " too many"}') == ""
        assert leak.stripped is True

    def test_unterminated_leak_is_discarded_at_eof(self) -> None:
        """Delivering it would mean showing half of the internal planning."""
        leak = PlanningLeakFilter()
        assert leak.feed('{"thought": "nunca fecha') == ""
        assert leak.flush() == ""
        assert leak.stripped is True

    def test_lone_brace_is_discarded_at_eof(self) -> None:
        """A lone brace has the signature of an incomplete leak; it is not text."""
        leak = PlanningLeakFilter()
        assert leak.feed("{") == ""
        assert leak.flush() == ""
        assert leak.stripped is True

    def test_non_leak_json_is_emitted_immediately(self) -> None:
        """Only what carries a leak signature enters the buffer; the rest passes at once.

        Holding back legitimate JSON would delay the stream for no reason.
        """
        leak = PlanningLeakFilter()
        assert leak.feed('{"result": ') == '{"result": '
        assert leak.stripped is False

    def test_empty_feed_is_a_noop(self) -> None:
        leak = PlanningLeakFilter()
        assert leak.feed("") == ""
        assert leak.flush() == ""
