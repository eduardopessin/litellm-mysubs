"""Planning-leak filter for the flash models.

A false negative hands the internal planning to the client; a false positive erases a
legitimate answer. Every detection test has a neighbour that must **not** fire.
"""

from __future__ import annotations

import pytest

from litellm_mysubs.wire.planning_leak import (
    consume_planning_buffer,
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


class TestConsumePlanningBuffer:
    """omp's `consumePlanningBuffer`: what the reader releases from its held text."""

    def test_text_that_cannot_open_a_leak_is_plain(self) -> None:
        assert consume_planning_buffer("hello world") == ("plain", "hello world")

    def test_whole_leak_is_dropped(self) -> None:
        assert consume_planning_buffer('{"thought": "let me read"}') == ("leak", "")

    def test_leak_split_across_deltas_is_held_until_it_closes(self) -> None:
        """Deciding per chunk let through everything that did not fit in a single one."""
        held = ""
        for delta in ['{"thou', 'ght": "let me read the fi', 'le"}']:
            held += delta
            outcome = consume_planning_buffer(held)
            if outcome[0] != "incomplete":
                break
        assert outcome == ("leak", "")

    def test_text_after_the_leak_survives(self) -> None:
        assert consume_planning_buffer('{"thought": "x"}visible answer') == (
            "leak",
            "visible answer",
        )

    def test_legitimate_json_is_not_dropped(self) -> None:
        """An object without a planning signature is an answer, not a leak."""
        assert consume_planning_buffer('{"result": 42}') == ("plain", '{"result": 42}')

    def test_only_an_object_opening_with_thought_is_suspected(self) -> None:
        """omp checks the first key: `{"thing": .., "command": ..}` is an answer."""
        text = '{"thing": 1, "command": "ls"}'
        assert consume_planning_buffer(text) == ("plain", text)

    def test_unbalanced_quotes_still_close_the_object(self) -> None:
        """A leak with unbalanced quotes would never close through the normal path."""
        assert consume_planning_buffer('{"thought": "quote " too many"}') == ("leak", "")

    def test_unterminated_leak_is_dropped_at_the_end(self) -> None:
        """Delivering it would mean showing half of the internal planning."""
        assert consume_planning_buffer('{"thought": "never closes', final=True) == ("leak", "")

    def test_unterminated_brace_without_a_signature_is_text_at_the_end(self) -> None:
        """omp: an unclosed prefix with no leak key in it is released as written."""
        assert consume_planning_buffer("{", final=True) == ("plain", "{")
        assert consume_planning_buffer('{"tho', final=True) == ("plain", '{"tho')

    def test_a_call_to_a_declared_tool_is_a_leak(self) -> None:
        text = '{"thought": 1, "call": "read"}'
        assert consume_planning_buffer(text, frozenset({"read"})) == ("leak", "")
        assert consume_planning_buffer(text) == ("plain", text)
