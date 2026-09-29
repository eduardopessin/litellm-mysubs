"""Healing of reasoning markup leaked into Gemini's visible text.

A leak left unhealed shows the user the model's scratch work; an over-eager heal hides
part of the answer. Each scenario is also replayed under every chunking (whole, one
character per delta, every two-way split): a delimiter split across deltas must heal
exactly as if it had arrived whole.
"""

from __future__ import annotations

import pytest

from litellm_mysubs.wire.thinking_markup import HealingEvent, StreamMarkupHealing


def heal(chunks: list[str]) -> list[HealingEvent]:
    """Feed ``chunks`` then flush; adjacent same-channel deltas are merged."""
    healer = StreamMarkupHealing()
    events = [e for chunk in chunks for e in healer.feed_events(chunk)]
    events += healer.flush_events()
    merged: list[HealingEvent] = []
    for kind, value in events:
        assert value, "events never carry empty payloads"
        if merged and merged[-1][0] == kind:
            merged[-1] = (kind, merged[-1][1] + value)
        else:
            merged.append((kind, value))
    return merged


def chunkings(text: str) -> list[list[str]]:
    return [[text], list(text)] + [[text[:i], text[i:]] for i in range(1, len(text))]


SCENARIOS: dict[str, tuple[str, list[HealingEvent]]] = {
    "thinking tag between text": (
        "before <thinking>mid</thinking> after",
        [("text", "before "), ("thinking", "mid"), ("text", " after")],
    ),
    "think tag": (
        "<think>plan</think>Answer.",
        [("thinking", "plan"), ("text", "Answer.")],
    ),
    "scratchpad": (
        "A<scratchpad>s</scratchpad>B",
        [("text", "A"), ("thinking", "s"), ("text", "B")],
    ),
    "gemma channel": (
        "<|channel>thought\nhmm<channel|>ok",
        [("thinking", "hmm"), ("text", "ok")],
    ),
    "harmony analysis": (
        "<|channel|>analysis<|message|>reason<|end|>answer",
        [("thinking", "reason"), ("text", "answer")],
    ),
    "fence with nested language block": (
        "Intro\n```thinking\nLet me plan.\n```python\nprint('<think>')\n```\n"
        "Done planning.\n```\nFinal answer.",
        [
            ("text", "Intro\n"),
            ("thinking", "Let me plan.\n```python\nprint('<think>')\n```\nDone planning.\n"),
            ("text", "\nFinal answer."),
        ],
    ),
    "fence closed inline by prose": (
        "```thinking\nplan\n```Visible reply",
        [("thinking", "plan\n"), ("text", "Visible reply")],
    ),
    "fence closed by a bare fence at end of stream": (
        "```thinking\nplan\n```",
        [("thinking", "plan\n")],
    ),
    "literal tag in inline code": (
        "Use `<think>` tags, then <think>hidden</think> done",
        [("text", "Use `<think>` tags, then "), ("thinking", "hidden"), ("text", " done")],
    ),
    "literal tag in double-backtick span": (
        "a ``x ` <think> y`` b",
        [("text", "a ``x ` <think> y`` b")],
    ),
    "literal tag in fenced code block": (
        "```python\nx = '<think>'\n```\nafter <think>hidden</think> end",
        [
            ("text", "```python\nx = '<think>'\n```\nafter "),
            ("thinking", "hidden"),
            ("text", " end"),
        ],
    ),
    "bare close tag stays visible": (
        "no open here</think> tail",
        [("text", "no open here</think> tail")],
    ),
    "comparison operator is not a tag": (
        "if a < b and c <thinker> d",
        [("text", "if a < b and c <thinker> d")],
    ),
}


@pytest.mark.parametrize("text,expected", SCENARIOS.values(), ids=SCENARIOS.keys())
def test_heals_identically_under_every_chunking(text: str, expected: list[HealingEvent]) -> None:
    for chunks in chunkings(text):
        assert heal(chunks) == expected, chunks


def test_partial_open_tag_is_held_until_resolved() -> None:
    healer = StreamMarkupHealing()
    assert healer.feed_events("Hello <thi") == [("text", "Hello ")]
    assert healer.feed_events("nking>secret</thin") == [("thinking", "secret")]
    assert healer.feed_events("king> world") == [("text", " world")]
    assert healer.flush_events() == []


def test_held_prefix_that_does_not_become_a_tag_is_released_as_text() -> None:
    healer = StreamMarkupHealing()
    assert healer.feed_events("a <") == [("text", "a ")]
    assert healer.feed_events("b") == [("text", "<b")]


def test_plain_text_is_not_held() -> None:
    healer = StreamMarkupHealing()
    assert healer.feed_events("hello world. ") == [("text", "hello world. ")]
    assert healer.feed_events("more > text") == [("text", "more > text")]
    assert healer.flush_events() == []


def test_empty_feed_is_a_no_op() -> None:
    healer = StreamMarkupHealing()
    assert healer.feed_events("") == []
    assert healer.flush_events() == []


def test_unterminated_thinking_is_flushed_as_thinking() -> None:
    healer = StreamMarkupHealing()
    assert healer.feed_events("Answer: <think>still going</thi") == [
        ("text", "Answer: "),
        ("thinking", "still going"),
    ]
    # The held partial close is reasoning after all: nothing may leak into the text.
    assert healer.flush_events() == [("thinking", "</thi")]


def test_unterminated_fenced_thinking_is_flushed_as_thinking() -> None:
    assert heal(["```thinking\nstep one\n", "``"]) == [("thinking", "step one\n``")]


def test_held_partial_open_is_flushed_as_text() -> None:
    healer = StreamMarkupHealing()
    assert healer.feed_events("x <thi") == [("text", "x ")]
    assert healer.flush_events() == [("text", "<thi")]


def test_several_sections_in_one_stream() -> None:
    assert heal(["<think>a</think>1", "<thinking>b</thinking>2```thinking\nc\n```3"]) == [
        ("thinking", "a"),
        ("text", "1"),
        ("thinking", "b"),
        ("text", "2"),
        ("thinking", "c\n"),
        ("text", "3"),
    ]
