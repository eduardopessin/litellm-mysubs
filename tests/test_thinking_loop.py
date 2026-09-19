"""Detection of runaway reasoning.

The thresholds come from OMP, which calibrated them against 536 thousand real reasoning
blocks. A false negative costs a turn's quota; a false positive kills a legitimate answer.
For every shape of runaway there is a neighbouring case that must **not** fire.
"""

from __future__ import annotations

import random
from typing import ClassVar

from litellm_mysubs.wire.thinking_loop import (
    EXACT_SHORT_MIN_REPEATED_CHARS,
    LEX_STALL_MIN_RUN,
    SEGMENT_CHAR_CAP,
    SEGMENT_MIN_CLUSTER,
    SEGMENT_MIN_COUNT,
    ThinkingLoopDetector,
    detect_exact_suffix_cycle,
    guard_for,
    jaccard,
    normalize_segment,
    trigram_shingles,
)


def paragraph(text: str) -> str:
    """A segment closes on a paragraph."""
    return f"{text}\n\n"


#: Sentence long enough to clear the floor of 60 normalized chars.
FILLER = (
    "so it is worth reviewing what had been planned for this point of the analysis "
    "under way and weighing the available alternatives before moving on"
)


class TestExactCycle:
    def test_short_cycle_needs_four_repeats(self) -> None:
        """Short cycles (<=60 chars) require 4 repeats and 180 characters."""
        unit = "let me check the file one more time. "
        assert detect_exact_suffix_cycle(unit * 3) is None
        found = detect_exact_suffix_cycle(unit * 8)
        assert found is not None and found[1] >= 4

    def test_too_short_to_judge(self) -> None:
        assert detect_exact_suffix_cycle("a" * (EXACT_SHORT_MIN_REPEATED_CHARS - 1)) is None

    def test_punctuation_only_cycle_ignored(self) -> None:
        """A cycle without letters is formatting, not runaway reasoning."""
        assert detect_exact_suffix_cycle("-=" * 200) is None

    def test_varied_text_is_fine(self) -> None:
        varied = "".join(
            f"step {i}: analyse the module and record what was observed. " for i in range(40)
        )
        assert detect_exact_suffix_cycle(varied) is None

    def test_detector_reports_the_cycle(self) -> None:
        detector = ThinkingLoopDetector()
        reason = detector.feed("let me check the file one more time. " * 10)
        assert reason is not None and "exact cycle" in reason

    def test_exact_detection_survives_disabled_semantics(self) -> None:
        """Exact detection always applies, even without the semantic heuristics."""
        detector = ThinkingLoopDetector(semantic_heuristics=False)
        assert detector.feed("repeat this over and over again please. " * 12) is not None


class TestNearDuplicateCluster:
    """Requires SEGMENT_MIN_CLUSTER similar segments, not two."""

    def test_cluster_fires_after_warmup(self) -> None:
        detector = ThinkingLoopDetector()
        base = (
            "analyse the server configuration file and check every duplicate entry "
            "that shows up in the current system listing"
        )
        reason = None
        for index in range(SEGMENT_MIN_COUNT + SEGMENT_MIN_CLUSTER + 2):
            reason = detector.feed(paragraph(f"{base} variante {index}"))
            if reason:
                break
        assert reason is not None and "near-identical" in reason

    def test_two_duplicates_are_not_enough(self) -> None:
        """OMP requires a cluster of 4; firing at 2 killed legitimate reasoning."""
        detector = ThinkingLoopDetector()
        base = (
            "analyse the server configuration file and check the duplicate entries "
            "in the current listing"
        )
        assert detector.feed(paragraph(base)) is None
        assert detector.feed(paragraph(base + " again")) is None

    def test_warmup_gate_blocks_early_detection(self) -> None:
        """Below SEGMENT_MIN_COUNT segments nothing fires, however repeated."""
        detector = ThinkingLoopDetector()
        base = (
            "exactly the same paragraph repeated over and over again to exercise the "
            "warmup of the semantic cycle detector"
        )
        for _ in range(SEGMENT_MIN_COUNT - 1):
            assert detector.feed(paragraph(base)) is None

    def test_distinct_segments_pass(self) -> None:
        """Paragraphs with genuinely different content never fire.

        Varying only a number does not make them distinct: the word trigrams stay
        identical and Jaccard gives 1.0 — which is exactly what the heuristic must catch.
        """
        topics = [
            "open the configuration file and confirm that the entries are sorted",
            "compare the totals returned by the database against the values in memory",
            "review the change history to work out when the divergence appeared",
            "write a test case that reproduces the failure observed in production",
            "measure the time spent in each stage of the data import pipeline",
            "document the decision that was taken and the alternatives discarded",
            "check whether the certificate used by the service is still valid",
            "delete the old records that nobody has queried for several months",
            "confirm that the alert fires once the configured limit is exceeded",
            "upgrade the dependency and run the whole integration test battery",
            "isolate the slow query and judge whether an index solves the problem",
            "prepare the rollback plan for a migration that fails halfway",
        ]
        detector = ThinkingLoopDetector()
        for topic in topics:
            assert detector.feed(paragraph(topic)) is None, topic


class TestLexicalStall:
    """Recycled vocabulary with no new anchors."""

    POOL: ClassVar[tuple[str, ...]] = (
        "therefore",
        "worth",
        "reviewing",
        "what",
        "planned",
        "moment",
        "analysis",
        "under",
        "weighing",
        "alternatives",
        "available",
        "before",
        "advancing",
        "situation",
        "present",
        "decision",
        "taken",
        "result",
        "expected",
        "process",
        "following",
        "stage",
        "consideration",
        "relevant",
    )

    def segment(self, seed: int, anchor: str | None = None) -> str:
        """Deterministic recombination of the same vocabulary."""
        words = random.Random(seed).sample(list(self.POOL), 16)
        text = " ".join(words)
        if anchor:
            text += f" {anchor}"
        return paragraph(text)

    def test_recycled_vocabulary_stalls(self) -> None:
        detector = ThinkingLoopDetector()
        reason = None
        for index in range(SEGMENT_MIN_COUNT + LEX_STALL_MIN_RUN + 4):
            reason = detector.feed(self.segment(500 + (index % 6)))
            if reason:
                break
        assert reason is not None and "low-information" in reason

    def test_fresh_anchor_resets_the_run(self) -> None:
        """A new file every paragraph is genuine work, not padding."""
        detector = ThinkingLoopDetector()
        for index in range(SEGMENT_MIN_COUNT + LEX_STALL_MIN_RUN + 4):
            reason = detector.feed(self.segment(500 + (index % 6), anchor=f"module_{index}.py"))
            assert reason is None, f"fired at {index}: {reason}"

    def test_same_anchor_every_paragraph_still_stalls(self) -> None:
        """Repeating a fixed reference is not progress: the anchor has to be new."""
        detector = ThinkingLoopDetector()
        reason = None
        for index in range(SEGMENT_MIN_COUNT + LEX_STALL_MIN_RUN + 4):
            reason = detector.feed(self.segment(500 + (index % 6), anchor="general_config.py"))
            if reason:
                break
        assert reason is not None


class TestSegmentation:
    def test_headings_are_stripped_before_analysis(self) -> None:
        """The always-different wording of headings would inflate novelty."""
        assert normalize_segment("## A Section\nreal text") == "a section real text"

    def test_runaway_paragraph_is_force_flushed(self) -> None:
        """A wall of text without blank lines still has to be segmented.

        The text has to be varied: repeating the same word would be caught first by the
        exact-cycle detector, and the flush would never get to run.
        """
        detector = ThinkingLoopDetector()
        wall = " ".join(f"term{index}" for index in range(SEGMENT_CHAR_CAP // 4))
        detector.feed(wall)
        assert detector._count > 0

    def test_partial_delta_does_not_close_a_segment(self) -> None:
        detector = ThinkingLoopDetector()
        detector.feed("sentence with no end")
        assert detector._count == 0

    def test_flush_processes_the_final_paragraph(self) -> None:
        """The last segment may be the one that completes the cluster."""
        detector = ThinkingLoopDetector()
        detector.feed(paragraph(FILLER))
        detector.feed(FILLER)
        before = detector._count
        detector.flush()
        assert detector._count == before + 1

    def test_short_segment_ignored(self) -> None:
        detector = ThinkingLoopDetector()
        detector.feed(paragraph("ok."))
        assert detector._count == 0


class TestTextHelpers:
    def test_trigrams_are_word_based(self) -> None:
        """Character trigrams gave high similarity to unrelated texts."""
        assert trigram_shingles("one two three four") == {"one two three", "two three four"}

    def test_short_text_is_one_shingle(self) -> None:
        assert trigram_shingles("one two") == {"one two"}

    def test_empty_text_has_no_shingles(self) -> None:
        assert trigram_shingles("") == set()

    def test_backticks_become_words(self) -> None:
        """Backtick content becomes text; punctuation (including `_`) is a separator."""
        assert normalize_segment("use `foo_bar` here") == "use foo bar here"

    def test_digits_alone_are_dropped(self) -> None:
        assert normalize_segment("step 2 of 3") == "step of"

    def test_jaccard_bounds(self) -> None:
        assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
        assert jaccard({"a"}, {"b"}) == 0.0
        assert jaccard(set(), {"a"}) == 0.0


class TestGuardSelection:
    def test_guards_the_families_that_run_away(self) -> None:
        for model in ("gemini-3-pro", "deepseek-v3", "grok-4"):
            assert isinstance(guard_for(model), ThinkingLoopDetector), model

    def test_other_families_unguarded(self) -> None:
        for model in ("claude-opus-5", "gpt-5.5", "qwen-agent-coder"):
            assert guard_for(model) is None, model
