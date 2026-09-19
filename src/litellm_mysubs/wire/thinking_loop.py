"""Runaway reasoning detection.

Direct port of ``utils/thinking-loop.ts``. The thresholds are not our choices: the source
comment says they were calibrated against **536 thousand real reasoning blocks**, and that
the longest legitimate run of low-information segments observed was 7 — hence the trigger
at 8. Changing them without an equivalent corpus is guesswork.

Three forms of runaway, with distinct purposes:

1. **Exact suffix cycle** — literal repetition. Two regimes: short cycles (≤60 chars)
   require 4 repetitions and ≥180 chars; long cycles require 3 and ≥1024. Always applied.
2. **Near-duplicate cluster** — the same paragraph rewritten with cosmetic drift, by
   word-trigram overlap.
3. **Lexical stall** — paragraphs that recycle recent vocabulary and introduce no new
   concrete reference.

The last two are semantic heuristics and can be switched off; the first cannot.
"""

from __future__ import annotations

import re
from typing import Final

#: Tail retained for exact cycle detection.
EXACT_TAIL_WINDOW: Final = 4096
#: Largest cycle considered.
EXACT_MAX_UNIT: Final = 1024
#: New characters between scans. Avoids quadratic work on every delta.
EXACT_CHECK_STRIDE: Final = 128
#: Boundary between the short and the long regime.
EXACT_SHORT_MAX_UNIT: Final = 60
EXACT_SHORT_MIN_REPEATED_CHARS: Final = 180
EXACT_LONG_MIN_REPEATED_CHARS: Final = 1024

#: Cap on a segment with no terminator; forces a flush so that a wall of text with no blank
#: lines still gets segmented.
SEGMENT_CHAR_CAP: Final = 700
#: Below this normalized length the segment is ignored — too short to be a paragraph with
#: meaning, and a lone heading must not be able to trigger detection.
SEGMENT_MIN_NORM_CHARS: Final = 60
SEGMENT_WINDOW: Final = 16
SEGMENT_SIMILARITY: Final = 0.8
#: Warm-up: substantial segments required before detection may fire.
SEGMENT_MIN_COUNT: Final = 8
#: Size of the near-duplicate cluster that fires.
SEGMENT_MIN_CLUSTER: Final = 4

#: Window whose vocabulary is the novelty baseline.
LEX_NOVELTY_WINDOW: Final = 8
LEX_STALL_NOVELTY_FLOOR: Final = 0.2
LEX_STALL_MIN_RUN: Final = 8

# A concrete reference the model is actually reasoning about: a code fragment, a dotted
# extension or member, a multi-segment path, or a snake/camel/Pascal identifier. Excludes
# bare digits, abbreviations and decimals ("Step 2", "i.e.", "1.2") so that numbered filler
# is not a self-anchor.
CONCRETE_ANCHOR: Final = re.compile(
    r"`[^`]+`"
    r"|\b\w{2,}\.[a-zA-Z]\w{0,4}\b"
    r"|[\w-]+(?:/[\w-]+){2,}"
    r"|\b\w+_\w+\b"
    r"|\b[a-z]+[A-Z]\w*\b"
    r"|\b[A-Z][a-z]+[A-Z]\w*\b"
)

# Summary headings ("**Keeping the Pace**", "## Section") are per-thought formatting, not
# reasoning. Their ever-changing wording would inflate novelty and mask a loop, so they are
# removed before analysis.
_HEADING = re.compile(r"^[ \t]*#{1,6}[ \t].*$", re.M)
_BOLD_TITLE = re.compile(r"^[ \t]*\*{2,3}.+?\*{2,3}[ \t]*$", re.M)

_PARAGRAPH_BOUNDARY = re.compile(r"\n\s*\n")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_BACKTICKED = re.compile(r"`([^`]*)`")
_HAS_LETTER = re.compile(r"[a-z]")
_LETTER_OR_EMOJI = re.compile(r"[^\W\d_]", re.UNICODE)


class ThinkingLoopError(Exception):
    """Runaway reasoning.

    Distinct from ``Exception`` so it passes through the handlers that tolerate malformed
    chunks: a detected loop is not a malformed chunk.
    """


def normalize_segment(segment: str) -> str:
    """Lowercase, no punctuation, only words that contain letters."""
    lowered = _BACKTICKED.sub(r" \1 ", segment.lower())
    tokens = _NON_ALNUM.sub(" ", lowered).split()
    return " ".join(token for token in tokens if _HAS_LETTER.search(token))


def trigram_shingles(normalized: str) -> set[str]:
    """Trigrams of **words** — not of characters.

    Character trigrams gave high similarity to completely unrelated texts, which makes
    detection fire on legitimate reasoning.
    """
    words = [word for word in normalized.split(" ") if word]
    if len(words) < 3:
        return {" ".join(words)} if words else set()
    return {" ".join(words[index : index + 3]) for index in range(len(words) - 2)}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    small, large = (left, right) if len(left) < len(right) else (right, left)
    intersection = sum(1 for item in small if item in large)
    union = len(left) + len(right) - intersection
    return intersection / union if union else 0.0


def detect_exact_suffix_cycle(text: str) -> tuple[str, int] | None:
    """Literal cycle repeated at the end of the text, by Z algorithm over the reverse.

    Two regimes: a short cycle must appear 4 times and cover 180 chars; a long one, 3 times
    and 1024. A cycle with no letters at all (only punctuation or spaces) does not count.
    """
    if len(text) < EXACT_SHORT_MIN_REPEATED_CHARS:
        return None

    reversed_text = text[::-1]
    length = len(reversed_text)
    z = [0] * length
    left = right = 0
    for index in range(1, length):
        if index <= right:
            z[index] = min(right - index + 1, z[index - left])
        while (
            index + z[index] < length and reversed_text[z[index]] == reversed_text[index + z[index]]
        ):
            z[index] += 1
        if index + z[index] - 1 > right:
            left, right = index, index + z[index] - 1

    max_unit = min(EXACT_MAX_UNIT, length // 3)
    for unit_length in range(2, max_unit + 1):
        count = 1 + z[unit_length] // unit_length
        short = unit_length <= EXACT_SHORT_MAX_UNIT
        min_count = 4 if short else 3
        min_chars = EXACT_SHORT_MIN_REPEATED_CHARS if short else EXACT_LONG_MIN_REPEATED_CHARS
        if count < min_count or unit_length * count < min_chars:
            continue
        unit = text[-unit_length:]
        if _LETTER_OR_EMOJI.search(unit):
            return unit, count
    return None


# omp: utils/thinking-loop.ts :: ThinkingLoopDetector
class ThinkingLoopDetector:
    """Fed with the reasoning deltas; returns the reason on the first runaway.

    Deliberate deviation from omp: there the trigger is a *retryable* error and the retry
    layer asks again. Here it raises. The generator has already flushed everything that is
    ``reasoning_content`` to the client before detection happens, and retrying would
    duplicate reasoning in the same stream — omp avoids that with a replay-safe window that
    does not exist here.
    """

    __slots__ = (
        "_anchor_window",
        "_count",
        "_exact_scanned_at",
        "_lex_stall_run",
        "_pending",
        "_semantic",
        "_tail",
        "_window",
        "_word_window",
        "chars",
    )

    def __init__(self, *, semantic_heuristics: bool = True) -> None:
        self._semantic = semantic_heuristics
        self._tail = ""
        self._exact_scanned_at = 0
        self._pending = ""
        self._window: list[set[str]] = []
        self._word_window: list[set[str]] = []
        self._anchor_window: list[set[str]] = []
        self._count = 0
        self._lex_stall_run = 0
        self.chars = 0

    def feed(self, delta: str) -> str | None:
        """Reason for the loop, or ``None``. Never raises."""
        if not delta:
            return None
        self.chars += len(delta)

        # 1. Exact cycles, scanned at a limited stride instead of quadratic work on every
        # token-sized delta.
        self._tail = (self._tail + delta)[-EXACT_TAIL_WINDOW:]
        self._exact_scanned_at += len(delta)
        if self._exact_scanned_at >= EXACT_CHECK_STRIDE or len(delta) >= EXACT_CHECK_STRIDE:
            self._exact_scanned_at = 0
            if reason := self._exact_reason():
                return reason

        if not self._semantic:
            return None

        # 2. Segments: accumulate and drain the ones that closed.
        self._pending += delta
        while True:
            boundary = _PARAGRAPH_BOUNDARY.search(self._pending)
            if boundary:
                raw = self._pending[: boundary.start()]
                self._pending = self._pending[boundary.end() :]
            elif len(self._pending) > SEGMENT_CHAR_CAP:
                # No boundary but too long: force a flush so that a wall of text with no
                # paragraphs still gets analysed.
                raw = self._pending[:SEGMENT_CHAR_CAP]
                self._pending = self._pending[SEGMENT_CHAR_CAP:]
            else:
                return None
            if reason := self._consume_chunks(raw):
                return reason

    def flush(self) -> str | None:
        """Process the final paragraph, which may be the one that completes a cluster.

        A stream can end before the next stride, so a final exact check is forced too.
        """
        if reason := self._exact_reason():
            return reason
        if not self._semantic or not self._pending:
            return None
        pending, self._pending = self._pending, ""
        return self._consume_chunks(pending)

    # -- internals -------------------------------------------------------------

    def _exact_reason(self) -> str | None:
        found = detect_exact_suffix_cycle(self._tail)
        if found is None:
            return None
        unit, times = found
        return f"exact cycle of {len(unit)} characters repeated {times}x in a row"

    def _consume_chunks(self, raw: str) -> str | None:
        """Split an over-long segment so that each chunk stays comparable."""
        rest = raw
        while rest:
            chunk, rest = rest[:SEGMENT_CHAR_CAP], rest[SEGMENT_CHAR_CAP:]
            if reason := self._consume_segment(chunk):
                return reason
        return None

    def _consume_segment(self, raw: str) -> str | None:
        segment = _BOLD_TITLE.sub("", _HEADING.sub("", raw))
        normalized = normalize_segment(segment)
        if len(normalized) < SEGMENT_MIN_NORM_CHARS:
            return None

        # (a) Near-duplicate cluster.
        fingerprint = trigram_shingles(normalized)
        cluster = 1 + sum(
            1 for previous in self._window if jaccard(fingerprint, previous) >= SEGMENT_SIMILARITY
        )

        # (b) Lexical stall: paragraphs that recycle recent vocabulary and add no *new*
        # concrete reference. Requiring a new anchor — and not merely any anchor — catches
        # the filler that repeats the same path in every paragraph, and spares genuine work
        # that names a different file each time.
        words = {word for word in normalized.split(" ") if word}
        prior_vocabulary: set[str] = set()
        for seen in self._word_window:
            prior_vocabulary |= seen
        unseen = sum(1 for word in words if word not in prior_vocabulary)
        novelty = 1.0 if not prior_vocabulary else unseen / len(words)

        # Canonicalize so that the same reference written as `Foo`, Foo or FOO is a single
        # anchor and cannot pass itself off as new.
        anchors = {
            match.group(0).replace("`", "").lower() for match in CONCRETE_ANCHOR.finditer(segment)
        }
        new_anchor = any(
            all(anchor not in seen for seen in self._anchor_window) for anchor in anchors
        )

        if novelty <= LEX_STALL_NOVELTY_FLOOR and not new_anchor:
            self._lex_stall_run += 1
        else:
            self._lex_stall_run = 0

        self._window.append(fingerprint)
        del self._window[:-SEGMENT_WINDOW]
        self._word_window.append(words)
        del self._word_window[:-LEX_NOVELTY_WINDOW]
        self._anchor_window.append(anchors)
        del self._anchor_window[:-LEX_NOVELTY_WINDOW]
        self._count += 1

        if self._count < SEGMENT_MIN_COUNT:
            return None
        if cluster >= SEGMENT_MIN_CLUSTER:
            return f"{cluster} near-identical segments in the last {SEGMENT_WINDOW}"
        if self._lex_stall_run >= LEX_STALL_MIN_RUN:
            return f"{self._lex_stall_run} low-information segments recycling recent text"
        return None


# omp: utils/thinking-loop.ts :: isLoopGuardedModel
def guard_for(model: str) -> ThinkingLoopDetector | None:
    """Guards the families that actually run away.

    omp guards Gemini, DeepSeek and xAI. Only Gemini is served here, but the list stays
    aligned with the source so that a new provider does not slip through unnoticed.
    """
    lowered = str(model).lower()
    guarded = ("gemini", "deepseek", "grok", "xai")
    return ThinkingLoopDetector() if any(name in lowered for name in guarded) else None
