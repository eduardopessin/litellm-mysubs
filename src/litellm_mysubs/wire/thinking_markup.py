"""Healing of reasoning markup leaked into a visible text stream.

Port of OMP's ``StreamMarkupHealing`` for the ``"thinking"`` pattern, the one
``providers/google-gemini-cli.ts`` uses: Gemini sometimes streams its reasoning in the
visible channel wrapped in a template idiom (``<think>``, ``<thinking>``,
``<scratchpad>``, a `` ```thinking `` fence, Gemma/Harmony channels) instead of as
``thought`` parts. The healer moves those sections into thinking events so the client
sees clean text.

Three properties matter and are what the scanners below are careful about:

- **Streaming-safe.** A delimiter split across deltas (``<thi`` + ``nk>``) is held at the
  buffer tail until it resolves, so no half tag leaks into either channel. Text with no
  delimiter prefix at its tail is never held.
- **Code-aware.** A Markdown code span or fence keeps a literal ``<think>`` visible: a
  model explaining the idiom is not reasoning.
- **Fence nesting.** Inside a `` ```thinking `` block, a language-tagged inner fence
  (`` ```python ``) is reasoning content, not the closer.

Deviation from OMP: ``ThinkingInbandScanner``'s ``impliedOpen`` option (a bare
``</think>`` reported as ``impliedThinkingEnd``) is not ported. ``StreamMarkupHealing``
never enables it, so it would be dead code here; the ``impliedOpen`` tag attribute and
the ``IMPLIED_OPEN_*`` tables go with it. ``MAX_DELIMITER_LENGTH`` is unchanged by
this, since the longest delimiter is an opener.

Indices are code points, not UTF-16 units as in the TypeScript. Every delimiter and
lead character is ASCII, so behaviour is identical; only offsets inside astral text
differ, and they never leave this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Literal, NamedTuple

# omp: dialect/types.ts :: InbandScanEvent
#: Scanner-level event. ``thinkingEnd`` carries the whole section; ``thinkingStart``
#: carries nothing (empty string).
ScanEvent = tuple[Literal["text", "thinkingStart", "thinkingDelta", "thinkingEnd"], str]

# omp: utils/stream-markup-healing.ts :: StreamMarkupHealingEvent
#: What callers see: cleaned visible text, or a non-empty thinking delta.
HealingEvent = tuple[Literal["text", "thinking"], str]

#: ``String.prototype.trim`` whitespace (WhiteSpace + LineTerminator). Python's default
#: ``str.strip`` differs (it strips ``\x1c``-``\x1f`` and ``\x85``, keeps ``\ufeff``),
#: and fence classification depends on what counts as blank.
_JS_WHITESPACE: Final = (
    "\t\n\v\f\r \u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006"
    "\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
)


def _trim(text: str) -> str:
    return text.strip(_JS_WHITESPACE)


# --- dialect/coercion.ts ---------------------------------------------------------------


# omp: dialect/coercion.ts :: partialSuffixOverlap
def partial_suffix_overlap(text: str, tag: str) -> int:
    """Length of the longest proper prefix of ``tag`` that ``text`` ends with."""
    for k in range(min(len(text), len(tag) - 1), 0, -1):
        if text.endswith(tag[:k]):
            return k
    return 0


# omp: dialect/coercion.ts :: partialSuffixOverlapAny
def partial_suffix_overlap_any(text: str, tags: list[str]) -> int:
    """Longest held-back tail across ``tags``."""
    return max((partial_suffix_overlap(text, tag) for tag in tags), default=0)


# --- dialect/fenced-thinking.ts --------------------------------------------------------

# JS regexes are ported with their JS meaning: ``.`` excludes every line terminator
# (``\r`` included, so a CRLF fence line is not a fence) and ``$`` without the ``m``
# flag is end of input only (``\Z``; Python's ``$`` also matches before a final ``\n``).

# omp: dialect/fenced-thinking.ts :: FENCE_LINE
#: A complete fence line: <=3 lead spaces, a run of >=3 backticks/tildes, then an info string.
FENCE_LINE: Final = re.compile("^ {0,3}(`{3,}|~{3,})([^\n\r\u2028\u2029]*)\\Z")
# omp: dialect/fenced-thinking.ts :: BACKTICK_LEAD
#: <=3 lead spaces then a (possibly partial) backtick run and whatever follows it.
BACKTICK_LEAD: Final = re.compile(r"^ {0,3}(`*)([\s\S]*)\Z")
# omp: dialect/fenced-thinking.ts :: LANG_TOKEN
#: A language-tag info string: one token, no whitespace. Used with ``fullmatch``.
LANG_TOKEN: Final = re.compile(r"[A-Za-z0-9_+#-]+")
_LEAD_SPACES_ONLY: Final = re.compile(" {0,3}")


# omp: dialect/fenced-thinking.ts :: FencedThinkingResult
class FencedThinkingResult(NamedTuple):
    """Result of one :meth:`FencedThinkingScanner.feed`."""

    thinking: str
    """Thinking text to emit for this feed (may be empty)."""
    closed: bool
    """True once the thinking closer has been consumed."""
    rest: str
    """Bytes after the closing fence (visible reply); only meaningful when ``closed``."""


# omp: dialect/fenced-thinking.ts :: FencedThinkingScanner
class FencedThinkingScanner:
    """Close-matcher for one `` ```thinking `` block that respects nested code fences.

    A naive search for the first `` ``` `` would end the reasoning at an inner
    `` ```python `` block and leak the rest into the visible channel. An inner opener is a
    fence followed by a single language token and a newline; the closer is a bare fence
    or a fence glued to prose (`` ```Visible reply ``). An info-less inner fence is
    indistinguishable from the closer and ends the block, as in OMP.

    Line-oriented: ordinary content streams as it arrives but stays buffered until its
    newline so the full line can be classified (``_emitted`` counts the bytes already
    returned). A top-level fence candidate is held until its info disambiguates.
    """

    def __init__(self) -> None:
        self._buffer = ""
        #: The fence run that opened the current nested code block, or "" at top level.
        self._inner = ""
        #: Bytes of the leading (incomplete) line already returned as thinking.
        self._emitted = 0

    def feed(self, text: str, final: bool) -> FencedThinkingResult:
        """Feed bytes; on ``final`` a held top-level fence resolves as the closer."""
        self._buffer += text
        thinking = ""
        while True:
            nl = self._buffer.find("\n")
            if nl == -1:
                break
            line = self._buffer[:nl]
            if not self._inner:
                close = self._close_rest(line)
                if close is not None:
                    # Closer bytes are always held, so _emitted is 0 and nothing leaked.
                    rest = close + self._buffer[nl:]  # keep the newline with the reply
                    self._reset()
                    return FencedThinkingResult(thinking, True, rest)
            # Content line (including an inner-fence open/close).
            thinking += self._buffer[self._emitted : nl + 1]
            self._update_inner(line)
            self._buffer = self._buffer[nl + 1 :]
            self._emitted = 0

        tail = self._buffer
        if self._inner:
            # Inside a nested block every byte is thinking: emit eagerly, keep buffered
            # until the newline classifies the line.
            thinking += tail[self._emitted :]
            self._emitted = len(tail)
            return FencedThinkingResult(thinking, False, "")

        if final:
            close = self._close_rest_final(tail)
            if close is not None:
                self._reset()
                return FencedThinkingResult(thinking, True, close)
        else:
            close = self._close_rest_streaming_tail(tail)
            if close is not None:
                self._reset()
                return FencedThinkingResult(thinking, True, close)
            if self._must_hold(tail):
                return FencedThinkingResult(thinking, False, "")
        # Either final (flush the remainder) or a line that can no longer be a fence.
        thinking += tail[self._emitted :]
        if final:
            self._reset()
        else:
            self._emitted = len(tail)
        return FencedThinkingResult(thinking, False, "")

    @staticmethod
    def _close_rest(line: str) -> str | None:
        """Complete line: bare fence closes, language token opens inner, prose is the reply."""
        m = BACKTICK_LEAD.match(line)
        if not m or len(m[1]) < 3:
            return None
        rest = m[2]
        if rest == "" or _trim(rest) == "":
            return ""
        if LANG_TOKEN.fullmatch(rest):
            return None
        return rest

    @staticmethod
    def _close_rest_final(tail: str) -> str | None:
        """End of input disambiguates any top-level backtick run as the closer."""
        m = BACKTICK_LEAD.match(tail)
        if not m or len(m[1]) < 3:
            return None
        rest = m[2]
        return "" if _trim(rest) == "" else rest

    @staticmethod
    def _close_rest_streaming_tail(tail: str) -> str | None:
        """Streaming tail: only a prose-like inline reply resolves the close."""
        m = BACKTICK_LEAD.match(tail)
        if not m or len(m[1]) < 3:
            return None
        rest = m[2]
        if rest == "" or _trim(rest) == "" or LANG_TOKEN.fullmatch(rest):
            return None
        return rest

    @staticmethod
    def _must_hold(tail: str) -> bool:
        """Whether a top-level trailing partial is still undecided."""
        m = BACKTICK_LEAD.match(tail)
        if not m:
            return False
        ticks = len(m[1])
        rest = m[2]
        # A growing backtick run could still reach a fence; a complete run plus a
        # language-token prefix waits for a newline (inner opener) or a non-token
        # character (inline close).
        if rest == "" or _trim(rest) == "":
            return ticks >= 1 or _LEAD_SPACES_ONLY.fullmatch(tail) is not None
        return ticks >= 3 and LANG_TOKEN.fullmatch(rest) is not None

    def _reset(self) -> None:
        self._buffer = ""
        self._inner = ""
        self._emitted = 0

    def _update_inner(self, line: str) -> None:
        """Toggle nested-fence state for a completed content line."""
        fence = FENCE_LINE.match(line)
        if not fence:
            return
        run = fence[1]
        info = _trim(fence[2])
        if not self._inner:
            # A top-level closer was handled by _close_rest, so this opens a nested block.
            self._inner = run
        elif run[0] == self._inner[0] and len(run) >= len(self._inner) and info == "":
            self._inner = ""


# --- dialect/thinking.ts ---------------------------------------------------------------


# omp: dialect/thinking.ts :: Tag
@dataclass(frozen=True, slots=True)
class Tag:
    """One dialect's in-band thinking delimiters."""

    open: str
    close: str
    fenced: bool = False


# omp: dialect/thinking.ts :: TAGS
#: Every dialect's canonical in-band thinking section. Plain delimiters only: attributed
#: or namespaced tags are the owned anthropic parser's job, not this fallback's.
TAGS: Final[tuple[Tag, ...]] = (
    Tag("<think>", "</think>"),  # deepseek, glm, hermes, kimi, qwen3
    Tag("<thinking>", "</thinking>"),  # anthropic, minimax, xml
    Tag("<scratchpad>", "</scratchpad>"),  # anthropic
    Tag("```thinking\n", "```", fenced=True),  # gemini fenced thinking
    Tag("<|channel>thought\n", "<channel|>"),  # gemma reasoning channel
    Tag("<|start|>assistant<|channel|>analysis<|message|>", "<|end|>"),  # harmony (rendered)
    Tag("<|channel|>analysis<|message|>", "<|end|>"),  # harmony analysis (bare leak)
)
# omp: dialect/thinking.ts :: OPENS
OPENS: Final[tuple[str, ...]] = tuple(tag.open for tag in TAGS)
# omp: dialect/thinking.ts :: MAX_DELIMITER_LENGTH
#: A hold needs the buffer tail to be a proper prefix of some delimiter, so shorter than this.
MAX_DELIMITER_LENGTH: Final = max(len(delimiter) for delimiter in OPENS)
# omp: dialect/thinking.ts :: BACKTICK
BACKTICK: Final = "`"
# omp: dialect/thinking.ts :: BOUNDARY_LEAD
#: Every character a scan boundary can start on: a delimiter's first character or the
#: backtick. The TS uses a char-code table; a character class lets ``re`` skip the rest.
BOUNDARY_LEAD: Final = re.compile(
    "[" + re.escape("".join(sorted({d[0] for d in OPENS} | {BACKTICK}))) + "]"
)


# omp: dialect/thinking.ts :: ThinkingInbandScanner
class ThinkingInbandScanner:
    """Heals reasoning leaked into visible text back into thinking events.

    Runs without OMP's ``impliedOpen``: a bare close tag stays visible text.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._close_tag = ""
        self._thinking = ""
        #: Fence-aware close-matcher while inside a `` ```thinking `` block.
        self._fenced: FencedThinkingScanner | None = None
        #: Backtick count that opened the code span/fence we are inside; 0 outside code.
        self._code_ticks = 0
        #: True when ``_code_ticks`` opened a fenced block (closes on a fence line).
        self._code_fenced = False
        #: Leading-space count on the current output line, -1 once a non-space appeared.
        #: Starts at 0 so a fence opening the stream (or indented <=3) is recognized.
        self._line_indent = 0

    def feed(self, text: str) -> list[ScanEvent]:
        if not text:
            return []
        self._buffer += text
        return self._consume(False)

    def flush(self) -> list[ScanEvent]:
        events = self._consume(True)
        if not self._buffer:
            return events
        if self._close_tag:
            self._emit_thinking(self._buffer, events)
            events.append(("thinkingEnd", self._thinking))
        else:
            events.append(("text", self._buffer))
        self._buffer = ""
        self._close_tag = ""
        return events

    def _consume(self, final: bool) -> list[ScanEvent]:
        events: list[ScanEvent] = []
        while True:
            if self._fenced is not None:
                # Run even with an empty buffer so a held partial close flushes on final.
                result = self._fenced.feed(self._buffer, final)
                self._buffer = result.rest if result.closed else ""
                self._emit_thinking(result.thinking, events)
                if result.closed or final:
                    events.append(("thinkingEnd", self._thinking))
                    self._thinking = ""
                    self._close_tag = ""
                    self._fenced = None
                if self._fenced is not None:
                    break
                continue
            if not self._buffer:
                break
            if self._close_tag:
                close = self._buffer.find(self._close_tag)
                if close == -1:
                    hold = (
                        0 if final else partial_suffix_overlap_any(self._buffer, [self._close_tag])
                    )
                    cut = len(self._buffer) - hold
                    self._emit_thinking(self._buffer[:cut], events)
                    self._buffer = self._buffer[cut:]
                    break
                self._emit_thinking(self._buffer[:close], events)
                self._buffer = self._buffer[close + len(self._close_tag) :]
                events.append(("thinkingEnd", self._thinking))
                self._thinking = ""
                self._close_tag = ""
                continue
            if self._code_ticks > 0:
                if self._emit_code(final, events):
                    continue
                break

            hit = scan_visible(self._buffer, final)
            if hit.kind == "none":
                self._emit_text(self._buffer, events)
                self._buffer = ""
                break
            if hit.index > 0:
                self._emit_text(self._buffer[: hit.index], events)
            if hit.kind == "hold":
                self._buffer = self._buffer[hit.index :]
                break
            if hit.kind == "code":
                fenced = hit.ticks >= 3 and 0 <= self._line_indent <= 3
                self._emit_text(self._buffer[hit.index : hit.index + hit.ticks], events)
                self._buffer = self._buffer[hit.index + hit.ticks :]
                self._code_ticks = hit.ticks
                self._code_fenced = fenced
                continue
            tag = hit.tag
            assert tag is not None
            self._buffer = self._buffer[hit.index + len(tag.open) :]
            self._close_tag = tag.close
            self._thinking = ""
            if tag.fenced:
                self._fenced = FencedThinkingScanner()
            events.append(("thinkingStart", ""))
        return events

    def _emit_code(self, final: bool, events: list[ScanEvent]) -> bool:
        """Emit content inside a code region with tag detection off.

        A fenced block closes only on a fence line (backticks >= the opener); an inline
        span on the first run of exactly the opener length. True when the region closed.
        """
        if self._code_fenced:
            end = find_fence_close_end(self._buffer, self._code_ticks, final)
            if end != -1:
                self._emit_text(self._buffer[:end], events)
                self._buffer = self._buffer[end:]
                self._code_ticks = 0
                self._code_fenced = False
                return True
            if final:
                self._emit_text(self._buffer, events)
                self._buffer = ""
                self._code_ticks = 0
                self._code_fenced = False
                return False
            # Stream committed lines; hold only the last (possibly partial) fence line.
            last_nl = self._buffer.rfind("\n")
            if last_nl != -1:
                self._emit_text(self._buffer[: last_nl + 1], events)
                self._buffer = self._buffer[last_nl + 1 :]
            return False
        close = find_backtick_run(self._buffer, 0, self._code_ticks)
        if close != -1 and (final or close + self._code_ticks < len(self._buffer)):
            self._emit_text(self._buffer[: close + self._code_ticks], events)
            self._buffer = self._buffer[close + self._code_ticks :]
            self._code_ticks = 0
            return True
        # No committed close yet: hold a trailing backtick run that may still grow into,
        # or past, the closing delimiter.
        hold = 0 if final else trailing_backtick_run(self._buffer)
        cut = len(self._buffer) - hold
        self._emit_text(self._buffer[:cut], events)
        self._buffer = self._buffer[cut:]
        if final:
            self._code_ticks = 0
        return False

    def _emit_text(self, text: str, events: list[ScanEvent]) -> None:
        if not text:
            return
        events.append(("text", text))
        self._line_indent = trailing_line_indent(text, self._line_indent)

    def _emit_thinking(self, delta: str, events: list[ScanEvent]) -> None:
        if not delta:
            return
        self._thinking += delta
        events.append(("thinkingDelta", delta))


# omp: dialect/thinking.ts :: VisibleHit
@dataclass(frozen=True, slots=True)
class VisibleHit:
    """Next boundary in idle visible text. ``tag`` is set only for ``kind == "tag"``."""

    kind: Literal["tag", "code", "hold", "none"]
    index: int = 0
    tag: Tag | None = None
    ticks: int = 0


_NO_HIT: Final = VisibleHit("none")


# omp: dialect/thinking.ts :: scanVisible
def scan_visible(buffer: str, final: bool) -> VisibleHit:
    """Earliest reasoning-tag open, code-span opener, or held partial delimiter.

    Tags win at any position so the `` ```thinking `` fence heals instead of reading as a
    code fence; backtick runs enter code mode so a literal ``<think>`` in code stays text.
    """
    # Only the tail can be a proper prefix of a delimiter.
    hold_from = len(buffer) if final else len(buffer) - MAX_DELIMITER_LENGTH + 1
    for lead in BOUNDARY_LEAD.finditer(buffer):
        i = lead.start()
        for tag in TAGS:
            if buffer.startswith(tag.open, i):
                return VisibleHit("tag", i, tag=tag)
        if i >= hold_from:
            rest = buffer[i:]
            if any(len(d) > len(rest) and d.startswith(rest) for d in OPENS):
                return VisibleHit("hold", i)
        if buffer[i] == BACKTICK:
            ticks = backtick_run(buffer, i)
            if not final and i + ticks == len(buffer):
                return VisibleHit("hold", i)
            return VisibleHit("code", i, ticks=ticks)
    return _NO_HIT


# omp: dialect/thinking.ts :: backtickRun
def backtick_run(buffer: str, start: int) -> int:
    """Length of the maximal backtick run beginning at ``start``."""
    end = start
    while end < len(buffer) and buffer[end] == BACKTICK:
        end += 1
    return end - start


# omp: dialect/thinking.ts :: findBacktickRun
def find_backtick_run(buffer: str, start: int, ticks: int) -> int:
    """Index of the first maximal backtick run of exactly ``ticks`` at/after ``start``."""
    i = buffer.find(BACKTICK, start)
    while i != -1:
        run = backtick_run(buffer, i)
        if run == ticks:
            return i
        i = buffer.find(BACKTICK, i + run)
    return -1


# omp: dialect/thinking.ts :: trailingBacktickRun
def trailing_backtick_run(buffer: str) -> int:
    """Length of a backtick run ending at the buffer tail."""
    return len(buffer) - len(buffer.rstrip(BACKTICK))


# omp: dialect/thinking.ts :: trailingLineIndent
def trailing_line_indent(text: str, prior: int) -> int:
    """Leading-space count of the tail line, continuing ``prior``; -1 after a non-space."""
    last_nl = text.rfind("\n")
    indent = prior if last_nl == -1 else 0
    for ch in text[last_nl + 1 :]:
        if indent == -1:
            break
        indent = indent + 1 if ch == " " else -1
    return indent


# omp: dialect/thinking.ts :: findFenceCloseEnd
def find_fence_close_end(buffer: str, ticks: int, final: bool) -> int:
    """Index past the first closing fence line (>= ``ticks`` backticks only), else -1.

    An unterminated line commits only when ``final``: more input could extend it.
    """
    start = 0
    while start <= len(buffer):
        nl = buffer.find("\n", start)
        terminated = nl != -1
        line = _trim(buffer[start : nl if terminated else len(buffer)])
        if len(line) >= ticks and is_all_backticks(line) and (terminated or final):
            return nl + 1 if terminated else len(buffer)
        if not terminated:
            break
        start = nl + 1
    return -1


# omp: dialect/thinking.ts :: isAllBackticks
def is_all_backticks(text: str) -> bool:
    """True when ``text`` is non-empty and only backticks."""
    return bool(text) and not text.strip(BACKTICK)


# --- utils/stream-markup-healing.ts ----------------------------------------------------


# omp: utils/stream-markup-healing.ts :: StreamMarkupHealing
class StreamMarkupHealing:
    """Streamed visible text in, cleaned text and thinking deltas out, in order.

    Feed one channel only (the visible text): mixing reasoning into the same instance
    would corrupt the held-back partial delimiters. Always call :meth:`flush_events` at
    stream end; an unterminated section comes out as thinking, a held partial as text.
    """

    def __init__(self) -> None:
        self._thinking_scanner = ThinkingInbandScanner()

    def feed_events(self, text: str) -> list[HealingEvent]:
        if not text:
            return []
        return _convert_scanner_events(self._thinking_scanner.feed(text))

    def flush_events(self) -> list[HealingEvent]:
        return _convert_scanner_events(self._thinking_scanner.flush())


def _convert_scanner_events(events: list[ScanEvent]) -> list[HealingEvent]:
    """Section markers are dropped: callers only need the ordered deltas."""
    out: list[HealingEvent] = []
    for kind, value in events:
        if kind == "text":
            out.append(("text", value))
        elif kind == "thinkingDelta" and value:
            out.append(("thinking", value))
    return out
