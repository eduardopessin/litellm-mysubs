"""Planning leak filter for the flash models.

Port of ``providers/google-gemini-cli.ts``. Antigravity's flash models sometimes spill the
internal planning object into the visible text — a JSON object with ``thought``, ``call``,
``paths`` or ``path``+``content`` that should never reach the client.

Two things make this harder than it looks:

1. **The object spans chunks.** One delta can carry `{"thou` and the next `ght": …}`.
   Deciding per chunk let through everything that did not fit in a single one; hence the
   buffering.
2. **Not every JSON object is a leak.** A model legitimately answering `{"command": "ls"}`
   must not have its answer erased — so the filter only applies to the family that does
   spill, and it requires a leak signature.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Final, Literal

#: Keys whose presence marks the object as internal planning.
LEAK_MARKERS: Final[tuple[str, ...]] = ("thought", "_i", "call", "paths", "command")

#: Maximum length of a still incomplete prefix accepted as a possible leak. Above this it
#: is legitimate text that happens to start with a brace.
MAX_PREFIX_CHARS: Final = 100


# omp: providers/google-gemini-cli.ts :: isFlashLeakModel
def is_flash_leak_model(model: str) -> bool:
    """Only the flash family spills planning into the visible text.

    Applying the filter to every model would make a `pro` legitimately answering
    ``{"command": "ls"}`` have its answer erased.
    """
    return "flash" in str(model).split("/")[-1].lower()


# omp: providers/google-gemini-cli.ts :: isPlanningLeakPrefix
def is_leak_prefix(text: str) -> bool:
    """Whether the text **may** turn into a planning object.

    It recognizes an incomplete prefix: `{`, `{"tho`, `{"thought"`. That is what allows
    holding the buffer instead of emitting half a leak.
    """
    trimmed = text.lstrip()
    if not trimmed.startswith("{"):
        return False

    after_brace = trimmed[1:].lstrip()
    if not after_brace:
        return len(trimmed) <= MAX_PREFIX_CHARS
    if after_brace[0] != '"':
        return False

    next_quote = after_brace.find('"', 1)
    if next_quote == -1:
        key_prefix = after_brace[1:]
        return "thought".startswith(key_prefix) and len(trimmed) <= MAX_PREFIX_CHARS

    if after_brace[1:next_quote] != "thought":
        return False

    after_key = after_brace[next_quote + 1 :].lstrip()
    return len(trimmed) <= MAX_PREFIX_CHARS if not after_key else after_key[0] == ":"


# omp: providers/google-gemini-cli.ts :: isPlanningLeakObject
def is_leak_object(parsed: object, tool_names: frozenset[str] = frozenset()) -> bool:
    """Whether the already decoded object has a planning signature."""
    if not isinstance(parsed, dict):
        return False
    if isinstance(parsed.get("thought"), str):
        return True
    call = parsed.get("call")
    if isinstance(call, str) and call in tool_names:
        return True
    if any(key in parsed for key in ("_i", "paths", "command")):
        return True
    return "path" in parsed and "content" in parsed


def _split_leading_object(text: str, *, honour_strings: bool = True) -> tuple[str, str] | None:
    """First brace-balanced JSON object, and whatever is left over.

    ``honour_strings=False`` is the fallback: a leak with unbalanced quotes would never
    close the object by the normal route, and letting it through was the worst outcome.
    """
    prefix = len(text) - len(text.lstrip())
    trimmed = text[prefix:]
    if not trimmed.startswith("{"):
        return None

    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(trimmed):
        if honour_strings and in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if honour_strings and char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return trimmed[: index + 1], trimmed[index + 1 :]
    return None


Outcome = Literal["incomplete", "plain", "leak"]


@dataclass(slots=True)
class PlanningLeakFilter:
    """Filter the visible text of a stream, holding back what may be a leak.

    Usage: ``feed`` for each delta, ``flush`` at the end. ``stripped`` tells whether
    anything was discarded — OMP uses that signal so as not to accept as "valid silence" a
    response whose content was thrown away entirely.
    """

    tool_names: frozenset[str] = frozenset()
    buffer: str = ""
    buffering: bool = False
    stripped: bool = False
    _emitted: list[str] = field(default_factory=list, repr=False)

    def feed(self, text: str) -> str:
        """Text to hand the client because of this delta. May be empty."""
        if not text:
            return ""
        if not self.buffering and not self.buffer:
            if not is_leak_prefix(text):
                return text
            self.buffering = True
        self.buffer += text
        return self._drain()

    def flush(self) -> str:
        """Whatever is left at the end of the stream.

        A buffer with a leak signature but no closing brace is discarded entirely: handing
        it over would mean showing half of the internal planning.
        """
        if not self.buffer:
            return ""
        pending, self.buffer = self.buffer, ""
        self.buffering = False
        emitted = self._consume(pending, final=True)
        return emitted

    def _drain(self) -> str:
        emitted = self._consume(self.buffer, final=False)
        return emitted

    def _consume(self, text: str, *, final: bool) -> str:
        split = _split_leading_object(text) or _split_leading_object(text, honour_strings=False)
        if split is None:
            if not final:
                # Object not closed yet: held back waiting for the next delta.
                self.buffer = text
                return ""
            # At the end of the stream, a prefix with a leak signature never becomes text.
            if is_leak_prefix(text):
                self.stripped = True
                return ""
            return text

        json_text, rest = split
        try:
            parsed: Any = json.loads(json_text)
        except ValueError:
            parsed = None

        if parsed is not None:
            leaked = is_leak_object(parsed, self.tool_names)
        else:
            # Malformed JSON — typically unbalanced quotes inside the leak. With no object
            # to inspect, the decision falls to the prefix signature; letting it through
            # because it failed to decode was the worst possible outcome.
            leaked = is_leak_prefix(json_text)

        if leaked:
            self.stripped = True
            visible = rest
        else:
            visible = json_text + rest

        self.buffer = ""
        self.buffering = False
        return visible
