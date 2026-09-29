"""Planning leak detection for the flash models.

Port of ``providers/google-gemini-cli.ts``. Antigravity's flash models sometimes spill the
internal planning object into the visible text — a JSON object opening with ``thought``
that should never reach the client.

Two things make this harder than it looks:

1. **The object spans chunks.** One delta can carry `{"thou` and the next `ght": …}`.
   Deciding per chunk let through everything that did not fit in a single one; hence the
   buffering, which the reader drives exactly where omp's stream loop does
   (`turns._AntigravityReader`).
2. **Not every JSON object is a leak.** A model legitimately answering `{"command": "ls"}`
   must not have its answer erased — so the filter only applies to the family that does
   spill, only to an object whose first key is ``thought``, and it requires a leak
   signature once the object closes.
"""

from __future__ import annotations

import json
from typing import Any, Final, Literal

#: Maximum length of a still incomplete prefix accepted as a possible leak. Above this it
#: is legitimate text that happens to start with a brace.
MAX_PREFIX_CHARS: Final = 100

#: Keys whose presence, alone, marks a decoded object as a tool-call planning leak.
_TOOL_SIGNATURE_KEYS: Final = ("_i", "paths", "command")


# omp: providers/google-gemini-cli.ts :: isFlashLeakModel
def is_flash_leak_model(model: str) -> bool:
    """Only the flash family spills planning into the visible text.

    omp reads ``flash-stream-leak-workaround`` off its catalog (``classes/gemini.kdl``:
    family ``flash`` on Antigravity and Gemini CLI). Applying the filter to every model
    would make a `pro` legitimately answering ``{"command": "ls"}`` have its answer erased.
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
    if any(key in parsed for key in _TOOL_SIGNATURE_KEYS):
        return True
    return "path" in parsed and "content" in parsed


def _has_leak_signature_text(text: str, tool_names: frozenset[str]) -> bool:
    """omp's substring fallback, for an object that will not decode or never closed."""
    return (
        '"thought"' in text
        or any(f'"{name}"' in text for name in tool_names)
        or any(f'"{key}"' in text for key in _TOOL_SIGNATURE_KEYS)
        or ('"path"' in text and '"content"' in text)
    )


# omp: providers/google-gemini-cli.ts :: splitLeadingJsonObject
# omp: providers/google-gemini-cli.ts :: splitLeadingJsonObjectIgnoringQuotes
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


# omp: providers/google-gemini-cli.ts :: consumePlanningBuffer
def consume_planning_buffer(
    text: str, tool_names: frozenset[str] = frozenset(), *, final: bool = False
) -> tuple[Outcome, str]:
    """What to do with the buffered visible text: keep holding it, or release it.

    ``("incomplete", "")`` holds the buffer for the next delta. ``("plain", text)``
    releases it whole; ``("leak", rest)`` drops the leading planning object and releases
    only what followed it. At the end of the stream (``final``) an object that never
    closed is a leak when it carries a leak signature, and text otherwise.
    """
    if not is_leak_prefix(text):
        return "plain", text

    split = _split_leading_object(text) or _split_leading_object(text, honour_strings=False)
    if split is None:
        if not final:
            return "incomplete", ""
        if _has_leak_signature_text(text.strip(), tool_names):
            return "leak", ""
        return "plain", text

    json_text, rest = split
    try:
        parsed: Any = json.loads(json_text)
    except ValueError:
        # Malformed JSON — typically unbalanced quotes inside the leak. With no object to
        # inspect, the decision falls to the substring signature; an object that cannot
        # be read is not safe to strip, so without one it goes out as text.
        if _has_leak_signature_text(json_text, tool_names):
            return "leak", rest
        return "plain", text

    return ("leak", rest) if is_leak_object(parsed, tool_names) else ("plain", text)
