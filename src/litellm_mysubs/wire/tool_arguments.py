"""Final tool-call arguments: omp's repair parser, and what a native call replays with.

Port of ``parseJsonWithRepair`` (``pi-utils`` ``json-parse.ts``) in its final, strict
mode only — the streaming mode (auto-closing a half-received buffer) serves omp's UI
previews, which have no counterpart here. Since 18.8.6 omp finalizes a tool call with
it (``parseToolCallArguments``) instead of the streaming parser, so the arguments a
client receives are repaired when the model wrote JSON5-ish text (single quotes,
unquoted keys, trailing commas, comments, Python literals, an unquoted path), and a
call whose text is truncated or ambiguous is not silently completed.

Indices are code points, not UTF-16 units as in the TypeScript. Every token the lexer
branches on is ASCII, so behaviour is identical.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Final

_WHITESPACE: Final = " \t\n\r"
#: Valid characters after `\` in strict JSON (`VALID_ESCAPE_CHAR`).
_ESCAPES: Final = {
    '"': '"',
    "'": "'",
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}
_NUMBER_CHARS: Final = frozenset("0123456789+-.abcdefABCDEFxX")
#: Keyword literals: JSON's, plus Python's.
_KEYWORDS: Final = (
    ("true", True),
    ("false", False),
    ("null", None),
    ("True", True),
    ("False", False),
    ("None", None),
)
#: JS-only atoms never recovered as bareword strings: a tool must not run with a
#: non-finite or undefined argument posing as a string.
_NON_RECOVERABLE_BAREWORDS: Final = frozenset(
    ("NaN", "Infinity", "-Infinity", "+Infinity", "undefined")
)
#: What JavaScript's ``Number(token)`` accepts out of the characters a numeric token
#: may hold: a signed decimal (leading or trailing dot, exponent), or an unsigned
#: ``0x``/``0b`` integer.
_DECIMAL: Final = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_HEX: Final = re.compile(r"0[xX][0-9a-fA-F]+")
_BINARY: Final = re.compile(r"0[bB][01]+")


def _is_ident_char(char: str) -> bool:
    return char.isascii() and (char.isalnum() or char in "_$")


def _skip_insignificant(source: str, index: int) -> int:
    """Past whitespace and ``//`` / ``/* */`` comments; a lone trailing ``/`` stays."""
    length = len(source)
    while True:
        while index < length and source[index] in _WHITESPACE:
            index += 1
        if index + 1 < length and source[index] == "/":
            following = source[index + 1]
            if following == "/":
                index += 2
                while index < length and source[index] != "\n":
                    index += 1
                continue
            if following == "*":
                index += 2
                while index + 1 < length and source[index : index + 2] != "*/":
                    index += 1
                index = min(index + 2, length)
                continue
        return index


def _js_number(token: str) -> float | int | None:
    """``Number(token)`` when finite, else ``None``; integral values as ``int`` so they
    serialize as ``JSON.stringify`` writes them (``1``, not ``1.0``)."""
    if _DECIMAL.fullmatch(token):
        value = float(token)
    elif _HEX.fullmatch(token) or _BINARY.fullmatch(token):
        try:
            value = float(int(token[2:], 16 if token[1] in "xX" else 2))
        except OverflowError:
            return None
    else:
        return None
    if not math.isfinite(value):
        return None
    return int(value) if value.is_integer() and abs(value) < 1e21 else value


def _join_surrogates(text: str) -> str:
    """``\\uD83D\\uDE00`` decoded unit by unit is one character in JavaScript; join the
    pairs the same way. A lone surrogate stays, as it does there."""
    return text.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "surrogatepass")


# omp: json-lexer.ts :: JsonLexer
class _Lexer:
    """The tolerant token readers, in the lexer's ``strict`` mode: every truncated or
    malformed token raises ``ValueError``."""

    __slots__ = ("pos", "src")

    def __init__(self, source: str) -> None:
        self.src = source
        self.pos = 0

    @property
    def at_end(self) -> bool:
        return self.pos >= len(self.src)

    def peek(self) -> str:
        return self.src[self.pos] if self.pos < len(self.src) else ""

    def ws(self) -> None:
        self.pos = _skip_insignificant(self.src, self.pos)

    def string(self, quote: str) -> str:
        """A string from its opening quote. A double quote closes on the first unescaped
        one, as in JSON; a single quote only where a value terminator follows, which
        recovers apostrophes (``'it's'``). Raw control characters and invalid escapes are
        kept literally."""
        source = self.src
        length = len(source)
        index = self.pos + 1
        out: list[str] = []
        run_start = index
        surrogates = False
        while index < length:
            char = source[index]
            if char != "\\" and char != quote:
                index += 1
                continue
            if char == quote:
                if quote == '"' or self._quote_closes(index + 1):
                    out.append(source[run_start:index])
                    self.pos = index + 1
                    text = "".join(out)
                    return _join_surrogates(text) if surrogates else text
                index += 1
                continue
            out.append(source[run_start:index])
            index += 1
            if index >= length:
                break
            escape = source[index]
            if escape in _ESCAPES:
                out.append(_ESCAPES[escape])
            elif escape == "u":
                digits = source[index + 1 : index + 5]
                if len(digits) == 4 and all(d in "0123456789abcdefABCDEF" for d in digits):
                    unit = int(digits, 16)
                    surrogates = surrogates or 0xD800 <= unit < 0xE000
                    out.append(chr(unit))
                    index += 4
                else:
                    out.append("\\u")
            else:
                out.append("\\" + escape)
            index += 1
            run_start = index
        raise ValueError("Unterminated string")

    def _quote_closes(self, start: int) -> bool:
        index = _skip_insignificant(self.src, start)
        return index >= len(self.src) or self.src[index] in ",}]:"

    def number(self) -> float | int:
        source = self.src
        start = index = self.pos
        while index < len(source) and source[index] in _NUMBER_CHARS:
            index += 1
        self.pos = index
        token = source[start:index]
        value = _js_number(token)
        if value is None:
            raise ValueError(f"Invalid number: {token}")
        return value

    def keyword(self) -> tuple[bool, bool | None]:
        """``(matched, value)``; consumes only on a match bounded by a non-identifier."""
        source = self.src
        for word, value in _KEYWORDS:
            end = self.pos + len(word)
            if source.startswith(word, self.pos) and not (
                end < len(source) and _is_ident_char(source[end])
            ):
                self.pos = end
                return True, value
        return False, None

    def unquoted_key(self) -> str:
        source = self.src
        start = index = self.pos
        while index < len(source) and source[index] not in ":,}" + _WHITESPACE:
            index += 1
        self.pos = index
        return source[start:index]

    def bareword(self) -> str:
        """An unquoted string value (``{"paths": packages/foo/*}``) up to ``,`` ``}``
        ``]`` or a newline. Refused when it runs to the end of the input, holds a ``"``,
        ``{``, ``[`` or a key-like ``:`` (a missed comma must not swallow the next
        field; ``:/`` and ``:\\`` stay, for URLs and Windows paths), or is a non-finite
        atom."""
        source = self.src
        start = index = self.pos
        while index < len(source):
            char = source[index]
            if char in ",}]\n\r":
                break
            following = source[index + 1] if index + 1 < len(source) else ""
            if char in '"{[' or (char == ":" and following not in ("/", "\\")):
                raise ValueError(f"Unexpected token at position {start}")
            index += 1
        if index >= len(source):
            raise ValueError(f"Unexpected token at position {start}")
        end = index
        while end > start and source[end - 1] in _WHITESPACE:
            end -= 1
        word = source[start:end]
        if word in _NON_RECOVERABLE_BAREWORDS:
            raise ValueError(f"Unexpected token at position {start}")
        self.pos = index
        return word


# omp: json-parse.ts :: RelaxedJson
class _RelaxedJson:
    """Recursive descent over `_Lexer`, final mode: stray commas are skipped, end of
    input mid-value and trailing garbage raise, so a half-formed call never parses."""

    __slots__ = ("_lex",)

    def __init__(self, source: str) -> None:
        self._lex = _Lexer(source)

    def parse(self) -> object:
        lex = self._lex
        lex.ws()
        if lex.at_end:
            raise ValueError("Unexpected end of JSON input")
        value = self._value(allow_bareword=False)
        lex.ws()
        if not lex.at_end:
            raise ValueError(f"Unexpected trailing characters at position {lex.pos}")
        return value

    def _value(self, *, allow_bareword: bool) -> object:
        lex = self._lex
        char = lex.peek()
        if char == "{":
            return self._object()
        if char == "[":
            return self._array()
        if char in ('"', "'"):
            return lex.string(char)
        if char and char in "+-.0123456789":
            return lex.number()
        matched, keyword = lex.keyword()
        if matched:
            return keyword
        if allow_bareword:
            return lex.bareword()
        raise ValueError(f"Unexpected token at position {lex.pos}")

    def _object(self) -> dict[str, object]:
        lex = self._lex
        lex.pos += 1
        out: dict[str, object] = {}
        while True:
            lex.ws()
            if lex.at_end:
                raise ValueError("Unterminated object")
            char = lex.peek()
            if char == "}":
                lex.pos += 1
                return out
            if char == ",":
                lex.pos += 1
                continue
            key = self._key()
            lex.ws()
            if lex.peek() != ":":
                raise ValueError("Expected ':' in object")
            lex.pos += 1
            lex.ws()
            if lex.at_end:
                raise ValueError("Expected value after ':'")
            out[key] = self._value(allow_bareword=True)
            lex.ws()
            char = lex.peek()
            if char == ",":
                lex.pos += 1
                continue
            if char == "}":
                lex.pos += 1
                return out
            raise ValueError("Expected ',' or '}' in object")

    def _array(self) -> list[object]:
        lex = self._lex
        lex.pos += 1
        out: list[object] = []
        while True:
            lex.ws()
            if lex.at_end:
                raise ValueError("Unterminated array")
            char = lex.peek()
            if char == "]":
                lex.pos += 1
                return out
            if char == ",":
                lex.pos += 1
                continue
            out.append(self._value(allow_bareword=True))
            lex.ws()
            char = lex.peek()
            if char == ",":
                lex.pos += 1
                continue
            if char == "]":
                lex.pos += 1
                return out
            raise ValueError("Expected ',' or ']' in array")

    def _key(self) -> str:
        lex = self._lex
        char = lex.peek()
        if char in ('"', "'"):
            return lex.string(char)
        key = lex.unquoted_key()
        if not key:
            raise ValueError("Expected object key")
        return key


def _reject_constant(name: str) -> object:
    raise ValueError(f"Unexpected token {name}")


def _js_float(token: str) -> float | int | None:
    """A JSON fraction or exponent as a JavaScript number reserializes: ``1.0`` is ``1``,
    and an overflow (``1e999``, ``Infinity`` there) is written ``null``."""
    value = float(token)
    if not math.isfinite(value):
        return None
    return int(value) if value.is_integer() and abs(value) < 1e21 else value


def json_parse(text: str) -> object:
    """``JSON.parse``: ``json.loads`` without the ``NaN``/``Infinity`` it alone accepts."""
    return json.loads(text, parse_constant=_reject_constant, parse_float=_js_float)


# omp: json-parse.ts :: parseJsonWithRepair
def parse_json_with_repair(text: str) -> object:
    """Strict JSON first, then the relaxed grammar; ``ValueError`` when neither parses."""
    try:
        return json_parse(text)
    except ValueError:
        return _RelaxedJson(text).parse()


@dataclass(frozen=True, slots=True)
class InvalidArguments:
    """Arguments no repair could parse: omp's ``{__parseError, __rawJson}`` marker."""

    error: str


# omp: utils/tool-call-arguments.ts :: parseToolCallArguments
def parse_tool_call_arguments(raw: str | None) -> object:
    """A finished call's arguments: ``{}`` when absent or blank (zero-argument tools),
    the repaired value, or `InvalidArguments`."""
    if raw is None or not raw.strip():
        return {}
    try:
        return parse_json_with_repair(raw)
    except (ValueError, RecursionError) as error:
        # RecursionError: nesting too deep to parse, JavaScript's stack RangeError.
        return InvalidArguments(str(error))


# omp: utils/tool-call-arguments.ts :: replayableToolCallArguments
def replayable_tool_call_arguments(raw: object, executed: object) -> object:
    """A native ``function_call.arguments`` worth replaying.

    Replay drops a call whose stored arguments do not parse, so a call that only parsed
    after repair is stored with the arguments it ran with (omp #14155); text that already
    parses, or that nothing could repair, is kept as it came.
    """
    if isinstance(raw, str):
        try:
            json_parse(raw)
            return raw
        except (ValueError, RecursionError):
            pass
    if isinstance(executed, InvalidArguments):
        return raw
    return json.dumps(executed, separators=(",", ":"), ensure_ascii=False)


_MAX_TOOL_CALL_NAME_LENGTH: Final = 128
#: JavaScript's ``/[\s\p{Cc}]/u``: Python's ``\s`` lacks only U+FEFF of JavaScript's.
_TOOL_CALL_NAME_SEPARATOR: Final = re.compile(r"[\s\ufeff\x00-\x1f\x7f-\x9f]")


# omp: providers/transform-messages.ts :: isMalformedToolCallName
def is_malformed_tool_call_name(name: object) -> bool:
    """A name no declared tool can have: missing, empty, longer than OpenAI's 128-unit
    replay limit, or holding whitespace or control characters — the name slot carrying
    invocation text. The length is counted in UTF-16 units, as in JavaScript."""
    return (
        not isinstance(name, str)
        or not name
        or len(name.encode("utf-16-le", "surrogatepass")) // 2 > _MAX_TOOL_CALL_NAME_LENGTH
        or _TOOL_CALL_NAME_SEPARATOR.search(name) is not None
    )
