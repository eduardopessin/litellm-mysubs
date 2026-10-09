"""Check the `# omp:` anchors against the real OMP package.

The wiring comes from ``@oh-my-pi/pi-ai``; this package is a port to Python. Without
checking, a rename on their side leaves the anchors pointing at nothing and nobody notices
until someone tries to follow one. A name that survives says nothing about the code behind
it, so every anchored declaration is also hashed and compared against
``tools/omp_anchors.lock``: a changed body is BODY DRIFT until someone reviews the port and
regenerates the lock.

    python tools/check_omp_drift.py                        # check the pinned version
    python tools/check_omp_drift.py --update               # regenerate the lock (pinned)
    python tools/check_omp_drift.py --omp-version 18.5.0   # preview a bump against the lock

Exits with 1 if any annotated symbol has disappeared, changed value, changed body, cannot be
extracted, or is missing from the lock. A new version on npm is a warning, not an error:
upgrading is a decision, not an obligation.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import io
import json
import re
import tarfile
import urllib.request
from pathlib import Path
from typing import Final, NamedTuple

#: OMP version the anchors were written against.
OMP_VERSION = "18.8.6"

#: The wiring is split across **three** packages: `pi-ai` has the logic, `pi-catalog` has
#: the wire constants (header values, pinned client versions), and `pi-utils` has what is
#: shared across every provider — the `USER_AGENT`, among others. An anchor may point at
#: any of them.
#:
#: Every missing package is an anchor that cannot be checked. That is how the Codex
#: `User-Agent` slipped through and ended up invented: it was looked up in the two
#: packages at hand, not found, and written by analogy instead of fetching the third.
PACKAGES: Final = ("@oh-my-pi/pi-ai", "@oh-my-pi/pi-catalog", "@oh-my-pi/pi-utils")
REGISTRY = "https://registry.npmjs.org"

#: `# omp: providers/anthropic.ts :: symbolA, symbolB`
ANCHOR_RE = re.compile(r"^\s*#\s*omp:\s*(?P<file>[\w./-]+)\s*::\s*(?P<symbols>[\w.,\s]+?)\s*$")

#: `# omp= providers/codex.ts :: CODEX_USAGE_PATH = "wham/usage"`, or `# omp= CODEX_USAGE_PATH
#: = "wham/usage"` right after an `# omp:` anchor — there the file is inherited from it.
#:
#: The plain anchor proves the **name** exists. It proves nothing about the **value**, and
#: the value is what goes on the wire: an endpoint path, a pinned client version, a client
#: id. Upstream can swap `"wham/usage"` for something else without touching the constant's
#: name, and the check would stay green while the plugin hit a 404.
#:
#: Measured: changing `claudeCodeSdkVersion` from `0.112.1` to `0.999.0` in the OMP tarball
#: left all 177 name anchors green. That is the hole this form closes.
VALUE_RE = re.compile(
    r"^\s*#\s*omp=\s*(?:(?P<file>[\w./-]+)\s*::\s*)?(?P<symbol>\w+)\s*=\s*(?P<value>.+?)\s*$"
)

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

#: Hash of every anchored declaration at `OMP_VERSION`. Committed: a pin bump changes the
#: hashes, and regenerating them with `--update` is the moment someone reads what changed.
LOCK = ROOT / "tools" / "omp_anchors.lock"


class Anchor(NamedTuple):
    path: Path
    line: int
    file: str
    symbol: str
    #: Literal the symbol must have on the OMP side. Empty: only the name is checked.
    value: str = ""

    def __str__(self) -> str:
        tail = f" = {self.value}" if self.value else ""
        return f"{self.path.name}:{self.line} -> {self.file} :: {self.symbol}{tail}"


def collect_anchors() -> list[Anchor]:
    """One anchor per symbol: several in the same comment count separately.

    Two forms. ``# omp:`` proves the symbol exists; ``# omp=`` also proves the value, for
    what goes on the wire and breaks silently when it changes. The value form may omit the
    file and inherit it from the name anchor immediately above - that is the common case,
    and repeating a long path on both lines only serves to blow the margin.
    """
    anchors: list[Anchor] = []
    for source in sorted(SRC.rglob("*.py")):
        previous = ""
        for number, text in enumerate(source.read_text("utf-8").splitlines(), start=1):
            if value_match := VALUE_RE.match(text):
                file = value_match.group("file") or previous
                if not file:
                    message = f"{source.name}:{number}: `# omp=` with no file and no anchor above"
                    raise SystemExit(message)
                anchors.append(
                    Anchor(
                        source,
                        number,
                        file,
                        value_match.group("symbol"),
                        value_match.group("value"),
                    )
                )
                previous = file
                continue
            match = ANCHOR_RE.match(text)
            if match is None:
                previous = ""
                continue
            file = match.group("file")
            previous = file
            for symbol in match.group("symbols").split(","):
                if stripped := symbol.strip():
                    anchors.append(Anchor(source, number, file, stripped))
    return anchors


def fetch_packages(version: str) -> dict[str, dict[str, str]]:
    """``src/`` files per package (``pi-ai``, ...), indexed by relative path.

    The three packages **share paths** — `index.ts` exists in all three, `stream.ts` in
    `pi-ai` and `pi-utils`, `types.ts` and `utils.ts` in `pi-ai` and `pi-catalog` — so they
    stay apart here: a body hash has to say which package it was taken from.
    """
    return {p.split("/")[-1]: _fetch_package_sources(p, version) for p in PACKAGES}


def merge_sources(packages: dict[str, dict[str, str]]) -> dict[str, str]:
    """One view over the three packages, for the name and value checks.

    Merging carelessly makes the last package shadow the earlier ones, and an anchor for a
    symbol in the shadowed file fails as if it did not exist. The content is concatenated
    instead of replaced: the question being asked is "does this symbol exist at this path",
    and the right answer is yes when it exists in any of the packages.
    """
    sources: dict[str, str] = {}
    for files in packages.values():
        for path, content in files.items():
            existing = sources.get(path)
            sources[path] = f"{existing}\n{content}" if existing else content
    return sources


def _fetch_package_sources(package: str, version: str) -> dict[str, str]:
    name = package.split("/")[-1]
    url = f"{REGISTRY}/{package}/-/{name}-{version}.tgz"
    with urllib.request.urlopen(url, timeout=120) as response:
        raw = response.read()

    sources: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or "/src/" not in member.name:
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            relative = member.name.split("/src/", 1)[1]
            sources[relative] = handle.read().decode("utf-8", "replace")
    return sources


def _literal(content: str, symbol: str) -> str:
    """The literal assigned to ``symbol``, normalized to double quotes.

    Only direct string assignments are recognized — ``const X = "y"``, with or without a
    type annotation, and with backticks or single quotes instead of double. A composed
    value (``` `${BASE}/x` ```) is not a literal and returns empty: the value anchor does
    not serve those, and saying so is better than faking a comparison.
    """
    match = re.search(
        rf"(?:const|let|var)\s+{re.escape(symbol)}\s*(?::[^=]+?)?=\s*"
        r"""(?P<quote>["'`])(?P<value>[^"'`$\\]*)(?P=quote)""",
        content,
    )
    return f'"{match.group("value")}"' if match else ""


# --- Body extraction ---------------------------------------------------------------------
#
# A name anchor stays green while the function behind it is rewritten: `applyHeadCaching`
# and `fetchClaudeUsage` changed between 18.3.2 and 18.4.1 without a rename. What follows
# cuts each anchored declaration out of the TypeScript (or KDL) source, normalizes away
# what a formatter can change, and hashes the rest.
#
# It is a scanner, not a parser: enough to know, for every character, whether it is code,
# a comment or the inside of a literal. Brackets are only matched in code, so a `{` inside
# a string, a template literal or a regex does not end a function early.

#: Previous significant character after which `/` opens a regex rather than dividing.
_REGEX_AFTER: Final = frozenset("(,=:[!&|?{};+-*%<>~^")
#: Keywords after which `/` opens a regex (`return /x/.test(s)`).
_REGEX_KEYWORDS: Final = frozenset(
    (
        "return",
        "typeof",
        "case",
        "do",
        "else",
        "in",
        "of",
        "new",
        "delete",
        "void",
        "throw",
        "yield",
        "await",
        "instanceof",
    )
)
_WORD: Final = re.compile(r"\w")
#: Words after which a `{` in a signature opens a type literal, not the body.
_TYPE_WORD: Final = re.compile(r"\b(?:is|readonly|keyof)\s*$")
#: Stand-in for literal characters in the masked text: neither a word character, whitespace
#: nor a bracket, so nothing inside a literal matches a pattern.
_FILL: Final = "\x01"
#: Stand-in for each literal in `normalize`, restored once the code around it is squeezed.
_SLOT: Final = "\x02"


class Span(NamedTuple):
    kind: str  # "code", "comment" or "literal"
    start: int
    end: int


def scan(text: str, kdl: bool = False) -> list[Span]:
    """Split ``text`` into code, comment and literal spans, in order, covering all of it.

    A template literal is literal up to ``${``, code inside the substitution (it may hold
    braces, strings and further templates), literal again from the closing ``}``. KDL has
    only double-quoted strings and C comments.
    """
    spans: list[Span] = []
    n = len(text)
    i = 0
    code_start = 0
    #: Open template substitutions: for each, the brace depth inside it.
    templates: list[int] = []
    last = ""  # last significant code character
    word = ""  # last identifier in code

    def push(kind: str, start: int, end: int) -> None:
        nonlocal code_start
        if code_start < start:
            spans.append(Span("code", code_start, start))
        spans.append(Span(kind, start, end))
        code_start = end

    def template_from(start: int) -> int:
        """Literal from ``start`` to the closing backtick or the next ``${``."""
        j = start
        while j < n:
            if text[j] == "\\":
                j += 2
            elif text[j] == "`":
                return j + 1
            elif text.startswith("${", j):
                templates.append(0)
                return j + 2
            else:
                j += 1
        return n

    while i < n:
        c = text[i]
        if text.startswith("//", i):
            end = text.find("\n", i)
            end = n if end < 0 else end
            push("comment", i, end)
            i = end
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            end = n if end < 0 else end + 2
            push("comment", i, end)
            i = end
        elif c == '"' or (not kdl and c == "'"):
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            end = min(j + 1, n)
            push("literal", i, end)
            i, last, word = end, c, ""
        elif not kdl and c == "`":
            end = template_from(i + 1)
            push("literal", i, end)
            i, last, word = end, c, ""
        elif not kdl and c == "}" and templates and templates[-1] == 0:
            templates.pop()
            end = template_from(i + 1)
            push("literal", i, end)
            i, last, word = end, "`", ""
        elif not kdl and c == "/" and (not last or last in _REGEX_AFTER or word in _REGEX_KEYWORDS):
            j, in_class = i + 1, False
            while j < n and text[j] != "\n":
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == "[":
                    in_class = True
                elif text[j] == "]":
                    in_class = False
                elif text[j] == "/" and not in_class:
                    break
                j += 1
            j += 1
            while j < n and _WORD.match(text[j]):
                j += 1
            push("literal", i, j)
            i, last, word = j, "/", ""
        else:
            if templates and c == "{":
                templates[-1] += 1
            elif templates and c == "}":
                templates[-1] -= 1
            if _WORD.match(c):
                j = i
                while j < n and _WORD.match(text[j]):
                    j += 1
                word, last = text[i:j], text[j - 1]
                i = j
                continue
            if not c.isspace():
                last, word = c, ""
            i += 1
    if code_start < n:
        spans.append(Span("code", code_start, n))
    return spans


def mask(text: str, spans: list[Span]) -> str:
    """``text`` with comments blanked and literal characters filled, newlines kept.

    Same length as the input, so offsets found in the mask cut the original.
    """
    parts: list[str] = []
    for span in spans:
        chunk = text[span.start : span.end]
        if span.kind == "code":
            parts.append(chunk)
        else:
            parts.append(re.sub(r"[^\n]", " " if span.kind == "comment" else _FILL, chunk))
    return "".join(parts)


def normalize(text: str, kdl: bool = False) -> str:
    """What is left of a declaration once the formatter's choices are taken out.

    Comments go; runs of whitespace in code become one space, kept only between two word
    characters (``const x``), so line wrapping and indentation do not count; a trailing
    comma before a closer goes, since Biome adds one exactly when it wraps a list. Literals
    are kept verbatim: a changed string is a changed body.
    """
    code: list[str] = []
    literals: list[str] = []
    for span in scan(text, kdl):
        chunk = text[span.start : span.end]
        if span.kind == "literal":
            literals.append(chunk)
            code.append(_SLOT)
        else:
            code.append(chunk if span.kind == "code" else " ")
    squeezed = re.sub(r"\s+", " ", "".join(code))
    # A space survives only between two word characters, which it keeps apart.
    squeezed = re.sub(r"(?<!\w) | (?!\w)", "", squeezed)
    squeezed = re.sub(r",(?=[)\]}])", "", squeezed)
    pieces = squeezed.split(_SLOT)
    return pieces[0] + "".join(lit + piece for lit, piece in zip(literals, pieces[1:], strict=True))


_MODIFIERS: Final = (
    r"(?:(?:export|default|declare|abstract|public|private|protected|static|readonly|"
    r"override|async|get|set|accessor)\s+)*"
)


#: A class field needs one of these before `name =`; without, it reads as an assignment.
_FIELD_MODIFIERS: Final = (
    r"(?:(?:declare|public|private|protected|static|readonly|override|accessor)\s+)+"
)


def _find_close(masked: str, start: int, opener: str, closer: str) -> int:
    """Index just past the ``closer`` matching the ``opener`` at ``start``; -1 if unmatched."""
    depth = 0
    for index in range(start, len(masked)):
        char = masked[index]
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return index + 1
    return -1


def _block_end(masked: str, start: int) -> int:
    """End of a function, method, class, interface or enum starting at ``start``.

    The signature is walked with brackets and generic angles counted; the body is the first
    ``{`` at depth zero that does not open a type (one right after ``:``, ``|``, ``&``,
    ``,``, ``<``, ``=>``, ``is`` or ``readonly`` is a return-type literal). A ``;`` before
    any body is an overload or a declaration without one, and ends there.
    """
    depth = angle = 0
    last = ""
    index = start
    while index < len(masked):
        char = masked[index]
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char == "<" and index > 0 and _WORD.match(masked[index - 1]):
            angle += 1
        elif char == ">" and angle and masked[index - 1] != "=":
            angle -= 1
        elif char == ";" and depth == 0 and angle == 0:
            return index + 1
        elif char == "{":
            opens_type = last in (":", "|", "&", ",", "<", "=>") or _TYPE_WORD.search(
                masked, max(start, index - 12), index
            )
            if depth == 0 and angle == 0 and not opens_type:
                end = _find_close(masked, index, "{", "}")
                return len(masked) if end < 0 else end
            close = _find_close(masked, index, "{", "}")
            if close < 0:
                return len(masked)
            index, last = close, "}"
            continue
        if not char.isspace():
            last = "=>" if char == ">" and masked[index - 1] == "=" else char
        index += 1
    return len(masked)


def _statement_end(masked: str, start: int, list_item: bool) -> int:
    """End of a ``const``/``type``/field declaration, or of an object property.

    A ``;`` at depth zero ends it, or the closer of the enclosing bracket; for an object
    property (``list_item``) a ``,`` at depth zero does too, generic angles counted so that
    ``Record<string, X>`` does not cut the type in half.
    """
    depth = angle = 0
    for index in range(start, len(masked)):
        char = masked[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth < 0:
                return index
        elif char == "<" and index > 0 and _WORD.match(masked[index - 1]):
            angle += 1
        elif char == ">" and angle and masked[index - 1] != "=":
            angle -= 1
        elif depth == 0 and (char == ";" or (list_item and char == "," and angle == 0)):
            return index + 1 if char == ";" else index
    return len(masked)


def _ts_declarations(masked: str, name: str) -> list[tuple[int, int]]:
    """Every declaration of ``name`` in ``masked``, most specific form first.

    Three tiers, and the first that finds anything wins, with all of its matches: a
    keyword declaration (``function``, ``const``, ``class``...), anywhere, nested ones
    included; a method (``name(...) {`` at the start of a line, told apart from a call by
    what follows the parameters); a property or field (``name:`` / ``name?:``, or
    ``name =`` behind a modifier). Several matches in a tier are all kept - overloads, a
    ``credential`` in both ``login`` and ``refresh`` - so a change in any of them counts.
    """
    word = re.escape(name)
    keyword = re.compile(
        rf"(?<![\w$.])(?:{_MODIFIERS})(?:function\s*\*?|class|interface|enum|namespace|type|"
        rf"const|let|var)\s+(?:\*\s*)?{word}(?![\w$])"
    )
    found = []
    for match in keyword.finditer(masked):
        head = match.group(0)
        block = re.search(r"\b(?:function|class|interface|enum|namespace)\b", head)
        end = (
            _block_end(masked, match.end())
            if block
            else _statement_end(masked, match.end(), list_item=False)
        )
        found.append((match.start(), end))
    if found:
        return found

    method = re.compile(rf"(?m)^[ \t]*(?:{_MODIFIERS})(?:\*\s*)?#?{word}\s*(?=[(<])")
    for match in method.finditer(masked):
        paren = masked.find("(", match.end())
        close = _find_close(masked, paren, "(", ")") if paren >= 0 else -1
        if close < 0:
            continue
        follow = masked[close:].lstrip()
        if follow.startswith((":", "{")):
            found.append(
                (
                    match.start() + len(match.group(0)) - len(match.group(0).lstrip()),
                    _block_end(masked, match.end()),
                )
            )
    if found:
        return found

    field = re.compile(
        rf"(?m)^[ \t]*(?:{_MODIFIERS}#?{word}\??\s*:(?!:)|{_FIELD_MODIFIERS}#?{word}\??\s*=(?![=>])"
        rf"|#{word}\s*=(?![=>]))"
    )
    for match in field.finditer(masked):
        start = match.start() + len(match.group(0)) - len(match.group(0).lstrip())
        found.append((start, _statement_end(masked, match.end(), list_item=True)))
    return found


def _kdl_nodes(masked: str, start: int, end: int) -> list[tuple[int, int, int]]:
    """Nodes between ``start`` and ``end``: (node start, end of its own line(s), node end).

    A node ends at a newline that is not escaped, at ``;``, or at the closer of its parent;
    a ``{`` opens its children, which belong to it.
    """
    nodes = []
    index = start
    while index < end:
        while index < end and (masked[index].isspace() or masked[index] == ";"):
            index += 1
        if index >= end or masked[index] == "}":
            break
        node_start = index
        head_end = -1
        while index < end:
            char = masked[index]
            if char == "\\":
                newline = masked.find("\n", index)
                index = end if newline < 0 else newline + 1
                continue
            if char in "\n;}":
                head_end = index
                break
            if char == "{":
                head_end = index
                close = _find_close(masked, index, "{", "}")
                index = end if close < 0 else close
                break
            index += 1
        if head_end < 0:
            head_end = end
        nodes.append((node_start, head_end, index))
        if masked[index : index + 1] == ";":
            index += 1
    return nodes


def _kdl_declarations(
    masked: str, spans: list[Span], text: str, name: str
) -> list[tuple[int, int]]:
    """KDL nodes named ``name``; failing those, nodes with a string argument containing it.

    The auth rules are KDL. An anchor there names a node (``callback``, ``credential``) or
    quotes a value (a base64 client id) - the node carrying the value is what is hashed.
    Names win: ``token`` is a node, and also a substring of ``"access_token"``.
    """
    named: list[tuple[int, int]] = []
    valued: list[tuple[int, int]] = []

    def walk(start: int, end: int) -> None:
        for node_start, head_end, node_end in _kdl_nodes(masked, start, end):
            node_name = re.match(r"[^\s{;=]+", masked[node_start:head_end])
            if node_name and node_name.group(0) == name:
                named.append((node_start, node_end))
                continue
            if any(
                name in text[span.start : span.end]
                for span in spans
                if span.kind == "literal" and node_start <= span.start < head_end
            ):
                valued.append((node_start, head_end))
            if node_end > head_end and masked[head_end] == "{":
                walk(head_end + 1, node_end - 1)

    walk(0, len(masked))
    return named or valued


@functools.lru_cache(maxsize=32)
def _scanned(text: str, kdl: bool) -> tuple[list[Span], str]:
    """Spans and mask of a chunk. A file carries dozens of anchors; it is scanned once."""
    spans = scan(text, kdl)
    return spans, mask(text, spans)


def extract(text: str, symbol: str, kdl: bool = False) -> str | None:
    """Normalized declaration(s) of ``symbol`` in ``text``; None when there is none.

    ``A.b`` narrows: ``b`` declared inside ``A``. Several declarations are joined in order.
    """
    parts = symbol.split(".")
    regions = [(0, len(text))]
    for part in parts:
        next_regions: list[tuple[int, int]] = []
        for start, end in regions:
            chunk = text[start:end]
            spans, masked = _scanned(chunk, kdl)
            finder = (
                _kdl_declarations(masked, spans, chunk, part)
                if kdl
                else _ts_declarations(masked, part)
            )
            next_regions.extend((start + a, start + b) for a, b in finder)
        if not next_regions:
            return None
        regions = next_regions
    return "\n".join(normalize(text[a:b], kdl) for a, b in regions)


def body_hash(text: str, symbol: str, kdl: bool = False) -> str | None:
    body = extract(text, symbol, kdl)
    return None if body is None else hashlib.sha256(body.encode("utf-8")).hexdigest()


def lock_key(package: str, file: str, symbol: str) -> str:
    return f"{package}/{file}::{symbol}"


def _pair(key: str) -> tuple[str, str]:
    """(file, symbol) of a lock key: `lock_key` undone, without the package."""
    path, _, symbol = key.partition("::")
    return path.partition("/")[2], symbol


def hash_anchors(
    anchors: list[Anchor], packages: dict[str, dict[str, str]]
) -> tuple[dict[str, str], dict[tuple[str, str], list[str]], list[tuple[str, str]]]:
    """Body hash of every anchored symbol, keyed ``package/file::symbol``.

    Returns the hashes, the lock keys each (file, symbol) resolved to, and the (file,
    symbol) pairs that could not be extracted from any package - those are reported, never
    skipped: an anchor whose body cannot be read protects nothing.
    """
    hashes: dict[str, str] = {}
    keys: dict[tuple[str, str], list[str]] = {}
    missing: list[tuple[str, str]] = []
    for file, symbol in sorted({(a.file, a.symbol) for a in anchors}):
        resolved = []
        for package, files in packages.items():
            content = files.get(file)
            if content is None:
                continue
            digest = body_hash(content, symbol, kdl=file.endswith(".kdl"))
            if digest is not None:
                key = lock_key(package, file, symbol)
                hashes[key] = digest
                resolved.append(key)
        if resolved:
            keys[(file, symbol)] = resolved
        else:
            missing.append((file, symbol))
    return hashes, keys, missing


def read_lock(path: Path = LOCK) -> tuple[str, dict[str, str]]:
    if not path.exists():
        return "", {}
    data = json.loads(path.read_text("utf-8"))
    return str(data["omp_version"]), {str(k): str(v) for k, v in data["bodies"].items()}


def write_lock(version: str, hashes: dict[str, str], path: Path = LOCK) -> None:
    data = {"omp_version": version, "bodies": dict(sorted(hashes.items()))}
    path.write_text(json.dumps(data, indent=2) + "\n", "utf-8")


def compare_bodies(
    anchors: list[Anchor],
    hashes: dict[str, str],
    keys: dict[tuple[str, str], list[str]],
    locked: dict[str, str],
) -> tuple[list[tuple[str, list[Anchor]]], list[tuple[str, list[Anchor]]], list[str]]:
    """(drifted keys, keys absent from the lock, lock keys no anchor uses any more).

    Each problem key carries the anchors that depend on it: the review starts there. A
    locked key whose anchor is still there but could not be extracted this time is not
    stale: it is already reported as not extractable.
    """
    by_pair: dict[tuple[str, str], list[Anchor]] = {}
    for anchor in anchors:
        by_pair.setdefault((anchor.file, anchor.symbol), []).append(anchor)
    drift: list[tuple[str, list[Anchor]]] = []
    unlocked: list[tuple[str, list[Anchor]]] = []
    for pair, pair_keys in sorted(keys.items()):
        for key in pair_keys:
            if key not in locked:
                unlocked.append((key, by_pair[pair]))
            elif locked[key] != hashes[key]:
                drift.append((key, by_pair[pair]))
    used = {key for pair_keys in keys.values() for key in pair_keys}
    stale = sorted(key for key in set(locked) - used if _pair(key) not in by_pair)
    return drift, unlocked, stale


def latest_version() -> str:
    with urllib.request.urlopen(f"{REGISTRY}/{PACKAGES[0]}", timeout=60) as response:
        return str(json.load(response)["dist-tags"]["latest"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the `# omp:` anchors against OMP.")
    parser.add_argument("--update", action="store_true", help="regenerate the lock")
    parser.add_argument("--omp-version", default=OMP_VERSION, help="OMP version to fetch")
    parser.add_argument("--lock", type=Path, default=LOCK, help="lock file")
    args = parser.parse_args(argv)
    version: str = args.omp_version

    anchors = collect_anchors()
    if not anchors:
        print("no `# omp:` anchors found")
        return 0

    packages = fetch_packages(version)
    sources = merge_sources(packages)
    names = " + ".join(p.split("/")[-1] for p in PACKAGES)
    print(f"{names}@{version}: {len(sources)} files, {len(anchors)} anchors\n")

    broken: list[tuple[Anchor, str]] = []
    for anchor in anchors:
        content = sources.get(anchor.file)
        if content is None:
            broken.append((anchor, "file does not exist"))
        elif anchor.symbol.split(".")[-1] not in content:
            broken.append((anchor, "symbol missing"))
        elif anchor.value and (found := _literal(content, anchor.symbol)) != anchor.value:
            actual = found or "(not a literal assignment)"
            broken.append((anchor, f"value changed: upstream has {actual}"))
        else:
            print(f"  ok    {anchor}")
    for anchor, reason in broken:
        print(f"  FAIL  {anchor}  ({reason})")

    hashes, keys, missing = hash_anchors(anchors, packages)
    failed = bool(broken)
    for file, symbol in missing:
        where = ", ".join(
            f"{a.path.name}:{a.line}" for a in anchors if (a.file, a.symbol) == (file, symbol)
        )
        print(f"  FAIL  {file} :: {symbol}  (body cannot be extracted; anchored at {where})")
        failed = True

    latest = latest_version()
    if latest != OMP_VERSION:
        print(f"\nwarning: {PACKAGES[0]}@{latest} available (pinned: {OMP_VERSION})")

    if args.update:
        if version != OMP_VERSION:
            print(f"\n--update writes the lock for the pinned {OMP_VERSION} only")
            return 1
        write_lock(version, hashes, args.lock)
        print(f"\n{len(hashes)} bodies written to {args.lock.name} for {version}")
        return 1 if failed else 0

    locked_version, locked = read_lock(args.lock)
    if locked_version != version:
        print(
            f"\n  FAIL  {args.lock.name} was generated for {locked_version or 'nothing'},"
            f" checking {version}: review the drift below, then run --update"
        )
        failed = True
    drift, unlocked, stale = compare_bodies(anchors, hashes, keys, locked)
    for key, dependants in drift:
        where = ", ".join(f"{a.path.name}:{a.line}" for a in dependants)
        print(f"  BODY DRIFT  {key}  (review {where})")
    for key, dependants in unlocked:
        where = ", ".join(f"{a.path.name}:{a.line}" for a in dependants)
        print(f"  FAIL  {key}  (not in {args.lock.name}: run --update; anchored at {where})")
    for key in stale:
        print(f"  FAIL  {key}  (in {args.lock.name} but no anchor uses it: run --update)")
    failed = failed or bool(drift or unlocked or stale)

    if failed:
        print(
            f"\n{len(broken)} anchor(s) without a match, {len(missing)} body(ies) not"
            f" extractable, {len(drift)} body drift, {len(unlocked) + len(stale)} lock"
            " mismatch(es)"
        )
        return 1
    print(f"\n{len(anchors)} anchors verified, {len(hashes)} bodies unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
