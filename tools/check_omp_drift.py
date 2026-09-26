"""Check the `# omp:` anchors against the real OMP package.

The wiring comes from ``@oh-my-pi/pi-ai``; this package is a port to Python. Without
checking, a rename on their side leaves the anchors pointing at nothing and nobody notices
until someone tries to follow one.

    python tools/check_omp_drift.py            # check the pinned version
    python tools/check_omp_drift.py --update   # only report the latest version

Exits with 1 if any annotated symbol has disappeared. A new version on npm is a warning,
not an error: upgrading is a decision, not an obligation.
"""

from __future__ import annotations

import io
import json
import re
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Final, NamedTuple

#: OMP version the anchors were written against.
OMP_VERSION = "18.3.2"

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

SRC = Path(__file__).resolve().parent.parent / "src"


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


def fetch_sources(version: str) -> dict[str, str]:
    """``src/`` files from the tarballs, indexed by relative path.

    The three packages **share paths** — `index.ts` exists in all three, `stream.ts` in
    `pi-ai` and `pi-utils`, `types.ts` and `utils.ts` in `pi-ai` and `pi-catalog`. Merging
    them carelessly makes the last one shadow the earlier ones, and an anchor for a symbol
    in the shadowed file fails as if it did not exist. The content is concatenated instead
    of replaced: the question being asked is "does this symbol exist at this path", and the
    right answer is yes when it exists in any of the packages.
    """
    sources: dict[str, str] = {}
    for package in PACKAGES:
        for path, content in _fetch_package_sources(package, version).items():
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


def latest_version() -> str:
    with urllib.request.urlopen(f"{REGISTRY}/{PACKAGES[0]}", timeout=60) as response:
        return str(json.load(response)["dist-tags"]["latest"])


def main() -> int:
    latest = latest_version()
    if "--update" in sys.argv:
        print(f"pinned: {OMP_VERSION}\nlatest: {latest}")
        return 0

    anchors = collect_anchors()
    if not anchors:
        print("no `# omp:` anchors found")
        return 0

    sources = fetch_sources(OMP_VERSION)
    names = " + ".join(p.split("/")[-1] for p in PACKAGES)
    print(f"{names}@{OMP_VERSION}: {len(sources)} files, {len(anchors)} anchors\n")

    broken: list[tuple[Anchor, str]] = []
    for anchor in anchors:
        content = sources.get(anchor.file)
        if content is None:
            broken.append((anchor, "file does not exist"))
        elif anchor.symbol not in content:
            broken.append((anchor, "symbol missing"))
        elif anchor.value and (found := _literal(content, anchor.symbol)) != anchor.value:
            actual = found or "(not a literal assignment)"
            broken.append((anchor, f"value changed: upstream has {actual}"))
        else:
            print(f"  ok    {anchor}")

    for anchor, reason in broken:
        print(f"  FAIL  {anchor}  ({reason})")

    if latest != OMP_VERSION:
        print(f"\nwarning: {PACKAGES[0]}@{latest} available (pinned: {OMP_VERSION})")

    if broken:
        print(f"\n{len(broken)} anchor(s) without a match")
        return 1
    print(f"\n{len(anchors)} anchors verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
