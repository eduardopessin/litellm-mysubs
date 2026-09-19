"""The anchor checker, tested without the network.

This file exists for a measured reason: the `# omp:` anchors prove that a **name** exists
on the OMP side, and nothing more. The failure mode that matters is the other one - the
name stays the same and the **value** changes. An endpoint path, a client id, a pinned
client version. Nothing breaks at import time; it breaks against the upstream, in
production, with CI green.

That is how `wham/usage` came to be wrong: the symbol existed, the call returned 403.

The tests run against synthetic sources - the network stays out of CI, and what is checked
is the checker's logic, not the availability of npm.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"


def _load() -> Any:
    """The checker lives in `tools/`, outside the package: it is loaded by path."""
    spec = importlib.util.spec_from_file_location("check_omp_drift", TOOLS / "check_omp_drift.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_omp_drift"] = module
    spec.loader.exec_module(module)
    return module


drift = _load()


def _anchors(tmp_path: Path, source: str) -> list[Any]:
    """Collect anchors from a hand-written file, pointing `SRC` at it."""
    (tmp_path / "module.py").write_text(source, encoding="utf-8")
    original = drift.SRC
    drift.SRC = tmp_path
    try:
        return list(drift.collect_anchors())
    finally:
        drift.SRC = original


class TestWireValue:
    """What the value anchor adds to the name anchor."""

    def test_an_unchanged_value_passes(self) -> None:
        content = 'const CODEX_USAGE_PATH = "wham/usage";'
        assert drift._literal(content, "CODEX_USAGE_PATH") == '"wham/usage"'

    def test_a_changed_value_is_visible(self) -> None:
        """The real case: the name survives, the value does not.

        Without this the checker said `ok` while the plugin hit a 404.
        """
        content = 'const CODEX_USAGE_PATH = "wham/v2/usage";'
        assert drift._literal(content, "CODEX_USAGE_PATH") == '"wham/v2/usage"'

    @pytest.mark.parametrize(
        "content",
        [
            'const X: string = "v";',
            "let X = 'v';",
            "var X = `v`;",
        ],
    )
    def test_declaration_forms(self, content: str) -> None:
        """TypeScript writes the same constant in several ways; all of them count."""
        assert drift._literal(content, "X") == '"v"'

    def test_a_composed_value_is_not_a_literal(self) -> None:
        """`${BASE}/x` is not comparable: saying it is unknown beats making it up."""
        assert drift._literal("const URL = `${BASE}/v1:load`;", "URL") == ""

    def test_a_missing_symbol(self) -> None:
        assert drift._literal("const OTHER = 'v';", "X") == ""

    def test_a_prefix_does_not_count_as_the_symbol(self) -> None:
        """`X` must not match `XY`: an accidental prefix gave a false green."""
        assert drift._literal("const XY = 'v';", "X") == ""


class TestCollection:
    """How the two anchor forms are read out of the code."""

    def test_a_name_anchor(self, tmp_path: Path) -> None:
        anchors = _anchors(tmp_path, "# omp: providers/codex.ts :: buildUrl\n")
        assert [(a.file, a.symbol, a.value) for a in anchors] == [
            ("providers/codex.ts", "buildUrl", "")
        ]

    def test_several_symbols_on_the_same_line(self, tmp_path: Path) -> None:
        anchors = _anchors(tmp_path, "# omp: providers/codex.ts :: a, b, c\n")
        assert [a.symbol for a in anchors] == ["a", "b", "c"]

    def test_a_complete_value_anchor(self, tmp_path: Path) -> None:
        anchors = _anchors(tmp_path, '# omp= usage/codex.ts :: PATH = "wham/usage"\n')
        assert [(a.file, a.symbol, a.value) for a in anchors] == [
            ("usage/codex.ts", "PATH", '"wham/usage"')
        ]

    def test_the_file_is_inherited_from_the_anchor_above(self, tmp_path: Path) -> None:
        """The short form: repeating a long path on both lines blows the margin."""
        anchors = _anchors(
            tmp_path,
            "# omp: usage/google-antigravity.ts :: FETCH_PATH\n"
            '# omp= FETCH_PATH = "/v1internal:fetchAvailableModels"\n',
        )
        assert [(a.file, a.symbol, a.value) for a in anchors] == [
            ("usage/google-antigravity.ts", "FETCH_PATH", ""),
            ("usage/google-antigravity.ts", "FETCH_PATH", '"/v1internal:fetchAvailableModels"'),
        ]

    def test_inheritance_does_not_cross_code(self, tmp_path: Path) -> None:
        """A line of code between the two cuts the inheritance.

        Without this, an `# omp=` stranded in the middle of the file inherited a path from
        another context and checked the wrong constant - a green that means nothing.
        """
        with pytest.raises(SystemExit, match="with no file and no anchor above"):
            _anchors(
                tmp_path,
                '# omp: usage/codex.ts :: A\ndef f() -> None: ...\n# omp= B = "v"\n',
            )


class TestVerification:
    """The checker's verdict on synthetic sources."""

    @staticmethod
    def _verdict(anchor: Any, sources: dict[str, str]) -> str:
        content = sources.get(anchor.file)
        if content is None:
            return "file does not exist"
        if anchor.symbol not in content:
            return "symbol missing"
        if anchor.value and drift._literal(content, anchor.symbol) != anchor.value:
            return "value changed"
        return "ok"

    def test_name_present_value_right(self, tmp_path: Path) -> None:
        anchor = _anchors(tmp_path, '# omp= u.ts :: P = "wham/usage"\n')[0]
        assert self._verdict(anchor, {"u.ts": 'const P = "wham/usage";'}) == "ok"

    def test_name_present_value_wrong(self, tmp_path: Path) -> None:
        """This is the line that separates this checker from the previous one."""
        anchor = _anchors(tmp_path, '# omp= u.ts :: P = "wham/usage"\n')[0]
        assert self._verdict(anchor, {"u.ts": 'const P = "wham/v2/usage";'}) == "value changed"

    def test_a_renamed_symbol(self, tmp_path: Path) -> None:
        anchor = _anchors(tmp_path, "# omp: u.ts :: buildUrl\n")[0]
        assert self._verdict(anchor, {"u.ts": "const other = 1;"}) == "symbol missing"

    def test_a_moved_file(self, tmp_path: Path) -> None:
        anchor = _anchors(tmp_path, "# omp: u.ts :: buildUrl\n")[0]
        assert self._verdict(anchor, {"other.ts": "buildUrl"}) == "file does not exist"


class TestRealAnchors:
    """The repository's own anchors, read without touching the network."""

    def test_value_anchors_exist(self) -> None:
        """If someone deletes them, the hole reopens in silence."""
        values = [a for a in drift.collect_anchors() if a.value]
        assert len(values) >= 9

    def test_pinned_values_are_literals(self) -> None:
        """A value with `${...}` would never match: catch it here and not in CI."""
        for anchor in drift.collect_anchors():
            if anchor.value:
                assert "${" not in anchor.value, anchor

    def test_the_codex_quota_path_is_pinned(self) -> None:
        """`wham/usage` was once wrong as `codex/wham/usage`, and returned 403.

        The symbol existed in both versions; only the value told them apart. That is the
        case that justifies the `# omp=` form existing.
        """
        pinned = {a.symbol: a.value for a in drift.collect_anchors() if a.value}
        assert pinned["CODEX_USAGE_PATH"] == '"wham/usage"'
