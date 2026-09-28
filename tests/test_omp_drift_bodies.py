"""Body drift: the anchored declaration is hashed, not just its name.

A name anchor stays green while OMP rewrites the function behind it - `applyHeadCaching` and
`fetchClaudeUsage` changed between 18.3.2 and 18.4.1 under the same names. The checker now
cuts each anchored declaration out of the TypeScript (or KDL) source, normalizes what a
formatter can change, and compares a SHA-256 against `tools/omp_anchors.lock`.

What these tests pin is what makes that comparison worth trusting: a comment or a re-wrap
must not read as drift, a real change must, a brace inside a string/template/regex must not
cut a function short, and a symbol that cannot be found must be reported, never skipped.
No network: synthetic sources, and `fetch_packages` replaced for the end-to-end runs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("check_omp_drift", TOOLS / "check_omp_drift.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_omp_drift"] = module
    spec.loader.exec_module(module)
    return module


drift = _load()

SOURCE = """\
import { x } from "./x";

// Leading comment that is not part of anything.
export function applyHeadCaching(blocks: Block[], ttl?: string): void {
\tconst marker = { type: "ephemeral", ...(ttl && { ttl }) };
\tfor (const block of blocks) {
\t\tblock.cache_control = marker;
\t}
}

function next(): number {
\treturn 1;
}
"""

REFORMATTED = """\
import { x } from "./x";

/** A doc comment added upstream. */
export function applyHeadCaching(
\tblocks: Block[],
\tttl?: string,
): void {
\t// explains the marker
\tconst marker = {
\t\ttype: "ephemeral",
\t\t...(ttl && { ttl }),
\t};
\tfor (const block of blocks) { block.cache_control = marker; } /* trailing */
}
"""


class TestNormalization:
    """What a formatter or a comment changes is not drift."""

    def test_comments_whitespace_and_wrapping_keep_the_hash(self) -> None:
        assert drift.body_hash(SOURCE, "applyHeadCaching") == drift.body_hash(
            REFORMATTED, "applyHeadCaching"
        )

    def test_a_changed_statement_changes_the_hash(self) -> None:
        changed = SOURCE.replace("block.cache_control = marker;", "block.cache = marker;")
        assert drift.body_hash(changed, "applyHeadCaching") != drift.body_hash(
            SOURCE, "applyHeadCaching"
        )

    def test_a_changed_literal_changes_the_hash(self) -> None:
        """The value is the wire: `"ephemeral"` -> `"persistent"` is a real change."""
        changed = SOURCE.replace('"ephemeral"', '"persistent"')
        assert drift.body_hash(changed, "applyHeadCaching") != drift.body_hash(
            SOURCE, "applyHeadCaching"
        )

    def test_whitespace_inside_a_literal_is_kept(self) -> None:
        """Only code is squeezed: a header value with a space is not one without."""
        before = 'const UA = "omp cli";'
        after = 'const UA = "omp  cli";'
        assert drift.body_hash(before, "UA") != drift.body_hash(after, "UA")

    def test_a_space_that_separates_two_words_is_kept(self) -> None:
        """`typeof x` and `typeofx` are different programs."""
        assert drift.extract("function f() { return typeof x; }", "f") == (
            "function f(){return typeof x;}"
        )


class TestExtraction:
    """Where a declaration ends, whatever its body holds."""

    def test_the_following_declaration_is_not_included(self) -> None:
        body = drift.extract(SOURCE, "applyHeadCaching")
        assert body is not None
        assert body.endswith("}}")
        assert "next" not in body

    @pytest.mark.parametrize(
        "statement",
        [
            'const s = "}";',
            "const s = '{{';",
            "const s = `}${a ? { b: 1 }.b : `}`}}`;",
            "const r = /[{]}/g;",
            "if (a / b > 1) return;",
            "// }\n",
            "/* } */",
        ],
    )
    def test_braces_in_literals_regexes_and_comments_do_not_count(self, statement: str) -> None:
        source = f"function f() {{\n\t{statement}\n\tlast();\n}}\n\nfunction g() {{}}\n"
        body = drift.extract(source, "f")
        assert body is not None
        assert body.endswith("last();}")
        assert "function g" not in body

    def test_a_return_type_literal_is_not_the_body(self) -> None:
        source = "function f(): { a: number } {\n\treturn { a: 1 };\n}\nconst after = 2;\n"
        assert drift.extract(source, "f") == "function f():{a:number}{return{a:1};}"

    def test_a_const_ends_at_its_semicolon(self) -> None:
        source = "const A: Record<string, number> = { a: 1, b: 2 };\nconst B = 3;\n"
        assert drift.extract(source, "A") == "const A:Record<string,number>={a:1,b:2};"

    def test_a_class_method(self) -> None:
        source = "class C {\n\tgenerateState(): string {\n\t\treturn 'x';\n\t}\n\tother() {}\n}\n"
        assert drift.extract(source, "generateState") == "generateState():string{return'x';}"

    def test_a_call_at_line_start_is_not_a_method(self) -> None:
        source = "function run() {\n\tcloseBlock(1);\n}\n"
        assert drift.extract(source, "closeBlock") is None

    def test_an_object_property(self) -> None:
        source = 'export const HEADERS = {\n\tBETA: "x",\n\tORIGINATOR: "omp",\n} as const;\n'
        assert drift.extract(source, "ORIGINATOR") == 'ORIGINATOR:"omp"'
        assert drift.extract(source, "HEADERS.BETA") == 'BETA:"x"'

    def test_a_missing_symbol_is_none(self) -> None:
        assert drift.extract(SOURCE, "applyHeadCachingV2") is None
        assert drift.body_hash(SOURCE, "applyHeadCachingV2") is None


class TestKdl:
    """The auth rules are KDL: nodes, not declarations."""

    RULE = """\
auth "x" {
\tlogin "oauth-code" {
\t\tclient-id "QUJDREVGR0hJSktM" encoding="base64"
\t\ttoken url="https://a/token" body="json"
\t\tcredential {
\t\t\taccess "access_token"
\t\t}
\t}
\trefresh {
\t\t// comment
\t\ttoken url="https://a/token" body="form"
\t}
}
"""

    def test_every_node_with_the_name_counts(self) -> None:
        body = drift.extract(self.RULE, "token", kdl=True)
        assert body == (
            'token url="https://a/token"body="json"\ntoken url="https://a/token"body="form"'
        )

    def test_a_name_wins_over_a_substring_of_a_value(self) -> None:
        """`token` is a node and also sits inside `"access_token"`."""
        body = drift.extract(self.RULE, "token", kdl=True)
        assert body is not None
        assert "access" not in body

    def test_a_value_prefix_selects_its_node(self) -> None:
        body = drift.extract(self.RULE, "QUJDREVG", kdl=True)
        assert body == 'client-id"QUJDREVGR0hJSktM"encoding="base64"'

    def test_children_belong_to_the_node(self) -> None:
        assert drift.extract(self.RULE, "credential", kdl=True) == (
            'credential{access"access_token"}'
        )


def _anchors(tmp_path: Path, source: str) -> list[Any]:
    (tmp_path / "module.py").write_text(source, encoding="utf-8")
    original = drift.SRC
    drift.SRC = tmp_path
    try:
        return list(drift.collect_anchors())
    finally:
        drift.SRC = original


class TestComparison:
    def test_drift_names_the_anchors_that_depend_on_it(self, tmp_path: Path) -> None:
        anchors = _anchors(tmp_path, "# omp: p.ts :: f\nx = 1\n# omp: p.ts :: f\n")
        packages = {"pi-ai": {"p.ts": "function f() { return 2; }"}}
        hashes, keys, missing = drift.hash_anchors(anchors, packages)
        locked = {"pi-ai/p.ts::f": drift.body_hash("function f() { return 1; }", "f")}
        found, unlocked, stale = drift.compare_bodies(anchors, hashes, keys, locked)
        assert missing == []
        assert [(key, [a.line for a in deps]) for key, deps in found] == [
            ("pi-ai/p.ts::f", [1, 3])
        ]
        assert unlocked == stale == []

    def test_an_unextractable_symbol_is_missing_not_stale(self, tmp_path: Path) -> None:
        """Reported once, as what it is: an anchor whose body cannot be read."""
        anchors = _anchors(tmp_path, "# omp: p.ts :: f\n")
        packages = {"pi-ai": {"p.ts": "f();"}}
        hashes, keys, missing = drift.hash_anchors(anchors, packages)
        found, unlocked, stale = drift.compare_bodies(anchors, hashes, keys, {"pi-ai/p.ts::f": "0"})
        assert missing == [("p.ts", "f")]
        assert found == unlocked == stale == []

    def test_a_new_anchor_and_a_dropped_one(self, tmp_path: Path) -> None:
        anchors = _anchors(tmp_path, "# omp: p.ts :: f\n")
        packages = {"pi-ai": {"p.ts": "function f() {}"}}
        hashes, keys, _ = drift.hash_anchors(anchors, packages)
        found, unlocked, stale = drift.compare_bodies(
            anchors, hashes, keys, {"pi-ai/p.ts::gone": "0"}
        )
        assert found == []
        assert [key for key, _ in unlocked] == ["pi-ai/p.ts::f"]
        assert stale == ["pi-ai/p.ts::gone"]

    def test_shared_paths_are_hashed_per_package(self, tmp_path: Path) -> None:
        """`index.ts` exists in all three packages: the key says which one was read."""
        anchors = _anchors(tmp_path, "# omp: index.ts :: f\n")
        packages = {
            "pi-ai": {"index.ts": "export * from './a';"},
            "pi-utils": {"index.ts": "export function f() {}"},
        }
        hashes, _, missing = drift.hash_anchors(anchors, packages)
        assert list(hashes) == ["pi-utils/index.ts::f"]
        assert missing == []


class TestRun:
    """`main` end to end, with the npm download replaced by synthetic packages."""

    @pytest.fixture
    def run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
        src = tmp_path / "src"
        src.mkdir()
        (src / "module.py").write_text("# omp: p.ts :: f\ndef f(): ...\n", encoding="utf-8")
        monkeypatch.setattr(drift, "SRC", src)
        monkeypatch.setattr(drift, "latest_version", lambda: drift.OMP_VERSION)
        lock = tmp_path / "omp_anchors.lock"
        body = {"text": "function f() { return 1; }"}

        def fetch(version: str) -> dict[str, dict[str, str]]:
            return {"pi-ai": {"p.ts": body["text"]}}

        monkeypatch.setattr(drift, "fetch_packages", fetch)

        def invoke(*args: str) -> int:
            return int(drift.main([*args, "--lock", str(lock)]))

        invoke.body = body  # type: ignore[attr-defined]
        return invoke

    def test_update_then_check_is_green(self, run: Any) -> None:
        assert run("--update") == 0
        assert run() == 0

    def test_a_changed_body_fails_with_body_drift(
        self, run: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert run("--update") == 0
        run.body["text"] = "function f() { return 2; }"
        assert run() == 1
        assert "BODY DRIFT  pi-ai/p.ts::f  (review module.py:1)" in capsys.readouterr().out

    def test_a_reformatted_body_stays_green(self, run: Any) -> None:
        assert run("--update") == 0
        run.body["text"] = "// new comment\nfunction f() {\n\treturn 1;\n}\n"
        assert run() == 0

    def test_an_unextractable_body_fails(
        self, run: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        run.body["text"] = "f();"
        assert run("--update") == 1
        assert "body cannot be extracted" in capsys.readouterr().out

    def test_no_lock_fails(self, run: Any) -> None:
        assert run() == 1

    def test_a_lock_for_another_version_fails(self, run: Any) -> None:
        """A pin bump has to regenerate the lock: equal hashes do not excuse the label."""
        assert run("--update") == 0
        assert run("--omp-version", "0.0.1") == 1
