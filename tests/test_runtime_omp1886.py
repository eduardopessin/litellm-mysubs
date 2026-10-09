"""omp 18.8.6: the thinking-loop heuristics judge prose only; an identical CAS is not rewritten.

The detector cases were run through omp's own `ThinkingLoopDetector` (18.4.4 and 18.8.6)
with Bun: every code-heavy case below tripped 18.4.4 ("8 near-identical segments within the
last 16") and passes 18.8.6, and both prose loops trip both — the same verdicts as here.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from litellm_mysubs.credentials.file_store import FileCredentialStore
from litellm_mysubs.credentials.store import Credential
from litellm_mysubs.wire.thinking_loop import _CODE_LINE, ThinkingLoopDetector


def _verdict(chunks: list[str]) -> str | None:
    detector = ThinkingLoopDetector()
    for chunk in chunks:
        if reason := detector.feed(chunk):
            return reason
    return detector.flush()


def _svg(i: int) -> str:
    return (
        "Drafting the next row of tiles.\n"
        f'<rect x="{i * 40}" y="{i * 20}" width="40" height="20" fill="#3366cc" '
        'stroke="black" stroke-width="2" />\n'
        f'<circle cx="{i * 40 + 20}" cy="{i * 20 + 10}" r="8" fill="white" stroke="black" />\n'
        f'<text x="{i * 40 + 4}" y="{i * 20 + 14}" font-size="12" '
        'font-family="sans-serif">Tile</text>\n\n'
    )


def _vrml(i: int) -> str:
    return (
        f"Transform {{\n  translation {i} {i * 2} {i * 3}\n  children [\n    Shape {{\n"
        f"      appearance Appearance {{ material Material {{ diffuseColor 0.{i} 0.2 0.8 }} }}\n"
        f"      geometry Box {{ size {i + 1} 1 1 }}\n    }}\n  ]\n}}\n\n"
    )


def _json(i: int) -> str:
    row = (
        f'{{"id": {i}, "name": "item", "price": {i * 3}.5, "tags": ["sale"], '
        f'"stock": {i * 7}}},\n'
    )
    return "```json\n" + row * 3 + "```\n\n"


_PROSE = (
    "I need to make sure the configuration file is loaded before the server starts "
    "and then verify the handler again.\n\n"
)


class TestCodeIsNotALoop:
    """omp 18.8: "false thinking-loop detections ... repetitive code or markup (VRML, SVG,
    JSON)". The same skeleton with different literals normalizes to the same trigrams."""

    @pytest.mark.parametrize(
        "chunks",
        [
            pytest.param([_svg(i) for i in range(12)], id="svg"),
            pytest.param([_vrml(i) for i in range(12)], id="vrml"),
            pytest.param([_json(i) for i in range(12)], id="json"),
            # Gemini summaries end inside a fence they never close: classified per line.
            pytest.param(["```svg\n" + _svg(i) for i in range(12)], id="unclosed-fence"),
        ],
    )
    def test_code_drafts_do_not_trip(self, chunks: list[str]) -> None:
        assert _verdict(chunks) is None

    def test_a_prose_loop_still_trips(self) -> None:
        assert _verdict([_PROSE] * 12) == "8 near-identical segments within the last 16"

    def test_a_prose_loop_between_code_still_trips(self) -> None:
        chunks = [_PROSE + _svg(i) for i in range(12)]
        assert _verdict(chunks) == "8 near-identical segments within the last 16"


class TestCodeLine:
    """Outputs of omp's `CODE_LINE` (JavaScript) on the same strings."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("  return x;\nplain prose", "\nplain prose"),
            ("  - nested item\n  1. step\n\tbody()", "  - nested item\n  1. step\n"),
            ("```python\nprose ends here.", "\nprose ends here."),
            ("a list of things,\nwith a wrapped line", "\nwith a wrapped line"),
            # JavaScript's `.` and multiline `$` stop at \r and U+2028 as well.
            ("foo;\r\nbar baz", "\r\nbar baz"),
            ("x = 1,\u2028prose here", "\u2028prose here"),
            ("  indented\r\n- item", "\r\n- item"),
        ],
    )
    def test_matches_omp(self, text: str, expected: str) -> None:
        assert _CODE_LINE.sub("", text) == expected


class TestIdenticalCompareAndSet:
    """`tryUpdateAuthCredentialIfMatches`: the same bytes again match without a write."""

    def test_identical_credential_is_not_rewritten(self, tmp_path: Path) -> None:
        path = tmp_path / "mysubs" / "credentials.json"
        store = FileCredentialStore(path)
        current = Credential("anthropic", "at-1", "rt-1", 2_000_000_000.0)
        store.set("anthropic", current)
        before = os.stat(path)

        assert store.update_if_matches("anthropic", current, Credential(**_fields(current)))

        after = os.stat(path)
        assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)

    def test_a_stale_expectation_still_loses(self, tmp_path: Path) -> None:
        store = FileCredentialStore(tmp_path / "mysubs" / "credentials.json")
        current = Credential("anthropic", "at-2", "rt-2", 2_000_000_000.0)
        store.set("anthropic", current)
        stale = Credential("anthropic", "at-1", "rt-1", 1_000_000_000.0)

        assert not store.update_if_matches("anthropic", stale, stale)
        assert store.get("anthropic") == current


def _fields(credential: Credential) -> dict[str, object]:
    return {
        "provider": credential.provider,
        "access_token": credential.access_token,
        "refresh_token": credential.refresh_token,
        "expires_at": credential.expires_at,
        "project_id": credential.project_id,
    }
