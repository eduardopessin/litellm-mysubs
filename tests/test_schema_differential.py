"""Differential testing of the schema normalizer against the real TypeScript.

The targeted tests in ``test_schema.py`` assert named rules — what must happen to an
`anyOf`, a `$ref`, a `not`. This one asserts something else: that for **any** schema the
result is byte-for-byte the one OMP produces.

The 400 entries were generated with a deterministic generator and passed through the real
``normalizeSchemaForCCA``, run in Node from the ``@oh-my-pi/pi-ai`` 18.2.6 tarball. The
recorded outputs are what the TypeScript produced, not what the Python produces —
regenerating them from the Python would make the test circular and useless.

Regenerate when the pinned OMP version goes up::

    cd /tmp/ccaharness
    cp tests/fixtures/cca_schema_cases.json cases.json
    node --experimental-strip-types run.ts > tests/fixtures/cca_schema_expected.json

A test like this pays for itself where the targeted ones do not reach: it caught a cycle
guard that used ``id()`` as identity and truncated distinct nodes that reused the same
memory address — a schema with a residual `not` became valid and went out on the wire.

The second corpus covers the full path a tool schema takes to Antigravity, as omp builds
it (``convertTools`` + ``normalizeAntigravityTools``): ``toolWireSchema`` first, then
``normalizeSchemaForGoogle`` and ``normalizeSchemaForCCA`` for Gemini, and
``normalizeSchemaForCCA`` alone for Claude. ``tool_schema_cases.json`` is every object case
above plus pydantic/ArkType shapes the generator does not produce (``T | None`` unions,
described ``const`` unions, bare enums, ``name /** doc */`` keys). The expectations in
``tool_schema_expected.json`` were produced by the TypeScript of 18.4.4, run under Bun with
two stubs — ``@oh-my-pi/pi-utils`` (``logger``, ``isRecord``, ``structuredCloneJSON``) and
``../../error`` (``ValidationError``) — around ``pi-ai/src/utils/schema/*.ts``::

    bun run.ts tests/fixtures 18.4.4 > tests/fixtures/tool_schema_expected.json

where ``run.ts`` maps ``normalizeSchemaForGoogle`` over ``cca_schema_cases.json`` and
``toolWireSchema`` / the two pipelines over ``tool_schema_cases.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from litellm_mysubs.wire.schema import normalize_for_cca, normalize_for_google, tool_wire_schema

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> list[Any]:
    return list(json.loads((FIXTURES / name).read_text("utf-8")))


CASES = _load("cca_schema_cases.json")
EXPECTED = _load("cca_schema_expected.json")
TOOL_CASES = _load("tool_schema_cases.json")
TOOL_EXPECTED: dict[str, Any] = json.loads(
    (FIXTURES / "tool_schema_expected.json").read_text("utf-8")
)


def _dump(value: object) -> str:
    return json.dumps(value, sort_keys=True)


def _drop_undefined_type(value: object) -> object:
    """omp assigns ``undefined`` to ``type`` when a type array holds no string, and
    ``JSON.stringify`` drops the key; the port holds ``None`` there. Same in-memory
    meaning — the CCA pass that always follows rejects both alike — so the Google-only
    comparison reads a ``None`` type as absent."""
    if isinstance(value, list):
        return [_drop_undefined_type(v) for v in value]
    if isinstance(value, dict):
        return {
            k: _drop_undefined_type(v)
            for k, v in value.items()
            if not (k == "type" and v is None)
        }
    return value


def test_corpus_is_paired() -> None:
    """An unpaired corpus silences the test instead of failing it."""
    assert len(CASES) == len(EXPECTED)
    assert CASES, "empty corpus"


@pytest.mark.parametrize("index", range(len(CASES)))
def test_matches_the_typescript_output(index: int) -> None:
    """Output identical to the real ``normalizeSchemaForCCA``."""
    produced = normalize_for_cca(CASES[index])
    assert json.dumps(produced, sort_keys=True) == json.dumps(EXPECTED[index], sort_keys=True)


def test_tool_corpus_is_paired() -> None:
    assert TOOL_EXPECTED["omp_version"] == "18.4.4"
    assert len(TOOL_EXPECTED["google"]) == len(CASES)
    for path in ("wire", "gemini", "claude"):
        assert len(TOOL_EXPECTED[path]) == len(TOOL_CASES), path


@pytest.mark.parametrize("index", range(len(CASES)))
def test_google_matches_the_typescript_output(index: int) -> None:
    """Output identical to the real ``normalizeSchemaForGoogle``."""
    produced = _drop_undefined_type(normalize_for_google(CASES[index]))
    assert _dump(produced) == _dump(TOOL_EXPECTED["google"][index])


@pytest.mark.parametrize("index", range(len(TOOL_CASES)))
def test_tool_wire_schema_matches_the_typescript_output(index: int) -> None:
    """Output identical to the real ``toolWireSchema``, with the input left untouched."""
    case = TOOL_CASES[index]
    before = _dump(case)
    assert _dump(tool_wire_schema(case)) == _dump(TOOL_EXPECTED["wire"][index])
    assert _dump(case) == before


@pytest.mark.parametrize("index", range(len(TOOL_CASES)))
def test_gemini_parameters_match_the_typescript_output(index: int) -> None:
    """``parameters`` omp sends for a Gemini model on Antigravity."""
    produced = normalize_for_cca(normalize_for_google(tool_wire_schema(TOOL_CASES[index])))
    assert _dump(produced) == _dump(TOOL_EXPECTED["gemini"][index])


@pytest.mark.parametrize("index", range(len(TOOL_CASES)))
def test_claude_parameters_match_the_typescript_output(index: int) -> None:
    """``parameters`` omp sends for a Claude model on Antigravity."""
    produced = normalize_for_cca(tool_wire_schema(TOOL_CASES[index]))
    assert _dump(produced) == _dump(TOOL_EXPECTED["claude"][index])
