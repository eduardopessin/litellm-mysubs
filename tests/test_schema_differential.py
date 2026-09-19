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
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from litellm_mysubs.wire.schema import normalize_for_cca

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> list[Any]:
    return list(json.loads((FIXTURES / name).read_text("utf-8")))


CASES = _load("cca_schema_cases.json")
EXPECTED = _load("cca_schema_expected.json")


def test_corpus_is_paired() -> None:
    """An unpaired corpus silences the test instead of failing it."""
    assert len(CASES) == len(EXPECTED)
    assert CASES, "empty corpus"


@pytest.mark.parametrize("index", range(len(CASES)))
def test_matches_the_typescript_output(index: int) -> None:
    """Output identical to the real ``normalizeSchemaForCCA``."""
    produced = normalize_for_cca(CASES[index])
    assert json.dumps(produced, sort_keys=True) == json.dumps(EXPECTED[index], sort_keys=True)
