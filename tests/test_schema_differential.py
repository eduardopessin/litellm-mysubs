"""Diferenciação do normalizador de schema contra o TypeScript real.

Os testes dirigidos em ``test_schema.py`` afirmam regras nomeadas — o que deve acontecer a
um `anyOf`, a um `$ref`, a um `not`. Este afirma outra coisa: que para **qualquer** schema
o resultado é byte-a-byte o do OMP.

As 400 entradas foram geradas com um gerador determinístico e passadas pelo
``normalizeSchemaForCCA`` real, corrido em Node a partir do tarball do
``@oh-my-pi/pi-ai`` 18.2.6. As saídas gravadas são o que o TypeScript produziu, não o que
o Python produz — regenerá-las a partir do Python tornaria o teste circular e inútil.

Regenerar quando a versão fixada do OMP subir::

    cd /tmp/ccaharness
    cp tests/fixtures/cca_schema_cases.json cases.json
    node --experimental-strip-types run.ts > tests/fixtures/cca_schema_expected.json

Um teste destes paga-se onde os dirigidos não chegam: apanhou uma guarda de ciclo que
usava ``id()`` como identidade e truncava nós distintos que reutilizavam o mesmo endereço
de memória — um schema com `not` residual passava a válido e seguia para o fio.
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
    """Um corpus desemparelhado silencia o teste em vez de o fazer falhar."""
    assert len(CASES) == len(EXPECTED)
    assert CASES, "corpus vazio"


@pytest.mark.parametrize("index", range(len(CASES)))
def test_matches_the_typescript_output(index: int) -> None:
    """Saída idêntica à do ``normalizeSchemaForCCA`` real."""
    produced = normalize_for_cca(CASES[index])
    assert json.dumps(produced, sort_keys=True) == json.dumps(EXPECTED[index], sort_keys=True)
