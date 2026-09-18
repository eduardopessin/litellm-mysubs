"""Equivalência entre ``wire/codex.py`` e o ``sitecustomize.py`` de onde foi extraído.

Não é um teste do pytest: precisa do ficheiro original, que vive noutro repositório e não
é dependência deste. Corre-se à mão durante uma extracção::

    python tools/check_equivalence_codex.py /caminho/para/sitecustomize.py

Compara o corpo do pedido gerado pelas duas implementações, mais a forma dos cabeçalhos.
O ``turn_id`` é excluído por ser aleatório a cada pedido.

Quando a extracção terminar e o original deixar de existir, apagar: a garantia passa então
a ser dos testes de contrato.
"""

from __future__ import annotations

import ast
import base64
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from litellm_mysubs.wire import codex as new

WANTED = {
    "_codex_token_claims",
    "_extract_account_id",
    "_codex_prompt_cache_key",
    "_content_to_text",
    "_codex_image_part",
    "_codex_file_part",
    "_content_to_codex_parts",
    "_codex_composite_call_id",
    "_codex_split_call_id",
    "_repair_codex_tool_pairs",
    "_messages_to_codex_input",
    "_tools_to_codex_tools",
    "_codex_tool_choice",
    "_codex_wire_generation",
    "_codex_request_body",
    "_normalize_effort",
    "_is_codex_model",
    "_build_codex_headers",
}

#: ``(nome, modelo, mensagens, tools, extra)``
CASES: list[tuple[str, str, list[Any], list[Any] | None, dict[str, Any]]] = [
    ("simples", "gpt-5.5", [{"role": "user", "content": "olá"}], None, {}),
    ("alias codex", "codex", [{"role": "user", "content": "x"}], None, {}),
    ("alias gpt-5", "gpt-5", [{"role": "user", "content": "x"}], None, {}),
    ("alias gpt-6", "gpt-6", [{"role": "user", "content": "x"}], None, {}),
    ("gpt-5.4 nao remapeado", "gpt-5.4", [{"role": "user", "content": "x"}], None, {}),
    (
        "system vira developer",
        "gpt-5.5",
        [{"role": "system", "content": "regra"}, {"role": "user", "content": "x"}],
        None,
        {},
    ),
    (
        "effort high",
        "gpt-5.5",
        [{"role": "user", "content": "x"}],
        None,
        {"reasoning_effort": "high"},
    ),
    (
        "effort none em 5.6 (item de juice)",
        "gpt-5.6-terra",
        [{"role": "user", "content": "x"}],
        None,
        {"reasoning_effort": "none"},
    ),
    (
        "effort none em 5.5 (sem juice)",
        "gpt-5.5",
        [{"role": "user", "content": "x"}],
        None,
        {"reasoning_effort": "none"},
    ),
    (
        "reasoning como objecto",
        "gpt-5.5",
        [{"role": "user", "content": "x"}],
        None,
        {"reasoning_effort": {"effort": "high", "summary": "detailed"}},
    ),
    (
        "summary invalido",
        "gpt-5.5",
        [{"role": "user", "content": "x"}],
        None,
        {"reasoning_effort": {"effort": "low", "summary": "xpto"}},
    ),
    (
        "tool function",
        "gpt-5.5",
        [{"role": "user", "content": "x"}],
        [
            {
                "type": "function",
                "function": {"name": "read", "description": "d", "parameters": {"type": "object"}},
            }
        ],
        {},
    ),
    ("tool hospedada", "gpt-5.5", [{"role": "user", "content": "x"}], [{"type": "web_search"}], {}),
    (
        "tool_choice funcao",
        "gpt-5.5",
        [{"role": "user", "content": "x"}],
        [{"type": "function", "function": {"name": "read", "parameters": {}}}],
        {"tool_choice": {"type": "function", "function": {"name": "read"}}},
    ),
    (
        "par de tool completo",
        "gpt-5.5",
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ],
        None,
        {},
    ),
    (
        "output orfao",
        "gpt-5.5",
        [{"role": "tool", "tool_call_id": "orphan", "content": "perdido"}],
        None,
        {},
    ),
    (
        "call sem output",
        "gpt-5.5",
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c9", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                ],
            }
        ],
        None,
        {},
    ),
    (
        "arguments como dict",
        "gpt-5.5",
        [
            {
                "role": "assistant",
                "tool_calls": [{"id": "c", "function": {"name": "f", "arguments": {"a": 1}}}],
            },
            {"role": "tool", "tool_call_id": "c", "content": "r"},
        ],
        None,
        {},
    ),
    (
        "imagem com detail original",
        "gpt-5.5",
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "cor?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,iVBOR", "detail": "original"},
                    },
                ],
            }
        ],
        None,
        {},
    ),
    (
        "ficheiro inline",
        "gpt-5.5",
        [
            {
                "role": "user",
                "content": [{"type": "file", "file": {"file_data": "b64", "filename": "a.pdf"}}],
            }
        ],
        None,
        {},
    ),
    (
        "id composto",
        "gpt-5.5",
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1|item-9",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1|item-9", "content": "r"},
        ],
        None,
        {},
    ),
    (
        "service_tier",
        "gpt-5.5",
        [{"role": "user", "content": "x"}],
        None,
        {"service_tier": "priority"},
    ),
    ("role desconhecido", "gpt-5.5", [{"role": "xpto", "content": "x"}], None, {}),
    (
        "conversa longa",
        "gpt-5.5",
        [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(8)],
        None,
        {},
    ),
]


def load_original(path: str) -> dict[str, Any]:
    """Compila só as funções do caminho do pedido, sem importar o módulo."""
    tree = ast.parse(Path(path).read_text("utf-8"))
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in WANTED
    ]
    missing = WANTED - {f.name for f in functions}
    if missing:
        raise SystemExit(f"funções ausentes no original: {sorted(missing)}")

    constants = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            getattr(target, "id", "").startswith(("_CODEX", "_ANTHROPIC"))
            for target in node.targets
        )
    ]
    namespace: dict[str, Any] = {
        "json": json,
        "hashlib": hashlib,
        "base64": base64,
        "sys": sys,
        "uuid": type(
            "Uuid",
            (),
            {
                "uuid4": staticmethod(
                    lambda: type("U", (), {"hex": "deadbeefcafe", "__str__": lambda s: "fixed"})()
                )
            },
        )(),
        "httpx": type("Httpx", (), {"Timeout": staticmethod(lambda *a, **k: None)})(),
        "os": type("Os", (), {"environ": {}})(),
    }
    exec(compile(ast.Module(body=constants + functions, type_ignores=[]), path, "exec"), namespace)
    return namespace


def compare_headers(namespace: dict[str, Any]) -> bool:
    """Chaves e valores iguais, excepto o ``turn_id`` aleatório."""
    token = "a.eyJzZXNzaW9uX2lkIjoicy0xIn0.c"
    before = namespace["_build_codex_headers"](token)
    after = new.build_headers(
        token,
        installation_id=namespace["_CODEX_INSTALLATION_ID"],
        window_id=namespace["_CODEX_WINDOW_ID"],
    )
    if set(before) != set(after):
        print(f"  FALHA headers: chaves diferentes {set(before) ^ set(after)}")
        return False
    differing = {
        key for key in before if before[key] != after[key] and key != "x-codex-turn-metadata"
    }
    if differing:
        print(f"  FALHA headers: { ({k: (before[k], after[k]) for k in differing}) }")
        return False
    print("  ok    headers (turn_id excluído)")
    return True


def main() -> int:
    namespace = load_original(sys.argv[1] if len(sys.argv) > 1 else "sitecustomize.py")
    original = namespace["_codex_request_body"]

    failures = 0
    for name, model, messages, tools, extra in CASES:
        before = original(
            model, copy.deepcopy(messages), copy.deepcopy(tools), copy.deepcopy(extra)
        )
        after = new.build_request_body(
            model, copy.deepcopy(messages), copy.deepcopy(tools), copy.deepcopy(extra)
        )
        dumped_before = json.dumps(before, sort_keys=True, default=str)
        dumped_after = json.dumps(after, sort_keys=True, default=str)
        if dumped_before == dumped_after:
            print(f"  ok    {name}")
            continue
        failures += 1
        print(f"  FALHA {name}")
        print(f"        original: {dumped_before[:300]}")
        print(f"        novo:     {dumped_after[:300]}")

    if not compare_headers(namespace):
        failures += 1

    print(f"\ndivergências: {failures} de {len(CASES) + 1}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
