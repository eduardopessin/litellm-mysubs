"""Equivalência entre ``wire/anthropic.py`` e o ``sitecustomize.py`` de onde foi extraído.

Não é um teste do pytest: precisa do ficheiro original, que vive noutro repositório e não
é dependência deste. Corre-se à mão durante uma extracção, para provar que o refactor não
mudou uma vírgula do que vai para o fio::

    python tools/check_equivalence.py /caminho/para/sitecustomize.py

Quando a extracção estiver completa e o original deixar de existir, este ficheiro deixa de
ter função e deve ser apagado — a garantia passa então a ser dos testes de contrato.
"""

from __future__ import annotations

import ast
import copy
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from litellm_mysubs.wire import anthropic as new

#: Funções do original que compõem o caminho do pedido Claude.
WANTED = {
    "_normalize_effort",
    "_is_anthropic_adaptive",
    "_anthropic_cache_control",
    "_anthropic_markable_message",
    "_anthropic_tool_call_anchor",
    "_count_cache_breakpoints",
    "_mark_cache_breakpoint",
    "_apply_conversation_cache",
    "_inject_claude_prompt",
}

CASES: list[tuple[str, dict[str, Any]]] = [
    ("simples", {"model": "claude-opus-5", "messages": [{"role": "user", "content": "olá"}]}),
    (
        "system+user",
        {
            "model": "claude-opus-5",
            "messages": [
                {"role": "system", "content": "regra"},
                {"role": "user", "content": "x"},
            ],
        },
    ),
    (
        "budget model",
        {
            "model": "claude-haiku-4-5",
            "reasoning_effort": "low",
            "messages": [{"role": "user", "content": "x"}],
        },
    ),
    (
        "adaptive high",
        {
            "model": "claude-opus-5",
            "reasoning_effort": "high",
            "messages": [{"role": "user", "content": "x"}],
        },
    ),
    (
        "temperatura com thinking",
        {
            "model": "claude-opus-5",
            "reasoning_effort": "medium",
            "temperature": 0.7,
            "messages": [{"role": "user", "content": "x"}],
        },
    ),
    (
        "top_p baixo",
        {
            "model": "claude-opus-5",
            "reasoning_effort": "medium",
            "top_p": 0.5,
            "messages": [{"role": "user", "content": "x"}],
        },
    ),
    (
        "tool_choice forcada (budget)",
        {
            "model": "claude-haiku-4-5",
            "reasoning_effort": "high",
            "tool_choice": "required",
            "messages": [{"role": "user", "content": "x"}],
        },
    ),
    (
        "tool_choice forcada (adaptive)",
        {
            "model": "claude-opus-5",
            "reasoning_effort": "high",
            "tool_choice": "required",
            "messages": [{"role": "user", "content": "x"}],
        },
    ),
    (
        "max_completion_tokens",
        {
            "model": "claude-haiku-4-5",
            "reasoning_effort": "low",
            "max_completion_tokens": 30000,
            "messages": [{"role": "user", "content": "x"}],
        },
    ),
    (
        "max_tokens grande",
        {
            "model": "claude-haiku-4-5",
            "reasoning_effort": "high",
            "max_tokens": 64000,
            "messages": [{"role": "user", "content": "x"}],
        },
    ),
    (
        "effort none",
        {
            "model": "claude-opus-5",
            "reasoning_effort": "none",
            "messages": [{"role": "user", "content": "x"}],
        },
    ),
    (
        "thinking disabled",
        {
            "model": "claude-opus-5",
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": "x"}],
        },
    ),
    (
        "reasoning como objecto (/v1/responses)",
        {
            "model": "claude-opus-5",
            "reasoning_effort": {"effort": "medium", "summary": "auto"},
            "messages": [{"role": "user", "content": "x"}],
        },
    ),
    (
        "tool result",
        {
            "model": "claude-opus-5",
            "messages": [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "r"},
            ],
        },
    ),
    (
        "cache do cliente no tecto",
        {
            "model": "claude-opus-5",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": c, "cache_control": {"type": "ephemeral"}}
                    ],
                }
                for c in "abc"
            ]
            + [{"role": "user", "content": "d"}, {"role": "user", "content": "e"}],
        },
    ),
    (
        "Continue. sintetico",
        {
            "model": "claude-opus-5",
            "messages": [
                {"role": "user", "content": "real"},
                {"role": "user", "content": "Continue."},
            ],
        },
    ),
    (
        "multi system",
        {
            "model": "claude-opus-5",
            "messages": [
                {"role": "system", "content": "um"},
                {"role": "user", "content": "x"},
                {"role": "system", "content": "dois"},
            ],
        },
    ),
    (
        "blocos de thinking",
        {
            "model": "claude-opus-5",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "text": "t"},
                        {"type": "text", "text": "visivel"},
                    ],
                },
                {"role": "user", "content": "x"},
            ],
        },
    ),
    (
        "modelo nao-claude",
        {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "x"}],
            "temperature": 0.2,
        },
    ),
    (
        "conversa longa",
        {
            "model": "claude-opus-5",
            "messages": [{"role": "user", "content": f"m{i}"} for i in range(12)],
        },
    ),
]


def load_original(path: str) -> Any:
    """Extrai ``_inject_claude_prompt`` e o que ele precisa, sem importar o módulo.

    O ``sitecustomize.py`` aplica monkey-patching ao ser importado; compilar só os nós
    que interessam é o que permite executá-lo aqui sem tocar no LiteLLM.
    """
    tree = ast.parse(Path(path).read_text("utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in WANTED
    ]
    missing = WANTED - {f.name for f in functions}
    if missing:
        raise SystemExit(f"funções ausentes no original: {sorted(missing)}")

    constants = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            getattr(target, "id", "").startswith(("_ANTHROPIC", "ANTHROPIC_", "CLAUDE_CODE"))
            for target in node.targets
        )
    ]
    namespace: dict[str, Any] = {
        "os": type("Os", (), {"environ": {}})(),
        "_token_manager": type("Tm", (), {"get_anthropic_token": staticmethod(lambda: "")})(),
    }
    module = ast.Module(body=constants + functions, type_ignores=[])
    exec(compile(module, path, "exec"), namespace)
    return namespace["_inject_claude_prompt"]


def main() -> int:
    original_path = sys.argv[1] if len(sys.argv) > 1 else "sitecustomize.py"
    original = load_original(original_path)

    failures = 0
    for name, payload in CASES:
        before = original(copy.deepcopy(payload))
        after = new.build_request(copy.deepcopy(payload), str(payload["model"]))
        dumped_before = json.dumps(before, sort_keys=True, default=str)
        dumped_after = json.dumps(after, sort_keys=True, default=str)
        if dumped_before == dumped_after:
            print(f"  ok    {name}")
            continue
        failures += 1
        print(f"  FALHA {name}")
        print(f"        original: {dumped_before[:300]}")
        print(f"        novo:     {dumped_after[:300]}")

    print(f"\ndivergências: {failures} de {len(CASES)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
