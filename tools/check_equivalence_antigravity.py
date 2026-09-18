"""Equivalência entre ``wire/antigravity.py`` e o ``sitecustomize.py`` de onde foi extraído.

Não é um teste do pytest: precisa do ficheiro original, que vive noutro repositório e não
é dependência deste. Corre-se à mão durante uma extracção::

    python tools/check_equivalence_antigravity.py /caminho/para/sitecustomize.py

Compara o envelope de ``:streamGenerateContent`` gerado pelas duas implementações. Onde
uma levanta, compara-se a **mensagem** e não o tipo: o original levantava ``Exception``
nua e aqui há ``ModelNotServedError``, com o mesmo texto.

Quando a extracção terminar e o original deixar de existir, apagar: a garantia passa então
a ser dos testes de contrato.
"""

from __future__ import annotations

import ast
import base64
import collections
import copy
import json
import mimetypes
import re
import sys
import urllib.parse
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from litellm_mysubs.wire import antigravity as new

WANTED = {
    "_normalize_effort",
    "_google_inline_part",
    "_google_media_from_url",
    "_google_media_part",
    "_google_content_parts",
    "_google_tool_choice",
    "_tools_to_antigravity_tools",
    "_tool_result_value",
    "_messages_to_antigravity_payload",
    "_map_antigravity_model",
    "_antigravity_base_family",
    "_google_model_supports_function_ids",
    "_antigravity_request_id",
}

REQUEST_ID = "agent/aaaabbbb/1000000/aaaabbbb/1"

PNG = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _call(call_id: str, name: str = "f", arguments: str = "{}") -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def _tool(name: str = "read", **extra: Any) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "parameters": {}, **extra}}


#: ``(nome, modelo, mensagens, tools, extra)``
CASES: list[tuple[str, str, list[Any], list[Any] | None, dict[str, Any]]] = [
    ("texto simples", "gemini-3-pro", [{"role": "user", "content": "olá"}], None, {}),
    (
        "system nativo",
        "gemini-3-pro",
        [{"role": "system", "content": "regra"}, {"role": "user", "content": "x"}],
        None,
        {},
    ),
    (
        "effort none",
        "gemini-3-pro",
        [{"role": "user", "content": "x"}],
        None,
        {"reasoning_effort": "none"},
    ),
    (
        "effort high",
        "gemini-3-pro",
        [{"role": "user", "content": "x"}],
        None,
        {"reasoning_effort": "high"},
    ),
    (
        "effort minimal",
        "gemini-3.5-flash",
        [{"role": "user", "content": "x"}],
        None,
        {"reasoning_effort": "minimal"},
    ),
    (
        "max_completion_tokens",
        "gemini-3-pro",
        [{"role": "user", "content": "x"}],
        None,
        {"max_completion_tokens": 1234},
    ),
    ("max_tokens", "gemini-3-pro", [{"role": "user", "content": "x"}], None, {"max_tokens": 999}),
    (
        "variante tiered explicita",
        "gemini-3.8-flash-tiered",
        [{"role": "user", "content": "x"}],
        None,
        {},
    ),
    ("geracao 2.5", "gemini-2.5-flash", [{"role": "user", "content": "x"}], None, {}),
    (
        "imagem em data uri",
        "gemini-3-pro",
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "cor?"},
                    {"type": "image_url", "image_url": {"url": PNG}},
                ],
            }
        ],
        None,
        {},
    ),
    (
        "ficheiro em gs://",
        "gemini-3-pro",
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {"file_uri": "gs://b/x.pdf", "mime_type": "application/pdf"},
                    }
                ],
            }
        ],
        None,
        {},
    ),
    (
        "audio",
        "gemini-3-pro",
        [
            {
                "role": "user",
                "content": [{"type": "input_audio", "input_audio": {"data": PNG, "format": "mp3"}}],
            }
        ],
        None,
        {},
    ),
    ("tools", "gemini-3-pro", [{"role": "user", "content": "x"}], [_tool(description="d")], {}),
    ("tools claude legacy", "claude-sonnet-4-6", [{"role": "user", "content": "x"}], [_tool()], {}),
    (
        "tool_choice por nome",
        "gemini-3-pro",
        [{"role": "user", "content": "x"}],
        [_tool()],
        {"tool_choice": {"type": "function", "function": {"name": "read"}}},
    ),
    (
        "tool_choice required",
        "gemini-3-pro",
        [{"role": "user", "content": "x"}],
        [_tool()],
        {"tool_choice": "required"},
    ),
    (
        "tool_choice none",
        "gemini-3-pro",
        [{"role": "user", "content": "x"}],
        [_tool()],
        {"tool_choice": "none"},
    ),
    (
        "chamada e resultado",
        "gemini-3-pro",
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "tool_calls": [_call("c1", arguments='{"a":1}')]},
            {"role": "tool", "tool_call_id": "c1", "content": "resultado"},
        ],
        None,
        {},
    ),
    (
        "sentinela de assinatura",
        "gemini-3-pro",
        [
            {"role": "assistant", "tool_calls": [_call("c1"), _call("c2", name="g")]},
            {"role": "tool", "tool_call_id": "c1", "content": "r1"},
            {"role": "tool", "tool_call_id": "c2", "content": "r2"},
        ],
        None,
        {},
    ),
    (
        "assinatura embebida no id",
        "gemini-3-pro",
        [
            {"role": "assistant", "tool_calls": [_call("c1|sig-real")]},
            {"role": "tool", "tool_call_id": "c1|sig-real", "content": "r"},
        ],
        None,
        {},
    ),
    (
        "arguments invalidos",
        "gemini-3-pro",
        [
            {"role": "assistant", "tool_calls": [_call("c1", arguments="nao-e-json")]},
            {"role": "tool", "tool_call_id": "c1", "content": "r"},
        ],
        None,
        {},
    ),
    (
        "media no tool result",
        "gemini-3-pro",
        [
            {"role": "assistant", "tool_calls": [_call("c1", name="shot")]},
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": [
                    {"type": "text", "text": "captura"},
                    {"type": "image_url", "image_url": {"url": PNG}},
                ],
            },
        ],
        None,
        {},
    ),
    (
        "tool result com erro",
        "gemini-3-pro",
        [
            {"role": "assistant", "tool_calls": [_call("c1")]},
            {"role": "tool", "tool_call_id": "c1", "content": "falhou", "is_error": True},
        ],
        None,
        {},
    ),
    (
        "geracao 2.5 sem ids",
        "gemini-2.5-flash",
        [
            {"role": "assistant", "tool_calls": [_call("c1")]},
            {"role": "tool", "tool_call_id": "c1", "content": "r"},
        ],
        None,
        {},
    ),
    (
        "conversa longa",
        "gemini-3-pro",
        [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(8)],
        None,
        {},
    ),
]


def load_original(path: str) -> Any:
    """Compila só as funções da conversão, sem importar o módulo."""
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
            getattr(target, "id", "").startswith(
                ("_GOOGLE", "_ANTIGRAVITY", "_antigravity", "_thought")
            )
            for target in node.targets
        )
    ]
    namespace: dict[str, Any] = {
        "json": json,
        "base64": base64,
        "re": re,
        "mimetypes": mimetypes,
        "collections": collections,
        "sys": sys,
        "_url": urllib.parse,
        "time": type("Time", (), {"time": staticmethod(lambda: 1000.0)})(),
        "uuid": type(
            "Uuid", (), {"uuid4": staticmethod(lambda: type("U", (), {"hex": "aaaabbbb"})())}
        )(),
        # O payload nunca deve abrir uma ligação: se abrir, o teste tem de falhar.
        "httpx": type(
            "Httpx",
            (),
            {
                "Timeout": staticmethod(lambda *a, **k: None),
                "get": staticmethod(lambda *a, **k: (_ for _ in ()).throw(AssertionError("rede"))),
            },
        )(),
    }
    exec(compile(ast.Module(body=constants + functions, type_ignores=[]), path, "exec"), namespace)
    namespace["_antigravity_request_id"] = lambda: REQUEST_ID
    return namespace["_messages_to_antigravity_payload"]


def _run(build: Any) -> dict[str, Any]:
    try:
        return dict(build())
    except Exception as error:  # a mensagem é o que se compara, não o tipo
        return {"__erro__": str(error)[:200]}


def main() -> int:
    original = load_original(sys.argv[1] if len(sys.argv) > 1 else "sitecustomize.py")

    failures = 0
    for name, model, messages, tools, extra in CASES:
        before = _run(
            lambda m=model, msg=messages, t=tools, e=extra: original(
                m, copy.deepcopy(msg), "proj-1", copy.deepcopy(t), copy.deepcopy(e)
            )
        )
        after = _run(
            lambda m=model, msg=messages, t=tools, e=extra: new.build_payload(
                m, copy.deepcopy(msg), "proj-1", REQUEST_ID, copy.deepcopy(t), copy.deepcopy(e)
            )
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

    print(f"\ndivergências: {failures} de {len(CASES)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
