"""Equivalência da contabilidade de usage e finish reason com o original.

Corre-se à mão durante a extracção::

    python tools/check_equivalence_usage.py /caminho/para/sitecustomize.py

Compara 8 metadados de usage e 26 combinações de finish reason. Apagar quando o original
deixar de existir.
"""

import ast
import sys
from pathlib import Path

ORIGINAL = sys.argv[1] if len(sys.argv) > 1 else "sitecustomize.py"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from litellm_mysubs.wire import usage as new  # noqa: E402

src = Path(ORIGINAL).read_text("utf-8")
tree = ast.parse(src)
WANT = {
    "_bridge_usage",
    "_google_usage",
    "_codex_usage",
    "_google_finish_reason",
    "_codex_finish_reason",
}
fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in WANT]
assert not (WANT - {f.name for f in fns})
consts = [
    n
    for n in tree.body
    if isinstance(n, ast.Assign)
    and any(getattr(t, "id", "").startswith("_GOOGLE_FINISH") for t in n.targets)
]


class FakeUsage:
    """Substitui litellm.types.utils.Usage: só interessam os atributos que se comparam."""

    def __init__(self, **fields: object) -> None:
        self.__dict__.update(fields)


ns = {
    "Usage": FakeUsage,
    "PromptTokensDetailsWrapper": lambda **k: k,
    "CompletionTokensDetailsWrapper": lambda **k: k,
}
exec(compile(ast.Module(body=consts + fns, type_ignores=[]), "o", "exec"), ns)


def norm(u):
    return (
        u.prompt_tokens,
        u.completion_tokens,
        u.total_tokens,
        getattr(u, "cache_read_input_tokens", 0),
        (getattr(u, "completion_tokens_details", None) or {}).get("reasoning_tokens", 0),
    )


def norm2(u):
    return (
        u.prompt_tokens,
        u.completion_tokens,
        u.total_tokens,
        u.cached_tokens,
        u.reasoning_tokens,
    )


CASES_G = [
    {
        "promptTokenCount": 1000,
        "cachedContentTokenCount": 400,
        "candidatesTokenCount": 50,
        "totalTokenCount": 1050,
    },
    {"candidatesTokenCount": 50, "thoughtsTokenCount": 120},
    {},
    {"promptTokenCount": 10},
]
CASES_C = [
    {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 400}},
    {"input_tokens": 10, "prompt_cache_hit_tokens": 7},
    {"output_tokens": 80, "output_tokens_details": {"reasoning_tokens": 60}},
    {},
]
fails = 0
for i, m in enumerate(CASES_G):
    a, b = norm(ns["_google_usage"](m)), norm2(new.google_usage(m))
    if a != b:
        fails += 1
        print("DIVERGE google", i, a, b)
for i, m in enumerate(CASES_C):
    a, b = norm(ns["_codex_usage"](m)), norm2(new.codex_usage(m))
    if a != b:
        fails += 1
        print("DIVERGE codex", i, a, b)
for raw in [
    None,
    "",
    "STOP",
    "MAX_TOKENS",
    "SAFETY",
    "RECITATION",
    "FINISH_REASON_UNSPECIFIED",
    "  max_tokens  ",
    "OTHER",
]:
    for tc in (True, False):
        a, b = ns["_google_finish_reason"](raw, tc), new.google_finish_reason(raw, tc)
        if a != b:
            fails += 1
            print("DIVERGE finish", raw, tc, a, b)
for st in [None, "completed", "incomplete", "failed"]:
    for tc in (True, False):
        a, b = ns["_codex_finish_reason"](st, tc), new.codex_finish_reason(st, tc)
        if a != b:
            fails += 1
            print("DIVERGE codex finish", st, tc, a, b)
print(f"divergencias: {fails}")
sys.exit(1 if fails else 0)
