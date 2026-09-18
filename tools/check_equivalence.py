"""Equivalência entre o wire/anthropic.py e o sitecustomize.py de onde foi extraído.

Não é um teste do pytest: precisa do ficheiro original, que vive noutro repositório e não
é dependência deste. Corre-se à mão durante uma extracção, para provar que o refactor não
mudou uma vírgula do que vai para o fio::

    python tools/check_equivalence.py /caminho/para/sitecustomize.py

Quando a extracção estiver completa e o original deixar de existir, este ficheiro deixa de
ter função e deve ser apagado — a garantia passa a ser dos testes de contrato.
"""

import ast, copy, json, sys

ORIGINAL = sys.argv[1] if len(sys.argv) > 1 else '/tmp/sc.py'
sys.path.insert(0,"src")
from litellm_mysubs.wire import anthropic as new

# --- carregar o original por AST, com o mínimo de ambiente ---
src = open(ORIGINAL).read()
tree = ast.parse(src)
want = {"_normalize_effort","_is_anthropic_adaptive","_anthropic_cache_control",
        "_anthropic_markable_message","_anthropic_tool_call_anchor","_count_cache_breakpoints",
        "_mark_cache_breakpoint","_apply_conversation_cache","_inject_claude_prompt"}
fns=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in want]
assert len(fns)==len(want), want-{f.name for f in fns}
consts=[n for n in tree.body if isinstance(n,ast.Assign) and any(
    getattr(t,'id','').startswith(('_ANTHROPIC','ANTHROPIC_','CLAUDE_CODE')) for t in n.targets)]
ns={"os":type("O",(),{"environ":{}})(),
    "_token_manager":type("T",(),{"get_anthropic_token":staticmethod(lambda: "")})()}
exec(compile(ast.Module(body=consts+fns,type_ignores=[]),"orig","exec"), ns)
orig = ns["_inject_claude_prompt"]

CASES = [
  ("simples", {"model":"claude-opus-5","messages":[{"role":"user","content":"olá"}]}),
  ("system+user", {"model":"claude-opus-5","messages":[{"role":"system","content":"regra"},{"role":"user","content":"x"}]}),
  ("budget model", {"model":"claude-haiku-4-5","reasoning_effort":"low","messages":[{"role":"user","content":"x"}]}),
  ("adaptive high", {"model":"claude-opus-5","reasoning_effort":"high","messages":[{"role":"user","content":"x"}]}),
  ("temp+thinking", {"model":"claude-opus-5","reasoning_effort":"medium","temperature":0.7,"messages":[{"role":"user","content":"x"}]}),
  ("top_p baixo", {"model":"claude-opus-5","reasoning_effort":"medium","top_p":0.5,"messages":[{"role":"user","content":"x"}]}),
  ("forced tool budget", {"model":"claude-haiku-4-5","reasoning_effort":"high","tool_choice":"required","messages":[{"role":"user","content":"x"}]}),
  ("forced tool adaptive", {"model":"claude-opus-5","reasoning_effort":"high","tool_choice":"required","messages":[{"role":"user","content":"x"}]}),
  ("max_completion", {"model":"claude-haiku-4-5","reasoning_effort":"low","max_completion_tokens":30000,"messages":[{"role":"user","content":"x"}]}),
  ("max_tokens grande", {"model":"claude-haiku-4-5","reasoning_effort":"high","max_tokens":64000,"messages":[{"role":"user","content":"x"}]}),
  ("effort none", {"model":"claude-opus-5","reasoning_effort":"none","messages":[{"role":"user","content":"x"}]}),
  ("thinking disabled", {"model":"claude-opus-5","thinking":{"type":"disabled"},"messages":[{"role":"user","content":"x"}]}),
  ("responses obj", {"model":"claude-opus-5","reasoning_effort":{"effort":"medium","summary":"auto"},"messages":[{"role":"user","content":"x"}]}),
  ("tool result", {"model":"claude-opus-5","messages":[
      {"role":"user","content":"q"},
      {"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]},
      {"role":"tool","tool_call_id":"c1","content":"r"}]}),
  ("cliente com cache", {"model":"claude-opus-5","messages":[
      {"role":"user","content":[{"type":"text","text":"a","cache_control":{"type":"ephemeral"}}]},
      {"role":"user","content":[{"type":"text","text":"b","cache_control":{"type":"ephemeral"}}]},
      {"role":"user","content":[{"type":"text","text":"c","cache_control":{"type":"ephemeral"}}]},
      {"role":"user","content":"d"},{"role":"user","content":"e"}]}),
  ("continue sintetico", {"model":"claude-opus-5","messages":[{"role":"user","content":"real"},{"role":"user","content":"Continue."}]}),
  ("multi system", {"model":"claude-opus-5","messages":[{"role":"system","content":"um"},{"role":"user","content":"x"},{"role":"system","content":"dois"}]}),
  ("thinking blocks", {"model":"claude-opus-5","messages":[
      {"role":"assistant","content":[{"type":"thinking","text":"t"},{"type":"text","text":"visivel"}]},
      {"role":"user","content":"x"}]}),
  ("nao-claude", {"model":"gpt-5.5","messages":[{"role":"user","content":"x"}],"temperature":0.2}),
  ("longa", {"model":"claude-opus-5","messages":[{"role":"user","content":f"m{i}"} for i in range(12)]}),
]

fails=0
for name, payload in CASES:
    a = orig(copy.deepcopy(payload))
    b = new.build_request(copy.deepcopy(payload), str(payload["model"]))
    # o original injecta headers sempre; comparar tudo
    ja, jb = json.dumps(a,sort_keys=True,default=str), json.dumps(b,sort_keys=True,default=str)
    if ja != jb:
        fails += 1
        print(f"DIVERGE: {name}")
        print("  orig:", ja[:400]); print("  novo:", jb[:400])
    else:
        print(f"  ok  {name}")
print()
print("DIVERGENCIAS:", fails, "de", len(CASES))
sys.exit(1 if fails else 0)
