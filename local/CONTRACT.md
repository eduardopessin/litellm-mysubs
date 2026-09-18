# Contrato — `transport/client.py` ⇄ `plugin.py`

Decidido à partida, não negociado entre agentes. Quem implementa um lado programa contra
isto, não contra o outro agente.

## A fronteira

`plugin.py` traduz OpenAI ⇄ provedor e não sabe falar HTTP.
`transport/client.py` fala HTTP e não sabe o que é uma `ModelResponse`.

```
litellm.main.acompletion
  └─ plugin.dispatch(kwargs)            # escolhe o provedor
       ├─ wire/*.build_*                # já existe, verificado
       ├─ client.stream(spec)           # → AsyncIterator[dict]  (eventos SSE crus)
       └─ client.request(spec)          # → Response            (não-streaming)
```

## `transport/client.py`

```python
@dataclass(frozen=True, slots=True)
class RequestSpec:
    url: str
    headers: Mapping[str, str]
    body: Mapping[str, Any]
    provider: Literal["codex", "antigravity"]
    model: str

@dataclass(frozen=True, slots=True)
class Response:
    status: int
    headers: Mapping[str, str]
    text: str

class Transport:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        refresh: RefreshCallback | None = None,
        rotation: HostRotation | None = None,
    ) -> None: ...

    async def request(self, spec: RequestSpec) -> Response: ...
    async def stream(self, spec: RequestSpec) -> AsyncIterator[dict[str, Any]]: ...
```

`RefreshCallback = Callable[[str], Awaitable[str | None]]` — recebe `provider`, devolve
novo access token ou `None`.

### Regras

- **Async nativo.** `httpx.AsyncClient`, nunca `run_in_executor` com cliente síncrono.
  O original ocupava um worker do executor durante toda a resposta — com N pedidos
  concorrentes esgota o pool e os pedidos seguintes ficam em fila sem sintoma visível.
- **Decisões vêm de `transport/retry.py`**, que já existe. O cliente aplica a `Action`;
  não reimplementa a classificação:
  - `REFRESH_TOKEN` → chama `refresh(provider)`, reconstrói o header `Authorization`,
    repete **uma** vez. Sem callback, ou segundo falhanço → levanta.
  - `REMAP_MODEL` / `REDEEM_CREDIT` → levanta `RemapRequired` / `RedeemRequired`; a
    decisão de que modelo usar é do `plugin.py`, não do transporte.
  - `FAIL` → levanta `UpstreamError(status, body)`.
  - `RETURN` → devolve.
- **`HostRotation`** (`transport/hosts.py`) decide a próxima URL com `urls()`,
  `mark_started()`, `can_failover()`, `commit()`. Usar, não duplicar.
- **`stream` corta o corpo em linhas e passa por `sse.iter_events`**, que já existe.
  Devolve os dicts dos eventos tal como vêm — a interpretação é do `plugin.py`.
- **Erro nunca inventa número.** `UpstreamError` leva `status` e corpo do upstream.

## `plugin.py`

```python
def install() -> None:              # idempotente
def uninstall() -> None:            # repõe os originais; usado nos testes
async def dispatch(**kwargs) -> Any # None => não é nosso, cai no original
```

### Regras

- **Patch de `litellm.main.acompletion` e `litellm.main.completion`.** Guarda os originais
  em módulo. `install()` duas vezes não encadeia dois wrappers.
- **Argumentos posicionais normalizados** para `model`/`messages` antes de qualquer coisa.
- **Despacho por modelo.** `antigravity.is_gemini_model` / `codex.is_codex_model` se
  existirem; caso contrário funções locais com âncora. Nada que não seja nosso é tocado:
  devolve-se o original **com** `anthropic.build_request` aplicado, que é o que o
  `sitecustomize.py` faz com `_inject_claude_prompt`.
- **Falhar alto.** Modelo que a subscrição não serve propaga o erro do upstream. Nunca
  substituir por outro modelo (princípio do README).
- **`ModelResponse` do LiteLLM** para a resposta não-streaming; o wrapper de streaming do
  LiteLLM para a outra. Uso apenas o que é público.

## Fronteiras que nenhum agente atravessa

| Ficheiro | Dono |
|---|---|
| `src/litellm_mysubs/transport/client.py` + `tests/test_transport_client.py` | Transport |
| `src/litellm_mysubs/plugin.py` + `tests/test_plugin.py` | Plugin |
| `wire/*`, `transport/{sse,retry,hosts}.py`, `credentials/*` | **ninguém** — já verificados |
| `pyproject.toml`, `README.md`, `docs/*` | eu, no fim |

Sem commits. Sem correr a suite inteira nem o formatter global — só os próprios testes.
