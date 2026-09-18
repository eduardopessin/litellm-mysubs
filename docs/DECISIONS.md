# Decisões de arquitectura

Cada entrada regista o que se decidiu, o que se mediu, e o que faria reabrir a questão.
Sem a medição não é uma decisão — é uma preferência.

---

## D1 — O streaming fica em monkey-patch, não em `CustomLLM`

**Data:** 2026-09-18 · **Contra:** `litellm[proxy]` 1.101.0 · **Estado:** decidido

### A questão

O LiteLLM tem uma via oficial para acrescentar um provedor — `custom_provider_map` com
uma classe `CustomLLM` (`litellm/utils.py :: custom_llm_setup`). Usá-la eliminaria a
dependência de símbolos internos (`Router.acompletion`, `route_llm_request.route_request`),
que podem mudar entre releases sem aviso.

### O que se mediu

Handler mínimo, sem upstream, a devolver `prompt_tokens=100, completion_tokens=5,
cached_tokens=80`:

| Forma de emitir o chunk final | Usage entregue |
|---|---|
| `GenericStreamingChunk` (o tipo que a assinatura declara) | não tem sequer campo de reasoning |
| `setattr(chunk, "usage", u)` | **8 / 2** |
| `usage=` como campo do modelo | **8 / 2** |
| chunk final com `choices=[]` | **8 / 2** |
| dict cru | `MidStreamFallbackError` |

Em todas: `cached_tokens` perdido.

Instrumentando `CustomStreamWrapper.chunk_creator`::

    [chunk_creator] entrada.usage=100 -> saida.usage=None
    chunks acumulados no wrapper: 1
    entregue ao cliente: 8

O ramo que preservaria o valor existe em
`litellm_core_utils/streaming_handler.py`::

    if hasattr(chunk, "usage") and chunk.usage is not None:
        model_response.usage = chunk.usage

…mas não é alcançado a partir de um handler custom. O `calculate_total_usage` que decide o
valor final lê `if "usage" in chunk` — acesso **por chave**, sobre um objecto que já perdeu
o campo. Verificado também que `_hidden_params["usage"]` fica `None` neste caminho, ao
contrário do que a issue [#12233] sugere.

Relacionado e ainda aberto no upstream: [#12970], "Streaming response reports incorrect
prompt_tokens usage — significantly inflated".

### Porque isto é bloqueante

Sem o usage real, o LiteLLM estima com `token_counter` e **todos os cache hits ficam
invisíveis** em `/spend/logs`. Numa conta de subscrição a cache é a diferença entre 8697 e
2876 tokens de prompt no mesmo pedido: perder essa contabilidade é perder a única forma de
saber porque a quota acabou.

### Decisão

Os geradores de streaming continuam em monkey-patch. É a única via que controla o usage
que chega ao spend logging.

### O que o spike deu de positivo

- **`/v1/responses` chega ao handler sem patch** em `route_request`. Um dos três patches
  actuais pode desaparecer numa migração futura.
- **Qualquer nome sob o prefixo chega ao handler**, que decide — confirma o desenho da
  guarda de nomes já implementada.
- **`reasoning_content` sobrevive** quando se emite `ModelResponseStream` cru.

### Consequência

A matriz de versões do LiteLLM no CI e o `tests/test_litellm_contract.py` deixam de ser
precaução e passam a ser obrigatórios: a dependência de internals é permanente, e é esse
teste que transforma um upgrade incompatível em CI vermelho em vez de avaria em produção.

### O que faria reabrir

O LiteLLM preservar o usage reportado por um `CustomLLM` — seja fechando a [#12970], seja
expondo um campo próprio para o efeito. Reverificar com
`python tools/spike_custom_llm.py` quando a versão fixada subir.

[#12233]: https://github.com/BerriAI/litellm/issues/12233
[#12970]: https://github.com/BerriAI/litellm/issues/12970

---

## D2 — Modelos declarados no `config.yaml`, wildcard como rede

**Data:** 2026-09-18 · **Estado:** decidido

### A questão

Se o wildcard `claude-*` serve a família toda, porquê manter entradas explícitas?

### O que se mediu

- Os aliases não são deriváveis: `claude-opus` → `claude-opus-4-8`,
  `claude-3-5-sonnet` → `claude-sonnet-4-6`. O wildcard mandaria `anthropic/claude-opus`
  ao upstream, que responde 404.
- O `model_info` desaparece nas entradas materializadas por wildcard (`id`,
  `max_input_tokens`, `supports_vision`) — era essa a diferença que tornava perigosa a
  sombra no `simple-shuffle`.
- O wildcard não enumera: sem entradas declaradas, `/v1/models` mostra apenas os padrões
  literais, e os clientes que descobrem modelos por lá ficam sem nada.

### Decisão

Config para nomes estáveis, aliases e metadados. Wildcard como rede de segurança para o
que ainda não foi declarado — é o que permite servir um modelo novo da família no dia um.

---

## D3 — Injecção no Router, não `POST /model/new`

**Data:** 2026-09-18 · **Estado:** decidido

### O que se mediu

    POST /model/new {"model_name": "zz-probe", ...}   -> 200, devolve model_id
    /model/info                                        -> não aparece
    /v1/models                                         -> não aparece
    POST /v1/chat/completions model=zz-probe           -> 400 no healthy deployments

O modelo grava em Postgres e nunca chega ao Router quando
`general_settings.supported_db_objects` não inclui `"models"` — e não incluir é a
configuração correcta em instalações que servem agentes A2A, onde a lista errada faz o
`_should_load_db_object` devolver `False` para tudo.

### Decisão

O plugin injecta via `Router.set_model_list()` e persiste por sua conta. Um "aplicar" que
devolve 200 e não faz nada é o pior modo de falha possível.
