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

## D4 — Divergências intencionais do `sitecustomize.py`

**Data:** 2026-09-18 · **Estado:** decidido

`tools/check_equivalence.py` nasceu para provar que a extracção não mudou nada do que vai
para o fio. Cumpriu: enquanto foi só refactor, as 20 amostras coincidiam byte a byte.

Deixou de ser verdade quando o porte passou a corrigir o original contra a fonte do OMP.
Um verificador que falha em 19 de 20 casos não distingue regressão de correcção, e nesse
estado é ruído com autoridade. As divergências abaixo são as esperadas; qualquer outra é
regressão.

### O prompt de sistema (19 casos)

O original mandava `You are a Claude agent, built on Anthropic's Claude Agent SDK.`; o
porte manda `You are Claude Code, Anthropic's official CLI for Claude.` — o do CLI real,
que é o que o `User-Agent` e a lista de betas dizem ser. Duas identidades no mesmo pedido
é o que faz a Anthropic tratá-lo como tráfego não-CLI.

A âncora de cache no bloco de sistema vem do mesmo sítio: o OMP fixa o prefixo
`tools`+`system`, que não muda entre turnos, em vez de reescrever a âncora de cauda a cada
pedido.

### `max_tokens` (6 casos)

O original fixava 16384 sempre. O porte deriva `budget + OUTPUT_FALLBACK_BUFFER`, limitado
por `MAX_OUTPUT_TOKENS`, que é a regra da fonte (`providers/anthropic.ts:3868`). Fixar o
valor dava 16384 a um pedido `minimal` — 12 mil tokens de output reservados sem uso — e
o mesmo 16384 a um `xhigh`, que fica sem espaço para responder depois de pensar.

### `budget_tokens` (2 casos)

Escala do OMP (`low` = 4096) em vez da tabela do original (`low` = 2048), com o tecto de
`THINKING_CEILING` aplicado **depois** do degrau, para preservar a ordem relativa.

### `output_config.effort` (1 caso)

Com `tool_choice` forçada num modelo adaptativo, o porte fixa `effort: "low"`. Omitir o
`thinking` não desliga o raciocínio nestes modelos — a API volta a ligá-lo por default — e
fixar o degrau mais baixo é a única forma de o reduzir sem o 400 de
`tool_choice` + `thinking`.

### `thinking.display`

Reposto depois de o verificador o apanhar em falta. O gate é geracional — opus ≥ 4.7,
sonnet/fable/mythos ≥ 5 (`compat/resolve.ts :: defaultSupportsDisplay`) — e não coincide
com `is_adaptive`: opus-4-6 e sonnet-4-6 são adaptativos e recusam o campo com 400.

## D5 — Como o botão "MySubs" entra na UI do LiteLLM

**Data:** 2026-09-18 · **Estado:** decidido

### O que se mediu

A UI do proxy é Next.js **pré-compilado**, servido de
`litellm/proxy/_experimental/out/`. O menu lateral é código React dentro de um chunk
minificado:

    out/_next/static/chunks/0c63y7umyjwi-.js
    …{key:"experimental",page:"experimental",label:"Experimental",
       icon:(0,a.jsx)($.FlaskConical,{…}),children:[{key:"prompts",…}]}

Dois factos que decidem:

1. O item `experimental` tem **`children`** — acrescentar uma entrada é acrescentar um
   elemento a essa lista.
2. Outras entradas usam **`external_url`** (`learning-resources` aponta para
   `models.litellm.ai/cookbook`). Há precedente para um item de menu que sai da SPA.

O nome do chunk é um hash de build: muda a cada versão do LiteLLM.

### Decisão

**Sub-app FastAPI montada em `/mysubs` com `app.mount()`**, servindo UI própria. O acesso
faz-se por URL directo e por um item de menu injectado.

**A injecção do item de menu é opcional e best-effort.** Um patch de string num bundle
minificado cujo nome é um hash de build parte em silêncio na próxima versão do LiteLLM —
e um botão que desaparece sem aviso é pior que um botão que nunca existiu. Portanto:

- `/mysubs` funciona sempre, por URL, sem depender de nenhum patch.
- A injecção procura o padrão; **se não o encontrar, não falha** — regista que a UI desta
  versão não foi reconhecida e diz ao utilizador o URL directo.
- Nunca se reescreve o ficheiro no `site-packages`: serve-se uma cópia alterada em
  memória, para um `pip install --force-reinstall` não deixar estado inconsistente.

### Alternativa rejeitada

Recompilar a UI do LiteLLM com a entrada incluída. Dá um botão nativo, mas obriga a
acompanhar cada release do upstream com um fork do frontend — custo permanente por um
ganho estético.
