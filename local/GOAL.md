# Objectivo da sessão — o produto, não o porte

Escrito a pedido do utilizador, para não se perder. **Isto é o objectivo; tudo o resto é
meio.** Sempre que houver dúvida sobre o que fazer a seguir, é contra esta lista que se
responde.

## O que o utilizador pediu, nas palavras dele

1. Um botão no **Experimental** do LiteLLM chamado **"MySubs"**.
2. Abre uma página onde ele pode **adicionar a sua sub** — Gemini, Anthropic ou OpenAI.
3. Essa tela tem os **mesmos cards do Quota Desktop**.
4. Ao clicar em **"conectar sub"**, vai para a página de autenticação upstream, o
   utilizador faz login e **cola a URL de retorno** na página.
5. A partir daí corre a **rotina de renovação** que já existe no Quota.
6. O utilizador **selecciona que modelos** quer adicionar ao LiteLLM e **aplica**.
7. Tudo num **pacote de instalação fácil**, que aplica o monkey-patch sozinho quando o
   utilizador adiciona um modelo da sua sub.

## Porque é que o porte existe

A pergunta inicial da sessão foi *"porque alguns modelos dão 400?"*. A causa raiz estava
no `sitecustomize.py` — um ficheiro único, sem testes, com formas de wire adivinhadas. O
porte para `litellm-mysubs` é o **meio** de tornar isso verificável, e é o passo 7 desta
lista. Não é o fim.

**Risco a vigiar:** o porte é absorvente. Há sempre mais uma âncora para verificar, mais
uma divergência para medir. Se uma sessão acabar com mais linhas portadas e nenhum passo
desta lista fechado, a sessão falhou.

## Estado por passo

| # | Passo | Estado |
|---|---|---|
| 1 | Botão "MySubs" no Experimental | ✅ `/mysubs` montado; injecção do item é best-effort (D5) |
| 2 | Página de adicionar sub | ✅ `ui/app.py` — verificada no browser |
| 3 | Cards do Quota Desktop | ✅ três cards com estado, validade e contagem no Router |
| 4 | OAuth + paste da URL de retorno | ✅ `credentials/oauth.py` — três formatos de paste, state verificado |
| 5 | Rotina de renovação | ✅ rotação do refresh aplicada, dono único respeitado |
| 6 | Seleccionar modelos e aplicar | ✅ `catalog/{discovery,deployments}.py` — sonda tri-estado |
| 7 | Instalação fácil + patch automático | ✅ `plugin.py` — smoke test ponta-a-ponta passa |

## O que já está verificado (e não precisa de ser revisitado)

- `wire/{anthropic,codex,antigravity,schema,usage,...}.py` — 5053 linhas, 117 âncoras ao
  OMP, 1083 testes, 3 verificadores de equivalência a zero.
- `transport/{sse,retry,hosts,client}.py` — o cliente async fechou com 26 testes e um bug
  real apanhado por mutação (`rotation.started` nunca reposto: o failover de host morria
  em silêncio depois do primeiro stream).
- `credentials/{store,file_store,env_store}.py` — leitura e escrita; falta o refresh.

## Decisões já medidas que condicionam o desenho

- **O `CustomLLM` oficial não dispensa o monkey-patch.** Medido: `provider_specific_fields`
  não sobrevive ao `astreaming`, portanto o `reasoning_content` desaparece. O `usage`
  passa, mas só com `stream_options={"include_usage": True}`. Os bridges continuam a
  precisar do patch.
- **`POST /model/new` não carrega o modelo** nesta instalação: grava em Postgres e nunca
  chega ao Router, porque `supported_db_objects` não inclui `"models"` — e incluí-lo
  reabriria o incidente A2A. A aplicação faz-se por `Router.set_model_list()` com
  persistência própria. Registado em `docs/DECISIONS.md` (D3).
- **O paste é a via principal, não o plano B.** Num LiteLLM em container ou cluster o
  browser do utilizador não alcança `localhost:54545`/`1455`/`51121`.
- **Não há catálogo para Anthropic nem Codex.** `/v1/models` dá 401 com token de
  subscrição, e o conjunto servido não é derivável da lista pública. Lista curada +
  verificação por sonda. Só o Google Antigravity tem catálogo real
  (`:fetchAvailableModels`, descontando `deprecatedModelIds`).

## Bloqueio de publicação, não uma fatia

`/api/credentials` do Quota Dashboard responde `GET` com access **e** refresh tokens dos
três provedores, em plaintext, **sem autenticação**. Numa LAN é tolerável; num pacote que
outros instalam expõe as subscrições de quem o instalar. **Tem de fechar antes de qualquer
release pública.**

## Próximo passo

**Só falta a UI (passos 1-3).** Tudo o que ela precisa existe e está verificado em
execução, não só em teste:

    oauth.begin(provider)              -> AuthRequest(url, state, verifier)
    oauth.complete(p, req, colado)     -> Credential          (passo 4)
    oauth.refresh(cred, store=…)       -> Credential          (passo 5)
    discovery.discover(cred)           -> list[DiscoveredModel]  (passo 6)
    deployments.to_deployments(…)      -> list[dict]
    registry.ModelRegistry.apply(…)    -> injecta no Router
    plugin.install()                   -> patch                (passo 7)

Falta a sub-app FastAPI em `/mysubs` que junta isto: cards por provedor, botão de
conectar, caixa de paste, lista de modelos com checkbox, botão de aplicar. Decisão de
como o botão entra no menu do LiteLLM está em `docs/DECISIONS.md` (D5).

## Bugs que os gates verdes não apanharam

Registo, porque o padrão repete-se e é o que justifica insistir em smoke tests reais:

1. **`install()` só patchava metade.** `litellm/__init__.py` faz
   `from .main import acompletion`, que **copia** a referência — `litellm.acompletion`
   ficava por patchar. 42 testes passavam porque afirmavam todos sobre o sítio que o
   código patcha. Um pedido real rebentava com `LLM Provider NOT provided`.
2. **`rotation.started` nunca reposto.** Depois do primeiro stream completo, o failover de
   host morria em silêncio para sempre. Apanhado por mutação.
3. **`_strip_output_limits` era código morto**, com um teste tautológico a cobri-lo —
   afirmava a ausência de chaves que nunca estariam presentes.

A lição comum: **um teste escrito a partir da implementação confirma o que ela faz; não
deteta o que ela esqueceu.**

## Referências

- Desenho completo: wiki, `pages/concepts/mysubs-plugin-litellm-desenho.md`
- Plano faseado: wiki, `pages/concepts/mysubs-plano-faseado.md`
- Contrato transporte ⇄ plugin: `local/CONTRACT.md`
- Divergências intencionais face ao original: `docs/DECISIONS.md`
