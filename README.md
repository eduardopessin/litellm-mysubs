# litellm-mysubs

Liga as tuas subscrições ao [LiteLLM](https://github.com/BerriAI/litellm) e serve-as como
modelos: **Claude Max**, **ChatGPT Plus (Codex)** e **Google Antigravity**.

> **Estado: alpha.** As fundações estão montadas e testadas (credenciais e registry).
> OAuth, bridges de wire protocol e UI estão em construção — ver [ROADMAP](#roadmap).

## Porquê

Uma subscrição de IA não é uma API key. Os tokens são OAuth de cliente first-party, o
conjunto de modelos servido não é o da API pública, e cada provedor fala um protocolo
próprio (Responses API, Cloud Code, Messages). Este pacote trata dessas diferenças para
que um cliente qualquer de OpenAI as veja como modelos normais do LiteLLM.

## Instalação

```bash
pip install litellm-mysubs
```

## Princípios

Vêm de incidentes medidos em produção, não de preferência de estilo.

**Falhar alto.** Um nome de modelo que a subscrição não serve devolve o erro do upstream.
Nunca se responde com outro modelo: a substituição silenciosa faz a facturação, as
comparações e a reprodutibilidade mentirem, e o cliente nunca sabe que falou com outro
modelo.

**Um só dono do refresh.** Anthropic e OpenAI emitem refresh tokens rotativos de uso
único. Dois renovadores independentes sobre o mesmo token produzem `invalid_grant` em
ciclo e forçam re-login manual. Um store que não é dono lê e nunca troca.

**Nunca inventar números.** Um provedor inalcançável mostra o erro ou o último
instantâneo real, etiquetado com a idade — nunca um valor plausível fabricado.

**Medir contra o backend.** As formas de wire vêm de medição contra o serviço real, não
de documentação. Os comentários no código trazem as medições que motivaram cada decisão.

## Arquitectura

```
src/litellm_mysubs/
├── credentials/     store plugável: ficheiro (0600), Secret, ambiente
├── wire/            um módulo por provedor; não se referenciam entre si
├── catalog/         descoberta de modelos servidos
├── registry.py      injecção no Router + guardas contra deployments fantasma
└── patch.py         o único módulo que muta estado global
```

Todos os módulos são importáveis sem efeitos colaterais. Só `patch.py` altera o LiteLLM,
e apenas quando invocado — é isso que torna o resto testável por unidades.

## Desenvolvimento

```bash
uv venv && uv pip install -e ".[dev]"
pytest                 # unitários
ruff check . && mypy   # lint + tipos
```

Testes que tocam no LiteLLM real precisam dos extras do proxy:

```bash
uv pip install "litellm[proxy]"
pytest tests/test_litellm_contract.py
```

Esse ficheiro afirma a existência dos símbolos internos de que o patch depende
(`Router.acompletion`, `route_llm_request.route_request`, `custom_provider_map`). É o que
transforma um upgrade incompatível do LiteLLM em CI vermelho em vez de numa avaria em
produção.

## Roadmap

| | Fatia | Estado |
|---|---|---|
| 0 | Fundações: credenciais, registry, CI | ✅ |
| 1 | Bridges de wire protocol (Anthropic, Codex, Antigravity) | — |
| 2 | Descoberta de modelos e aplicação | — |
| 3 | OAuth com paste do código | — |
| 4 | UI `/mysubs` montada no proxy | — |

## Licença

MIT
