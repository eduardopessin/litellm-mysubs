# Rastreabilidade com o OMP

O wiring é do [`@oh-my-pi/pi-ai`](https://www.npmjs.com/package/@oh-my-pi/pi-ai)
(`can1357/oh-my-pi`). Este pacote é uma porta para Python do que o OMP faz no fio, e o OMP
é a fonte de verdade: quando um provedor muda, a correcção aparece lá primeiro.

Está repartido por **três** pacotes, e as âncoras podem apontar para qualquer um:

| Pacote | O que tem |
|---|---|
| `@oh-my-pi/pi-ai` | a lógica — `providers/`, `utils/`, `stream.ts` |
| `@oh-my-pi/pi-catalog` | as constantes de fio — valores de headers, versões de cliente fixadas |
| `@oh-my-pi/pi-utils` | o que é transversal a todos os provedores — `USER_AGENT`, `VERSION`, caminhos |

Os dois últimos são fáceis de esquecer, e esquecê-los custou duas vezes:

- `providers/openai-codex-responses.ts` importa do `pi-catalog` o `OPENAI_HEADERS`,
  `OPENAI_HEADER_VALUES` e `CODEX_CLIENT_VERSION`. Sem ele, o `originator` ficou `"pi"` em
  vez de `"omp"` e o header `version` nem existia — e a auditoria deu-os por
  "inverificáveis" em vez de os ir buscar.
- O mesmo ficheiro importa do `pi-utils` o `USER_AGENT`. Sem ele, o valor foi **inventado**
  por analogia (`codex/0.153.0 (external, cli)`) quando o real é `omp/18.2.6`.

O padrão é o mesmo nos dois casos: procurar nos pacotes que se tem, não encontrar, e
escrever um valor plausível em vez de procurar no que falta. Um pacote em falta não produz
uma âncora falhada — produz uma âncora que **nunca chega a ser escrita**, e portanto nada
acusa. O `check_omp_drift.py` descarrega os três.

Isto só é útil se, ao ver uma mudança no OMP, se souber em dez segundos o que actualizar
aqui. Daí uma convenção única.

## A convenção

Uma linha por função portada, imediatamente antes do `def`:

```python
# omp: providers/anthropic.ts :: ensureMaxTokensForThinking
def apply_thinking_params(kwargs, model): ...
```

Regras:

- caminho relativo a `src/` no tarball do npm;
- `::` separa ficheiro e símbolo;
- **sem número de linha** — muda a cada release e o símbolo não;
- **uma linha, um símbolo**; vários símbolos, várias linhas;
- sem prosa. A explicação do *porquê* fica no docstring.

A versão vive num sítio só, `OMP_VERSION` em `tools/check_omp_drift.py`.

## Verificar

```bash
python tools/check_omp_drift.py
```

Descarrega a versão fixada, confirma que cada símbolo anotado ainda existe e compara com a
`latest` do npm. Corre no CI: um rename do lado do OMP fica vermelho em vez de derivar em
silêncio.

Actualizar: subir `OMP_VERSION`, correr o script, tratar o que acusar.

## Divergências deliberadas

Onde não seguimos o OMP, e porquê. Cada uma foi medida contra o serviço real.

| Onde | OMP | Aqui | Porquê |
|---|---|---|---|
| Betas da Anthropic | inclui `redact-thinking-2026-02-12` (`usage/claude.ts`) | omitida | Com ela a Anthropic devolve blocos thinking assinados mas vazios: medido em sonnet-4-6, 74 chars sem a beta, 0 com ela. |
| Betas da Anthropic | inclui `context-1m-2025-08-07` | omitida | Dá 429 de crédito em tokens de subscrição. |
| Orçamento de thinking | até 32768 | tecto 8192 | Janela TPM curta da subscrição Max; 32768 dá 429. |
| Loop de raciocínio | erro *retryable*, a camada de retry volta a pedir | levanta | Não há janela replay-safe aqui: o `reasoning_content` já foi despejado ao cliente antes da detecção, e retentar duplicava-o no mesmo stream. |
| Variantes `-thinking` | descascáveis | só `gemini-2.5-flash-thinking` | `gemini-3.7/3.8-flash-thinking` não existem no upstream; descascá-los servia `-low` em silêncio para um nome inventado. |
| Nome não servido | fallback para modelo próximo | levanta | Responder com outro modelo faz a facturação e as comparações mentirem, e o cliente nunca sabe. |

## Divergências corrigidas ao comparar com a fonte

Estas não eram decisões: eram erros de terem sido portadas de uma cópia intermédia em vez
da fonte. Ficam registadas porque o modo de falha é instrutivo.

| Onde | Estava | Corrigido para | Como se notou |
|---|---|---|---|
| `google_finish_reason` | enumerava as razões de **erro** | enumera as **normais** (`STOP`, `MAX_TOKENS`) e trata o resto como erro, como `mapStopReasonString` | Cinco razões (`FINISH_REASON_UNSPECIFIED`, `LANGUAGE`, `IMAGE_OTHER`, `IMAGE_PROHIBITED_CONTENT`, `IMAGE_RECITATION`) passavam por `stop`: uma resposta bloqueada pelo servidor chegava ao cliente como se estivesse completa. |
| `thinking_loop` | trigramas de **caracteres**, aglomerado de 2, sem aquecimento | trigramas de **palavras**, `SEGMENT_MIN_CLUSTER=4`, `SEGMENT_MIN_COUNT=8`, dois regimes de ciclo exacto, âncoras canonicalizadas | Trigramas de caracteres dão semelhança alta a textos sem relação; disparar a 2 segmentos matava raciocínio legítimo que o OMP deixa passar. |

**Lição de método:** portar da fonte e verificar contra o intermediário — nunca o inverso.
Uma cópia de segunda mão herda os erros da primeira sem os assinalar.

Uma divergência sem medição não é uma divergência: é um bug por corrigir.
