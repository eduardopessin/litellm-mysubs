"""Descoberta dos modelos que uma subscrição serve, para o utilizador escolher.

A capacidade real difere por provedor, e o desenho reflecte isso em vez de fingir uma
interface uniforme:

* **Google Antigravity** tem catálogo consultável (``:fetchAvailableModels``). A resposta
  é a verdade da conta, incluindo as variantes que já não respondem
  (``deprecatedModelIds``). O desconto é feito por `ModelCatalog.update`, não aqui.
* **Anthropic** e **OpenAI Codex** não têm catálogo. ``/v1/models`` devolve 401 com um
  token de subscrição, e o conjunto servido **não é derivável** da lista pública:
  ``claude-sonnet-4-20250514`` existe na API da Anthropic e devolve 404 numa conta Max.
  Resta uma lista curada de nomes medidos e uma sonda real a cada um.

Três regras governam o resultado, e todas vêm do mesmo princípio — nunca inventar um facto
sobre a conta de outra pessoa:

1. ``verified=True`` só quando o upstream respondeu mesmo. Um nome que ninguém conseguiu
   perguntar aparece com ``verified=False`` e com a razão em ``note``.
2. Uma sonda que falha **por rede** não marca o modelo como não servido. "O upstream disse
   que não" e "não consegui perguntar" são factos diferentes: só o primeiro remove o
   modelo da lista, o segundo deixa-o lá por verificar.
3. Um catálogo inalcançável não produz uma lista plausível. Ou se devolve o instantâneo
   real etiquetado com a idade, ou se levanta `DiscoveryError`.

Nota sobre o OMP: o `pi-catalog` tem `discovery/codex.ts :: fetchCodexModels`, que lê
``/backend-api/codex/models``. Não é usado aqui porque o que esse endpoint anuncia não foi
medido contra uma conta de subscrição nesta instalação, e a medição que existe é a
contrária (o catálogo público não prevê o que a sub serve). A sonda mede; a lista do
upstream, por enquanto, seria suposição.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Final

import httpx

from ..credentials.store import Credential
from ..transport import hosts
from ..transport.retry import is_unsupported_model
from ..wire import anthropic, codex
from ..wire.antigravity_models import BROKEN_WIRE, ModelCatalog

#: Sondas em voo ao mesmo tempo. O limite existe porque uma lista curada dispara um pedido
#: por nome contra o mesmo backend: sem tecto, ligar uma subscrição abria meia dúzia de
#: ligações simultâneas ao upstream só para desenhar um ecrã de selecção.
PROBE_CONCURRENCY: Final = 4

#: Endpoint de inferência da Anthropic. Duplica o valor que o `plugin.py` deriva através do
#: LiteLLM; importá-lo de lá traria o LiteLLM inteiro para dentro da descoberta, que é
#: precisamente a dependência que este pacote separa.
ANTHROPIC_MESSAGES_URL: Final = "https://api.anthropic.com/v1/messages"

#: Valor de ``anthropic-version``. Fixado em `providers/anthropic.ts` do OMP (não leva
#: âncora porque o verificador de âncoras só aceita símbolos identificadores).
ANTHROPIC_API_VERSION: Final = "2023-06-01"

#: Responses API servida pela subscrição ChatGPT. Mesma razão de duplicação que acima:
#: `plugin.py` tem a constante gémea e importa o LiteLLM.
CODEX_RESPONSES_URL: Final = "https://chatgpt.com/backend-api/codex/responses"

# omp: wire/gemini-headers.ts :: getAntigravityUserAgent
#: O backend do Cloud Code Assist fecha a disponibilidade de modelos contra a versão do
#: cliente; o `cl` não é validado.
ANTIGRAVITY_USER_AGENT: Final = (
    "antigravity/hub/2.8.0 (aidev_client; os_type=darwin; arch=arm64; cl=963137146)"
)

#: Marca textual da recusa de nome pela Anthropic. O estado sozinho não chega: a rota OAuth
#: devolve 404 também para caminhos errados, e é o corpo que nomeia o modelo.
ANTHROPIC_NOT_FOUND_MARKER: Final = "not_found_error"

# Lista curada da Anthropic: **só** nomes com resposta 200 medida contra um token de
# subscrição. A fonte de cada um está no próprio pacote, o que faz desta lista uma
# consequência de medições e não de uma escolha de gosto:
#
#   opus-5, fable-5, sonnet-5, opus-4-8, opus-4-6, sonnet-4-6, opus-4-5, sonnet-4-5,
#   haiku-4-5  -> `wire/anthropic.py`, tabela de `ADAPTIVE_EFFORT`, onde cada linha traz os
#                 caracteres de raciocínio que o upstream devolveu. Um modelo que não
#                 responde não produz essa contagem.
#   opus-4-8    -> também o modelo do bootstrap do Claude Code
#                  (`registry/oauth/anthropic.ts :: CLAUDE_CODE_BOOTSTRAP_MODEL`).
#   haiku-4-5   -> o modelo que o original usa na sonda de saúde do proxy.
#
# Fora de propósito: `claude-sonnet-4-20250514` (existe na API pública, 404 na conta Max —
# é o contra-exemplo que justifica este módulo) e os nomes do catálogo do OMP sem medição
# nossa (`claude-mythos-5`, `claude-fable-5-1`, ...). Acrescentá-los é uma linha, depois de
# medidos; pô-los cá agora fazia a sonda parecer confirmação de um palpite.
CURATED_ANTHROPIC: Final[tuple[str, ...]] = (
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-opus-4-8",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
)

# Lista curada do Codex:
#
#   gpt-5.5       -> alvo dos aliases `gpt-5`/`gpt5`/`codex` em `wire/codex.py`, e o modelo
#                    da sonda de saúde do original. Servido, medido.
#   gpt-6-astra   -> alvo dos aliases `gpt-6`/`gpt6` na mesma tabela.
#   gpt-5.6-luna, gpt-5.6-sol, gpt-5.6-terra, gpt-daybreak-blue-latest
#                 -> únicas entradas do provedor `openai-codex` no catálogo agregado do OMP
#                    (`pi-catalog`, models.json). É um catálogo específico do backend da
#                    subscrição, não da API pública — evidência mais fraca que uma medição
#                    nossa, e por isso a sonda é que decide.
#
# Fora de propósito: `gpt-5.4` e `gpt-5.4-mini`. Medido: "The 'gpt-5.4' model is not
# supported when using Codex with a ChatGPT account". Estavam no original a apontar para
# gpt-5.5 e o cliente era facturado contra um modelo que nunca correu.
CURATED_CODEX: Final[tuple[str, ...]] = (
    "gpt-5.5",
    "gpt-6-astra",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-daybreak-blue-latest",
)


class DiscoveryError(RuntimeError):
    """Não foi possível saber o que a conta serve.

    Levantada só onde a alternativa seria inventar: um catálogo inalcançável e sem
    instantâneo anterior não tem resposta honesta em forma de lista.
    """


@dataclass(frozen=True, slots=True)
class DiscoveredModel:
    """Um modelo que a subscrição pode servir.

    ``verified`` é a única coisa que distingue um facto de uma hipótese, e é por isso que
    ``note`` é obrigatório na prática sempre que ``verified`` é falso: um ecrã de selecção
    que mostre os dois casos iguais transforma a lista curada em promessa.
    """

    wire_name: str
    """Nome tal como vai no fio. Nu, sem prefixo de provedor."""

    suggested_name: str
    """Nome público sugerido no LiteLLM. Sugestão: o utilizador muda-o na UI."""

    verified: bool
    """Se o upstream respondeu mesmo a este nome nesta descoberta."""

    note: str = ""
    """Porque não foi verificado, quando aplicável."""


def suggested_name(wire_name: str) -> str:
    """Nome público a partir do nome de fio: ``anthropic/claude-opus-5`` -> ``claude-opus-5``.

    O prefixo de provedor pertence a ``litellm_params["model"]``, não ao ``model_name``: é
    o ``model_name`` que ecoa no spend log, e um nome prefixado aí nomeia algo que o
    cliente nunca pediu.
    """
    return str(wire_name).split("/")[-1]


@dataclass(frozen=True, slots=True)
class _Probe:
    """Resultado de uma sonda. ``served=None`` é "não consegui perguntar"."""

    served: bool | None
    note: str = ""


#: Uma sonda: cliente, credencial, nome de fio -> veredicto.
Probe = Callable[[httpx.AsyncClient, Credential, str], Awaitable[_Probe]]


async def discover(
    credential: Credential,
    *,
    client: httpx.AsyncClient,
    catalog: ModelCatalog | None = None,
    now: float | None = None,
) -> list[DiscoveredModel]:
    """Modelos que esta subscrição serve.

    ``catalog`` só é usado pelo Google: passar o catálogo vivo do processo é o que permite
    que uma resposta vazia do endpoint não apague o que já se sabia. ``now`` existe para
    tornar a idade do instantâneo determinística nos testes.
    """
    if credential.provider == "google-antigravity":
        return await _discover_google(
            credential, client=client, catalog=catalog or ModelCatalog(), now=now
        )
    if credential.provider == "anthropic":
        return await _discover_probed(credential, CURATED_ANTHROPIC, _probe_anthropic, client)
    return await _discover_probed(credential, CURATED_CODEX, _probe_codex, client)


# -- Google Antigravity: catálogo real -----------------------------------------


# omp: discovery/antigravity.ts :: fetchAntigravityDiscoveryModels
async def _discover_google(
    credential: Credential,
    *,
    client: httpx.AsyncClient,
    catalog: ModelCatalog,
    now: float | None,
) -> list[DiscoveredModel]:
    """Catálogo da conta, com os dois endpoints por ordem.

    Só o desconto de `BROKEN_WIRE` é feito aqui: o de ``deprecatedModelIds`` é do
    `ModelCatalog`, que é onde a forma do payload está verificada.
    """
    moment = time.time() if now is None else now
    payload = await _fetch_catalog(credential, client=client)

    # O que conta é se o catálogo *absorveu* alguma coisa, não se houve resposta: um
    # payload cujos modelos estão todos em ``deprecatedModelIds`` é uma resposta 200 que
    # não acrescenta nada, e `ModelCatalog.update` deixa o instantâneo anterior intacto
    # de propósito. Compara-se o par (ids, instante) porque nenhum dos dois sozinho
    # distingue os casos. Degenerescência conhecida: um chamador que passe ``now`` igual
    # ao instante da recolha anterior *e* receba exactamente os mesmos ids vê a lista
    # rotulada como instantâneo de 0 s. Com um relógio real não acontece, e o erro é para
    # o lado seguro — etiqueta a mais, nunca modelo a mais.
    before = catalog.ids, catalog.fetched_at
    if payload is not None:
        catalog.update(payload, now=moment)
    absorbed = payload is not None and (catalog.ids, catalog.fetched_at) != before

    if not catalog.ids:
        raise DiscoveryError(
            "Google Antigravity: o catálogo não respondeu e não há instantâneo anterior; "
            "listar modelos aqui seria inventá-los"
        )

    age = ""
    if not absorbed:
        # Há instantâneo anterior e o endpoint não o renovou. Devolvê-lo é legítimo — foi
        # medido —, mas sem a idade passaria por actual, que é exactamente o número
        # plausível que este pacote não inventa.
        seconds = int(max(0.0, moment - catalog.fetched_at))
        reason = (
            "o endpoint não respondeu"
            if payload is None
            else "o endpoint respondeu sem modelos utilizáveis"
        )
        age = f"instantâneo do catálogo com {seconds} s; {reason} agora"

    discovered: list[DiscoveredModel] = []
    for wire in catalog.ids:
        notes = [age] if age else []
        broken = wire in BROKEN_WIRE
        if broken:
            # O catálogo anuncia-as e o streamGenerateContent recusa-as. Escondê-las
            # perderia informação real; anunciá-las como servidas repetia o defeito.
            notes.append(
                "o catálogo anuncia-o mas o streamGenerateContent devolve 400 "
                "INVALID_ARGUMENT para esta variante"
            )
        discovered.append(
            DiscoveredModel(
                wire_name=wire,
                suggested_name=suggested_name(wire),
                verified=not broken,
                note="; ".join(notes),
            )
        )
    return discovered


# omp: discovery/antigravity.ts :: FETCH_AVAILABLE_MODELS_PATH
async def _fetch_catalog(
    credential: Credential, *, client: httpx.AsyncClient
) -> dict[str, Any] | None:
    """Payload de ``:fetchAvailableModels``, ou ``None`` se nenhum endpoint respondeu.

    Percorre os dois hosts como o resto do pacote: um host em baixo não é uma conta sem
    modelos.
    """
    headers = {
        "Authorization": f"Bearer {credential.access_token}",
        "Content-Type": "application/json",
        "User-Agent": ANTIGRAVITY_USER_AGENT,
    }
    for host in hosts.HOSTS:
        try:
            response = await client.post(host + hosts.MODELS_PATH, json={}, headers=headers)
        except httpx.HTTPError:
            continue
        if response.status_code != 200:
            continue
        try:
            payload = response.json()
        except ValueError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


# -- Anthropic e Codex: lista curada + sonda -----------------------------------


async def _discover_probed(
    credential: Credential,
    curated: tuple[str, ...],
    probe: Probe,
    client: httpx.AsyncClient,
) -> list[DiscoveredModel]:
    """Sonda a lista curada, em paralelo e com tecto de concorrência.

    Um nome recusado pelo upstream sai da lista — não é servido, e oferecê-lo produzia um
    deployment que só sabe dar 404. Um nome que a sonda não conseguiu perguntar fica, com
    ``verified=False`` e a razão: a rede desta máquina não é um facto sobre a conta.
    """
    limit = asyncio.Semaphore(PROBE_CONCURRENCY)

    async def guarded(wire: str) -> _Probe:
        async with limit:
            return await probe(client, credential, wire)

    results = await asyncio.gather(*(guarded(wire) for wire in curated))

    discovered: list[DiscoveredModel] = []
    for wire, result in zip(curated, results, strict=True):
        if result.served is False:
            continue
        discovered.append(
            DiscoveredModel(
                wire_name=wire,
                suggested_name=suggested_name(wire),
                verified=result.served is True,
                note=result.note,
            )
        )
    return discovered


async def _post_status(
    client: httpx.AsyncClient,
    url: str,
    *,
    body: dict[str, Any],
    headers: dict[str, str],
) -> tuple[int, str]:
    """Estado, e corpo apenas quando não é 200.

    Em stream para que uma sonda bem sucedida feche a ligação logo a seguir aos
    cabeçalhos: o que interessa é o veredicto, não os tokens gerados.
    """
    async with client.stream("POST", url, json=body, headers=headers) as response:
        if response.status_code == 200:
            return 200, ""
        return response.status_code, (await response.aread()).decode("utf-8", "replace")


def _unreachable(exc: httpx.HTTPError) -> _Probe:
    """Falha de transporte: por verificar, e nunca "não servido"."""
    return _Probe(
        None,
        f"a sonda não chegou ao upstream ({type(exc).__name__}: {exc}); "
        f"por verificar, não recusado",
    )


async def _probe_anthropic(client: httpx.AsyncClient, credential: Credential, wire: str) -> _Probe:
    """Pedido mínimo de mensagens: ``max_tokens=1`` e um turno de um caractere.

    O bloco de identidade tem de vir primeiro mesmo numa sonda — medido: ``system`` só com
    o prompt do cliente devolve 429, e um 429 aqui era indistinguível de quota.
    """
    body: dict[str, Any] = {
        "model": wire,
        "max_tokens": 1,
        "system": anthropic.build_system_blocks(""),
        "messages": [{"role": "user", "content": "."}],
    }
    headers = {
        "Authorization": f"Bearer {credential.access_token}",
        "Content-Type": "application/json",
        "accept": "application/json",
        "anthropic-version": ANTHROPIC_API_VERSION,
        "anthropic-beta": anthropic.build_betas(thinking=False),
        **anthropic.CLIENT_HEADERS,
    }
    try:
        status, text = await _post_status(
            client, ANTHROPIC_MESSAGES_URL, body=body, headers=headers
        )
    except httpx.HTTPError as exc:
        return _unreachable(exc)

    if status == 200:
        return _Probe(True)
    if status == 404 and ANTHROPIC_NOT_FOUND_MARKER in text:
        return _Probe(False, "o upstream recusou o nome com not_found_error")
    return _Probe(
        None,
        f"o upstream respondeu HTTP {status}, que não distingue modelo inexistente de "
        f"recusa temporária; por verificar",
    )


async def _probe_codex(client: httpx.AsyncClient, credential: Credential, wire: str) -> _Probe:
    """Turno mínimo na Responses API, com o raciocínio desligado.

    A recusa de nome do Codex é um 400 com marca própria, e é `transport.retry` que a
    reconhece — a mesma função que o transporte usa em produção, para que a sonda e o
    caminho real não possam divergir na definição de "não servido".
    """
    body = codex.build_request_body(
        wire,
        [{"role": "user", "content": "."}],
        extra={"reasoning_effort": "none"},
    )
    headers = codex.build_headers(
        credential.access_token,
        window_id=_probe_window_id(credential),
        model=str(body.get("model") or wire),
    )
    try:
        status, text = await _post_status(client, CODEX_RESPONSES_URL, body=body, headers=headers)
    except httpx.HTTPError as exc:
        return _unreachable(exc)

    if status == 200:
        return _Probe(True)
    if status == 404 or (status == 400 and is_unsupported_model(text)):
        return _Probe(False, "a conta ChatGPT não serve este modelo")
    return _Probe(
        None,
        f"o upstream respondeu HTTP {status}, que não distingue modelo inexistente de "
        f"recusa temporária; por verificar",
    )


def _probe_window_id(credential: Credential) -> str:
    """Identidade de janela das sondas.

    Derivada da conta e não aleatória: o backend usa o ``window_id`` para o cache de
    prompt, e um id novo por sonda envelhecia o cache da sessão real do utilizador.
    """
    return f"mysubs-discovery-{codex.account_id(credential.access_token) or 'anon'}"
