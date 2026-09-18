"""Camada HTTP assíncrona: a única parte do pacote que abre sockets.

O original abria o stream dentro do próprio tradutor, e a versão "async" era a síncrona
enrolada num ``run_in_executor``: cada pedido ocupava um worker do pool durante toda a
resposta — que num stream de subscrição são minutos — e o pedido N+1 ficava em fila sem
sintoma nenhum do lado de fora. Aqui o transporte é `httpx.AsyncClient` nativo.

Não há âncora ao OMP nesta camada: o OMP fala com o `fetch` do runtime e a forma do laço
não tem correspondência directa em ``providers/*.ts``. O que **é** portado — a
classificação de erros e a rotação de endpoints — vive em ``retry.py`` e ``hosts.py``, com
âncoras lá. Este módulo executa decisões, não as toma.

Fronteira: entra um `RequestSpec` (URL, headers e corpo já construídos pelo `wire/`), sai
um `Response` ou uma sequência de eventos SSE crus. O transporte não sabe o que é uma
`ModelResponse` nem que modelo substituir por qual.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Final, Literal

import httpx

from . import sse
from .hosts import HostRotation
from .retry import Action, Decision, decide_antigravity, decide_codex

#: Watchdogs do omp: 300 s para o primeiro evento e 300 s de inactividade entre eventos.
#: O `read` do httpx é exactamente o segundo; o total tem de ser ``None`` — um stream
#: legítimo de raciocínio longo passa dos limites de um timeout global.
TIMEOUT: Final = httpx.Timeout(None, connect=30.0, read=300.0, write=60.0)

#: Recebe o nome do provedor, devolve um access token novo ou ``None`` se não for
#: renovável. Assíncrono porque a renovação é ela própria um pedido HTTP.
RefreshCallback = Callable[[str], Awaitable[str | None]]

Provider = Literal["codex", "antigravity"]


@dataclass(frozen=True, slots=True)
class RequestSpec:
    """Um pedido já traduzido para o fio do provedor."""

    url: str
    headers: Mapping[str, str]
    body: Mapping[str, Any]
    provider: Provider
    model: str


@dataclass(frozen=True, slots=True)
class Response:
    """Resposta não-streaming, por interpretar."""

    status: int
    headers: Mapping[str, str]
    text: str


class UpstreamError(Exception):
    """Erro propagado do upstream, com o estado e o corpo **reais**.

    Nunca se inventa um número: um 500 fabricado por cima de um 429 apaga a única
    informação que diz ao utilizador que a quota acabou.
    """

    __slots__ = ("body", "status")

    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


class RemapRequired(UpstreamError):  # noqa: N818 — nome fixado pelo contrato da fronteira
    """A conta não serve este nome de modelo, e o nome pode ser um alias resolúvel.

    Deriva de `UpstreamError` de propósito: quem não a trate propaga o erro real do
    upstream em vez de um erro inventado pelo transporte. Que modelo usar — se algum — é
    decisão do `plugin.py`.
    """

    __slots__ = ()


class RedeemRequired(UpstreamError):  # noqa: N818 — nome fixado pelo contrato da fronteira
    """Quota esgotada; pode haver crédito de reset por resgatar.

    Mesma regra: resgatar crédito é um efeito com custo, não é decisão do transporte.
    """

    __slots__ = ()


def _drain(pending: list[str]) -> Iterator[str]:
    """Iterável que se esvazia à medida que é consumido: o que sobrar é observável."""
    while pending:
        yield pending.pop(0)


def _decode(line: str) -> tuple[list[dict[str, Any]], bool]:
    """Eventos de uma linha, e se o stream terminou.

    `sse.iter_events` assinala o ``[DONE]`` **parando**, não devolvendo marca nenhuma —
    do lado de fora é indistinguível de uma linha sem dados. Acrescenta-se por isso uma
    linha em branco a seguir à real: se ela ficar por consumir, o `iter_events` parou a
    meio, o que só acontece no ``[DONE]``.

    Uma linha rende no máximo um evento, logo materializá-los não bufferiza nada.
    """
    pending = [line, ""]
    events = list(sse.iter_events(_drain(pending)))
    return events, bool(pending)


class Transport:
    """Abre ligações, aplica a decisão de `retry.py` e roda endpoints via `hosts.py`."""

    __slots__ = ("_client", "_owns_client", "_refresh", "_rotation")

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        refresh: RefreshCallback | None = None,
        rotation: HostRotation | None = None,
    ) -> None:
        #: Um cliente injectado é de quem o injectou — fechá-lo partia o dono, que pode
        #: ainda ter pedidos em voo. Só se fecha o que se criou aqui.
        self._owns_client = client is None
        self._client = httpx.AsyncClient(timeout=TIMEOUT) if client is None else client
        self._refresh = refresh
        self._rotation = rotation

    async def __aenter__(self) -> Transport:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def request(self, spec: RequestSpec) -> Response:
        """Pedido não-streaming: abre, lê o corpo todo e fecha."""
        response = await self._open(spec)
        try:
            text = (await response.aread()).decode("utf-8", "replace")
        finally:
            await response.aclose()
        self._commit(response)
        return Response(
            status=response.status_code,
            headers=dict(response.headers),
            text=text,
        )

    async def stream(self, spec: RequestSpec) -> AsyncIterator[dict[str, Any]]:
        """Eventos SSE crus, pela ordem em que chegam.

        O laço de abertura — com renovação de token e failover — corre **antes** do
        primeiro `yield`, e só uma vez. Depois de o chamador ter visto um evento não há
        volta atrás: reabrir noutro host ou com outro token reenviava o prefixo que ele já
        consumiu. Um corpo cortado a meio termina o iterador com os eventos completos que
        chegaram; a linha truncada não produz evento nenhum.
        """
        response = await self._open(spec)
        try:
            async for line in response.aiter_lines():
                events, done = _decode(line)
                for event in events:
                    self._mark_started()
                    yield event
                if done:
                    break
            self._commit(response)
        finally:
            await response.aclose()

    async def _open(self, spec: RequestSpec) -> httpx.Response:
        """Resposta 200 ainda por ler; fechá-la é de quem chama.

        Um 401 renova o token e repete **uma** vez no mesmo endpoint — repetir sem limite
        com uma credencial que o servidor recusa é um laço de rejeições à velocidade da
        rede. Esgotado o endpoint, tenta-se o seguinte enquanto `hosts.py` o autorizar.
        """
        headers = dict(spec.headers)
        urls = self._candidate_urls(spec)
        # Este pedido ainda não emitiu nada, e a rotação sobrevive ao pedido anterior:
        # sem repor a marca, um stream completo deixava o `can_failover` a `False` para
        # sempre e o pedido seguinte perdia o failover em silêncio.
        self._clear_started()
        last = 0, ""
        for position, url in enumerate(urls):
            refreshed = False
            while True:
                response = await self._client.send(
                    self._client.build_request(
                        "POST", url, json=dict(spec.body), headers=headers
                    ),
                    stream=True,
                )
                if response.status_code == 200:
                    return response

                status = response.status_code
                body = (await response.aread()).decode("utf-8", "replace")
                await response.aclose()

                decision = self._decide(spec, status, body)
                if decision.action is Action.REMAP_MODEL:
                    raise RemapRequired(status, body)
                if decision.action is Action.REDEEM_CREDIT:
                    raise RedeemRequired(status, body)
                if decision.action is Action.REFRESH_TOKEN and not refreshed:
                    token = await self._refresh(spec.provider) if self._refresh else None
                    if token:
                        headers["Authorization"] = f"Bearer {token}"
                        refreshed = True
                        continue
                last = status, body
                break

            # Guarda da invariante de `hosts.py`: enquanto `_open` corre antes do
            # primeiro `yield`, `started` é sempre falso aqui e o `is_last` já é imposto
            # pelo `for` — a condição é hoje redundante das duas maneiras. Fica porque é
            # a única coisa que impede a invariante de se perder em silêncio se alguém
            # vier a chamar `_open` a meio de um stream: aí a resposta certa é parar, não
            # reabrir noutro host e duplicar o que o cliente já viu.
            if not self._can_failover(is_last=position == len(urls) - 1):
                break
        raise UpstreamError(*last)

    def _decide(self, spec: RequestSpec, status: int, body: str) -> Decision:
        """Classificação delegada; aqui não se decide nada sobre o conteúdo do erro.

        ``can_remap``/``can_redeem`` vão a ``True``: são capacidades do chamador, e o
        transporte não conhece os aliases nem tem autoridade para gastar crédito. Ao
        passá-las afirmativas, a possibilidade chega ao `plugin.py` como excepção própria
        — que, por derivar de `UpstreamError`, ainda propaga o estado e o corpo reais se
        ninguém a tratar.
        """
        if spec.provider == "codex":
            return decide_codex(status, body, can_remap=True, can_redeem=True)
        return decide_antigravity(status)

    def _candidate_urls(self, spec: RequestSpec) -> list[str]:
        """URLs a tentar, por ordem. Sem rotação, só a que veio no pedido."""
        if self._rotation is None:
            return [spec.url]
        for host in self._rotation.hosts:
            if spec.url.startswith(host):
                return self._rotation.urls(spec.url[len(host) :])
        return [spec.url]

    def _can_failover(self, *, is_last: bool) -> bool:
        return self._rotation is not None and self._rotation.can_failover(is_last=is_last)

    def _mark_started(self) -> None:
        if self._rotation is not None:
            self._rotation.mark_started()

    def _clear_started(self) -> None:
        """Repõe a marca de emissão no início de cada pedido.

        `hosts.py` não expõe um reset — a marca lá é o campo `started`, e a rotação
        existe para ser reutilizada entre pedidos. Escreve-se o campo directamente em vez
        de acrescentar um método a um ficheiro já verificado.
        """
        if self._rotation is not None:
            self._rotation.started = False

    def _commit(self, response: httpx.Response) -> None:
        """Memoriza o endpoint só depois de a resposta ter sido consumida por inteiro."""
        if self._rotation is not None:
            self._rotation.commit(str(response.request.url))
