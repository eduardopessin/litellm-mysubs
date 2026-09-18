"""Ponto de entrada: intercepta o LiteLLM e serve os modelos da subscrição.

Este módulo é a única parte do pacote que conhece o LiteLLM. Traduz OpenAI ⇄ provedor e
delega o HTTP no `transport/client.py`; a construção dos corpos é toda de `wire/*`.

Porquê monkey-patch e não `custom_provider_map`: o mapa oficial exige que o nome do modelo
traga um prefixo de provedor (``mysubs/gpt-5.5``). Os clientes pedem ``gpt-5.5``, e
reescrever o nome no caminho faria o spend log registar um modelo que ninguém pediu. O
patch apanha o nome tal como chega.

Decisões tomadas onde o contrato deixa margem
---------------------------------------------

``RemapRequired`` **propaga**. O transporte levanta-a quando a conta recusa o nome do
modelo e sinaliza que pode haver alias. Mas `codex.resolve_model` já aplicou a tabela de
aliases *antes* de enviar: se o upstream recusou o resultado, não sobra nome nenhum por
tentar, e repetir mandaria exactamente o mesmo pedido. Substituir por outro modelo é o que
o README proíbe — a resposta vinha com o campo ``model`` a ecoar o pedido e a facturação
passava a mentir. Como `RemapRequired` deriva de `UpstreamError`, o estado e o corpo reais
chegam ao cliente.

``RedeemRequired`` **propaga** pela mesma ordem de razões: resgatar um crédito de reset
gasta saldo do utilizador, e o plugin não tem mandato para o fazer sem lho pedirem. Um 429
honesto é melhor que um débito silencioso.

O caminho síncrono (`litellm.main.completion`) corre a mesma rotina assíncrona num loop
privado: o transporte é async nativo por decisão do contrato, e duplicar a lógica em
versão síncrona foi exactamente o que fez as duas derivarem no original.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any, Final, Protocol

import litellm
import litellm.main
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.types.utils import Delta, ModelResponse, ModelResponseStream, StreamingChoices

from .credentials.store import CredentialStore, ProviderId
from .transport import hosts
from .transport.client import RequestSpec, Transport
from .wire import anthropic, antigravity, codex, planning_leak, thinking_loop
from .wire.usage import Usage, codex_finish_reason, codex_usage, google_finish_reason, google_usage

#: Endpoint da Responses API servida pela subscrição ChatGPT.
CODEX_URL: Final = "https://chatgpt.com/backend-api/codex/responses"

ANTIGRAVITY_USER_AGENT: Final = (
    "antigravity/hub/2.8.0 (aidev_client; os_type=darwin; arch=arm64; cl=963137146)"
)

#: Assinaturas de raciocínio das tool calls do Gemini, para reenviar no turno seguinte. O
#: tecto existe porque uma sessão longa acumularia uma entrada por chamada até ao fim do
#: processo.
_SIGNATURE_LIMIT: Final = 512

#: Identidade de transporte desta instância. É por processo, como no cliente real: um
#: `window_id` novo a cada pedido invalidava o cache de prompt do backend.
_WINDOW_ID: Final = str(uuid.uuid4())
_AGENT_ID: Final = uuid.uuid4().hex[:16]
_TRAJECTORY_ID: Final = uuid.uuid4().hex[:16]


class _State:
    """Estado do módulo, num objecto só para que `uninstall` não deixe pontas soltas."""

    __slots__ = ("original_acompletion", "original_completion", "signatures", "step", "store",
                 "transport")

    def __init__(self) -> None:
        self.original_acompletion: Callable[..., Any] | None = None
        self.original_completion: Callable[..., Any] | None = None
        self.store: CredentialStore | None = None
        self.transport: Transport | None = None
        self.signatures: OrderedDict[str, str] = OrderedDict()
        self.step = 0


_state = _State()


def configure(
    *, store: CredentialStore | None = None, transport: Transport | None = None
) -> None:
    """Liga as dependências. Chamar antes de `install`.

    Ambas são injectadas em vez de descobertas: é o que permite exercer o despacho inteiro
    sem tocar na rede nem no disco.
    """
    if store is not None:
        _state.store = store
    if transport is not None:
        _state.transport = transport


def _transport() -> Transport:
    """Transporte em uso; cria o de produção à primeira necessidade."""
    if _state.transport is None:
        _state.transport = Transport(refresh=_refresh, rotation=hosts.HostRotation())
    return _state.transport


async def _refresh(provider: str) -> str | None:
    """Relê a credencial depois de um 401.

    Não renova por sua conta: quem detém o refresh token é o store (ver a regra do dono
    único em `credentials/store.py`). Aqui só se volta a ler a fonte, que outro processo
    pode entretanto ter rodado.
    """
    store = _state.store
    if store is None:
        return None
    store.reload()
    credential = store.get(_PROVIDER_IDS[provider])
    return credential.access_token if credential else None


_PROVIDER_IDS: Final[dict[str, ProviderId]] = {
    "codex": "openai-codex",
    "antigravity": "google-antigravity",
    "anthropic": "anthropic",
}


def _access_token(provider: str) -> str:
    store = _state.store
    if store is None:
        return ""
    credential = store.get(_PROVIDER_IDS[provider])
    return credential.access_token if credential else ""


def is_gemini_model(model: str) -> bool:
    """Modelos servidos pela subscrição Google Antigravity.

    Sem âncora ao OMP de propósito: lá a distinção é feita por ``model.provider`` num
    catálogo tipado (`google-gemini-cli.ts`), não por um predicado sobre o nome. Aqui o
    nome é tudo o que chega do cliente. A forma vem do original, `sitecustomize.py:1549`.
    """
    lowered = str(model).lower()
    return "gemini" in lowered or "antigravity" in lowered


def _remember_signature(call_id: str, signature: str) -> None:
    signatures = _state.signatures
    signatures[call_id] = signature
    signatures.move_to_end(call_id)
    while len(signatures) > _SIGNATURE_LIMIT:
        signatures.popitem(last=False)


def _request_id() -> str:
    """``agent/<id>/<ts>/<traj>/<passo>`` — o formato que o CCA espera."""
    _state.step += 1
    return f"agent/{_AGENT_ID}/{int(time.time() * 1000)}/{_TRAJECTORY_ID}/{_state.step}"


def _codex_spec(model: str, messages: list[Any], extra: dict[str, Any]) -> RequestSpec:
    # Sem tectos de output a remover: `build_request_body` constrói o corpo de raiz e não
    # lê `max_tokens`/`max_output_tokens`/`max_completion_tokens` dos kwargs. O original
    # tinha de os apagar porque passava os kwargs adiante; aqui nunca chegam ao fio.
    token = _access_token("codex")
    body = codex.build_request_body(
        model,
        messages,
        tools=extra.get("tools"),
        extra=extra,
        session_id=extra.get("litellm_session_id") or extra.get("user"),
    )
    headers = codex.build_headers(
        token,
        window_id=_WINDOW_ID,
        session_id=extra.get("litellm_session_id") or extra.get("user"),
        model=str(body.get("model") or model),
        service_tier=extra.get("service_tier"),
    )
    return RequestSpec(
        url=CODEX_URL, headers=headers, body=body, provider="codex", model=model
    )


def _antigravity_spec(model: str, messages: list[Any], extra: dict[str, Any]) -> RequestSpec:
    token = _access_token("antigravity")
    store = _state.store
    credential = store.get("google-antigravity") if store else None
    body = antigravity.build_payload(
        model,
        messages,
        project_id=credential.project_id if credential else "",
        request_id=_request_id(),
        tools=extra.get("tools"),
        extra=extra,
        thought_signatures=_state.signatures,
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "User-Agent": ANTIGRAVITY_USER_AGENT,
    }
    return RequestSpec(
        url=hosts.HOSTS[0] + hosts.STREAM_PATH,
        headers=headers,
        body=body,
        provider="antigravity",
        model=model,
    )


# -- interpretação dos eventos -------------------------------------------------


class _Turn:
    """Acumulador do que um stream de eventos produziu.

    O mesmo objecto serve os dois caminhos: no não-streaming lê-se no fim, no streaming
    vai-se emitindo. Ter duas rotinas de interpretação foi o que fez as versões síncrona e
    assíncrona do original divergirem.
    """

    __slots__ = ("finish_raw", "reasoning", "terminal", "text", "tool_calls", "usage_meta")

    def __init__(self) -> None:
        self.text: list[str] = []
        self.reasoning: list[str] = []
        self.tool_calls: list[dict[str, Any]] = []
        self.usage_meta: dict[str, Any] = {}
        self.finish_raw: object = None
        self.terminal = False


class StreamError(RuntimeError):
    """Falha dentro de um stream com HTTP 200.

    Tanto o Codex (``response.failed``) como o CCA (``error`` in-band) reportam erros no
    corpo de uma resposta bem-sucedida. Engoli-los entregava um turno vazio como sucesso.
    """


class _CodexReader:
    """Traduz eventos da Responses API para chunks OpenAI, actualizando um `_Turn`.

    Interface ``feed``/``close`` em vez de gerador sobre um iterável: o mesmo objecto
    serve o caminho streaming (emite-se o que ``feed`` devolve) e o não-streaming
    (descarta-se), sem que um evento que produza vários chunks fique retido.
    """

    __slots__ = ("_active", "_index_of", "_turn", "_ws_bytes", "_ws_events")

    def __init__(self, turn: _Turn) -> None:
        self._turn = turn
        self._active: dict[str, dict[str, Any]] = {}
        self._index_of: dict[str, int] = {}
        self._ws_events = 0
        self._ws_bytes = 0

    def feed(self, event: dict[str, Any]) -> list[ModelResponseStream]:
        kind = event.get("type")
        turn = self._turn

        if kind == "response.output_item.added":
            item = event.get("item") or {}
            if item.get("type") != "function_call":
                return []
            item_id = str(item.get("id") or "")
            call_id = codex.composite_call_id(item.get("call_id"), item.get("id"))
            name = str(item.get("name") or "")
            index = len(self._index_of)
            self._index_of[item_id] = index
            self._active[item_id] = {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": ""},
            }
            return [_tool_open_chunk(index, call_id, name)]

        if kind == "response.function_call_arguments.delta":
            item_id = str(event.get("item_id") or "")
            delta = str(event.get("delta") or "")
            if not delta:
                return []
            # O backend entra por vezes num ciclo a emitir só espaços nos argumentos; sem
            # travão o stream nunca fecha. Limites do OMP: 256 eventos / 16 KB.
            if not delta.strip():
                self._ws_events += 1
                self._ws_bytes += len(delta)
                if self._ws_events > 256 or self._ws_bytes > 16384:
                    raise StreamError("Codex: ciclo de espaços nos argumentos de tool call")
            if item_id in self._active:
                self._active[item_id]["function"]["arguments"] += delta
            return [_tool_delta_chunk(self._index_of.get(item_id, 0), delta)]

        if kind == "response.output_item.done":
            item = event.get("item") or {}
            item_id = str(item.get("id") or "")
            if item.get("type") == "function_call" and item_id in self._active:
                call = self._active.pop(item_id)
                if item.get("arguments"):
                    call["function"]["arguments"] = str(item["arguments"])
                turn.tool_calls.append(call)
            return []

        if kind in ("response.reasoning_text.delta", "response.reasoning_summary_text.delta"):
            delta = str(event.get("delta") or "")
            if not delta:
                return []
            turn.reasoning.append(delta)
            return [_delta_chunk(Delta(reasoning_content=delta))]

        if kind in ("response.output_text.delta", "response.refusal.delta"):
            # O OMP trata `refusal` como texto visível; sem este ramo um turno recusado
            # chegava ao cliente com content vazio e stop limpo.
            delta = str(event.get("delta") or "")
            if not delta:
                return []
            turn.text.append(delta)
            return [_delta_chunk(Delta(content=delta))]

        if kind in ("response.completed", "response.incomplete"):
            payload = event.get("response") or {}
            turn.terminal = True
            turn.usage_meta = payload.get("usage") or {}
            turn.finish_raw = payload.get("status") or (
                "incomplete" if kind == "response.incomplete" else "completed"
            )
            return []

        if kind in ("response.failed", "error"):
            payload = event.get("response") or {}
            detail = payload.get("error") or event.get("message") or "erro desconhecido"
            raise StreamError(f"Codex: {detail}")

        return []

    def close(self) -> list[ModelResponseStream]:
        """Só `response.completed`/`response.incomplete` fecham a resposta.

        Um stream cortado antes disso é falha de transporte: devolvê-lo como sucesso
        entregava output truncado como se estivesse completo.
        """
        if not self._turn.terminal:
            raise StreamError(
                "Codex: stream terminou sem response.completed/response.incomplete"
            )
        return []


def _raise_in_band(event: dict[str, Any]) -> None:
    """O CCA devolve erros dentro do stream com HTTP 200."""
    error = event.get("error")
    if isinstance(error, dict) and int(error.get("code") or 0) >= 400:
        raise StreamError(f"Antigravity {error.get('code')}: {error.get('message') or error}")
    feedback = (event.get("response") or {}).get("promptFeedback")
    if isinstance(feedback, dict) and feedback.get("blockReason"):
        raise StreamError(f"Antigravity: conteúdo bloqueado ({feedback['blockReason']})")


class _AntigravityReader:
    """Traduz eventos de ``:streamGenerateContent``, actualizando um `_Turn`."""

    __slots__ = ("_guard", "_leak", "_tool_index", "_turn", "_wire_model")

    def __init__(self, turn: _Turn, *, wire_model: str) -> None:
        self._turn = turn
        self._wire_model = wire_model
        self._guard = thinking_loop.guard_for(wire_model)
        self._leak = (
            planning_leak.PlanningLeakFilter()
            if planning_leak.is_flash_leak_model(wire_model)
            else None
        )
        self._tool_index = 0

    def feed(self, event: dict[str, Any]) -> list[ModelResponseStream]:
        _raise_in_band(event)
        turn = self._turn
        payload = event.get("response") or {}
        turn.usage_meta = payload.get("usageMetadata") or turn.usage_meta
        candidates = payload.get("candidates") or []
        if not candidates:
            return []
        turn.finish_raw = candidates[0].get("finishReason") or turn.finish_raw

        chunks: list[ModelResponseStream] = []
        for part in (candidates[0].get("content") or {}).get("parts") or []:
            text = str(part.get("text") or "")
            if text:
                chunks.extend(self._text(text, thought=bool(part.get("thought"))))
            call = part.get("functionCall")
            if call:
                chunks.extend(self._call(call, part.get("thoughtSignature")))
        return chunks

    def _text(self, text: str, *, thought: bool) -> list[ModelResponseStream]:
        turn = self._turn
        if thought:
            if self._guard is not None and (reason := self._guard.feed(text)):
                raise thinking_loop.ThinkingLoopError(
                    f"Antigravity: raciocínio em loop ({reason}) após "
                    f"{self._guard.chars} chars em {self._wire_model}; abortado em vez "
                    "de facturar o resto"
                )
            turn.reasoning.append(text)
            return [_delta_chunk(Delta(reasoning_content=text))]
        visible = self._leak.feed(text) if self._leak is not None else text
        if not visible:
            return []
        turn.text.append(visible)
        return [_delta_chunk(Delta(content=visible))]

    def _call(self, call: dict[str, Any], signature: object) -> list[ModelResponseStream]:
        call_id = str(call.get("id") or f"call_{uuid.uuid4().hex[:8]}")
        if signature:
            _remember_signature(call_id, str(signature))
        name = str(call.get("name") or "")
        arguments = json.dumps(call.get("args") or {})
        self._turn.tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
        index = self._tool_index
        self._tool_index += 1
        return [_tool_open_chunk(index, call_id, name), _tool_delta_chunk(index, arguments)]

    def close(self) -> list[ModelResponseStream]:
        """Despeja o que o filtro de leak reteve e afinal não era planeamento."""
        if self._leak is None:
            return []
        tail = self._leak.flush()
        if not tail:
            return []
        self._turn.text.append(tail)
        return [_delta_chunk(Delta(content=tail))]


# -- forma que o LiteLLM espera ------------------------------------------------


def _delta_chunk(delta: Delta) -> ModelResponseStream:
    return ModelResponseStream(
        choices=[StreamingChoices(index=0, delta=delta, finish_reason=None)]
    )


def _tool_open_chunk(index: int, call_id: str, name: str) -> ModelResponseStream:
    """Abre uma tool call. O ``role`` viaja aqui porque pode ser o primeiro chunk do turno."""
    return _delta_chunk(
        Delta(
            role="assistant",
            tool_calls=[
                {
                    "index": index,
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": ""},
                }
            ],
        )
    )


def _tool_delta_chunk(index: int, arguments: str) -> ModelResponseStream:
    return _delta_chunk(
        Delta(tool_calls=[{"index": index, "function": {"arguments": arguments}}])
    )


def _finish_chunk(reason: str) -> ModelResponseStream:
    return ModelResponseStream(
        choices=[StreamingChoices(index=0, delta=Delta(), finish_reason=reason)]
    )


def _usage_chunk(usage: Usage) -> ModelResponseStream:
    """Chunk final com o usage real; sem ele o LiteLLM estima por contagem de tokens.

    ``choices`` leva uma entrada vazia em vez de vir a ``[]``: o iterador da rota
    ``/v1/responses`` faz ``chunk.choices[0].delta`` sem guarda, e uma lista vazia mata o
    stream antes do evento terminal — o cliente fica à espera para sempre.
    """
    chunk = ModelResponseStream(
        choices=[StreamingChoices(index=0, delta=Delta(), finish_reason=None)]
    )
    chunk.usage = _litellm_usage(usage)
    return chunk


def _litellm_usage(usage: Usage) -> litellm.Usage:
    """``cached_tokens`` tem de ir também no atributo que o spend logging lê."""
    out = litellm.Usage(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        prompt_tokens_details={"cached_tokens": usage.cached_tokens},
    )
    out.cache_read_input_tokens = usage.cached_tokens
    return out


def _message(turn: _Turn) -> dict[str, Any]:
    """Mensagem assistant no shape OpenAI, com o raciocínio no campo padronizado."""
    message: dict[str, Any] = {"role": "assistant"}
    if turn.tool_calls:
        message["tool_calls"] = turn.tool_calls
    else:
        message["content"] = "".join(turn.text)
    if turn.reasoning:
        message["reasoning_content"] = "".join(turn.reasoning)
    return message


def _model_response(model: str, turn: _Turn, *, finish_reason: str, usage: Usage) -> ModelResponse:
    return ModelResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[{"index": 0, "message": _message(turn), "finish_reason": finish_reason}],
        usage=_litellm_usage(usage),
    )


# -- despacho ------------------------------------------------------------------


def _normalize(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """``(model, messages)`` posicionais passam a kwargs.

    O LiteLLM aceita ambas as formas; sem isto o despacho via ``kwargs["model"]`` não via
    o modelo e todos os pedidos posicionais caíam no original.
    """
    if args and "model" not in kwargs:
        kwargs["model"] = args[0]
    if len(args) > 1 and "messages" not in kwargs:
        kwargs["messages"] = args[1]
    return kwargs


class _Reader(Protocol):
    def feed(self, event: dict[str, Any]) -> list[ModelResponseStream]: ...
    def close(self) -> list[ModelResponseStream]: ...


async def _drive(events: AsyncIterator[dict[str, Any]], reader: _Reader) -> None:
    """Caminho não-streaming: consome tudo pelo mesmo leitor e descarta os chunks.

    Uma rotina de interpretação só, partilhada com o streaming — ter duas foi o que fez as
    versões síncrona e assíncrona do original divergirem.
    """
    async for event in events:
        reader.feed(event)
    reader.close()


async def _pump(
    events: AsyncIterator[dict[str, Any]], reader: _Reader
) -> AsyncIterator[ModelResponseStream]:
    """Caminho streaming: emite os chunks de cada evento à medida que chegam."""
    async for event in events:
        for chunk in reader.feed(event):
            yield chunk
    for chunk in reader.close():
        yield chunk


async def _codex_turn(model: str, messages: list[Any], extra: dict[str, Any]) -> ModelResponse:
    spec = _codex_spec(model, messages, extra)
    turn = _Turn()
    await _drive(_transport().stream(spec), _CodexReader(turn))
    return _model_response(
        model,
        turn,
        finish_reason=codex_finish_reason(turn.finish_raw, bool(turn.tool_calls)),
        usage=codex_usage(turn.usage_meta),
    )


async def _antigravity_turn(
    model: str, messages: list[Any], extra: dict[str, Any]
) -> ModelResponse:
    spec = _antigravity_spec(model, messages, extra)
    turn = _Turn()
    reader = _AntigravityReader(turn, wire_model=str(spec.body.get("model") or model))
    await _drive(_transport().stream(spec), reader)
    return _model_response(
        model,
        turn,
        finish_reason=google_finish_reason(turn.finish_raw, bool(turn.tool_calls)),
        usage=google_usage(turn.usage_meta),
    )


async def _codex_stream(
    model: str, messages: list[Any], extra: dict[str, Any]
) -> AsyncIterator[ModelResponseStream]:
    spec = _codex_spec(model, messages, extra)
    turn = _Turn()
    async for chunk in _pump(_transport().stream(spec), _CodexReader(turn)):
        yield chunk
    yield _finish_chunk(codex_finish_reason(turn.finish_raw, bool(turn.tool_calls)))
    yield _usage_chunk(codex_usage(turn.usage_meta))


async def _antigravity_stream(
    model: str, messages: list[Any], extra: dict[str, Any]
) -> AsyncIterator[ModelResponseStream]:
    spec = _antigravity_spec(model, messages, extra)
    turn = _Turn()
    reader = _AntigravityReader(turn, wire_model=str(spec.body.get("model") or model))
    async for chunk in _pump(_transport().stream(spec), reader):
        yield chunk
    yield _finish_chunk(google_finish_reason(turn.finish_raw, bool(turn.tool_calls)))
    yield _usage_chunk(google_usage(turn.usage_meta))


async def dispatch(**kwargs: Any) -> Any:
    """Serve o pedido se o modelo for de uma subscrição nossa; ``None`` se não for.

    ``None`` é a única forma de dizer "não é meu" sem fabricar resposta: o chamador
    delega no original. Um modelo nosso que o upstream recuse propaga o erro — `RemapRequired`
    e `RedeemRequired` incluídas, pelas razões no topo do módulo.
    """
    model = str(kwargs.get("model") or "")
    messages = kwargs.get("messages") or []
    streaming = bool(kwargs.get("stream"))

    if is_gemini_model(model):
        if streaming:
            return _wrap_stream(_antigravity_stream(model, messages, kwargs), model, kwargs)
        return await _antigravity_turn(model, messages, kwargs)

    # Depois do Gemini: `codex.is_codex_model` faz match em qualquer nome com "gpt-", e um
    # hipotético "gemini-gpt" pertence ao Google.
    if codex.is_codex_model(model):
        if streaming:
            return _wrap_stream(_codex_stream(model, messages, kwargs), model, kwargs)
        return await _codex_turn(model, messages, kwargs)

    return None


def _wrap_stream(
    chunks: AsyncIterator[ModelResponseStream], model: str, kwargs: dict[str, Any]
) -> litellm.CustomStreamWrapper:
    """Embrulha no iterador do LiteLLM: é ele que o proxy sabe consumir.

    Devolver o gerador cru dava ao cliente objectos sem o protocolo que a rota
    ``/v1/chat/completions`` espera — e nenhum callback de spend log dispararia.
    """
    return litellm.CustomStreamWrapper(
        completion_stream=chunks,
        model=model,
        custom_llm_provider="custom_openai",
        logging_obj=kwargs.get("litellm_logging_obj") or _logging_obj(model, kwargs),
    )


def _logging_obj(model: str, kwargs: dict[str, Any]) -> Logging:
    """Objecto de logging para quando o chamador não traz o dele.

    O `CustomStreamWrapper` desreferencia-o no construtor — passar ``None`` rebenta antes
    do primeiro chunk. O proxy injecta sempre o seu; uma chamada directa à biblioteca não.
    """
    return Logging(
        model=model,
        messages=kwargs.get("messages") or [],
        stream=True,
        call_type="acompletion",
        start_time=datetime.datetime.now(),
        litellm_call_id=str(kwargs.get("litellm_call_id") or uuid.uuid4()),
        function_id=str(kwargs.get("id") or uuid.uuid4()),
    )


def _delegate_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Kwargs para o original, com o prompt Claude aplicado quando é um modelo Claude.

    É o que o ``_inject_claude_prompt`` do original faz: a subscrição Anthropic só valida
    a identidade do Claude Code como system message, e sem isto o pedido é recusado.
    """
    model = str(kwargs.get("model") or "")
    return anthropic.build_request(kwargs, model, _access_token("anthropic"))


async def _wrapped_acompletion(*args: Any, **kwargs: Any) -> Any:
    kwargs = _normalize(args, kwargs)
    served = await dispatch(**kwargs)
    if served is not None:
        return served
    original = _state.original_acompletion
    assert original is not None
    return await original(**_delegate_kwargs(kwargs))


def _wrapped_completion(*args: Any, **kwargs: Any) -> Any:
    """Caminho síncrono: corre o mesmo `dispatch` num loop privado.

    Duplicar a lógica numa versão síncrona foi o que fez as duas derivarem no original. O
    loop é privado porque `asyncio.run` recusa correr dentro de um loop já activo, e o
    proxy chama isto de threads sem loop nenhum.
    """
    kwargs = _normalize(args, kwargs)
    served = _run_sync(dispatch(**kwargs))
    if served is not None:
        return served
    original = _state.original_completion
    assert original is not None
    return original(**_delegate_kwargs(kwargs))


def _run_sync(coroutine: Coroutine[Any, Any, Any]) -> Any:
    """Corre uma corotina a partir de código síncrono, haja ou não loop activo."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    # Chamada síncrona de dentro de um loop: correr numa thread com loop próprio é a
    # única saída que não bloqueia o loop do chamador contra si mesmo.
    result: list[Any] = []
    error: list[BaseException] = []

    def run() -> None:
        try:
            result.append(asyncio.run(coroutine))
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result[0]


#: Sítios que referem as funções de entrada. `litellm/__init__.py` faz
#: `from .main import acompletion`, o que **copia** a referência: rebindar só
#: `litellm.main` deixa `litellm.acompletion` a apontar para a função original, e um
#: cliente que chame `litellm.acompletion(...)` — a forma documentada — nunca passa pelo
#: despacho. O `sitecustomize.py` original patcha os dois (linhas 2916-2917) e o porte
#: começou por patchar só um: o pedido rebentava com "LLM Provider NOT provided", porque
#: chegava ao caminho nativo com um nome que nenhum provedor conhece.
_ASYNC_TARGETS: Final = ((litellm, "acompletion"), (litellm.main, "acompletion"))
_SYNC_TARGETS: Final = ((litellm, "completion"), (litellm.main, "completion"))


def install() -> None:
    """Aplica o patch. Idempotente.

    A guarda não é cosmética: instalar duas vezes encadeava dois wrappers, e o segundo
    guardava o primeiro como "original" — `uninstall` deixava então o patch meio posto e
    cada pedido passava duas vezes pelo despacho.
    """
    if _state.original_acompletion is not None:
        return
    _state.original_acompletion = litellm.main.acompletion
    _state.original_completion = litellm.main.completion
    for module, name in _ASYNC_TARGETS:
        setattr(module, name, _wrapped_acompletion)
    for module, name in _SYNC_TARGETS:
        setattr(module, name, _wrapped_completion)


def uninstall() -> None:
    """Repõe os originais em todos os sítios. Sem patch posto, não faz nada."""
    if _state.original_acompletion is None:
        return
    for module, name in _ASYNC_TARGETS:
        setattr(module, name, _state.original_acompletion)
    if _state.original_completion is not None:
        for module, name in _SYNC_TARGETS:
            setattr(module, name, _state.original_completion)
    _state.original_acompletion = None
    _state.original_completion = None


__all__ = [
    "ANTIGRAVITY_USER_AGENT",
    "CODEX_URL",
    "StreamError",
    "configure",
    "dispatch",
    "install",
    "is_gemini_model",
    "uninstall",
]
