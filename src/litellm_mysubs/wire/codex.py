"""Wire protocol do OpenAI Codex (Responses API) sobre subscrição ChatGPT Plus.

Extraído sem alteração de comportamento do ``sitecustomize.py`` original. Tudo neste
módulo é construção de payload — puro e testável sem rede. O transporte (SSE, quota,
renovação de token) fica fora.

Diferença estrutural face ao bridge da Anthropic: aqui o pedido não é um dicionário de
kwargs do LiteLLM que se ajusta, é um corpo da Responses API construído de raiz. As
mensagens em formato chat completions são traduzidas para ``input`` items.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import uuid
from typing import Any, Final, NamedTuple

# A conta ChatGPT rejeita a família 5.4 com "The 'gpt-5.4' model is not supported when
# using Codex with a ChatGPT account".
#
# Aliases de família, não de versão: "codex"/"gpt-5"/"gpt-6" não prometem uma versão
# concreta, logo resolvê-los para a servida é honesto. `gpt-5.4` e `gpt-5.4-mini` já
# estiveram aqui a apontar para gpt-5.5 — nomeiam uma versão que esta conta não serve, e o
# cliente era facturado e registado contra um modelo que nunca correu. Quem os peça recebe
# a recusa do upstream.
WIRE_ALIASES: Final[dict[str, str]] = {
    "gpt-6": "gpt-6-astra",
    "gpt6": "gpt-6-astra",
    "gpt-5": "gpt-5.5",
    "gpt5": "gpt-5.5",
    "codex": "gpt-5.5",
}

# Com o reasoning desligado, os Responses do GPT-5.6+ continuam a reservar "juice"; o omp
# fixa-o com um item developer no fim do input (getJuiceValue).
JUICE: Final[dict[str, int]] = {
    "none": 0,
    "minimal": 2,
    "low": 4,
    "medium": 8,
    "high": 48,
    "xhigh": 112,
    "max": 960,
}

#: A partir desta geração o item de juice é necessário para desligar mesmo o reasoning.
JUICE_MIN_GENERATION: Final = 5.6

#: ``original`` é um valor válido da API; alguns backends de Responses (o GitHub Copilot,
#: por exemplo) recusam-no com 400, e aí degrada-se para "auto" — a fidelidade mais próxima
#: que passa. Forçar sempre "auto" perdia detalhe em screenshots contra hosts que o servem.
IMAGE_DETAILS: Final[tuple[str, ...]] = ("auto", "low", "high", "original")

# Tools hospedadas pelo backend (web search, geração de imagem, shell…) não têm `function`:
# viajam com o spec próprio e eram descartadas antes de isto existir.
HOSTED_TOOL_TYPES: Final[tuple[str, ...]] = (
    "web_search",
    "web_search_preview",
    "image_generation",
    "code_interpreter",
    "local_shell",
    "computer",
    "computer_use_preview",
    "custom",
    "mcp",
    "file_search",
)

TEXT_PART_TYPES: Final[tuple[str, ...]] = ("text", "input_text", "output_text")


def is_codex_model(model: str) -> bool:
    lowered = str(model).lower()
    return "gpt-" in lowered or "codex" in lowered or lowered.startswith("gpt")


def wire_generation(model: str) -> float:
    """Geração numérica de um nome de modelo: ``gpt-5.6-terra`` -> ``5.6``."""
    pieces = str(model).lower().split("-")
    try:
        return float(pieces[1]) if len(pieces) > 1 else 0.0
    except ValueError:
        return 0.0


def resolve_model(model: str, unsupported: dict[str, str] | None = None) -> str:
    """Nome que vai no fio, depois de aliases e de recusas aprendidas."""
    name = str(model).split("/")[-1]
    name = WIRE_ALIASES.get(name.lower(), name)
    if unsupported:
        name = unsupported.get(name.lower(), name)
    return name


# -- identidade do token -------------------------------------------------------


def token_claims(token: str) -> dict[str, Any]:
    """Claims de um JWT, sem verificar assinatura.

    Não se valida porque não se emite: o token vem do fluxo OAuth e o backend é quem o
    verifica. Aqui só se lê o account id e a residência.
    """
    try:
        parts = str(token).split(".")
        if len(parts) != 3:
            return {}
        padding = len(parts[1]) % 4
        padded = parts[1] + ("=" * (4 - padding) if padding else "")
        decoded = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
        claims = json.loads(decoded)
        return claims if isinstance(claims, dict) else {}
    except Exception:
        return {}


def account_id(token: str) -> str | None:
    auth = token_claims(token).get("https://api.openai.com/auth") or {}
    return auth.get("chatgpt_account_id")


# As constantes de fio do Codex vivem no `pi-catalog`, não no `pi-ai`. A auditoria inicial
# deu-as por inverificáveis por só ter o segundo à mão — estão no npm, e é de lá que estes
# valores vêm.

# omp: wire/codex.ts :: ORIGINATOR_CODEX
#: O port emitia "pi". O backend usa este valor para identificar o cliente.
ORIGINATOR: Final = "omp"

# omp: wire/codex.ts :: CODEX_CLIENT_VERSION
#: O backend fecha a disponibilidade de modelos contra esta versão, em `/models` e em
#: `/responses` — `gpt-6-astra` exige >= 0.153.0. Uma versão antiga esconde SKUs novos da
#: descoberta, em silêncio.
CLIENT_VERSION: Final = "0.153.0"

# omp: wire/codex.ts :: OPENAI_HEADER_VALUES
BETA_RESPONSES: Final = "responses=experimental"

#: User-Agent do cliente. Mantido alinhado com a versão fixada acima.
USER_AGENT: Final = f"codex/{CLIENT_VERSION} (external, cli)"

# omp: providers/openai-codex-responses.ts :: OpenAICodexRequestKind
#: Vocabulário fechado: "turn" | "prewarm" | "compaction". O port emitia "chat", que não
#: pertence ao conjunto.
REQUEST_KIND_TURN: Final = "turn"


# omp: wire/codex.ts :: codexRoutingHint
def routing_hint(model: str, service_tier: str | None = None) -> str:
    """Valor de ``x-codex-routing-hint``: o modelo pedido e, quando há, o tier."""
    return f"model={model};tier={service_tier}" if service_tier else f"model={model}"


def build_headers(
    token: str,
    *,
    window_id: str,
    session_id: str | None = None,
    turn_id: str | None = None,
    turn_state: str | None = None,
    model: str | None = None,
    service_tier: str | None = None,
) -> dict[str, str]:
    """Cabeçalhos de um pedido ao backend do Codex.

    Os ids de transporte entram por argumento em vez de virem de estado global: são por
    processo, e injectá-los é o que permite afirmar a forma sem os adivinhar.
    """
    claims = token_claims(token)
    auth = claims.get("https://api.openai.com/auth") or {}
    resolved_session = session_id or str(claims.get("session_id") or window_id)

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "accept": "text/event-stream",
        "originator": ORIGINATOR,
        "OpenAI-Beta": BETA_RESPONSES,
        "version": CLIENT_VERSION,
        "User-Agent": USER_AGENT,
        "conversation_id": resolved_session,
        "session_id": resolved_session,
        "session-id": resolved_session,
        "x-client-request-id": resolved_session,
        "x-codex-window-id": window_id,
        # O `installation_id` viaja só no envelope de metadata; o OMP apaga-o
        # explicitamente dos cabeçalhos antes de enviar.
        "x-codex-turn-metadata": json.dumps(
            {
                "session_id": resolved_session,
                "thread_id": resolved_session,
                "turn_id": turn_id or str(uuid.uuid4()),
                "window_id": window_id,
                "request_kind": REQUEST_KIND_TURN,
            }
        ),
    }

    account = auth.get("chatgpt_account_id")
    if account:
        headers["chatgpt-account-id"] = account

    # Pista de encaminhamento: o backend usa-a para escolher a rota do modelo. Viaja em
    # todos os pedidos ChatGPT-OAuth; tráfego de API key nunca a leva.
    if model:
        headers["x-codex-routing-hint"] = routing_hint(model, service_tier)

    # O backend devolve x-codex-turn-state e espera-o de volta no turno seguinte: é o
    # estado de transporte da sessão.
    if turn_state:
        headers["x-codex-turn-state"] = turn_state

    # Workspaces enterprise com residência fixada respondem 401 "Workspace is not
    # authorized in this region" a pedidos de outra região. O header só viaja quando o
    # token traz a claim: contas pessoais não a têm.
    residency = auth.get("chatgpt_data_residency") or auth.get("chatgpt_compute_residency")
    if residency and str(residency) != "no_constraint":
        headers["x-openai-internal-codex-residency"] = str(residency)
    return headers


# -- conteúdo ------------------------------------------------------------------


def content_to_text(content: object) -> str:
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in TEXT_PART_TYPES
        )
    return str(content) if content is not None else ""


PROMPT_CACHE_KEY_MAX_CHARS: Final = 64


# omp: providers/openai-shared.ts :: getOpenAIPromptCacheKey
def prompt_cache_key(session_id: str | None, *, cache_retention: str | None = None) -> str | None:
    """Chave de cache de prompt, derivada da **identidade da sessão**.

    Não se deriva do conteúdo: duas conversas distintas que partilhem o prompt de sistema
    e a primeira mensagem colidiriam na mesma chave — entre sessões e entre utilizadores —
    e uma conversa cuja cabeça fosse editada perderia o hit sem razão.

    ``cache_retention="none"`` desliga o cache; sem isto não havia forma de o chamador o
    dispensar.
    """
    if cache_retention == "none" or not session_id:
        return None
    if len(session_id) <= PROMPT_CACHE_KEY_MAX_CHARS:
        return session_id
    return f"pc_{_stable_hash(session_id)}"


# omp: providers/openai-shared.ts :: clampResponsesImageDetail
def clamp_image_detail(detail: object, *, supports_detail_original: bool = True) -> str:
    """Normaliza ``detail``, degradando ``original`` só onde o host o recusa."""
    resolved = str(detail or "auto").lower()
    if resolved not in IMAGE_DETAILS:
        return "auto"
    if resolved == "original" and not supports_detail_original:
        return "auto"
    return resolved


# omp: providers/openai-shared.ts :: convertResponsesInputImage
def image_part(
    part: dict[str, Any], *, supports_detail_original: bool = True
) -> dict[str, str] | None:
    """``image_url`` do chat completions -> ``input_image`` do Responses.

    Uma imagem já carregada para o backend viaja por ``file_id`` e não tem ``url``: sem
    este ramo devolvia-se ``None`` e a imagem era descartada em silêncio.
    """
    image = part.get("image_url")
    spec: dict[str, Any] = image if isinstance(image, dict) else part
    detail = clamp_image_detail(
        spec.get("detail") or part.get("detail"),
        supports_detail_original=supports_detail_original,
    )
    if file_id := spec.get("file_id"):
        return {"type": "input_image", "detail": detail, "file_id": str(file_id)}
    url = image.get("url") if isinstance(image, dict) else image
    if not url:
        return None
    return {"type": "input_image", "image_url": str(url), "detail": detail}


def file_part(part: dict[str, Any]) -> dict[str, str] | None:
    """``file`` do chat completions -> ``input_file`` do Responses."""
    nested = part.get("file")
    spec: dict[str, Any] = nested if isinstance(nested, dict) else part
    data = spec.get("file_data") or spec.get("data")
    file_id = spec.get("file_id")
    if not data and not file_id:
        return None
    item: dict[str, str] = {"type": "input_file"}
    if spec.get("filename"):
        item["filename"] = str(spec["filename"])
    if file_id:
        item["file_id"] = str(file_id)
    else:
        item["file_data"] = str(data)
    return item


#: Uma imagem ou ficheiro que não converta é descartado, mas nunca leva o resto do turno
#: com ele.
IMAGE_PART_TYPES: Final[tuple[str, ...]] = ("image_url", "input_image")
FILE_PART_TYPES: Final[tuple[str, ...]] = ("file", "input_file")


def content_to_parts(
    content: object, *, assistant: bool = False, supports_detail_original: bool = True
) -> list[dict[str, str]]:
    """Preserva imagens e ficheiros em vez de os deixar cair.

    Antes disto, qualquer pedido multimodal chegava ao modelo só com o texto — e a
    resposta falava de uma imagem que ele nunca viu.
    """
    text_type = "output_text" if assistant else "input_text"
    if not isinstance(content, list):
        text = str(content) if content is not None else ""
        return [{"type": text_type, "text": text}] if text else []

    parts: list[dict[str, str]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        kind = str(part.get("type"))
        if kind in TEXT_PART_TYPES:
            if part.get("text"):
                parts.append({"type": text_type, "text": str(part["text"])})
            continue
        built: dict[str, str] | None = None
        if kind in IMAGE_PART_TYPES:
            built = image_part(part, supports_detail_original=supports_detail_original)
        elif kind in FILE_PART_TYPES:
            built = file_part(part)
        if built:
            parts.append(built)
    return parts


# -- tool calls ----------------------------------------------------------------


# omp: providers/transform-messages.ts :: normalizeResponsesToolCallId
def composite_call_id(call_id: str | None, item_id: str | None) -> str:
    """Junta ``(call_id, item_id)`` num só identificador.

    O Responses identifica cada tool call pelo par. Juntá-los faz o replay reconstituir o
    par exacto — sem isso, chamadas paralelas desalinham-se.
    """
    if call_id and item_id and call_id != item_id:
        return f"{call_id}|{item_id}"
    return call_id or item_id or f"call_{uuid.uuid4().hex[:8]}"


#: O backend recusa ids fora deste conjunto ou acima deste comprimento.
CALL_ID_MAX_CHARS: Final = 64
_INVALID_CALL_ID_CHARS: Final = re.compile(r"[^a-zA-Z0-9_-]")
_TRAILING_UNDERSCORES: Final = re.compile(r"_+$")
#: Separadores: `|` é o nosso composto, `\n` aparece em ids reencaminhados de outro provedor.
_CALL_ID_SEPARATOR: Final = re.compile(r"[\n|]")


def _stable_hash(text: str) -> str:
    """Hash curto e determinístico, em base36 como o do OMP."""
    digest = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    out = ""
    while digest:
        digest, remainder = divmod(digest, 36)
        out = alphabet[remainder] + out
    return out or "0"


# omp: providers/openai-codex/request-transformer.ts :: sanitizeCodexCallId
def split_call_id(value: object) -> str:
    """Id de chamada saneado para o wire do Codex.

    Um id vindo de outro provedor traz frequentemente caracteres que o backend recusa, ou
    passa dos 64 caracteres; deixá-lo passar cru dá 400. Quando é preciso alterar o id,
    acrescenta-se um hash para que dois ids diferentes não colapsem no mesmo.
    """
    raw = str(value or "")
    if not raw:
        return f"call_{_stable_hash('empty')}"

    match = _CALL_ID_SEPARATOR.search(raw)
    if match is None:
        base = raw
    elif match.start() == 0:
        base = raw[1:]
    else:
        base = raw[: match.start()]

    sanitized = _TRAILING_UNDERSCORES.sub("", _INVALID_CALL_ID_CHARS.sub("_", base))
    if 0 < len(sanitized) <= CALL_ID_MAX_CHARS and sanitized == base:
        return sanitized

    digest = _stable_hash(base or raw)
    effective = sanitized or "call"
    prefix_length = max(0, CALL_ID_MAX_CHARS - 1 - len(digest))
    return f"{effective[:prefix_length]}_{digest}"[:CALL_ID_MAX_CHARS]


# omp: providers/openai-codex/request-transformer.ts :: CODEX_ORPHAN_OUTPUT_LIMIT
#: Um resultado órfão gigante (a leitura de um ficheiro de 2 MB, por exemplo) rebentava o
#: limite do corpo do pedido em vez de ser cortado.
ORPHAN_OUTPUT_LIMIT: Final = 16_000

# omp: providers/openai-codex/request-transformer.ts :: CODEX_INTERRUPTED_TOOL_OUTPUT
INTERRUPTED_TOOL_OUTPUT: Final = (
    "[No tool output recorded: the tool call was interrupted before it produced a result.]"
)


def _orphan_output_text(item: dict[str, Any]) -> str:
    """Texto de um resultado cuja chamada se perdeu, truncado."""
    output = item.get("output")
    if isinstance(output, str):
        text = output
    else:
        try:
            text = json.dumps(output)
        except (TypeError, ValueError):
            text = str(output if output is not None else "")
    if len(text) > ORPHAN_OUTPUT_LIMIT:
        text = f"{text[:ORPHAN_OUTPUT_LIMIT]}\n...[truncated]"
    return text


#: Texto literal da fonte (lá está inline no ramo `computer` de `repairToolCallPairs`, sem
#: nome próprio). Uma `computer_call` não tem output sintetizável: a screenshot que faltou
#: não se inventa, logo a chamada passa a nota que o modelo lê.
INTERRUPTED_COMPUTER_CALL: Final = (
    "[Computer call interrupted before a screenshot was recorded; call_id={call_id}]"
)

#: Item de chamada -> tipo da tool. O par só fecha entre itens do **mesmo** tipo: o
#: Responses recusa um ``custom_tool_call_output`` a fechar um ``function_call``.
_CALL_KINDS: Final[dict[str, str]] = {
    "function_call": "function",
    "custom_tool_call": "custom",
    "computer_call": "computer",
}
_OUTPUT_KINDS: Final[dict[str, str]] = {
    "function_call_output": "function",
    "custom_tool_call_output": "custom",
    "computer_call_output": "computer",
}


# omp: providers/openai-codex/request-transformer.ts :: repairToolCallPairs, toolCallKind
# omp: providers/openai-codex/request-transformer.ts :: toolOutputKind
# omp: providers/openai-codex/request-transformer.ts :: orphanFunctionOutputToMessage
def repair_tool_pairs(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fecha metades soltas de uma troca com tool, indexadas por **tipo** de tool.

    O Responses rejeita com 400 um output sem a chamada e uma chamada sem output. Um
    histórico truncado pelo cliente (ou um turno abortado depois de a chamada ter sido
    emitida) traz exactamente isso, e reparar é preferível a 400 por algo que o modelo
    interpreta. Indexar só por ``call_id`` emparelhava tipos diferentes — um
    ``custom_tool_call_output`` a "fechar" um ``function_call`` volta a dar 400.
    """
    call_kinds: dict[str, str] = {}
    output_kinds: dict[str, str] = {}
    for item in items:
        call_id = item.get("call_id")
        if not isinstance(call_id, str):
            continue
        item_type = str(item.get("type"))
        if kind := _CALL_KINDS.get(item_type):
            call_kinds[call_id] = kind
        if kind := _OUTPUT_KINDS.get(item_type):
            output_kinds[call_id] = kind

    repaired: list[dict[str, Any]] = []
    for item in items:
        call_id = item.get("call_id")
        call_id = call_id if isinstance(call_id, str) else None
        item_type = str(item.get("type"))
        call_kind = _CALL_KINDS.get(item_type)
        output_kind = _OUTPUT_KINDS.get(item_type)

        if output_kind and call_id is not None and call_kinds.get(call_id) != output_kind:
            # O nome da tool vem do próprio item: sem ele o modelo não sabe o que produziu
            # o resultado órfão.
            tool_name = item.get("name") if isinstance(item.get("name"), str) else "tool"
            repaired.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": (
                        f"[Previous {tool_name} result; call_id={call_id}]: "
                        f"{_orphan_output_text(item)}"
                    ),
                }
            )
            continue
        if call_kind and call_id is not None and output_kinds.get(call_id) != call_kind:
            if call_kind == "computer":
                repaired.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": INTERRUPTED_COMPUTER_CALL.format(call_id=call_id),
                    }
                )
                continue
            repaired.append(item)
            repaired.append(
                {
                    "type": (
                        "custom_tool_call_output"
                        if call_kind == "custom"
                        else "function_call_output"
                    ),
                    "call_id": call_id,
                    "output": INTERRUPTED_TOOL_OUTPUT,
                }
            )
            continue
        repaired.append(item)
    return repaired


class CodexInput(NamedTuple):
    """``instructions`` e ``input`` são campos distintos do pedido, não um só."""

    instructions: str | None
    items: list[dict[str, Any]]


def _last_developer_text(items: list[dict[str, Any]]) -> str | None:
    """Último texto developer do input, do fim para o princípio."""
    for item in reversed(items):
        if item.get("role") != "developer":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in reversed(content):
            if not isinstance(part, dict) or part.get("type") != "input_text":
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                return text
    return None


# omp: providers/openai-codex-responses.ts :: buildTransformedCodexRequestBody
# omp: providers/openai-codex/request-transformer.ts :: transformRequestBody
# omp: utils.ts :: normalizeSystemPrompts
def messages_to_input(messages: list[Any], *, supports_detail_original: bool = True) -> CodexInput:
    """Traduz mensagens do chat completions para ``instructions`` + ``input`` items.

    O **primeiro** system prompt vai para ``instructions``, que o backend trata como prompt
    base cacheável; mandá-lo como item developer perdia esse tratamento. Os restantes não
    cabem lá (o campo é uma string) e viajam como itens developer no topo do input, antes
    da conversa.
    """
    instructions: str | None = None
    developer_items: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content")

        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": split_call_id(message.get("tool_call_id")),
                    "output": content_to_text(content),
                }
            )
            continue

        if role == "system":
            text = content_to_text(content)
            if not text.strip():
                continue
            if instructions is None:
                instructions = text
            else:
                developer_items.append(
                    {
                        "type": "message",
                        "role": "developer",
                        "content": [{"type": "input_text", "text": text}],
                    }
                )
            continue

        codex_role = role if role in ("user", "assistant", "developer") else "user"
        parts = content_to_parts(
            content,
            assistant=codex_role == "assistant",
            supports_detail_original=supports_detail_original,
        )
        if parts:
            items.append({"type": "message", "role": codex_role, "content": parts})

        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            arguments = function.get("arguments") or "{}"
            items.append(
                {
                    "type": "function_call",
                    "call_id": split_call_id(tool_call.get("id")),
                    "name": function.get("name") or "",
                    "arguments": json.dumps(arguments)
                    if isinstance(arguments, dict)
                    else arguments,
                }
            )

    repaired = repair_tool_pairs([*developer_items, *items])

    # Um input só com itens developer (prompt de sistema sem turno de utilizador) faz o
    # backend devolver resposta vazia: promove-se a última instrução a turno `user` para
    # haver algo a que responder.
    if not any(item.get("role") != "developer" for item in repaired):
        final = _last_developer_text(developer_items) or _last_developer_text(repaired)
        final = final or (instructions if instructions and instructions.strip() else None)
        if final is not None:
            repaired.append(
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": final}],
                }
            )
    return CodexInput(instructions, repaired)


def tools_to_codex_tools(tools: list[Any] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") in HOSTED_TOOL_TYPES:
            converted.append(dict(tool))
            continue
        nested = tool.get("function")
        function: dict[str, Any] = nested if isinstance(nested, dict) else tool
        if not function.get("name"):
            continue
        converted.append(
            {
                "type": "function",
                "name": function["name"],
                "description": function.get("description") or "",
                "parameters": function.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return converted or None


def tool_choice(choice: object) -> object:
    """O Responses usa ``{"type": "function", "name": …}``, sem o nível ``function``."""
    if not isinstance(choice, dict):
        return choice
    function = choice.get("function")
    if choice.get("type") == "function" and isinstance(function, dict) and function.get("name"):
        return {"type": "function", "name": function["name"]}
    return choice


# -- corpo do pedido -----------------------------------------------------------


def normalize_effort(value: object) -> tuple[str | None, str | None]:
    """Devolve ``(effort, summary)``; ver a mesma função em ``wire.anthropic``."""
    if isinstance(value, dict):
        effort = value.get("effort")
        summary = value.get("summary")
    else:
        effort = value
        summary = None
    return (
        str(effort or "").strip().lower() or None,
        str(summary).strip().lower() if summary else None,
    )


# omp: providers/openai-responses.ts :: getJuiceValue
def juice_for(effort: str | None) -> int:
    """Orçamento de raciocínio reservado quando o thinking é desligado.

    O valor é o do effort **pedido pelo cliente**, não zero: desligar o raciocínio não
    significa que o modelo deva ficar sem orçamento nenhum. Um effort desconhecido cai no
    default de ``medium``.
    """
    return JUICE.get(str(effort or "medium").strip().lower(), JUICE["medium"])


def build_request_body(
    model: str,
    messages: list[Any],
    tools: list[Any] | None = None,
    extra: dict[str, Any] | None = None,
    unsupported: dict[str, str] | None = None,
    session_id: str | None = None,
    supports_detail_original: bool = True,
) -> dict[str, Any]:
    """Corpo de um pedido à Responses API."""
    req_model = resolve_model(model, unsupported)
    extra = extra or {}
    instructions, items = messages_to_input(
        messages, supports_detail_original=supports_detail_original
    )
    body: dict[str, Any] = {
        "model": req_model,
        "store": False,
        "stream": True,
        "input": items,
        # Sem isto o backend não devolve o raciocínio encriptado, e num histórico
        # stateless (`store: false`) o modelo recomeça a raciocinar a cada turno.
        "include": ["reasoning.encrypted_content"],
    }
    if instructions is not None:
        body["instructions"] = instructions

    cache_key = prompt_cache_key(session_id, cache_retention=extra.get("cache_retention"))
    if cache_key:
        body["prompt_cache_key"] = cache_key
    if codex_tools := tools_to_codex_tools(tools):
        body["tools"] = codex_tools

    choice = tool_choice(extra.get("tool_choice"))
    if choice is not None:
        body["tool_choice"] = choice

    # O backend só devolve texto de reasoning quando o pedido traz o objecto `reasoning`
    # (verificado: sem ele, zero eventos response.reasoning_summary_text.delta). O omp
    # manda sempre um effort, por isso o default aqui é "medium" em vez de omitir.
    effort, summary = normalize_effort(extra.get("reasoning_effort"))
    summary = summary if summary in ("auto", "detailed", "concise") else "auto"

    if effort == "none":
        # Desligar o raciocínio não dispensa o item: as gerações recentes continuam a
        # reservar juice, e é ele que o fixa no valor pedido.
        if wire_generation(req_model) >= JUICE_MIN_GENERATION:
            body["input"] = [
                *body["input"],
                {
                    "type": "message",
                    "role": "developer",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                f"# Juice: "
                                f"{juice_for(extra.get('juice_effort') or effort)} !important"
                            ),
                        }
                    ],
                },
            ]
    else:
        body["reasoning"] = {"effort": effort or "medium", "summary": summary}

    if extra.get("service_tier") is not None:
        body["service_tier"] = extra["service_tier"]
    return body
