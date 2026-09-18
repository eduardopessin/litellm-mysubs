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
import uuid
from collections.abc import Callable
from typing import Any, Final

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

#: O Codex recusa ``detail: "original"``; reescreve-se para "auto".
IMAGE_DETAILS: Final[tuple[str, ...]] = ("auto", "low", "high")

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


def build_headers(
    token: str,
    *,
    installation_id: str,
    window_id: str,
    turn_id: str | None = None,
    turn_state: str | None = None,
) -> dict[str, str]:
    """Cabeçalhos de um pedido ao backend do Codex.

    Os ids de transporte entram por argumento em vez de virem de estado global do módulo:
    são por processo, e injectá-los é o que permite testar a forma sem os adivinhar.
    """
    claims = token_claims(token)
    auth = claims.get("https://api.openai.com/auth") or {}
    session_id = str(claims.get("session_id") or window_id)

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "accept": "text/event-stream",
        "originator": "pi",
        "OpenAI-Beta": "responses=experimental",
        "User-Agent": "pi (linux; x86_64)",
        "session_id": session_id,
        "session-id": session_id,
        "x-codex-installation-id": installation_id,
        "x-codex-window-id": window_id,
        "x-codex-turn-metadata": json.dumps(
            {
                "turn_id": turn_id or str(uuid.uuid4()),
                "installation_id": installation_id,
                "request_kind": "chat",
            }
        ),
    }

    account = auth.get("chatgpt_account_id")
    if account:
        headers["chatgpt-account-id"] = account

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


def prompt_cache_key(messages: list[Any]) -> str | None:
    """Chave estável por conversa.

    O backend só reaproveita o prefixo em cache quando o pedido repete a mesma chave, e o
    prefixo estável é a cabeça da conversa (developer/system + primeiro turno do
    utilizador). Incluir a cauda mudaria a chave a cada turno e nunca haveria hit.
    """
    digest = hashlib.sha256()
    seen = 0
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role", "user")
        if role not in ("system", "developer", "user"):
            continue
        digest.update(str(role).encode("utf-8"))
        digest.update(content_to_text(message.get("content")).encode("utf-8"))
        seen += 1
        if seen >= 2:
            break
    return digest.hexdigest()[:32] if seen else None


def image_part(part: dict[str, Any]) -> dict[str, str] | None:
    """``image_url`` do chat completions -> ``input_image`` do Responses."""
    image = part.get("image_url")
    url = image.get("url") if isinstance(image, dict) else image
    if not url:
        return None
    detail = (image.get("detail") if isinstance(image, dict) else None) or part.get("detail")
    detail = str(detail or "auto").lower()
    if detail not in IMAGE_DETAILS:
        detail = "auto"
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


#: Tipo de bloco -> construtor. Uma imagem ou ficheiro que não converta é descartado,
#: mas nunca leva o resto do turno com ele.
_PART_BUILDERS: Final[dict[str, Callable[[dict[str, Any]], dict[str, str] | None]]] = {
    "image_url": image_part,
    "input_image": image_part,
    "file": file_part,
    "input_file": file_part,
}


def content_to_parts(content: object, *, assistant: bool = False) -> list[dict[str, str]]:
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
        kind = part.get("type")
        if kind in TEXT_PART_TYPES:
            if part.get("text"):
                parts.append({"type": text_type, "text": str(part["text"])})
            continue
        builder = _PART_BUILDERS.get(str(kind))
        if builder is None:
            continue
        if built := builder(part):
            parts.append(built)
    return parts


# -- tool calls ----------------------------------------------------------------


def composite_call_id(call_id: str | None, item_id: str | None) -> str:
    """Junta ``(call_id, item_id)`` num só identificador.

    O Responses identifica cada tool call pelo par. Juntá-los faz o replay reconstituir o
    par exacto — sem isso, chamadas paralelas desalinham-se.
    """
    if call_id and item_id and call_id != item_id:
        return f"{call_id}|{item_id}"
    return call_id or item_id or f"call_{uuid.uuid4().hex[:8]}"


def split_call_id(value: object) -> str:
    return str(value or "").split("|", 1)[0]


def repair_tool_pairs(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fecha pares ``function_call`` / ``function_call_output`` incompletos.

    O Responses rejeita um output sem a chamada correspondente e uma chamada sem output.
    Um histórico truncado pelo cliente traz exactamente isso, e a alternativa a reparar
    seria devolver 400 por algo que o modelo consegue interpretar.
    """
    calls = {
        item.get("call_id")
        for item in items
        if item.get("type") == "function_call" and item.get("call_id")
    }
    outputs = {
        item.get("call_id")
        for item in items
        if item.get("type") == "function_call_output" and item.get("call_id")
    }

    repaired: list[dict[str, Any]] = []
    for item in items:
        call_id = item.get("call_id")
        if item.get("type") == "function_call_output" and call_id not in calls:
            repaired.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": (
                        f"[Previous tool result; call_id={call_id}]: "
                        f"{content_to_text(item.get('output'))}"
                    ),
                }
            )
        else:
            repaired.append(item)
        if item.get("type") == "function_call" and call_id not in outputs:
            repaired.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": (
                        "[No tool output recorded: the tool call was interrupted "
                        "before it produced a result.]"
                    ),
                }
            )
    return repaired


def messages_to_input(messages: list[Any]) -> list[dict[str, Any]]:
    """Traduz mensagens do chat completions para ``input`` items do Responses."""
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

        codex_role = "developer" if role == "system" else role
        if codex_role not in ("user", "assistant", "developer"):
            codex_role = "user"
        parts = content_to_parts(content, assistant=codex_role == "assistant")
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
    return repair_tool_pairs(items)


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


def build_request_body(
    model: str,
    messages: list[Any],
    tools: list[Any] | None = None,
    extra: dict[str, Any] | None = None,
    unsupported: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Corpo de um pedido à Responses API."""
    req_model = resolve_model(model, unsupported)
    body: dict[str, Any] = {
        "model": req_model,
        "store": False,
        "stream": True,
        "input": messages_to_input(messages),
    }

    if cache_key := prompt_cache_key(messages):
        body["prompt_cache_key"] = cache_key
    if codex_tools := tools_to_codex_tools(tools):
        body["tools"] = codex_tools

    extra = extra or {}
    choice = tool_choice(extra.get("tool_choice"))
    if choice is not None:
        body["tool_choice"] = choice

    # O backend só devolve texto de reasoning quando o pedido traz o objecto `reasoning`
    # (verificado: sem ele, zero eventos response.reasoning_summary_text.delta). O omp
    # manda sempre um effort, por isso o default aqui é "medium" em vez de omitir.
    effort, summary = normalize_effort(extra.get("reasoning_effort"))
    effort = effort or "medium"
    summary = summary if summary in ("auto", "detailed", "concise") else "auto"

    if effort == "none":
        if wire_generation(req_model) >= JUICE_MIN_GENERATION:
            body["input"] = [
                *body["input"],
                {
                    "type": "message",
                    "role": "developer",
                    "content": [
                        {"type": "input_text", "text": f"# Juice: {JUICE['none']} !important"}
                    ],
                },
            ]
    else:
        body["reasoning"] = {"effort": effort, "summary": summary, "context": "all_turns"}

    if extra.get("service_tier") is not None:
        body["service_tier"] = extra["service_tier"]
    return body
