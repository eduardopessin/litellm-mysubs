"""Wire protocol da Anthropic sobre OAuth de subscrição (Claude Max).

Extraído sem alteração de comportamento do ``sitecustomize.py`` original. As medições
nos comentários são o que justifica cada decisão e vêm do serviço real, não de
documentação — são a parte mais valiosa deste módulo e não devem ser apagadas.

Divisão interna: as funções de cache e de parâmetros são puras e testáveis sozinhas;
``build_request`` é a única que precisa de um token, e recebe-o como argumento em vez de
ir buscá-lo a estado global.
"""

from __future__ import annotations

import re
from typing import Any, Final

# omp: providers/claude-code-fingerprint.ts :: claudeCodeSystemInstruction
#: Bloco de identidade que o runtime do Claude Code antepõe. A medição que existia antes
#: comparava *identidade vs ausência de identidade*, não *esta string vs a do Claude Code*
#: — e a string herdada do intermediário não era a que o CLI real põe no fio.
CLAUDE_CODE_PROMPT: Final = "You are Claude Code, Anthropic's official CLI for Claude."

# omp: stream.ts :: ANTHROPIC_THINKING
# Efeito -> orçamento de thinking. Os degraus são os do OMP; só o topo é que difere, e a
# razão está em `THINKING_CEILING`.
EFFORT_BUDGET: Final[dict[str, int]] = {
    "minimal": 1024,
    "low": 4096,
    "medium": 8192,
    "high": 16384,
    "xhigh": 32768,
    "max": 32768,
}

#: A janela TPM curta da subscrição Max não aguenta os 32768 do OMP: pedidos acima disto
#: devolvem 429. O tecto aplica-se depois de escolher o degrau, para que a escala do OMP
#: seja preservada em vez de ser achatada na tabela.
THINKING_CEILING: Final = 8192

# omp: stream.ts :: OUTPUT_FALLBACK_BUFFER
#: Margem de output reservada para lá do orçamento de raciocínio. Um pedido cujo
#: `max_tokens` fique abaixo de `budget + isto` não tem espaço para responder depois de
#: pensar, e a resposta sai truncada.
OUTPUT_FALLBACK_BUFFER: Final = 4000

# omp: providers/claude-code-fingerprint.ts :: CLAUDE_CODE_MAX_OUTPUT_TOKENS
MAX_OUTPUT_TOKENS: Final = 64000

# Medido no upstream (max_tokens=2048, display="summarized", pergunta que exige
# raciocínio): xhigh e max são aceites e rendem mais output que high (out=164 em high,
# 273 em xhigh, 275 em max), logo colapsá-los em "high" escondia dois degraus reais.
ADAPTIVE_EFFORT: Final[dict[str, str]] = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}

# O default é adaptive, e esta lista enumera quem o *rejeita*, não quem o aceita.
# Medido modelo a modelo (max_tokens 1024/4096, display="summarized"), chars de thinking
# devolvidos com cada forma:
#   opus-5       adaptive  82 | budget    0   <- adaptive obrigatório
#   fable-5      adaptive  83 | budget    0   <- adaptive obrigatório
#   sonnet-5     adaptive  58 | budget   59
#   opus-4-8     adaptive  61 | budget   62
#   opus-4-6     adaptive 101 | budget  101
#   sonnet-4-6   adaptive 104 | budget  105
#   opus-4-5     adaptive 400 | budget  233   <- adaptive rejeitado
#   sonnet-4-5   adaptive 400 | budget  367   <- adaptive rejeitado
#   haiku-4-5    adaptive 400 | budget  413   <- adaptive rejeitado
# A assimetria é o que decide o default: errar para adaptive dá 400 "adaptive thinking is
# not supported on this model"; errar para budget dá 200 com 0 chars de raciocínio. Assim
# um alias (claude-opus -> opus-4-8) ou um modelo novo ainda não enumerado cai no lado que
# se detecta.
BUDGET_ONLY_MODELS: Final[tuple[str, ...]] = (
    "opus-4-5",
    "sonnet-4-5",
    "haiku-4-5",
    "opus-4-1",
    "opus-4-0",
    "sonnet-4-1",
    "sonnet-4-0",
    "3-7-sonnet",
    "sonnet-3-7",
    "3-5-sonnet",
    "3-5-haiku",
    "3-opus",
    "opus-3",
)

# A Anthropic faz cache de tudo *até* um breakpoint, e a ordem canónica no fio é
# tools -> system -> messages. Dois marcadores adjacentes na cauda (não um) mantêm uma
# entrada válida para estender à medida que a conversa cresce.
CACHE_BREAKPOINT_MESSAGES: Final = 2

# omp: providers/anthropic.ts :: ANTHROPIC_DECIMATION_INTERVAL
# Checkpoint histórico estável a cada 15 turnos de utilizador (15.º, 30.º, 45.º...). As
# duas âncoras de cauda movem-se a cada turno, logo quando a janela de 5 min expira não
# há nenhuma entrada viva que cubra o prefixo antigo e ele é relido a preço cheio. Um
# marcador em posição fixa sobrevive ao churn da cauda e apanha esse prefixo.
DECIMATION_INTERVAL: Final = 15

# omp: providers/anthropic.ts :: VOLATILE_SYSTEM_SEGMENT_MARKERS
#: Segmentos de system que mudam a cada turno. A âncora de system fica no último bloco
#: *antes* deles, para que um refresh de memória re-facture só o sufixo em vez do head
#: inteiro. A detecção é pela nossa própria marcação, e só conta em início de bloco: um
#: `<memories>` citado a meio de um bloco estável não o torna volátil.
VOLATILE_SYSTEM_MARKERS: Final[tuple[str, ...]] = ("<memories>",)

# Um cliente que faça o seu próprio caching chega aqui com marcadores postos. Medido com
# claude-sonnet-4-6: 4 marcadores -> 200, 5 -> 400 "A maximum of 4 blocks with
# cache_control may be provided. Found 5." Três marcadores do cliente fora da nossa janela
# de cauda mais os nossos dois davam exactamente esse 400.
CACHE_BREAKPOINT_CEILING: Final = 4

#: Blocos que carregam raciocínio nunca são âncoras válidas.
UNCACHEABLE_BLOCKS: Final[tuple[str, ...]] = ("thinking", "redacted_thinking", "fallback")

# Ferramentas hospedadas: o LiteLLM emite `server_tool_use` sem cache_control
# (factory.py:1971), logo uma chamada destas não serve de âncora.
SERVER_TOOL_PREFIX: Final = "srvtoolu_"

# omp: providers/anthropic.ts :: claudeCodeAgentBetaDefaults
# Ordem e conteúdo da fonte. Notas sobre o que **não** está aqui:
#  - `context-1m-2025-08-07`: credenciais OAuth não têm saldo de contexto longo, e a
#    Anthropic devolve 429 duro em qualquer modelo com a beta, independentemente do
#    tamanho do prompt. O OMP também nunca a anuncia.
#  - `redact-thinking-2026-02-12`: faz devolver blocos thinking assinados mas sem texto
#    (medido em sonnet-4-6: 74 chars sem a beta, 0 com ela). O OMP também não a envia na
#    inferência — só no cabeçalho da rota de usage.
#  - `structured-outputs-2025-12-15`: é da lista de utilitário, não da de agente.
AGENT_BETAS: Final[tuple[str, ...]] = (
    "claude-code-20250219",
    # A única específica de credencial OAuth. Sem ela o servidor classifica o pedido
    # como sendo de API key — faltava por ter sido portada do intermediário.
    "oauth-2025-04-20",
    "interleaved-thinking-2025-05-14",
    "thinking-token-count-2026-05-13",
    "context-management-2025-06-27",
    "prompt-caching-scope-2026-01-05",
    "mid-conversation-system-2026-04-07",
)

#: Acrescentada só quando o pedido pede raciocínio.
EFFORT_BETA: Final = "effort-2025-11-24"
#: Acrescentada a todos os pedidos de agente.
FALLBACK_CREDIT_BETA: Final = "fallback-credit-2026-06-01"
#: Acrescentada quando alguma âncora leva `ttl: "1h"`.
EXTENDED_CACHE_TTL_BETA: Final = "extended-cache-ttl-2025-04-11"


# omp: providers/anthropic.ts :: buildClaudeCodeBetas
def build_betas(*, thinking: bool) -> str:
    """Cabeçalho ``anthropic-beta`` para um pedido de agente."""
    betas = [*AGENT_BETAS]
    if thinking:
        betas.append(EFFORT_BETA)
    betas.append(FALLBACK_CREDIT_BETA)
    # A beta NÃO viaja no caminho OAuth. O OMP só a junta quando `!isOAuth`
    # (`providers/anthropic.ts`), e `getCacheControl` mostra porquê: para OAuth o default
    # já é `ttl: "1h"` em modelos que o suportam, sem beta nenhuma.
    #
    # O cabeçalho de `usage/claude.ts` traz esta beta e podia parecer o contra-exemplo,
    # mas é da rota de *usage* e traz também `redact-thinking-2026-02-12` — que medimos a
    # devolver blocos de thinking vazios. Copiá-lo para a inferência partia o raciocínio.
    return ",".join(betas)


# omp: providers/claude-code-fingerprint.ts :: claudeCodeUserAgent
CLAUDE_CODE_VERSION: Final = "2.1.257"
#: O entrypoint tem de ser `cli` para ser coerente com o `x-app` que segue no mesmo pedido.
CLAUDE_CODE_USER_AGENT: Final = f"claude-cli/{CLAUDE_CODE_VERSION} (external, cli)"

CLIENT_HEADERS: Final[dict[str, str]] = {
    "User-Agent": CLAUDE_CODE_USER_AGENT,
    "anthropic-dangerous-direct-browser-access": "true",
    "x-app": "cli",
}


def is_anthropic_model(model: str) -> bool:
    lowered = str(model).lower()
    return "claude" in lowered or "anthropic" in lowered


def is_adaptive(model: str) -> bool:
    """Se o modelo usa ``thinking: adaptive`` em vez de ``budget_tokens``."""
    lowered = str(model).lower()
    return not any(marker in lowered for marker in BUDGET_ONLY_MODELS)


#: Modelos que aceitam ``thinking.display``, por ordem de especificidade.
#:
#: A regra da fonte é geracional, não uma lista: opus a partir de 4.7, e
#: sonnet/fable/mythos a partir de 5. Não coincide com ``is_adaptive`` — opus-4-6 e
#: sonnet-4-6 são adaptativos mas **não** aceitam ``display``, e mandá-lo dá 400.
_DISPLAY_FLOORS: Final[tuple[tuple[str, float], ...]] = (
    ("opus", 4.7),
    ("sonnet", 5.0),
    ("fable", 5.0),
    ("mythos", 5.0),
)


# omp: compat/resolve.ts :: defaultSupportsDisplay
def supports_display(model: str) -> bool:
    """Se o modelo aceita ``thinking.display``.

    ``display: "summarized"`` é o que faz o raciocínio voltar em texto legível: a partir
    do Opus 4.7 o conteúdo é omitido da resposta por default. O campo é estritamente
    fechado por modelo — quem não o suporta responde 400 — por isso não basta ser
    adaptativo.
    """
    lowered = str(model).lower()
    for family, floor in _DISPLAY_FLOORS:
        if family not in lowered:
            continue
        match = re.search(rf"{family}[^0-9]*(\d+)(?:[.-](\d+))?", lowered)
        if not match:
            return False
        major = int(match.group(1))
        minor = int(match.group(2) or 0)
        return major + minor / 10 >= floor
    return False


def normalize_effort(value: object) -> tuple[str | None, str | None]:
    """Devolve ``(effort, summary)`` a partir de ``reasoning_effort``.

    A rota ``/v1/responses`` entrega ``reasoning: {effort, summary}`` e o tradutor do
    LiteLLM reencaminha o objecto inteiro. Tratá-lo como string punha
    ``"{'effort': 'medium', …}"`` no fio, o que dá 400 Invalid value.
    """
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


#: Retenção default. O OMP defaulta a "long" para OAuth em modelo com
#: `supportsLongCacheRetention`, "matching Claude Code's native policy"; este módulo só
#: serve o caminho OAuth do Claude Code, logo a condição colapsa no default.
LONG_CACHE_TTL: Final = "1h"


# omp: providers/anthropic.ts :: getCacheControl
def cache_control(ttl: str | None = LONG_CACHE_TTL) -> dict[str, str]:
    """Marcador de cache, com ``ttl`` de 1 h por default e ``None`` para os 5 min base.

    O trade-off é de custo contra frequência de reescrita: uma escrita de 1 h factura 2x o
    preço base do token contra 1.25x para os 5 min. Numa sessão de agente o prefixo é
    relido dezenas de vezes e as pausas entre turnos passam facilmente dos 5 min, logo
    pagar 2x uma vez sai mais barato do que pagar 1.25x a cada reescrita a frio — que é
    exactamente a razão pela qual o Claude Code nativo defaulta a 1 h.
    """
    if not ttl:
        return {"type": "ephemeral"}
    return {"type": "ephemeral", "ttl": ttl}


# -- âncoras de cache ----------------------------------------------------------


def tool_call_anchor(message: dict[str, Any]) -> int | None:
    """Índice do último tool call que o LiteLLM aceita marcar.

    ``convert_to_anthropic_tool_invoke:1952`` salta o que não é ``type: "function"``.
    """
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return None
    for index in range(len(calls) - 1, -1, -1):
        call = calls[index]
        if not isinstance(call, dict) or call.get("type") != "function":
            continue
        if str(call.get("id") or "").startswith(SERVER_TOOL_PREFIX):
            continue
        return index
    return None


def is_markable(message: object) -> bool:
    """Se um breakpoint pode ser preso a esta mensagem.

    O omp marca o wire da Anthropic, onde um tool result é um bloco ``tool_result`` dentro
    de um turno ``user``, logo a janela rolante dele cai sempre nos dois últimos turnos.
    Aqui vê-se a forma OpenAI: o tool result é uma mensagem ``role: "tool"`` própria e o
    tool call do assistant traz ``content: None``. O LiteLLM propaga o breakpoint em
    ambos, mas lê-o de níveis diferentes
    (``litellm_core_utils/prompt_templates/factory.py``):

      - ``role: "tool"``  -> nível-mensagem, ``convert_to_anthropic_tool_result:1844``
      - ``tool_calls[i]`` -> dentro da chamada, ``convert_to_anthropic_tool_invoke:2003``
      - blocos de texto   -> no próprio bloco

    Recusar os dois primeiros prendia a janela à cabeça da conversa: num turno terminado
    em tool result, 67% do prompt era relido a preço cheio (medido: opus-5 pt=8697,
    read=2876).
    """
    if not isinstance(message, dict):
        return False
    role = message.get("role")
    if role == "tool" or message.get("tool_call_id"):
        return True
    if role not in ("user", "assistant", "developer"):
        return False
    if tool_call_anchor(message) is not None:
        return True
    content = message.get("content")
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        return any(
            isinstance(block, dict)
            and block.get("type") not in UNCACHEABLE_BLOCKS
            and str(block.get("text", "")).strip()
            for block in content
        )
    return False


def count_breakpoints(messages: list[Any]) -> int:
    """Marcadores já presentes, venham de onde vierem."""
    total = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("cache_control"):
            total += 1
        for call in message.get("tool_calls") or ():
            if isinstance(call, dict) and call.get("cache_control"):
                total += 1
        content = message.get("content")
        if isinstance(content, list):
            total += sum(
                1 for block in content if isinstance(block, dict) and block.get("cache_control")
            )
    return total


def mark_breakpoint(message: dict[str, Any]) -> bool:
    """Marca a última âncora não-raciocínio; desiste se já houver uma."""
    control = cache_control()
    if message.get("role") == "tool" or message.get("tool_call_id"):
        if message.get("cache_control") is not None:
            return False
        message["cache_control"] = control
        return True

    # Um assistant pode trazer texto e tool calls; no wire da Anthropic o `tool_use` vem
    # depois do texto, logo é ele a âncora que cobre mais prefixo.
    call_index = tool_call_anchor(message)
    if call_index is not None:
        call = message["tool_calls"][call_index]
        if call.get("cache_control") is not None:
            return False
        call["cache_control"] = control
        return True

    content = message.get("content")
    if isinstance(content, str):
        message["content"] = [{"type": "text", "text": content, "cache_control": control}]
        return True
    if not isinstance(content, list):
        return False
    for index in range(len(content) - 1, -1, -1):
        block = content[index]
        if not isinstance(block, dict) or block.get("type") in UNCACHEABLE_BLOCKS:
            continue
        if block.get("cache_control") is not None:
            return False
        if not str(block.get("text", "")).strip():
            continue
        block["cache_control"] = control
        return True
    return False


# omp: providers/anthropic.ts :: stableSystemSuffixStart
def stable_system_suffix_start(blocks: list[Any]) -> int:
    """Índice onde começa o sufixo volátil de system; ``len(blocks)`` se não houver."""
    start = len(blocks)
    while start > 0:
        block = blocks[start - 1]
        text = str(block.get("text", "")) if isinstance(block, dict) else ""
        if not any(text.startswith(marker) for marker in VOLATILE_SYSTEM_MARKERS):
            break
        start -= 1
    return start


def _is_deferred_tool(tool: Any) -> bool:
    """O LiteLLM aceita ``defer_loading`` no topo ou dentro de ``function``
    (``transformation.py:843``), logo os dois sítios contam."""
    if not isinstance(tool, dict):
        return False
    if tool.get("defer_loading"):
        return True
    nested = tool.get("function")
    return bool(isinstance(nested, dict) and nested.get("defer_loading"))


# omp: providers/anthropic.ts :: countHeadBreakpoints
def count_head_breakpoints(system_blocks: list[Any] | None, tools: list[Any] | None) -> int:
    """Marcadores presentes em system e em tools."""
    total = 0
    for block in system_blocks or ():
        if isinstance(block, dict) and block.get("cache_control") is not None:
            total += 1
    for tool in tools or ():
        if isinstance(tool, dict) and tool.get("cache_control") is not None:
            total += 1
    return total


# omp: providers/anthropic.ts :: applyHeadCaching
def apply_head_cache(system_blocks: list[Any] | None, tools: list[Any] | None) -> int:
    """Ancora o head estável — última tool não-deferred e último bloco estável de system.

    Devolve quantos marcadores ficaram no head. A ordem no fio é tools -> system ->
    messages, logo um marcador no último bloco estável de system faz cache do prefixo
    tools+system inteiro; o marcador nas tools mantém as definições em cache mesmo quando
    o texto de system muda. Sem isto o head só era coberto pela âncora de cauda, que se
    move a cada turno, e portanto era reescrito a preço cheio a cada pedido.
    """
    if tools and not any(
        isinstance(tool, dict) and tool.get("cache_control") is not None for tool in tools
    ):
        # Uma tool deferred não entra no prefixo verificado enquanto não for referida, por
        # isso ancorar nela deixaria de fora tudo o que vem antes.
        for tool in reversed(tools):
            if not isinstance(tool, dict) or _is_deferred_tool(tool):
                continue
            tool["cache_control"] = cache_control()
            break

    if system_blocks:
        suffix_start = stable_system_suffix_start(system_blocks)
        if suffix_start == len(system_blocks):
            if not any(
                isinstance(b, dict) and b.get("cache_control") is not None for b in system_blocks
            ):
                last = system_blocks[-1]
                if isinstance(last, dict):
                    last["cache_control"] = cache_control()
        else:
            # Com sufixo volátil o marcador de fronteira entra mesmo que já haja um mais
            # atrás: caso contrário o único marcador de system fica antes do prompt
            # estável e um refresh de memória re-factura-o.
            anchor_index = len(system_blocks) - 1 if suffix_start == 0 else suffix_start - 1
            anchor = system_blocks[anchor_index]
            if isinstance(anchor, dict) and anchor.get("cache_control") is None:
                anchor["cache_control"] = cache_control()

    return count_head_breakpoints(system_blocks, tools)


def _decimation_indices(messages: list[Any], end: int) -> list[int]:
    """Índices dos turnos de utilizador cujo ordinal é múltiplo de ``DECIMATION_INTERVAL``.

    Limitação face à fonte: o OMP conta turnos por `isConversationalUser`, um marcador de
    proveniência que distingue um turno humano de um `developer` serializado ou de um
    "Continue." interior. Recebemos kwargs em forma OpenAI e esse marcador não existe no
    fio, logo a aproximação é ``role == "user"`` — uma mensagem `user` sintetizada conta
    como turno onde o OMP não a contaria, o que desloca os checkpoints para posições mais
    recentes do que as canónicas. Eles continuam a cair em posições fixas ao longo da
    conversa, que é o que lhes dá valor.
    """
    indices: list[int] = []
    ordinal = 0
    for index in range(min(end + 1, len(messages))):
        message = messages[index]
        if isinstance(message, dict) and message.get("role") == "user":
            ordinal += 1
            if ordinal % DECIMATION_INTERVAL == 0:
                indices.append(index)
    return indices


# omp: providers/anthropic.ts :: applyPromptCaching
# omp: providers/anthropic.ts :: cloneAnthropicCacheControl
def apply_conversation_cache(messages: list[Any], head_breakpoints: int = 0) -> int:
    """Ancora os breakpoints nas mensagens. Muta ``messages``.

    ``head_breakpoints`` é o que `apply_head_cache` já gastou em tools e system: sai do
    orçamento porque o tecto de 4 é por pedido, não por secção.
    """
    anchors = [i for i, m in enumerate(messages) if is_markable(m)]
    if not anchors:
        return 0

    # Um "Continue." sintético no fim não é uma âncora útil.
    last = messages[anchors[-1]]
    if last.get("role") == "user" and last.get("content") == "Continue." and len(anchors) > 1:
        anchors = anchors[:-1]

    # O que o cliente já gastou e o que o head consumiu saem ambos do nosso orçamento.
    budget = CACHE_BREAKPOINT_CEILING - head_breakpoints - count_breakpoints(messages)
    if budget <= 0:
        return 0

    trailing = list(reversed(anchors[-CACHE_BREAKPOINT_MESSAGES:]))
    markable = set(anchors)
    decimation = [i for i in _decimation_indices(messages, anchors[-1]) if i in markable]

    # Prioridade da fonte: cauda mais recente, depois os checkpoints de decimação do mais
    # novo para o mais velho, e só então a segunda âncora de cauda. Com orçamento curto é
    # o checkpoint estável que sobrevive, não a redundância da cauda.
    candidates: list[int] = []
    for index in (*trailing[:1], *reversed(decimation), *trailing[1:]):
        if index not in candidates:
            candidates.append(index)

    marked = 0
    for index in candidates:
        if marked >= budget:
            break
        message = dict(messages[index])
        if isinstance(message.get("content"), list):
            message["content"] = [
                dict(block) if isinstance(block, dict) else block for block in message["content"]
            ]
        if isinstance(message.get("tool_calls"), list):
            message["tool_calls"] = [
                dict(call) if isinstance(call, dict) else call for call in message["tool_calls"]
            ]
        if mark_breakpoint(message):
            messages[index] = message
            marked += 1
    return marked


# -- parâmetros de thinking ----------------------------------------------------


# omp: providers/anthropic.ts :: disableThinkingIfToolChoiceForced
def _forced_tool_choice(choice: object) -> bool:
    """Se a escolha de ferramenta força o modelo a chamar uma.

    Só `any` e `tool` contam: são os dois valores do wire da Anthropic que forçam. A forma
    OpenAI ``{"type": "function", ...}`` é uma *selecção* de ferramenta, não uma
    imposição, e tratá-la como forçada desligava o raciocínio sem razão de wire.
    """
    if isinstance(choice, dict):
        return choice.get("type") in ("any", "tool")
    return isinstance(choice, str) and choice in ("required", "any")


# omp: providers/anthropic.ts :: ensureMaxTokensForThinking
# omp: providers/anthropic.ts :: supportsSamplingParams
# omp: providers/anthropic.ts :: disableThinkingIfToolChoiceForced
def apply_thinking_params(kwargs: dict[str, Any], model: str) -> dict[str, Any]:
    """Normaliza thinking, temperatura, top_p e tectos de tokens. Muta ``kwargs``.

    Separado de ``build_request`` porque é pura: não toca em credenciais nem em mensagens.
    """
    reasoning, _summary = normalize_effort(kwargs.get("reasoning_effort"))
    thinking = kwargs.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") == "disabled":
        kwargs.pop("thinking", None)
        thinking = None
    if reasoning == "none":
        kwargs.pop("reasoning_effort", None)
        kwargs.pop("thinking", None)
        reasoning = thinking = None

    thinking_active = bool(thinking or reasoning in EFFORT_BUDGET)

    # Medido: com thinking activo a Anthropic devolve 400 para temperature != 1 ("may only
    # be set to 1 when thinking is enabled") e para top_p < 0.95 ("`top_p` must be greater
    # than or equal to 0.95 or unset").
    temperature = kwargs.get("temperature")
    if temperature is not None and float(temperature) != 1.0:
        if thinking_active:
            kwargs["temperature"] = 1.0
        else:
            kwargs.pop("reasoning_effort", None)
            kwargs.pop("thinking", None)
            thinking_active = False
    if thinking_active:
        top_p = kwargs.get("top_p")
        if top_p is not None and float(top_p) < 0.95:
            kwargs.pop("top_p", None)

    # Uma tool_choice que força ferramenta é incompatível com budget thinking: medido em
    # claude-sonnet-4-6 -> 400 "Thinking may not be enabled when tool_choice forces tool
    # use".
    forced = _forced_tool_choice(kwargs.get("tool_choice"))
    forced_adaptive = False
    if thinking_active and forced:
        if is_adaptive(model):
            # Omitir thinking num modelo adaptive não o desliga — a API volta a ligá-lo
            # por default. A única forma de o baixar é fixar o effort.
            forced_adaptive = True
        else:
            kwargs.pop("thinking", None)
            kwargs.pop("reasoning_effort", None)
            thinking = None
            thinking_active = False

    if not thinking_active:
        return kwargs

    # `display: "summarized"` é o que faz o raciocínio voltar em texto legível: a partir
    # do Opus 4.7 o conteúdo é omitido por default, e sem o campo os deltas de thinking
    # chegam vazios. Respeita-se o que o cliente mandou; o gate é por modelo porque quem
    # não o suporta responde 400.
    display = thinking.get("display") if isinstance(thinking, dict) else None
    show = str(display or "summarized")

    if isinstance(thinking, dict):
        budget = min(thinking.get("budget_tokens") or EFFORT_BUDGET["medium"], THINKING_CEILING)
        if thinking.get("type") != "adaptive":
            thinking["budget_tokens"] = budget
    else:
        budget = min(EFFORT_BUDGET.get(reasoning or "", EFFORT_BUDGET["medium"]), THINKING_CEILING)
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
    kwargs.pop("reasoning_effort", None)

    if is_adaptive(model):
        # budget_tokens é rejeitado/ignorado nestes modelos; o par adaptive +
        # output_config.effort é a única forma suportada.
        adaptive: dict[str, Any] = {"type": "adaptive"}
        if supports_display(model):
            adaptive["display"] = show
        kwargs["thinking"] = adaptive
        effort = "low" if forced_adaptive else ADAPTIVE_EFFORT.get(reasoning or "", "medium")
        kwargs["output_config"] = {"effort": effort}
    elif isinstance(kwargs.get("thinking"), dict) and supports_display(model):
        kwargs["thinking"]["display"] = show

    # Mexe-se só na chave que o cliente mandou: preencher as duas fazia a cópia de baixo
    # sobrepor o valor do cliente com o default.
    token_key = "max_completion_tokens" if "max_completion_tokens" in kwargs else "max_tokens"
    token_value = kwargs.get(token_key)
    if token_value is None or int(token_value) < budget + OUTPUT_FALLBACK_BUFFER:
        # Sobe-se até haver margem de output para lá do raciocínio; nunca se baixa o que o
        # cliente pediu, a não ser pelo tecto do Claude Code.
        kwargs[token_key] = min(budget + OUTPUT_FALLBACK_BUFFER, MAX_OUTPUT_TOKENS)
    else:
        kwargs[token_key] = min(int(token_value), MAX_OUTPUT_TOKENS)
    if "max_completion_tokens" in kwargs:
        kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
    return kwargs


def split_system_messages(messages: list[Any]) -> tuple[str, list[Any]]:
    """Separa as instruções de system do resto da conversa."""
    system_parts: list[str] = []
    rest: list[Any] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "system":
            rest.append(message)
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            system_parts.append(content)
        elif isinstance(content, list):
            system_parts.extend(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
    client_prompt = "\n\n".join(
        part.strip().replace(CLAUDE_CODE_PROMPT, "").strip()
        for part in system_parts
        if part.strip()
    )
    return client_prompt, rest


def build_system_blocks(client_prompt: str) -> list[dict[str, Any]]:
    """Identidade do Agent SDK primeiro, instruções do cliente a seguir.

    Medido no upstream com token OAuth (opus-5/sonnet-4-6/opus-4-8/opus-4-6,
    max_tokens=64)::

        system=[identidade]          -> 200
        system=[identidade, cliente] -> 200, e a instrução do cliente é obedecida
                                        (marcador ZX9-ACK nos quatro modelos)
        system=[cliente]             -> 429 rate_limit_error

    Ou seja, a rejeição do OAuth depende de a identidade ser o **primeiro** bloco, não de
    haver só um bloco. Enfiar o prompt do cliente no primeiro turno user tirava-lhe a
    autoridade de system sem necessidade alguma.
    """
    blocks: list[dict[str, Any]] = [{"type": "text", "text": CLAUDE_CODE_PROMPT}]
    if client_prompt:
        blocks.append({"type": "text", "text": client_prompt})
    return blocks


def _wants_thinking(kwargs: dict[str, Any]) -> bool:
    """Se o pedido pede raciocínio, antes de qualquer normalização.

    A beta de effort só viaja quando há raciocínio — enviá-la sempre é ruído de
    fingerprint face ao que o Claude Code real emite.
    """
    effort, _ = normalize_effort(kwargs.get("reasoning_effort"))
    if effort == "none":
        return False
    thinking = kwargs.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") == "disabled":
        return False
    return bool(thinking or effort in EFFORT_BUDGET)


def build_request(kwargs: dict[str, Any], model: str, access_token: str = "") -> dict[str, Any]:
    """Prepara os kwargs de um pedido Claude. Muta e devolve ``kwargs``.

    O token entra por argumento: manter a leitura de credenciais fora deste módulo é o que
    o torna testável sem estado global.
    """
    if not is_anthropic_model(model):
        return kwargs

    if access_token:
        kwargs["api_key"] = access_token

    headers = kwargs.setdefault("extra_headers", {})
    if isinstance(headers, dict):
        headers.update(CLIENT_HEADERS)
        # A beta de effort só viaja quando o pedido pede raciocínio, como no OMP; a de TTL
        # estendido acompanha a retenção que as âncoras deste pedido realmente levam.
        headers["anthropic-beta"] = build_betas(thinking=_wants_thinking(kwargs))

    apply_thinking_params(kwargs, model)

    messages = kwargs.get("messages")
    if not isinstance(messages, list):
        return kwargs

    # O LiteLLM faz pop de todas as mensagens system e junta-as à frente
    # (llms/anthropic/chat/transformation.py:1686), por isso a beta
    # mid-conversation-system-2026-04-07 que enviamos não se consegue honrar por aqui.
    client_prompt, rest = split_system_messages(messages)
    system_blocks = build_system_blocks(client_prompt)
    identity = {"role": "system", "content": system_blocks}

    # O head é ancorado primeiro para que o orçamento das mensagens já desconte o que ele
    # gastou: o tecto de 4 é por pedido, e um quinto marcador dá 400.
    tools = kwargs.get("tools")
    head = apply_head_cache(system_blocks, tools if isinstance(tools, list) else None)
    apply_conversation_cache(rest, head)
    kwargs["messages"] = [identity, *rest]
    return kwargs
