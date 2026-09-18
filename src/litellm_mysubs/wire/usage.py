"""Normalização de usage e de finish reason.

Os bridges respondem aos pedidos eles próprios, por isso o que não for reportado aqui
perde-se: o LiteLLM cai para estimativas do ``token_counter`` e **todos os cache hits
ficam invisíveis** em ``/spend/logs``. Uma conta de subscrição sem contabilidade de cache
é uma conta que não se sabe gerir.

Estruturas neutras em vez dos tipos do LiteLLM: a conversão para ``litellm.types.utils``
vive na camada de transporte. Isto mantém o módulo testável sem o LiteLLM instalado, e é
também o que permite reutilizá-lo se o transporte mudar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final


@dataclass(frozen=True, slots=True)
class Usage:
    """Contagem de tokens de um turno.

    ``cached_tokens`` é lido pelo spend logging do LiteLLM por um atributo próprio
    (``cache_read_input_tokens``), não pelo wrapper de detalhes — a conversão trata disso.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0


def make_usage(
    prompt_tokens: object = 0,
    completion_tokens: object = 0,
    cached_tokens: object = 0,
    reasoning_tokens: object = 0,
    total_tokens: object = None,
) -> Usage:
    """Normaliza contagens vindas do fio, que chegam em formatos e tipos variados."""

    def count(value: object) -> int:
        """Um contador partido vale zero: rebentar aqui perdia o turno inteiro."""
        if value is None or isinstance(value, bool):
            return 0
        if isinstance(value, int):
            return max(0, value)
        if isinstance(value, float):
            return max(0, int(value))
        if isinstance(value, str):
            try:
                return max(0, int(value.strip()))
            except ValueError:
                return 0
        return 0

    prompt = count(prompt_tokens)
    completion = count(completion_tokens)
    total = count(total_tokens) if total_tokens else prompt + completion
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        cached_tokens=count(cached_tokens),
        reasoning_tokens=count(reasoning_tokens),
        total_tokens=total,
    )


# omp: providers/google-shared.ts :: mapStopReason
def google_usage(meta: dict[str, Any]) -> Usage:
    """``promptTokenCount`` **inclui** os tokens em cache.

    Subtrai-se para não os contar duas vezes, e os pensamentos contam como output.
    """
    cached = meta.get("cachedContentTokenCount") or 0
    thinking = meta.get("thoughtsTokenCount") or 0
    return make_usage(
        prompt_tokens=(meta.get("promptTokenCount") or 0) - cached,
        completion_tokens=(meta.get("candidatesTokenCount") or 0) + thinking,
        cached_tokens=cached,
        reasoning_tokens=thinking,
        total_tokens=meta.get("totalTokenCount"),
    )


def codex_usage(meta: dict[str, Any]) -> Usage:
    """Ao contrário do Google, ``input_tokens`` **não** é reduzido pelos tokens em cache."""
    details = meta.get("input_tokens_details") or {}
    output_details = meta.get("output_tokens_details") or {}
    cached = details.get("cached_tokens")
    if cached is None:
        cached = meta.get("prompt_cache_hit_tokens") or 0
    return make_usage(
        prompt_tokens=meta.get("input_tokens") or 0,
        completion_tokens=meta.get("output_tokens") or 0,
        cached_tokens=cached,
        reasoning_tokens=output_details.get("reasoning_tokens") or 0,
        total_tokens=meta.get("total_tokens"),
    )


# Sem isto o finalizador dava sempre "stop", e um corte por limite de tokens ou um bloqueio
# de segurança chegava ao cliente como uma resposta normal e curta.
GOOGLE_FINISH_ERROR: Final[tuple[str, ...]] = (
    "SAFETY",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "SPII",
    "IMAGE_SAFETY",
    "RECITATION",
    "MALFORMED_FUNCTION_CALL",
    "UNEXPECTED_TOOL_CALL",
    "NO_IMAGE",
    "OTHER",
)

#: Razões em que uma tool call pendente ainda é o desfecho correcto do turno.
_TOOL_CALL_COMPATIBLE: Final[tuple[str, ...]] = (
    "",
    "STOP",
    "MAX_TOKENS",
    "FINISH_REASON_UNSPECIFIED",
)


# omp: providers/google-shared.ts :: mapStopReason
def google_finish_reason(raw: object, has_tool_calls: bool) -> str:
    """Traduz ``candidates[0].finishReason`` para a forma OpenAI."""
    reason = str(raw or "").strip().upper()
    if has_tool_calls and reason in _TOOL_CALL_COMPATIBLE:
        return "tool_calls"
    if reason == "MAX_TOKENS":
        return "length"
    if reason in GOOGLE_FINISH_ERROR:
        # `content_filter` é o único valor OpenAI que não mente sobre um corte imposto
        # pelo servidor; o nome cru vai no erro in-band quando existe.
        return "content_filter"
    return "stop"


def codex_finish_reason(status: object, has_tool_calls: bool) -> str:
    """``response.incomplete`` é truncatura por limite de output.

    Sem isto uma resposta cortada chegava ao cliente como um ``stop`` limpo.
    """
    if has_tool_calls:
        return "tool_calls"
    return "length" if str(status or "completed").strip().lower() == "incomplete" else "stop"
