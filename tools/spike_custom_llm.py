"""Reverifica a decisão D1: o `CustomLLM` oficial preserva o usage do provider?

Ver `docs/DECISIONS.md`. A resposta em 2026-09-18 foi **não** — o usage reportado é
substituído por uma estimativa do `token_counter` e os cache hits desaparecem. Enquanto
isso for verdade, os geradores de streaming têm de ficar em monkey-patch.

    python tools/spike_custom_llm.py

Sai com 0 se a limitação se mantém (nada a fazer) e com 1 se **deixou de existir** — nesse
caso a decisão D1 pode ser reaberta e o monkey-patch encolhido.

Não é um teste do pytest: mexe em estado global do LiteLLM (`custom_provider_map`), o que
o torna hostil a correr ao lado de outros testes.
"""

from __future__ import annotations

import asyncio
from typing import Any

try:
    import litellm
    from litellm import CustomLLM
    from litellm.types.utils import (
        Delta,
        ModelResponseStream,
        PromptTokensDetailsWrapper,
        StreamingChoices,
        Usage,
    )
except ImportError:
    raise SystemExit("litellm não instalado: pip install 'litellm[proxy]'") from None

#: Valores que o handler reporta. Distintos de qualquer estimativa plausível para um
#: prompt de duas palavras, para não haver coincidência.
REPORTED_PROMPT = 100
REPORTED_COMPLETION = 5
REPORTED_CACHED = 80


def _usage() -> Usage:
    usage = Usage(
        prompt_tokens=REPORTED_PROMPT,
        completion_tokens=REPORTED_COMPLETION,
        total_tokens=REPORTED_PROMPT + REPORTED_COMPLETION,
    )
    usage.prompt_tokens_details = PromptTokensDetailsWrapper(cached_tokens=REPORTED_CACHED)
    return usage


class Probe(CustomLLM):
    """Handler mínimo: emite reasoning, conteúdo e um chunk final com usage."""

    async def astreaming(self, *args: Any, **kwargs: Any) -> Any:
        yield ModelResponseStream(
            id="spike",
            created=1,
            model="probe",
            choices=[
                StreamingChoices(
                    index=0, delta=Delta(reasoning_content="PENSO"), finish_reason=None
                )
            ],
        )
        yield ModelResponseStream(
            id="spike",
            created=1,
            model="probe",
            choices=[StreamingChoices(index=0, delta=Delta(content="Olá"), finish_reason=None)],
        )
        yield ModelResponseStream(
            id="spike",
            created=1,
            model="probe",
            choices=[StreamingChoices(index=0, delta=Delta(), finish_reason="stop")],
            usage=_usage(),
        )


async def run() -> int:
    litellm.custom_provider_map = [{"provider": "probe", "custom_handler": Probe()}]
    from litellm.utils import custom_llm_setup

    custom_llm_setup()

    stream = await litellm.acompletion(
        model="probe/x",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    chunks = [chunk async for chunk in stream]

    reasoning = [
        getattr(chunk.choices[0].delta, "reasoning_content", None)
        for chunk in chunks
        if chunk.choices
    ]
    usages = [getattr(chunk, "usage", None) for chunk in chunks]
    final = next((usage for usage in reversed(usages) if usage), None)

    cached = getattr(getattr(final, "prompt_tokens_details", None), "cached_tokens", None)
    got_prompt = getattr(final, "prompt_tokens", None)
    got_completion = getattr(final, "completion_tokens", None)
    reasoning_ok = any(reasoning)
    usage_ok = final is not None and final.prompt_tokens == REPORTED_PROMPT
    cached_ok = cached == REPORTED_CACHED

    from importlib.metadata import version

    print(f"litellm {version('litellm')}")
    print(f"  reasoning_content preservado: {'sim' if reasoning_ok else 'NÃO'}")
    print(
        f"  usage do provider preservado: {'sim' if usage_ok else 'NÃO'}"
        f"  (reportado {REPORTED_PROMPT}/{REPORTED_COMPLETION},"
        f" recebido {got_prompt}/{got_completion})"
    )
    print(
        f"  cached_tokens preservado: {'sim' if cached_ok else 'NÃO'}"
        f"  (reportado {REPORTED_CACHED}, recebido {cached})"
    )
    print()

    if usage_ok and cached_ok:
        print("A limitação de D1 DEIXOU DE EXISTIR — a decisão pode ser reaberta.")
        return 1
    print("A limitação de D1 mantém-se: o streaming fica em monkey-patch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
