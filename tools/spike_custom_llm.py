"""Rechecks decision D1: does the official `CustomLLM` preserve the provider's usage?

See `docs/DECISIONS.md`. The answer on 2026-09-18 was **no** — the reported usage is
replaced by a `token_counter` estimate and the cache hits disappear. While that holds, the
streaming generators have to stay in a monkey-patch.

    python tools/spike_custom_llm.py

Exits 0 if the limitation still holds (nothing to do) and 1 if it is **gone** — in which
case D1 can be reopened and the monkey-patch shrunk.

Not a pytest test: it mutates LiteLLM global state (`custom_provider_map`), which makes it
hostile to running alongside other tests.
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
    raise SystemExit("litellm is not installed: pip install 'litellm[proxy]'") from None

#: Values the handler reports. Distinct from any plausible estimate for a two-word prompt,
#: so a match cannot be a coincidence.
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
    """Minimal handler: emits reasoning, content, and a final chunk carrying usage."""

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
            choices=[StreamingChoices(index=0, delta=Delta(content="Hi"), finish_reason=None)],
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
    print(f"  reasoning_content preserved: {'yes' if reasoning_ok else 'NO'}")
    print(
        f"  provider usage preserved: {'yes' if usage_ok else 'NO'}"
        f"  (reported {REPORTED_PROMPT}/{REPORTED_COMPLETION},"
        f" received {got_prompt}/{got_completion})"
    )
    print(
        f"  cached_tokens preserved: {'yes' if cached_ok else 'NO'}"
        f"  (reported {REPORTED_CACHED}, received {cached})"
    )
    print()

    if usage_ok and cached_ok:
        print("The D1 limitation is GONE — the decision can be reopened.")
        return 1
    print("The D1 limitation still holds: streaming stays in a monkey-patch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
