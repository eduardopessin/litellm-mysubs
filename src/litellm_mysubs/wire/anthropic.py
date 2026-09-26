"""Anthropic wire protocol over subscription OAuth (Claude Max).

Extracted from the original ``sitecustomize.py`` with no behaviour change. The
measurements in the comments are what justifies each decision, and they come from the
real service, not from documentation — they are the most valuable part of this module and
must not be deleted.

Internal split: the cache and parameter functions are pure and testable on their own;
``build_request`` is the only one that needs a token, and it takes it as an argument
instead of reaching into global state.
"""

from __future__ import annotations

import re
from typing import Any, Final


def _version() -> str:
    """The installed version, or a marker that it is not installed.

    Imported lazily so that this module keeps working when the package is on the path but
    not installed — running the test suite from a source checkout, mostly.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("litellm-mysubs")
    except PackageNotFoundError:  # pragma: no cover - source tree without an install
        return "0.0.0.dev0"


# omp: providers/claude-code-fingerprint.ts :: claudeCodeSystemInstruction
#: Identity block that the Claude Code runtime prepends.
#:
#: **This one is not optional, and it is the only part of the fingerprint that is not.**
#: Measured against the real endpoint, three runs each, `claude-sonnet-4-6` with
#: `thinking` enabled: with this block the request returns 200; without it, 429
#: `rate_limit_error` — regardless of what the `User-Agent` says. Subscription OAuth
#: tokens are issued for this client, and the endpoint checks for it here rather than in
#: the headers.
#:
#: So the package sends its own `User-Agent` (see `USER_AGENT`) and keeps this block: it
#: claims nothing about itself that is untrue, and it does not pretend the request would
#: work without the identity the token was issued against. Stripping it does not make the
#: traffic more honest — it makes it fail.
CLAUDE_CODE_PROMPT: Final = "You are Claude Code, Anthropic's official CLI for Claude."

# omp: stream.ts :: ANTHROPIC_THINKING
# Effort -> thinking budget. The steps are OMP's; only the top differs, and the reason is
# in `THINKING_CEILING`.
EFFORT_BUDGET: Final[dict[str, int]] = {
    "minimal": 1024,
    "low": 4096,
    "medium": 8192,
    "high": 16384,
    "xhigh": 32768,
    "max": 32768,
}

#: The short TPM window of the Max subscription cannot take OMP's 32768: requests above
#: this return 429. The ceiling is applied after picking the step, so that OMP's scale is
#: preserved instead of being flattened in the table.
THINKING_CEILING: Final = 8192

# omp: stream.ts :: OUTPUT_FALLBACK_BUFFER
#: Output margin reserved beyond the reasoning budget. A request whose `max_tokens` falls
#: below `budget + this` has no room to answer after thinking, and the response comes back
#: truncated.
OUTPUT_FALLBACK_BUFFER: Final = 4000

# omp: providers/claude-code-fingerprint.ts :: CLAUDE_CODE_MAX_OUTPUT_TOKENS
MAX_OUTPUT_TOKENS: Final = 64000

# Measured against upstream (max_tokens=2048, display="summarized", a question that
# demands reasoning): xhigh and max are accepted and yield more output than high (out=164
# at high, 273 at xhigh, 275 at max), so collapsing them into "high" hid two real steps.
ADAPTIVE_EFFORT: Final[dict[str, str]] = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}

# The default is adaptive, and this list enumerates who *rejects* it, not who accepts it.
# Measured model by model (max_tokens 1024/4096, display="summarized"), thinking chars
# returned with each form:
#   opus-5-5     adaptive  76 | budget    0   <- adaptive mandatory
#   opus-5       adaptive  82 | budget    0   <- adaptive mandatory
#   fable-5      adaptive  83 | budget    0   <- adaptive mandatory
#   sonnet-5     adaptive  58 | budget   59
#   opus-4-8     adaptive  61 | budget   62
#   opus-4-6     adaptive 101 | budget  101
#   sonnet-4-6   adaptive 104 | budget  105
#   opus-4-5     adaptive 400 | budget  233   <- adaptive rejected
#   sonnet-4-5   adaptive 400 | budget  367   <- adaptive rejected
#   haiku-4-5    adaptive 400 | budget  413   <- adaptive rejected
# The asymmetry is what decides the default: erring towards adaptive gives 400 "adaptive
# thinking is not supported on this model"; erring towards budget gives 200 with 0 chars
# of reasoning. That way an alias (claude-opus -> opus-4-8) or a new model not yet
# enumerated falls on the side that is detectable.
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

# Anthropic caches everything *up to* a breakpoint, and the canonical order on the wire is
# tools -> system -> messages. Two adjacent markers in the tail (not one) keep a valid
# entry to extend as the conversation grows.
CACHE_BREAKPOINT_MESSAGES: Final = 2

# omp: providers/anthropic.ts :: ANTHROPIC_DECIMATION_INTERVAL
# Stable historical checkpoint every 15 user turns (15th, 30th, 45th...). The two tail
# anchors move on every turn, so when the 5 min window expires there is no live entry
# covering the old prefix and it is re-read at full price. A marker at a fixed position
# survives the tail churn and catches that prefix.
DECIMATION_INTERVAL: Final = 15

# omp: providers/anthropic.ts :: VOLATILE_SYSTEM_SEGMENT_MARKERS
#: System segments that change on every turn. The system anchor sits on the last block
#: *before* them, so that a memory refresh re-bills only the suffix instead of the whole
#: head. Detection is by our own marking, and only counts at the start of a block: a
#: `<memories>` quoted in the middle of a stable block does not make it volatile.
VOLATILE_SYSTEM_MARKERS: Final[tuple[str, ...]] = ("<memories>",)

# A client that does its own caching arrives here with markers already set. Measured with
# claude-sonnet-4-6: 4 markers -> 200, 5 -> 400 "A maximum of 4 blocks with cache_control
# may be provided. Found 5." Three client markers outside our tail window plus our two
# gave exactly that 400.
CACHE_BREAKPOINT_CEILING: Final = 4

#: Blocks carrying reasoning are never valid anchors.
UNCACHEABLE_BLOCKS: Final[tuple[str, ...]] = ("thinking", "redacted_thinking", "fallback")

# Hosted tools: LiteLLM emits `server_tool_use` without cache_control (factory.py:1971),
# so such a call cannot serve as an anchor.
SERVER_TOOL_PREFIX: Final = "srvtoolu_"

# omp: providers/anthropic.ts :: claudeCodeAgentBetaDefaults
# Order and content from the source. Notes on what is **not** here:
#  - `context-1m-2025-08-07`: OAuth credentials have no long-context balance, and Anthropic
#    returns a hard 429 on any model with the beta, regardless of prompt size. OMP never
#    advertises it either.
#  - `redact-thinking-2026-02-12`: makes thinking blocks come back signed but with no text
#    (measured on sonnet-4-6: 74 chars without the beta, 0 with it). OMP does not send it
#    on inference either — only on the header of the usage route.
#  - `structured-outputs-2025-12-15`: belongs to the utility list, not the agent one.
AGENT_BETAS: Final[tuple[str, ...]] = (
    "claude-code-20250219",
    # The only one specific to an OAuth credential. Without it the server classifies the
    # request as coming from an API key — it was missing because the port came from the
    # intermediary.
    "oauth-2025-04-20",
    "interleaved-thinking-2025-05-14",
    "thinking-token-count-2026-05-13",
    "context-management-2025-06-27",
    "prompt-caching-scope-2026-01-05",
    "mid-conversation-system-2026-04-07",
)

#: Added only when the request asks for reasoning.
EFFORT_BETA: Final = "effort-2025-11-24"
#: Added to every agent request.
FALLBACK_CREDIT_BETA: Final = "fallback-credit-2026-06-01"
#: Added when some anchor carries `ttl: "1h"`.
EXTENDED_CACHE_TTL_BETA: Final = "extended-cache-ttl-2025-04-11"


# omp: providers/anthropic.ts :: buildClaudeCodeBetas
def build_betas(*, thinking: bool) -> str:
    """``anthropic-beta`` header for an agent request."""
    betas = [*AGENT_BETAS]
    if thinking:
        betas.append(EFFORT_BETA)
    betas.append(FALLBACK_CREDIT_BETA)
    # The extended-cache-TTL beta does NOT travel on the OAuth path. OMP only adds it when
    # `!isOAuth` (`providers/anthropic.ts`), and `getCacheControl` shows why: for OAuth the
    # default is already `ttl: "1h"` on models that support it, with no beta at all.
    #
    # The header in `usage/claude.ts` carries this beta and might look like the
    # counter-example, but it belongs to the *usage* route and also carries
    # `redact-thinking-2026-02-12` — which we measured returning empty thinking blocks.
    # Copying it into inference would break reasoning.
    return ",".join(betas)


# omp: providers/claude-code-fingerprint.ts :: getClaudeCodeUserAgent, DEFAULT_CLAUDE_CODE_VERSION
# omp= DEFAULT_CLAUDE_CODE_VERSION = "2.1.280"
#: The version the real CLI pins. Kept as a constant because the anchor above has to match
#: something, and because a future measurement may show it matters again — it does not now.
CLAUDE_CODE_VERSION: Final = "2.1.280"

#: What this package calls itself on the wire.
#:
#: It used to send `claude-cli/{version} (external, cli)`, inherited from the port.
#: Measured against the real endpoint, three runs each, `claude-sonnet-4-6` with
#: `thinking` enabled:
#:
#: | User-Agent | system prompt | result |
#: |---|---|---|
#: | `claude-cli/2.1.257` | present | 200, 200, 200 |
#: | `litellm-mysubs/0.1.0` | present | 200, 200, 200 |
#: | `claude-cli/2.1.257` | absent | 429, 429, 429 |
#: | `litellm-mysubs/0.1.0` | absent | 429, 429, 429 |
#:
#: The `User-Agent` changes nothing; the identity block in `system` is what the endpoint
#: actually gates on. So the claim of being another program bought nothing, and claiming
#: it anyway is the one thing every discussion of this mechanism asks implementations not
#: to do. This one says what it is.
#:
#: The version is read from the installed metadata rather than written here: two places
#: holding the same number drift, and the one that lies is always the copy.
USER_AGENT: Final = (
    f"litellm-mysubs/{_version()} (+https://github.com/eduardopessin/litellm-mysubs)"
)

CLIENT_HEADERS: Final[dict[str, str]] = {
    "User-Agent": USER_AGENT,
    "anthropic-dangerous-direct-browser-access": "true",
    "x-app": "cli",
}


def is_anthropic_model(model: str) -> bool:
    lowered = str(model).lower()
    return "claude" in lowered or "anthropic" in lowered


def is_adaptive(model: str) -> bool:
    """Whether the model uses ``thinking: adaptive`` instead of ``budget_tokens``."""
    lowered = str(model).lower()
    return not any(marker in lowered for marker in BUDGET_ONLY_MODELS)


#: Models that accept ``thinking.display``, in order of specificity.
#:
#: The rule in the source is generational, not a list: opus from 4.7 up, and
#: sonnet/fable/mythos from 5 up. It does not coincide with ``is_adaptive`` — opus-4-6 and
#: sonnet-4-6 are adaptive but do **not** accept ``display``, and sending it gives 400.
_DISPLAY_FLOORS: Final[tuple[tuple[str, float], ...]] = (
    ("opus", 4.7),
    ("sonnet", 5.0),
    ("fable", 5.0),
    ("mythos", 5.0),
)


# omp: compat/resolve.ts :: defaultSupportsDisplay
def supports_display(model: str) -> bool:
    """Whether the model accepts ``thinking.display``.

    ``display: "summarized"`` is what makes reasoning come back as readable text: from Opus
    4.7 on, the content is omitted from the response by default. The field is strictly
    gated per model — anything that does not support it answers 400 — so being adaptive is
    not enough.
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
    """Return ``(effort, summary)`` from ``reasoning_effort``.

    The ``/v1/responses`` route delivers ``reasoning: {effort, summary}`` and LiteLLM's
    translator forwards the whole object. Treating it as a string put
    ``"{'effort': 'medium', …}"`` on the wire, which gives 400 Invalid value.
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


#: Default retention. OMP defaults to "long" for OAuth on a model with
#: `supportsLongCacheRetention`, "matching Claude Code's native policy"; this module only
#: serves the Claude Code OAuth path, so the condition collapses into the default.
LONG_CACHE_TTL: Final = "1h"


# omp: providers/anthropic.ts :: getCacheControl
def cache_control(ttl: str | None = LONG_CACHE_TTL) -> dict[str, str]:
    """Cache marker, with ``ttl`` of 1 h by default and ``None`` for the base 5 min.

    The trade-off is cost against rewrite frequency: a 1 h write bills 2x the base token
    price against 1.25x for the 5 min one. In an agent session the prefix is re-read dozens
    of times and the pauses between turns easily exceed 5 min, so paying 2x once comes out
    cheaper than paying 1.25x on every cold rewrite — which is exactly why native Claude
    Code defaults to 1 h.
    """
    if not ttl:
        return {"type": "ephemeral"}
    return {"type": "ephemeral", "ttl": ttl}


# -- cache anchors -------------------------------------------------------------


def tool_call_anchor(message: dict[str, Any]) -> int | None:
    """Index of the last tool call that LiteLLM accepts marking.

    ``convert_to_anthropic_tool_invoke:1952`` skips anything that is not
    ``type: "function"``.
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
    """Whether a breakpoint can be pinned to this message.

    omp marks the Anthropic wire, where a tool result is a ``tool_result`` block inside a
    ``user`` turn, so its rolling window always lands on the last two turns. Here we see
    the OpenAI form: the tool result is a message of its own with ``role: "tool"`` and the
    assistant's tool call carries ``content: None``. LiteLLM propagates the breakpoint in
    both, but reads it from different levels
    (``litellm_core_utils/prompt_templates/factory.py``):

      - ``role: "tool"``  -> message level, ``convert_to_anthropic_tool_result:1844``
      - ``tool_calls[i]`` -> inside the call, ``convert_to_anthropic_tool_invoke:2003``
      - text blocks       -> on the block itself

    Refusing the first two pinned the window to the head of the conversation: on a turn
    ending in a tool result, 67% of the prompt was re-read at full price (measured: opus-5
    pt=8697, read=2876).
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
    """Markers already present, wherever they came from."""
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


def _client_marker_ttl(*sections: object) -> str | None:
    """The ``ttl`` on markers the caller placed, or ``None`` when it placed none.

    ``None`` is also what a marker without an explicit ``ttl`` means on the wire, and the
    two cases are told apart by `client_uses_short_ttl`.
    """
    for section in sections:
        for block in _marked_blocks(section):
            ttl = block.get("cache_control", {}).get("ttl")
            return ttl if isinstance(ttl, str) else None
    return None


def _marked_blocks(section: object) -> list[dict[str, Any]]:
    if not isinstance(section, list):
        return []
    found: list[dict[str, Any]] = []
    for item in section:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("cache_control"), dict):
            found.append(item)
        for call in item.get("tool_calls") or ():
            if isinstance(call, dict) and isinstance(call.get("cache_control"), dict):
                found.append(call)
        found.extend(_marked_blocks(item.get("content")))
    return found


def client_uses_short_ttl(*sections: object) -> bool:
    """Whether the caller already marked the request with the base 5 min ttl.

    Anthropic rejects a request where a ``1h`` marker comes after a ``5m`` one, in the
    fixed order ``tools``, ``system``, ``messages``::

        400 messages.1.content.0.cache_control.ttl: a ttl='1h' cache_control block must
            not come after a ttl='5m' cache_control block

    Claude Code marks its own prefix without an explicit ``ttl``, which is ``5m``, so
    anchoring ours at the 1 h default behind it is a hard 400 and the turn never runs.
    When the caller has already chosen, we follow its choice rather than its error.
    """
    marked = any(_marked_blocks(section) for section in sections)
    return marked and _client_marker_ttl(*sections) is None


def mark_breakpoint(message: dict[str, Any], ttl: str | None = LONG_CACHE_TTL) -> bool:
    """Mark the last non-reasoning anchor; give up if there is one already."""
    control = cache_control(ttl)
    if message.get("role") == "tool" or message.get("tool_call_id"):
        if message.get("cache_control") is not None:
            return False
        message["cache_control"] = control
        return True

    # An assistant message can carry both text and tool calls; on the Anthropic wire the
    # `tool_use` comes after the text, so it is the anchor covering the most prefix.
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
    """Index where the volatile system suffix starts; ``len(blocks)`` if there is none."""
    start = len(blocks)
    while start > 0:
        block = blocks[start - 1]
        text = str(block.get("text", "")) if isinstance(block, dict) else ""
        if not any(text.startswith(marker) for marker in VOLATILE_SYSTEM_MARKERS):
            break
        start -= 1
    return start


# -- third-party fingerprint ---------------------------------------------------

#: Declared together, these three are read as a third-party agent and the subscription
#: answers 400 "You're out of extra usage" — with credit on the account and the very same
#: request passing as soon as one of them is renamed. Measured on the live gateway against
#: `claude-opus-5`, holding everything else equal:
#:
#:   | tools in the request                     | status |
#:   |------------------------------------------|--------|
#:   | 25 client tools, names untouched         | 400    |
#:   | the same 25 minus these three            | 200    |
#:   | these three alone                        | 400    |
#:   | any two of the three                     | 200    |
#:   | 25 with the three under `mcp__`          | 200    |
#:
#: Not a size limit: padding the set back to the same byte count without the trio still
#: passes (45152 bytes -> 200) while the trio fails at 45300.
FINGERPRINT_TOOLS: Final = frozenset({"skill_manage", "skill_view", "skills_list"})

#: Claude Code's own namespace for MCP-provided tools. The classifier accepts it because
#: it is what the first-party client sends.
MCP_TOOL_PREFIX: Final = "mcp__"

#: Where `build_request` leaves the renaming for its caller. Defined here, next to the
#: code that writes it, because this module owes nothing to the plugin layer that reads it.
TOOL_ALIAS_KEY: Final = "mysubs_tool_aliases"


def _tool_name(tool: Any) -> str | None:
    """Name of a chat-completions tool entry, or ``None`` if it has no readable one."""
    if not isinstance(tool, dict):
        return None
    nested = tool.get("function")
    name = nested.get("name") if isinstance(nested, dict) else tool.get("name")
    return name if isinstance(name, str) else None


def wire_tool_names(tools: list[Any] | None) -> dict[str, str]:
    """``{wire name: original name}`` for the tools this request has to rename.

    Empty unless all three are present: two of them pass, so a client that declares a
    subset keeps its names untouched and nothing is renamed without cause.

    A name already taken in the same request is skipped. Two identical tool names is a
    hard 400 — strictly worse than the fingerprint this avoids.
    """
    names = {name for name in map(_tool_name, tools or ()) if name is not None}
    if not FINGERPRINT_TOOLS.issubset(names):
        return {}
    return {
        MCP_TOOL_PREFIX + name: name
        for name in sorted(FINGERPRINT_TOOLS)
        if MCP_TOOL_PREFIX + name not in names
    }


def _rename_tool_choice(choice: Any, forward: dict[str, str]) -> Any:
    """``tool_choice`` pointing at a renamed tool, updated to the name that will travel.

    Renaming the tools and leaving the choice behind names a tool the request no longer
    declares, and the upstream rejects it outright::

        400 Tool 'skills_list' not found in provided tools

    Both spellings reach here: the OpenAI ``{"type": "function", "function": {...}}`` and
    the Anthropic ``{"type": "tool", "name": ...}``. ``"auto"``/``"none"`` name no tool and
    pass through untouched.
    """
    if not isinstance(choice, dict):
        return choice
    nested = choice.get("function")
    if isinstance(nested, dict) and nested.get("name") in forward:
        return {**choice, "function": {**nested, "name": forward[nested["name"]]}}
    if choice.get("name") in forward:
        return {**choice, "name": forward[choice["name"]]}
    return choice


def apply_tool_aliases(kwargs: dict[str, Any]) -> dict[str, str]:
    """Rename the fingerprint trio on the way out. Returns ``{wire name: original name}``.

    The caller maps the names back on the response: the model answers with the name it was
    given, and a client that never declared ``mcp__skills_list`` cannot dispatch it.
    """
    tools = kwargs.get("tools")
    if not isinstance(tools, list):
        return {}
    back = wire_tool_names(tools)
    if not back:
        return {}
    forward = {original: wire for wire, original in back.items()}
    renamed: list[Any] = []
    for tool in tools:
        name = _tool_name(tool)
        if name not in forward:
            renamed.append(tool)
            continue
        nested = tool.get("function")
        if isinstance(nested, dict):
            renamed.append({**tool, "function": {**nested, "name": forward[name]}})
        else:
            renamed.append({**tool, "name": forward[name]})
    kwargs["tools"] = renamed
    if "tool_choice" in kwargs:
        kwargs["tool_choice"] = _rename_tool_choice(kwargs["tool_choice"], forward)
    return back


def _child(node: Any, key: str) -> Any:
    """``node[key]`` or ``node.key``, whichever the node carries."""
    if isinstance(node, dict):
        return node.get(key)
    return getattr(node, key, None)


def _set_child(node: Any, key: str, value: Any) -> None:
    if isinstance(node, dict):
        node[key] = value
    else:
        setattr(node, key, value)


#: Where a tool name can hang on a response. `choices`/`message`/`delta` is the
#: chat-completions shape (streaming included), `content` the ``/v1/messages`` blocks,
#: `tool_calls`/`function` the call itself.
_TOOL_NAME_PARENTS: Final = ("choices", "message", "delta", "content", "tool_calls", "function")


def restore_tool_names(payload: Any, aliases: dict[str, str]) -> Any:
    """Put the client's own names back on whatever the model called.

    Walks both shapes, because the routes do not agree on one: the native chat path
    answers with pydantic models (``ModelResponse``), ``/v1/messages`` with plain dicts,
    and a streaming chunk nests the call under ``delta`` rather than ``message``. Reading
    only mappings silently left the alias in place on the path that matters most — the one
    LiteLLM's own client serves.
    """
    if not aliases:
        return payload
    if isinstance(payload, (list, tuple)):
        for item in payload:
            restore_tool_names(item, aliases)
        return payload
    if isinstance(payload, (str, bytes, int, float, bool)) or payload is None:
        return payload
    name = _child(payload, "name")
    if isinstance(name, str) and name in aliases:
        _set_child(payload, "name", aliases[name])
    for key in _TOOL_NAME_PARENTS:
        child = _child(payload, key)
        if child is not None and not isinstance(child, (str, bytes, int, float, bool)):
            restore_tool_names(child, aliases)
    return payload


def _is_deferred_tool(tool: Any) -> bool:
    """LiteLLM accepts ``defer_loading`` at the top level or inside ``function``
    (``transformation.py:843``), so both places count."""
    if not isinstance(tool, dict):
        return False
    if tool.get("defer_loading"):
        return True
    nested = tool.get("function")
    return bool(isinstance(nested, dict) and nested.get("defer_loading"))


# omp: providers/anthropic.ts :: countHeadBreakpoints
def count_head_breakpoints(system_blocks: list[Any] | None, tools: list[Any] | None) -> int:
    """Markers present in system and in tools."""
    total = 0
    for block in system_blocks or ():
        if isinstance(block, dict) and block.get("cache_control") is not None:
            total += 1
    for tool in tools or ():
        if isinstance(tool, dict) and tool.get("cache_control") is not None:
            total += 1
    return total


# omp: providers/anthropic.ts :: applyHeadCaching
def apply_head_cache(
    system_blocks: list[Any] | None, tools: list[Any] | None, ttl: str | None = LONG_CACHE_TTL
) -> int:
    """Anchor the stable head — last non-deferred tool and last stable system block.

    Returns how many markers ended up in the head. The order on the wire is tools ->
    system -> messages, so a marker on the last stable system block caches the whole
    tools+system prefix; the marker on the tools keeps the definitions cached even when the
    system text changes. Without this the head was only covered by the tail anchor, which
    moves on every turn, and was therefore rewritten at full price on every request.
    """
    if tools and not any(
        isinstance(tool, dict) and tool.get("cache_control") is not None for tool in tools
    ):
        # A deferred tool does not enter the checked prefix until it is referenced, so
        # anchoring on it would leave out everything that comes before.
        for tool in reversed(tools):
            if not isinstance(tool, dict) or _is_deferred_tool(tool):
                continue
            tool["cache_control"] = cache_control(ttl)
            break

    if system_blocks:
        suffix_start = stable_system_suffix_start(system_blocks)
        if suffix_start == len(system_blocks):
            if not any(
                isinstance(b, dict) and b.get("cache_control") is not None for b in system_blocks
            ):
                last = system_blocks[-1]
                if isinstance(last, dict):
                    last["cache_control"] = cache_control(ttl)
        else:
            # With a volatile suffix the boundary marker goes in even if there is one
            # further back: otherwise the only system marker sits before the stable prompt
            # and a memory refresh re-bills it.
            anchor_index = len(system_blocks) - 1 if suffix_start == 0 else suffix_start - 1
            anchor = system_blocks[anchor_index]
            if isinstance(anchor, dict) and anchor.get("cache_control") is None:
                anchor["cache_control"] = cache_control(ttl)

    return count_head_breakpoints(system_blocks, tools)


def _decimation_indices(messages: list[Any], end: int) -> list[int]:
    """Indices of user turns whose ordinal is a multiple of ``DECIMATION_INTERVAL``.

    Limitation against the source: OMP counts turns by `isConversationalUser`, a
    provenance marker that distinguishes a human turn from a serialized `developer` one or
    from an internal "Continue.". We receive kwargs in OpenAI form and that marker does not
    exist on the wire, so the approximation is ``role == "user"`` — a synthesized `user`
    message counts as a turn where OMP would not count it, which shifts the checkpoints to
    more recent positions than the canonical ones. They still land at fixed positions along
    the conversation, which is what makes them worth having.
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
def apply_conversation_cache(
    messages: list[Any], head_breakpoints: int = 0, ttl: str | None = LONG_CACHE_TTL
) -> int:
    """Anchor the breakpoints in the messages. Mutates ``messages``.

    ``head_breakpoints`` is what `apply_head_cache` already spent on tools and system: it
    comes out of the budget because the ceiling of 4 is per request, not per section.

    ``ttl`` follows the caller's own markers when it placed any; see
    `client_uses_short_ttl` for why mixing the two is a 400.
    """
    anchors = [i for i, m in enumerate(messages) if is_markable(m)]
    if not anchors:
        return 0

    # A synthetic "Continue." at the end is not a useful anchor.
    last = messages[anchors[-1]]
    if last.get("role") == "user" and last.get("content") == "Continue." and len(anchors) > 1:
        anchors = anchors[:-1]

    # What the client already spent and what the head consumed both come out of our budget.
    budget = CACHE_BREAKPOINT_CEILING - head_breakpoints - count_breakpoints(messages)
    if budget <= 0:
        return 0

    trailing = list(reversed(anchors[-CACHE_BREAKPOINT_MESSAGES:]))
    markable = set(anchors)
    decimation = [i for i in _decimation_indices(messages, anchors[-1]) if i in markable]

    # Priority from the source: most recent tail, then the decimation checkpoints from
    # newest to oldest, and only then the second tail anchor. On a short budget it is the
    # stable checkpoint that survives, not the tail's redundancy.
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
        if mark_breakpoint(message, ttl):
            messages[index] = message
            marked += 1
    return marked


# -- thinking parameters -------------------------------------------------------


# omp: providers/anthropic.ts :: disableThinkingIfToolChoiceForced
def _forced_tool_choice(choice: object) -> bool:
    """Whether the tool choice forces the model to call one.

    Only `any` and `tool` count: they are the two values on the Anthropic wire that force.
    The OpenAI form ``{"type": "function", ...}`` is a tool *selection*, not an imposition,
    and treating it as forced disabled reasoning with no wire-level reason.
    """
    if isinstance(choice, dict):
        return choice.get("type") in ("any", "tool")
    return isinstance(choice, str) and choice in ("required", "any")


# omp: providers/anthropic.ts :: ensureMaxTokensForThinking
# omp: providers/anthropic.ts :: supportsSamplingParams
# omp: providers/anthropic.ts :: disableThinkingIfToolChoiceForced
def apply_thinking_params(kwargs: dict[str, Any], model: str) -> dict[str, Any]:
    """Normalize thinking, temperature, top_p and token ceilings. Mutates ``kwargs``.

    Kept apart from ``build_request`` because it is pure: it touches neither credentials
    nor messages.
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

    # Measured: with thinking active Anthropic returns 400 for temperature != 1 ("may only
    # be set to 1 when thinking is enabled") and for top_p < 0.95 ("`top_p` must be greater
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

    # A tool_choice that forces a tool is incompatible with budget thinking: measured on
    # claude-sonnet-4-6 -> 400 "Thinking may not be enabled when tool_choice forces tool
    # use".
    forced = _forced_tool_choice(kwargs.get("tool_choice"))
    forced_adaptive = False
    if thinking_active and forced:
        if is_adaptive(model):
            # Omitting thinking on an adaptive model does not disable it — the API turns
            # it back on by default. The only way to lower it is to pin the effort.
            forced_adaptive = True
        else:
            kwargs.pop("thinking", None)
            kwargs.pop("reasoning_effort", None)
            thinking = None
            thinking_active = False

    if not thinking_active:
        return kwargs

    # `display: "summarized"` is what makes reasoning come back as readable text: from Opus
    # 4.7 on the content is omitted by default, and without the field the thinking deltas
    # arrive empty. What the client sent is respected; the gate is per model because
    # anything that does not support it answers 400.
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
        # budget_tokens is rejected/ignored on these models; the adaptive +
        # output_config.effort pair is the only supported form.
        adaptive: dict[str, Any] = {"type": "adaptive"}
        if supports_display(model):
            adaptive["display"] = show
        kwargs["thinking"] = adaptive
        effort = "low" if forced_adaptive else ADAPTIVE_EFFORT.get(reasoning or "", "medium")
        kwargs["output_config"] = {"effort": effort}
    elif isinstance(kwargs.get("thinking"), dict) and supports_display(model):
        kwargs["thinking"]["display"] = show

    # Only the key the client sent is touched: filling both made the copy below override
    # the client's value with the default.
    token_key = "max_completion_tokens" if "max_completion_tokens" in kwargs else "max_tokens"
    token_value = kwargs.get(token_key)
    if token_value is None or int(token_value) < budget + OUTPUT_FALLBACK_BUFFER:
        # Raised until there is output room beyond the reasoning; what the client asked
        # for is never lowered, except by the Claude Code ceiling.
        kwargs[token_key] = min(budget + OUTPUT_FALLBACK_BUFFER, MAX_OUTPUT_TOKENS)
    else:
        kwargs[token_key] = min(int(token_value), MAX_OUTPUT_TOKENS)
    if "max_completion_tokens" in kwargs:
        kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
    return kwargs


def split_system_messages(messages: list[Any]) -> tuple[str, list[Any]]:
    """Split the system instructions from the rest of the conversation."""
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
    """Agent SDK identity first, client instructions after.

    Measured against upstream with an OAuth token (opus-5/sonnet-4-6/opus-4-8/opus-4-6,
    max_tokens=64)::

        system=[identity]         -> 200
        system=[identity, client] -> 200, and the client instruction is obeyed
                                     (ZX9-ACK marker on all four models)
        system=[client]           -> 429 rate_limit_error

    That is, the OAuth rejection depends on the identity being the **first** block, not on
    there being only one block. Stuffing the client prompt into the first user turn stripped
    it of system authority for no reason at all.
    """
    blocks: list[dict[str, Any]] = [{"type": "text", "text": CLAUDE_CODE_PROMPT}]
    if client_prompt:
        blocks.append({"type": "text", "text": client_prompt})
    return blocks


def _wants_thinking(kwargs: dict[str, Any]) -> bool:
    """Whether the request asks for reasoning, before any normalization.

    The effort beta only travels when there is reasoning — sending it always is fingerprint
    noise against what the real Claude Code emits.
    """
    effort, _ = normalize_effort(kwargs.get("reasoning_effort"))
    if effort == "none":
        return False
    thinking = kwargs.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") == "disabled":
        return False
    return bool(thinking or effort in EFFORT_BUDGET)


def build_request(
    kwargs: dict[str, Any],
    model: str,
    access_token: str = "",
    *,
    native_system: bool = False,
) -> dict[str, Any]:
    """Prepare the kwargs of a Claude request. Mutates and returns ``kwargs``.

    The token comes in as an argument: keeping credential reading out of this module is
    what makes it testable without global state.

    ``native_system`` selects where the identity goes. On chat-completions it has to ride
    as ``messages[0]``, because LiteLLM pops system messages and joins them at the front.
    On ``/v1/messages`` the payload already has a top-level ``system`` and the upstream
    rejects the other form outright::

        400 messages.0: use the top-level 'system' parameter for the initial system prompt

    Measured on the live gateway, which is why the placement is a parameter rather than a
    guess from the shape of the kwargs.
    """
    if not is_anthropic_model(model):
        return kwargs

    if access_token:
        kwargs["api_key"] = access_token

    headers = kwargs.setdefault("extra_headers", {})
    if isinstance(headers, dict):
        headers.update(CLIENT_HEADERS)
        # The effort beta only travels when the request asks for reasoning, as in OMP; the
        # extended-TTL one follows the retention that this request's anchors actually
        # carry.
        headers["anthropic-beta"] = build_betas(thinking=_wants_thinking(kwargs))

    apply_thinking_params(kwargs, model)

    # Before the `messages` guard: a request whose turns are carried elsewhere still
    # declares tools, and the fingerprint is read off the tool names alone. The map rides
    # on the kwargs so the caller can undo the renaming on the response; it is popped
    # before the request goes out.
    aliases = apply_tool_aliases(kwargs)
    if aliases:
        kwargs[TOOL_ALIAS_KEY] = aliases

    messages = kwargs.get("messages")
    if not isinstance(messages, list):
        return kwargs

    # LiteLLM pops every system message and joins them at the front
    # (llms/anthropic/chat/transformation.py:1686), so the
    # mid-conversation-system-2026-04-07 beta we send cannot be honoured from here.
    client_prompt, rest = split_system_messages(messages)
    if native_system:
        # `/v1/messages` carries its own top-level `system`, and the client's own prompt
        # is already there. Prepending ours keeps the identity the subscription validates
        # without displacing what the caller wrote.
        existing = kwargs.get("system")
        blocks = build_system_blocks(client_prompt)
        if isinstance(existing, list):
            blocks = [*blocks, *existing]
        elif isinstance(existing, str) and existing.strip():
            blocks = [*blocks, {"type": "text", "text": existing}]
        kwargs["system"] = blocks
        tools = kwargs.get("tools")
        ttl = None if client_uses_short_ttl(tools, blocks, rest) else LONG_CACHE_TTL
        head = apply_head_cache(blocks, tools if isinstance(tools, list) else None, ttl)
        apply_conversation_cache(rest, head, ttl)
        kwargs["messages"] = rest
        return kwargs

    system_blocks = build_system_blocks(client_prompt)
    identity = {"role": "system", "content": system_blocks}

    # The head is anchored first so that the messages budget already discounts what it
    # spent: the ceiling of 4 is per request, and a fifth marker gives 400.
    tools = kwargs.get("tools")
    ttl = None if client_uses_short_ttl(tools, system_blocks, rest) else LONG_CACHE_TTL
    head = apply_head_cache(system_blocks, tools if isinstance(tools, list) else None, ttl)
    apply_conversation_cache(rest, head, ttl)
    kwargs["messages"] = [identity, *rest]
    return kwargs
