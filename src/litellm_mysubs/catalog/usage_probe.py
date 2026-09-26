"""Active quota probe: ask for usage instead of waiting for traffic.

`usage.from_headers` only knows what the upstream attached to an inference response. On a
subscriptions page that is little: whoever has just connected an account has made no
request at all, and the card would say the provider publishes no usage — which is false.
Anthropic and Codex both have a quota endpoint, and that is what this module queries.

Both were **measured** with HTTP 200 against real OAuth tokens:

    GET https://api.anthropic.com/api/oauth/usage
        -> {"five_hour": {"utilization": 15.0, "resets_at": "2026-09-19T12:30:00+00:00"}, ...}
    GET https://chatgpt.com/backend-api/wham/usage
        -> {"plan_type": "plus", "rate_limit": {"primary_window": {"used_percent": 0, ...}}}

The Codex path does **not** carry the `/codex/` segment: with it, measured, the backend
returns 403. The Antigravity one is not here — it is a POST with `project` in the body and
a list of alternative hosts, and lives in `ui/service.py :: fetch_usage`.

The rule that governs this module: never invent numbers. A probe that fails returns an
empty `UsageSnapshot()`, and the caller keeps the last known snapshot. An empty one means
"I don't know"; a bar at zero would mean "the account is free", which is a different
claim.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Final

import httpx

from ..credentials.store import Credential
from ..wire.anthropic import USER_AGENT
from .usage import UsageSnapshot, from_anthropic_usage, from_codex_usage

# omp: usage/claude.ts :: fetchClaudeUsage
# omp: usage/claude-api.ts :: claudeOAuthBaseUrl, DEFAULT_CLAUDE_API_BASE_URL
# omp= DEFAULT_CLAUDE_API_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_USAGE_URL: Final = "https://api.anthropic.com/api/oauth/usage"

# omp: usage/openai-codex.ts :: CODEX_USAGE_PATH, buildCodexUsageUrl
# omp= usage/openai-codex.ts :: CODEX_USAGE_PATH = "wham/usage"
#: Without the `/codex/` segment: measured, `/backend-api/codex/wham/usage` returns 403.
CODEX_USAGE_URL: Final = "https://chatgpt.com/backend-api/wham/usage"

#: The `oauth-2025-04-20` beta is what classifies the request as coming from an OAuth
#: credential — without it the server treats it as an API key, which this token is not.
#:
#: The `User-Agent` is **not** part of that: measured against the real endpoint, this route
#: answers 200 with the Claude Code string, with this package's own, and with no
#: `User-Agent` at all. An earlier comment here claimed the CLI identity was required; it
#: was never measured, and it is wrong.
_ANTHROPIC_BETA: Final = "oauth-2025-04-20"

#: Per-probe ceiling. High enough for a slow upstream, short enough not to hold the page
#: load hostage to a provider that does not answer.
_TIMEOUT_S: Final = 15.0


async def probe(credential: Credential, *, client: httpx.AsyncClient) -> UsageSnapshot:
    """The usage the provider publishes, or an empty snapshot.

    Never raises. Network down, a 401 from a rotated token, a 500 from the upstream or a
    body that is not JSON all give the same result — empty — because taking down the
    subscriptions page over one quota bar is worse than not having the bar.

    `google-antigravity` leaves here empty on purpose and with **no request at all**: its
    endpoint is a POST with a body and an alternative host, handled in
    `ui/service.py :: fetch_usage`.
    """
    parse: Callable[[Mapping[str, Any]], UsageSnapshot]
    if credential.provider == "anthropic":
        url = ANTHROPIC_USAGE_URL
        headers = {
            "Authorization": f"Bearer {credential.access_token}",
            "anthropic-beta": _ANTHROPIC_BETA,
            "User-Agent": USER_AGENT,
            "accept": "application/json",
        }
        parse = from_anthropic_usage
    elif credential.provider == "openai-codex":
        url = CODEX_USAGE_URL
        headers = {
            "Authorization": f"Bearer {credential.access_token}",
            "accept": "application/json",
        }
        parse = from_codex_usage
    else:
        return UsageSnapshot()

    try:
        response = await client.get(url, headers=headers, timeout=_TIMEOUT_S)
        if response.status_code != 200:
            return UsageSnapshot()
        payload = response.json()
    except Exception:
        return UsageSnapshot()
    if not isinstance(payload, dict):
        return UsageSnapshot()
    return parse(payload)
