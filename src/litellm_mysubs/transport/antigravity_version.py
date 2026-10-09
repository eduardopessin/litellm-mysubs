"""The Antigravity client identity: the ``User-Agent`` every Cloud Code Assist call carries.

The backend gates models on the version in this header — the catalog included, not just
inference. Measured in production on 2026-10-09: with the fixed ``antigravity/hub/2.8.0``
every Claude id dropped out of ``:fetchAvailableModels`` (served on 2026-09-30), while the
Gemini ids kept answering. omp tracks the latest Antigravity release through the
electron-builder update manifest and falls back to a pinned version when it cannot read it;
this module does the same, once per process.

One module for every caller — inference, discovery, usage, the OAuth control plane — so
the identity cannot diverge between them again (it used to be three copied constants). It
imports nothing but ``httpx``: ``credentials/oauth.py`` and discovery must stay free of
LiteLLM.

Overrides, as in omp: ``PI_AI_ANTIGRAVITY_VERSION`` (also skips the manifest lookup),
``PI_AI_ANTIGRAVITY_CL``, ``PI_AI_ANTIGRAVITY_OS``, ``PI_AI_ANTIGRAVITY_ARCH``. An empty
value counts as unset, like omp's ``||``.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from typing import Final

import httpx

# omp: wire/gemini-headers.ts :: DEFAULT_ANTIGRAVITY_VERSION
# omp= wire/gemini-headers.ts :: DEFAULT_ANTIGRAVITY_VERSION = "2.19.1"
#: What goes out when the manifest cannot be read and no override is set.
DEFAULT_VERSION: Final = "2.19.1"

# omp: wire/gemini-headers.ts :: ANTIGRAVITY_VERSION_MANIFEST_URL
MANIFEST_URL: Final = (
    "https://antigravity-hub-auto-updater-974169037036.us-central1.run.app"
    "/manifest/latest-arm64-mac.yml"
)
#: The headers the Antigravity updater itself sends.
MANIFEST_HEADERS: Final[dict[str, str]] = {
    "Cache-Control": "no-cache",
    "User-Agent": "electron-builder",
}

# omp: wire/gemini-headers.ts :: ANTIGRAVITY_VERSION_FETCH_TIMEOUT_MS
#: Bounds the whole lookup, body included.
FETCH_TIMEOUT_S: Final = 5.0

# omp: wire/gemini-headers.ts :: ANTIGRAVITY_VERSION_RETRY_MS
#: A failed lookup is not retried before this delay, so the request path never pays the
#: timeout on every call while the manifest is unreachable.
RETRY_S: Final = 10 * 60.0

#: The changelist the version was captured with. The backend does not validate it (omp,
#: verified live: stale, zero and absent ``cl`` all pass model gating) and the manifest
#: carries none, so it stays.
DEFAULT_CL: Final = "963137146"

_ENV_VERSION: Final = "PI_AI_ANTIGRAVITY_VERSION"

_MANIFEST_VERSION_LINE: Final = re.compile(
    r"""^\s*version\s*:\s*(?:"([^"]*)"|'([^']*)'|([^\s#]+))\s*(?:#.*)?$"""
)
_SEMVER: Final = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")


class _Lookup:
    """Process state of the manifest lookup: what it found, what is in flight, when it
    last failed."""

    __slots__ = ("discovered", "failed_at", "task")

    def __init__(self) -> None:
        self.discovered: str | None = None
        self.task: asyncio.Task[None] | None = None
        self.failed_at: float | None = None


_lookup = _Lookup()


# omp: wire/gemini-headers.ts :: getAntigravityVersion
def version() -> str:
    """Override, else the manifest's version, else `DEFAULT_VERSION`."""
    return os.environ.get(_ENV_VERSION) or _lookup.discovered or DEFAULT_VERSION


# omp: wire/gemini-headers.ts :: getAntigravityUserAgent
def user_agent() -> str:
    """The ``User-Agent`` value, rebuilt on every call so a discovered version takes effect
    at once. ``os_type``/``arch`` are pinned to the darwin/arm64 client the format and the
    manifest come from, independent of this host."""
    cl = os.environ.get("PI_AI_ANTIGRAVITY_CL") or DEFAULT_CL
    os_type = os.environ.get("PI_AI_ANTIGRAVITY_OS") or "darwin"
    arch = os.environ.get("PI_AI_ANTIGRAVITY_ARCH") or "arm64"
    return f"antigravity/hub/{version()} (aidev_client; os_type={os_type}; arch={arch}; cl={cl})"


# omp: wire/gemini-headers.ts :: parseAntigravityManifestVersion
def parse_manifest_version(text: str) -> str | None:
    """The ``version:`` of an electron-builder manifest. The first ``version`` line decides:
    a malformed one gives ``None`` rather than a later line's value."""
    for line in re.split(r"\r?\n", text):
        match = _MANIFEST_VERSION_LINE.match(line)
        if match is None:
            continue
        found = (match.group(1) or match.group(2) or match.group(3) or "").strip()
        return found if _SEMVER.match(found) else None
    return None


# omp: wire/gemini-headers.ts :: ensureAntigravityVersion
async def ensure_version(client: httpx.AsyncClient | None = None) -> None:
    """Resolve the latest Antigravity release before a request goes out.

    Success is kept for the life of the process; a failure is silent (the fallback stays
    valid) and suppresses further lookups for `RETRY_S`. Skipped when the version override
    is set. Concurrent callers share one lookup, bounded only by its own timeout: a caller
    cancelled while waiting stops waiting without cancelling it (``shield``), as omp's
    ``signal`` does. ``client`` is the caller's, as omp passes its fetcher; without one the
    lookup opens its own.
    """
    if os.environ.get(_ENV_VERSION) or _lookup.discovered:
        return
    loop = asyncio.get_running_loop()
    task = _lookup.task
    # A task from another event loop (a CLI's `asyncio.run` next to the proxy's loop)
    # cannot be awaited here.
    if task is None or task.done() or task.get_loop() is not loop:
        if _lookup.failed_at is not None and time.monotonic() - _lookup.failed_at < RETRY_S:
            return
        task = _lookup.task = loop.create_task(_read_manifest(client))
    await asyncio.shield(task)


async def _read_manifest(client: httpx.AsyncClient | None) -> None:
    try:
        if client is None:
            async with httpx.AsyncClient() as own:
                found = await _fetch_manifest(own)
        else:
            found = await _fetch_manifest(client)
        if found:
            _lookup.discovered = found
    except Exception:  # silent by design: the pinned fallback stays valid
        pass
    finally:
        if _lookup.task is asyncio.current_task():
            _lookup.task = None
        if not _lookup.discovered:
            _lookup.failed_at = time.monotonic()


async def _fetch_manifest(client: httpx.AsyncClient) -> str | None:
    response = await asyncio.wait_for(
        client.get(MANIFEST_URL, headers=MANIFEST_HEADERS, timeout=FETCH_TIMEOUT_S),
        FETCH_TIMEOUT_S,
    )
    return parse_manifest_version(response.text) if response.is_success else None
