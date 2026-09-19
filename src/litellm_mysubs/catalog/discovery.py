"""Discovery of the models a subscription serves, for the user to choose from.

Real capability differs per provider, and the design reflects that instead of pretending a
uniform interface:

* **Google Antigravity** has a queryable catalog (``:fetchAvailableModels``). The response
  is the account's truth, including the variants that no longer answer
  (``deprecatedModelIds``). Subtracting those is `ModelCatalog.update`'s job, not this
  module's.
* **Anthropic** and **OpenAI Codex** have no catalog. ``/v1/models`` returns 401 with a
  subscription token, and the served set **is not derivable** from the public list:
  ``claude-sonnet-4-20250514`` exists in the Anthropic API and returns 404 on a Max
  account. What is left is a curated list of measured names and a real probe of each one.

Three rules govern the result, and all of them come from the same principle — never invent
a fact about someone else's account:

1. ``verified=True`` only when the upstream actually answered. A name nobody managed to ask
   about shows up with ``verified=False`` and the reason in ``note``.
2. A probe that fails **for network reasons** does not mark the model as unserved. "The
   upstream said no" and "I could not ask" are different facts: only the first removes the
   model from the list, the second leaves it there as unverified.
3. An unreachable catalog does not produce a plausible list. Either the real snapshot is
   returned labelled with its age, or `DiscoveryError` is raised.

Note about OMP: `pi-catalog` has `discovery/codex.ts :: fetchCodexModels`, which reads
``/backend-api/codex/models``. It is not used here because what that endpoint advertises
has not been measured against a subscription account on this installation, and the
measurement that does exist says the opposite (the public catalog does not predict what the
subscription serves). The probe measures; the upstream list would, for now, be a guess.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Final

import httpx

from ..credentials.store import Credential
from ..transport import hosts, sse
from ..transport.retry import is_unsupported_model
from ..wire import anthropic, codex
from ..wire.antigravity import is_retired_response
from ..wire.antigravity_models import BROKEN_WIRE, ModelCatalog

#: Probes in flight at once. The limit exists because a curated list fires one request per
#: name against the same backend: without a ceiling, connecting a subscription opened half a
#: dozen simultaneous connections to the upstream just to draw a selection screen.
PROBE_CONCURRENCY: Final = 4

#: Wait ceiling per Antigravity probe. Short on purpose: what the probe produces is a mark
#: on a selection screen, not a response for the user to read. Anything exceeding this is a
#: fact about this machine's network, and the verdict for those is always "could not
#: probe" — never "not served".
PROBE_TIMEOUT_S: Final = 10.0

#: Output ceiling for Antigravity probes. Each probe is a billed turn; the verdict comes
#: from the status and the in-band warning, not from the generated text, so there is no
#: reason to pay for more than a handful of tokens per catalog name.
PROBE_MAX_OUTPUT_TOKENS: Final = 8

#: Anthropic inference endpoint. It duplicates the value `plugin.py` derives through
#: LiteLLM; importing it from there would drag the whole of LiteLLM into discovery, which is
#: precisely the dependency this package keeps apart.
ANTHROPIC_MESSAGES_URL: Final = "https://api.anthropic.com/v1/messages"

#: Value of ``anthropic-version``. Fixed in OMP's `providers/anthropic.ts` (it carries no
#: anchor because the anchor checker only accepts identifier symbols).
ANTHROPIC_API_VERSION: Final = "2023-06-01"

#: Responses API served by the ChatGPT subscription. Same reason for the duplication as
#: above: `plugin.py` has the twin constant and imports LiteLLM.
CODEX_RESPONSES_URL: Final = "https://chatgpt.com/backend-api/codex/responses"

# omp: wire/gemini-headers.ts :: getAntigravityUserAgent
#: The Cloud Code Assist backend gates model availability on the client version; `cl` is
#: not validated.
ANTIGRAVITY_USER_AGENT: Final = (
    "antigravity/hub/2.8.0 (aidev_client; os_type=darwin; arch=arm64; cl=963137146)"
)

#: Textual marker of Anthropic refusing a name. The status alone is not enough: the OAuth
#: route returns 404 for wrong paths too, and it is the body that names the model.
ANTHROPIC_NOT_FOUND_MARKER: Final = "not_found_error"

#: Antigravity catalog ``modelProvider`` -> model family.
#:
#: Antigravity is a reseller: it serves three model families, and it is the family — not
#: who serves it — that decides the pricing prefix in LiteLLM. Measured on the real
#: account: of the catalog's 32 models, `claude-sonnet-4-6` and `claude-opus-4-6-thinking`
#: come as ``MODEL_PROVIDER_ANTHROPIC``, `gpt-oss-120b-medium` comes as
#: ``MODEL_PROVIDER_OPENAI``, and the rest as ``MODEL_PROVIDER_GOOGLE`` — including opaque
#: names such as `chat_23310` and `tab_flash_lite_preview`, which no name-based heuristic
#: would classify.
#:
#: An enum outside this table yields an empty family, never a guessed one: the consumer
#: (`catalog/deployments.py`) has a safe fallback prefix for that case, and a guess here
#: would only trade a zero price for a wrong one.
MODEL_FAMILY_BY_PROVIDER: Final[dict[str, str]] = {
    "MODEL_PROVIDER_GOOGLE": "google",
    "MODEL_PROVIDER_ANTHROPIC": "anthropic",
    "MODEL_PROVIDER_OPENAI": "openai",
}

# Curated Anthropic list: **only** names with a measured 200 response against a
# subscription token. The source of each one is inside the package itself, which makes this
# list a consequence of measurements and not of taste:
#
#   opus-5, fable-5, sonnet-5, opus-4-8, opus-4-6, sonnet-4-6, opus-4-5, sonnet-4-5,
#   haiku-4-5  -> `wire/anthropic.py`, the `ADAPTIVE_EFFORT` table, where every row carries
#                 the reasoning characters the upstream returned. A model that does not
#                 answer does not produce that count.
#   opus-4-8    -> also the Claude Code bootstrap model
#                  (`registry/oauth/anthropic.ts :: CLAUDE_CODE_BOOTSTRAP_MODEL`).
#   haiku-4-5   -> the model the original uses in the proxy health probe.
#
# Deliberately out: `claude-sonnet-4-20250514` (it exists in the public API, 404 on the Max
# account — the counter-example that justifies this module) and the OMP catalog names with
# no measurement of ours (`claude-mythos-5`, `claude-fable-5-1`, ...). Adding them is one
# line, once measured; putting them here now would make the probe look like confirmation of
# a guess.
CURATED_ANTHROPIC: Final[tuple[str, ...]] = (
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-opus-4-8",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
)

# Curated Codex list:
#
#   gpt-5.5       -> target of the `gpt-5`/`gpt5`/`codex` aliases in `wire/codex.py`, and
#                    the model of the original's health probe. Served, measured.
#   gpt-6-astra   -> target of the `gpt-6`/`gpt6` aliases in the same table.
#   gpt-5.6-luna, gpt-5.6-sol, gpt-5.6-terra, gpt-daybreak-blue-latest
#                 -> the only `openai-codex` provider entries in OMP's aggregated catalog
#                    (`pi-catalog`, models.json). That is a catalog specific to the
#                    subscription backend, not to the public API — weaker evidence than a
#                    measurement of ours, which is why the probe decides.
#
# Deliberately out: `gpt-5.4` and `gpt-5.4-mini`. Measured: "The 'gpt-5.4' model is not
# supported when using Codex with a ChatGPT account". They were in the original pointing at
# gpt-5.5, and the client was billed against a model that never ran.
CURATED_CODEX: Final[tuple[str, ...]] = (
    "gpt-5.5",
    "gpt-6-astra",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-daybreak-blue-latest",
)


class DiscoveryError(RuntimeError):
    """It was not possible to learn what the account serves.

    Raised only where the alternative would be to invent: an unreachable catalog with no
    earlier snapshot has no honest answer in list form.
    """


@dataclass(frozen=True, slots=True)
class DiscoveredModel:
    """A model the subscription may serve.

    ``verified`` is the only thing that separates a fact from a hypothesis, which is why
    ``note`` is in practice mandatory whenever ``verified`` is false: a selection screen
    that shows both cases alike turns the curated list into a promise.
    """

    wire_name: str
    """Name exactly as it goes on the wire. Bare, with no provider prefix."""

    suggested_name: str
    """Suggested public name in LiteLLM. A suggestion: the user changes it in the UI."""

    verified: bool
    """Whether the upstream actually answered this name during this discovery."""

    note: str = ""
    """Why it was not verified, where applicable."""

    family: str = ""
    """Model family: ``"google"``, ``"anthropic"``, ``"openai"``, or ``""``.

    Not who serves it, but what is being served — the distinction only shows up in
    Antigravity, which resells all three. It is what decides the pricing prefix in
    `catalog/deployments.py`: measured with `litellm.completion_cost`, `gemini-2.5-pro`
    only has a table under ``gemini/`` (or ``vertex_ai/``) and `claude-sonnet-4-6` under
    ``anthropic/``; under ``openai/`` both cost zero. Empty means "I don't know", and the
    default is empty so that older callers do not start asserting a family nobody measured.
    """


def suggested_name(wire_name: str) -> str:
    """Public name from the wire name: ``anthropic/claude-opus-5`` -> ``claude-opus-5``.

    The provider prefix belongs in ``litellm_params["model"]``, not in ``model_name``: it
    is ``model_name`` that echoes in the spend log, and a prefixed name there names
    something the client never asked for.
    """
    return str(wire_name).split("/")[-1]


@dataclass(frozen=True, slots=True)
class _Probe:
    """The result of a probe. ``served=None`` means "I could not ask"."""

    served: bool | None
    note: str = ""


#: A probe: client, credential, wire name -> verdict.
Probe = Callable[[httpx.AsyncClient, Credential, str], Awaitable[_Probe]]


async def discover(
    credential: Credential,
    *,
    client: httpx.AsyncClient,
    catalog: ModelCatalog | None = None,
    now: float | None = None,
    probe: bool = False,
) -> list[DiscoveredModel]:
    """The models this subscription serves.

    ``catalog`` is only used by Google: passing the process's live catalog is what keeps an
    empty response from the endpoint from erasing what was already known. ``now`` exists to
    make the snapshot age deterministic in tests.

    ``probe`` only affects Google, and it **spends quota**: it fires a minimal
    ``:streamGenerateContent`` turn per catalog name — on the measured account that is 32
    names, hence 32 billed turns per discovery. That is why it is off by default: the
    unprobed list is what the account advertises, which is legitimate information, only
    unconfirmed. Turned on, it replaces that advertisement with measurement — it is the only
    way to tell `gemini-3.5-flash-lite` (answers "2 + 2 = 4") from `gemini-3.5-flash-low`
    (200 with a retirement warning), or `tab_flash_lite_preview` (answers) from
    `tab_jump_flash_lite_preview` (400). Neither pair is separable by name, and no model is
    removed from the list because of the probe — see `_discover_google`.

    The probe lives here as a parameter and not as a separate `probe_models(...)` because
    the verdict has to land on the same `DiscoveredModel` the catalog produces: a separate
    function returned a second object every caller would have to match up with the first,
    and a caller that forgot went back to showing the catalog as truth — the very defect
    this corrects.
    """
    if credential.provider == "google-antigravity":
        return await _discover_google(
            credential, client=client, catalog=catalog or ModelCatalog(), now=now, probe=probe
        )
    if credential.provider == "anthropic":
        # These two have no catalog with ``modelProvider``, and they do not need one: the
        # one who serves is the owner of the family. The Max subscription only serves
        # `claude-*`, the Codex one only serves `gpt-*`. The constant here is measured by
        # the curated list above.
        return await _discover_probed(
            credential, CURATED_ANTHROPIC, _probe_anthropic, client, family="anthropic"
        )
    return await _discover_probed(credential, CURATED_CODEX, _probe_codex, client, family="openai")


# -- Google Antigravity: real catalog ------------------------------------------


# omp: discovery/antigravity.ts :: fetchAntigravityDiscoveryModels
async def _discover_google(
    credential: Credential,
    *,
    client: httpx.AsyncClient,
    catalog: ModelCatalog,
    now: float | None,
    probe: bool = False,
) -> list[DiscoveredModel]:
    """The account's catalog, trying both endpoints in order.

    Only the `BROKEN_WIRE` subtraction happens here: the ``deprecatedModelIds`` one belongs
    to `ModelCatalog`, which is where the payload shape is verified.

    With ``probe``, ``verified`` stops coming from the catalog and starts coming from the
    response: the catalog advertises `chat_23310` (400 INVALID_ARGUMENT) next to
    `tab_flash_lite_preview` (answers), and being advertised was never proof of being
    served. No name leaves the list because of the probe — the user sees what the account
    advertises, marked with what was measured; hiding an advertised model would be deciding
    for them.
    """
    moment = time.time() if now is None else now
    payload = await _fetch_catalog(credential, client=client)

    # What counts is whether the catalog *absorbed* anything, not whether there was a
    # response: a payload whose models are all in ``deprecatedModelIds`` is a 200 that adds
    # nothing, and `ModelCatalog.update` leaves the earlier snapshot intact on purpose. The
    # pair (ids, instant) is compared because neither alone distinguishes the cases. Known
    # degeneracy: a caller that passes ``now`` equal to the instant of the previous
    # collection *and* receives exactly the same ids sees the list labelled as a 0 s
    # snapshot. With a real clock it does not happen, and the error falls on the safe side
    # — a label too many, never a model too many.
    before = catalog.ids, catalog.fetched_at
    if payload is not None:
        catalog.update(payload, now=moment)
    absorbed = payload is not None and (catalog.ids, catalog.fetched_at) != before

    if not catalog.ids:
        raise DiscoveryError(
            "Google Antigravity: the catalog did not respond and there is no earlier "
            "snapshot; listing models here would mean inventing them"
        )

    age = ""
    if not absorbed:
        # There is an earlier snapshot and the endpoint did not refresh it. Returning it
        # is legitimate — it was measured — but without the age it would pass for current,
        # which is exactly the plausible number this package does not invent.
        seconds = int(max(0.0, moment - catalog.fetched_at))
        reason = (
            "the endpoint did not respond"
            if payload is None
            else "the endpoint responded with no usable models"
        )
        age = f"catalog snapshot {seconds} s old; {reason} now"

    verdicts = await _probe_catalog(credential, catalog.ids, client=client) if probe else {}

    discovered: list[DiscoveredModel] = []
    for wire in catalog.ids:
        notes = [age] if age else []
        verdict = verdicts.get(wire)
        if verdict is None:
            broken = wire in BROKEN_WIRE
            if broken:
                # The catalog advertises them and streamGenerateContent refuses them.
                # Hiding them would lose real information; advertising them as served would
                # repeat the defect.
                notes.append(
                    "the catalog advertises it but streamGenerateContent returns 400 "
                    "INVALID_ARGUMENT for this variant"
                )
            served = not broken
        else:
            # Measured beats advertised, both ways: a `BROKEN_WIRE` name that answers
            # becomes verified, and a clean name that returns the retirement warning does
            # not. The name decides nothing here.
            served = verdict.served is True
            if verdict.note:
                notes.append(verdict.note)
        discovered.append(
            DiscoveredModel(
                wire_name=wire,
                suggested_name=suggested_name(wire),
                verified=served,
                note="; ".join(notes),
                family=_catalog_family(catalog.info.get(wire)),
            )
        )
    return discovered


def _catalog_family(entry: Any) -> str:
    """The family of a catalog entry, from its ``modelProvider``.

    ``modelProvider`` is read and not ``apiProvider``: the second says which way Antigravity
    speaks (measured: ``API_PROVIDER_GOOGLE_GEMINI`` even for the `claude-*` ones), the
    first says whose the model is — and the LiteLLM pricing table depends on whose the model
    is. An entry with an unexpected shape or an enum outside the table gives ``""``: the
    caller has a fallback prefix, and inventing a family here would trade zero cost for
    wrong cost.
    """
    if not isinstance(entry, dict):
        return ""
    return MODEL_FAMILY_BY_PROVIDER.get(str(entry.get("modelProvider") or ""), "")


# omp: discovery/antigravity.ts :: FETCH_AVAILABLE_MODELS_PATH
# omp= discovery/antigravity.ts :: FETCH_AVAILABLE_MODELS_PATH = "/v1internal:fetchAvailableModels"
async def _fetch_catalog(
    credential: Credential, *, client: httpx.AsyncClient
) -> dict[str, Any] | None:
    """The ``:fetchAvailableModels`` payload, or ``None`` if no endpoint answered.

    It walks both hosts like the rest of the package: a host being down is not an account
    with no models.
    """
    headers = {
        "Authorization": f"Bearer {credential.access_token}",
        "Content-Type": "application/json",
        "User-Agent": ANTIGRAVITY_USER_AGENT,
    }
    for host in hosts.HOSTS:
        try:
            response = await client.post(host + hosts.MODELS_PATH, json={}, headers=headers)
        except httpx.HTTPError:
            continue
        if response.status_code != 200:
            continue
        try:
            payload = response.json()
        except ValueError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


async def _probe_catalog(
    credential: Credential,
    wires: tuple[str, ...],
    *,
    client: httpx.AsyncClient,
) -> dict[str, _Probe]:
    """Probe every catalog name, in parallel and under the same ceiling as the curated list.

    The ceiling matters more here than on the curated path: the measured account's catalog
    has 32 names, and without a semaphore connecting the subscription opened 32 connections
    at once to the same backend — which Google treats as a spike and answers with 503.
    """
    limit = asyncio.Semaphore(PROBE_CONCURRENCY)

    async def guarded(wire: str) -> _Probe:
        async with limit:
            return await _probe_antigravity(client, credential, wire)

    results = await asyncio.gather(*(guarded(wire) for wire in wires))
    return dict(zip(wires, results, strict=True))


async def _probe_antigravity(
    client: httpx.AsyncClient, credential: Credential, wire: str
) -> _Probe:
    """Minimal ``:streamGenerateContent`` turn for one catalog name.

    ``"2 + 2"`` is sent rather than a lone character because the difference only shows up
    with an answerable request: measured, `gemini-3.5-flash-lite` returns "2 + 2 = 4" with
    12 tokens, while `gemini-3.5-flash-low` returns, for the same request and with HTTP
    200, "Gemini 3.5 Flash is no longer available. Please switch to Gemini 3.7 Flash..."
    with usage at zero. An empty prompt left the two indistinguishable — both "answered
    200".

    Four verdicts, and the distinction between them is this module's value:

    * 200 with real content -> served;
    * 200 with the retirement warning and usage at zero -> not served, with its own note;
    * 400 INVALID_ARGUMENT (measured: `chat_23310`, `chat_20706`,
      `tab_jump_flash_lite_preview`) -> not served, the upstream refused;
    * anything else — 503 "No capacity available" (measured on `gpt-oss-120b-medium` and
      `gemini-2.5-pro`), a timeout, the network being down — -> unprobed. That is Google's
      capacity or this machine's network, never a fact about the account, and treating it
      as a refusal switched off a good model until the next discovery.
    """
    body = {
        "project": credential.project_id,
        "requestId": _probe_request_id(wire),
        "model": wire,
        "userAgent": "antigravity",
        "requestType": "agent",
        "request": {
            "contents": [{"role": "user", "parts": [{"text": "2 + 2"}]}],
            "generationConfig": {"maxOutputTokens": PROBE_MAX_OUTPUT_TOKENS},
        },
    }
    headers = {
        "Authorization": f"Bearer {credential.access_token}",
        "Content-Type": "application/json",
        "User-Agent": ANTIGRAVITY_USER_AGENT,
        "accept": "text/event-stream",
    }
    url = hosts.HOSTS[0] + hosts.STREAM_PATH
    try:
        async with client.stream(
            "POST", url, json=body, headers=headers, timeout=PROBE_TIMEOUT_S
        ) as response:
            if response.status_code != 200:
                return _antigravity_status(response.status_code)
            lines = [line async for line in response.aiter_lines()]
    except httpx.HTTPError as exc:
        return _unprobed(f"{type(exc).__name__}: {exc}")

    return _antigravity_stream(lines)


def _antigravity_status(status: int) -> _Probe:
    """Verdict from a status that is not 200."""
    if status == 400:
        # CCA refusing a name. Measured: `chat_23310` and `tab_jump_flash_lite_preview`
        # give this, while `tab_flash_lite_preview` — same prefix — answers.
        return _Probe(False, "upstream refused: HTTP 400")
    return _Probe(
        None,
        f"could not probe: upstream responded HTTP {status}, which does not distinguish "
        f"an unserved model from a temporary outage",
    )


def _antigravity_stream(lines: list[str]) -> _Probe:
    """Verdict from the events of a stream with HTTP 200.

    The text of every event is joined before deciding because the retirement warning arrives
    split across several ``parts`` like any other response: looking only at the first event
    classified a dead model as alive.
    """
    text: list[str] = []
    usage: dict[str, Any] | None = None
    for event in sse.iter_events(lines):
        if isinstance(error := event.get("error"), dict) and int(error.get("code") or 0) >= 400:
            # In-band error: CCA returns it inside a 200, as `plugin.py` documents. The
            # status alone said "served".
            return _Probe(False, f"upstream refused: HTTP {error.get('code')} in band")
        payload = event.get("response") or {}
        if isinstance(meta := payload.get("usageMetadata"), dict):
            usage = meta
        for candidate in payload.get("candidates") or []:
            for part in (candidate.get("content") or {}).get("parts") or []:
                text.append(str(part.get("text") or ""))

    joined = "".join(text)
    if is_retired_response(joined, usage):
        return _Probe(False, "model retired by upstream")
    if joined.strip():
        return _Probe(True)
    # A 200 with no text at all. Measured: `gemini-pro-agent` answers empty to "hi" and is
    # still served, so this is not a refusal — it is a probe that measured nothing.
    return _Probe(None, "could not probe: the stream closed with no content")


def _unprobed(detail: str) -> _Probe:
    """Transport failure on a catalog probe: unprobed, never "not served"."""
    return _Probe(None, f"could not probe: {detail}")


def _probe_request_id(wire: str) -> str:
    """``requestId`` in the format CCA requires (`plugin.py :: _request_id`).

    The step is the probed name instead of a counter: the probes run in parallel and a
    shared counter would impose no order at all, only different ids.
    """
    return f"agent/mysubs-discovery/{int(time.time() * 1000)}/probe/{wire}"


# -- Anthropic and Codex: curated list + probe ---------------------------------


async def _discover_probed(
    credential: Credential,
    curated: tuple[str, ...],
    probe: Probe,
    client: httpx.AsyncClient,
    *,
    family: str,
) -> list[DiscoveredModel]:
    """Probe the curated list, in parallel and under a concurrency ceiling.

    A name the upstream refuses leaves the list — it is not served, and offering it produced
    a deployment that only knows how to return 404. A name the probe could not ask about
    stays, with ``verified=False`` and the reason: this machine's network is not a fact
    about the account.
    """
    limit = asyncio.Semaphore(PROBE_CONCURRENCY)

    async def guarded(wire: str) -> _Probe:
        async with limit:
            return await probe(client, credential, wire)

    results = await asyncio.gather(*(guarded(wire) for wire in curated))

    discovered: list[DiscoveredModel] = []
    for wire, result in zip(curated, results, strict=True):
        if result.served is False:
            continue
        discovered.append(
            DiscoveredModel(
                wire_name=wire,
                suggested_name=suggested_name(wire),
                verified=result.served is True,
                note=result.note,
                family=family,
            )
        )
    return discovered


async def _post_status(
    client: httpx.AsyncClient,
    url: str,
    *,
    body: dict[str, Any],
    headers: dict[str, str],
) -> tuple[int, str]:
    """The status, and the body only when it is not 200.

    Streamed so that a successful probe closes the connection right after the headers: what
    matters is the verdict, not the generated tokens.
    """
    async with client.stream("POST", url, json=body, headers=headers) as response:
        if response.status_code == 200:
            return 200, ""
        return response.status_code, (await response.aread()).decode("utf-8", "replace")


def _unreachable(exc: httpx.HTTPError) -> _Probe:
    """Transport failure: unverified, and never "not served"."""
    return _Probe(
        None,
        f"the probe did not reach the upstream ({type(exc).__name__}: {exc}); "
        f"unverified, not refused",
    )


async def _probe_anthropic(client: httpx.AsyncClient, credential: Credential, wire: str) -> _Probe:
    """Minimal messages request: ``max_tokens=1`` and a one-character turn.

    The identity block has to come first even in a probe — measured: a ``system`` carrying
    only the client prompt returns 429, and a 429 here was indistinguishable from quota.
    """
    body: dict[str, Any] = {
        "model": wire,
        "max_tokens": 1,
        "system": anthropic.build_system_blocks(""),
        "messages": [{"role": "user", "content": "."}],
    }
    headers = {
        "Authorization": f"Bearer {credential.access_token}",
        "Content-Type": "application/json",
        "accept": "application/json",
        "anthropic-version": ANTHROPIC_API_VERSION,
        "anthropic-beta": anthropic.build_betas(thinking=False),
        **anthropic.CLIENT_HEADERS,
    }
    try:
        status, text = await _post_status(
            client, ANTHROPIC_MESSAGES_URL, body=body, headers=headers
        )
    except httpx.HTTPError as exc:
        return _unreachable(exc)

    if status == 200:
        return _Probe(True)
    if status == 404 and ANTHROPIC_NOT_FOUND_MARKER in text:
        return _Probe(False, "upstream refused the name with not_found_error")
    return _Probe(
        None,
        f"upstream responded HTTP {status}, which does not distinguish a nonexistent "
        f"model from a temporary refusal; unverified",
    )


async def _probe_codex(client: httpx.AsyncClient, credential: Credential, wire: str) -> _Probe:
    """Minimal turn on the Responses API, with reasoning switched off.

    Codex refusing a name is a 400 with its own marker, and it is `transport.retry` that
    recognises it — the same function the transport uses in production, so that the probe
    and the real path cannot diverge on the definition of "not served".
    """
    body = codex.build_request_body(
        wire,
        [{"role": "user", "content": "."}],
        extra={"reasoning_effort": "none"},
    )
    headers = codex.build_headers(
        credential.access_token,
        window_id=_probe_window_id(credential),
        model=str(body.get("model") or wire),
    )
    try:
        status, text = await _post_status(client, CODEX_RESPONSES_URL, body=body, headers=headers)
    except httpx.HTTPError as exc:
        return _unreachable(exc)

    if status == 200:
        return _Probe(True)
    if status == 404 or (status == 400 and is_unsupported_model(text)):
        return _Probe(False, "this ChatGPT account does not serve this model")
    return _Probe(
        None,
        f"upstream responded HTTP {status}, which does not distinguish a nonexistent "
        f"model from a temporary refusal; unverified",
    )


def _probe_window_id(credential: Credential) -> str:
    """Window identity of the probes.

    Derived from the account and not random: the backend uses ``window_id`` for the prompt
    cache, and a new id per probe aged the cache of the user's real session.
    """
    return f"mysubs-discovery-{codex.account_id(credential.access_token) or 'anon'}"
