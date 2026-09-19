"""Paste-based OAuth flow and token renewal, for the three providers.

Why paste is the main route and not the fallback: these clients' native flow pins a
callback server to ``localhost:1455`` (Codex), ``:54545`` (Anthropic) or ``:51121``
(Antigravity). On a containerized or clustered LiteLLM the user's browser reaches none of
those ports. What it does reach is the page's text box. So the redirect URI stays whatever
the provider has registered — what changes is where the code is read, not where it is sent.

What this module does **not** do: it opens no sockets waiting for callbacks, opens no
browsers and decides nothing about when to renew. It takes what the user pasted, or an
expiring credential, and returns a ``Credential``.

Two rules inherited from measured incidents:

* **A single refresh owner** (see ``store.py``). ``refresh()`` accepts an optional
  ``CredentialStore``; if one is given and it is not the owner, it refuses to exchange
  instead of running the race.
* **Rotation replaces.** Anthropic and OpenAI issue single-use refresh tokens. Keeping the
  old one after the provider returns a new one guarantees ``invalid_grant`` on the next
  renewal — so the new token replaces it, and only when the provider does not rotate is the
  previous one preserved.

Provider errors surface with the real body. An upstream ``error_description`` is the only
thing that tells the user what to do next; swapping it for a generic message erases it.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import secrets
import time
import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Final, Literal

import httpx

from .store import Credential, CredentialStore, ProviderId

__all__ = [
    "ANTHROPIC_GRANT_TTL_S",
    "AuthRequest",
    "NotRefreshOwnerError",
    "OAuthError",
    "begin",
    "callback_origin",
    "complete",
    "refresh",
]


class OAuthError(RuntimeError):
    """An OAuth flow failure, carrying the **real** upstream status and body.

    ``status`` is zero when the failure is local (invalid paste, mismatched ``state``) and
    there was no provider response.
    """

    __slots__ = ("body", "provider", "status")

    def __init__(self, provider: str, message: str, *, status: int = 0, body: str = "") -> None:
        super().__init__(message)
        self.provider = provider
        self.status = status
        self.body = body


class NotRefreshOwnerError(OAuthError):
    """A renewal requested from a store that does not own the refresh."""


# omp: registry/oauth/anthropic-constants.ts :: ANTHROPIC_OAUTH_GRANT_TTL_MS
#: Absolute lifetime of the Anthropic grant, anchored at the interactive login. Rotation
#: does **not** extend it: ~30 days later the endpoint returns ``invalid_grant`` for even
#: the most recent token and only a fresh login recovers the account. It is a warning
#: heuristic, not a wire contract.
ANTHROPIC_GRANT_TTL_S: Final = 30 * 24 * 60 * 60.0

# omp: providers/claude-code-fingerprint.ts :: claudeCodeSdkVersion
# omp= providers/claude-code-fingerprint.ts :: claudeCodeSdkVersion = "0.112.1"
#: Goes in the ``User-Agent`` of the Anthropic renewal. Claude Code sends these headers on
#: the renewal but not on the initial code exchange.
CLAUDE_CODE_SDK_VERSION: Final = "0.112.1"

# omp: wire/gemini-headers.ts :: getAntigravityUserAgent
#: ``User-Agent`` for the Antigravity control plane (``loadCodeAssist``/``onboardUser``).
#: The backend does not validate ``cl``; only the version gates. Repeated here instead of
#: imported from ``plugin.py`` because that module drags in the whole of LiteLLM, and
#: discovering a project does not need it.
ANTIGRAVITY_USER_AGENT: Final = (
    "antigravity/hub/2.8.0 (aidev_client; os_type=darwin; arch=arm64; cl=963137146)"
)

# omp: registry/oauth/google-antigravity.ts :: ANTIGRAVITY_LOAD_CODE_ASSIST_METADATA
_ANTIGRAVITY_METADATA: Final[dict[str, str]] = {"ideType": "ANTIGRAVITY"}

# omp: registry/oauth/google-antigravity.ts :: LOAD_CODE_ASSIST_URL, ONBOARD_USER_URL
_CLOUD_CODE_ENDPOINT: Final = "https://daily-cloudcode-pa.googleapis.com"
_LOAD_CODE_ASSIST_URL: Final = f"{_CLOUD_CODE_ENDPOINT}/v1internal:loadCodeAssist"
_ONBOARD_USER_URL: Final = f"{_CLOUD_CODE_ENDPOINT}/v1internal:onboardUser"
_OPERATIONS_URL: Final = f"{_CLOUD_CODE_ENDPOINT}/v1internal"

# omp: registry/oauth/google-antigravity.ts :: FREE_TIER_ID
# omp= registry/oauth/google-antigravity.ts :: FREE_TIER_ID = "free-tier"
_FREE_TIER_ID: Final = "free-tier"

# omp: registry/oauth/google-antigravity.ts :: ONBOARD_TIMEOUT_MS, ONBOARD_POLL_INTERVAL_MS
_ONBOARD_TIMEOUT_S: Final = 30.0
_ONBOARD_POLL_INTERVAL_S: Final = 1.0


@dataclass(frozen=True, slots=True)
class _Provider:
    """What distinguishes one provider from another in the authorization code flow."""

    client_id: str
    authorize_url: str
    token_url: str
    scopes: tuple[str, ...]
    redirect_uri: str
    pkce: bool
    authorize_params: tuple[tuple[str, str], ...]
    token_body: Literal["json", "form"]
    #: Extra parameters on the code exchange. ``{state}`` is replaced by the validated
    #: state.
    exchange_params: tuple[tuple[str, str], ...] = ()
    refresh_headers: tuple[tuple[str, str], ...] = ()
    client_secret: str = ""
    #: Margin subtracted from ``expires_in``, to renew before the token dies.
    expiry_skew_s: float = 0.0


# The three provider records come from the OMP declarative rules
# (`compat/rules/auth/<provider>.kdl`), which is where the client ids, URLs and extra
# parameters actually live. The KDL nodes quoted in the comments — `authorize-url`,
# `token url`, `authorize-params`, `callback` — do not fit in an anchor (the checker does
# not accept hyphens in symbols), so the anchor points at a literal symbol from the same
# file and the node is named here.

# omp: registry/oauth/openai-codex.ts :: CLIENT_ID, AUTHORIZE_URL, TOKEN_URL, SCOPE
# omp= registry/oauth/openai-codex.ts :: CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
# omp: compat/rules/auth/openai-codex.kdl :: callback, token, credential
_CODEX: Final = _Provider(
    client_id="app_EMoamEEZ73f0CkXaXp7hrann",
    authorize_url="https://auth.openai.com/oauth/authorize",
    token_url="https://auth.openai.com/oauth/token",
    scopes=(
        "openid",
        "profile",
        "email",
        "offline_access",
        "api.connectors.read",
        "api.connectors.invoke",
    ),
    # OpenAI only authorizes this exact URI. An occupied port has to fail, not fall back to
    # another: the rule's `port-fallback=#false` says so, and in the paste flow that becomes
    # a fixed URI instead of one derived from a server we never opened.
    redirect_uri="http://localhost:1455/auth/callback",
    pkce=True,
    authorize_params=(
        ("id_token_add_organizations", "true"),
        ("codex_cli_simplified_flow", "true"),
        # omp: wire/codex.ts :: ORIGINATOR_CODEX
        ("originator", "omp"),
    ),
    token_body="form",
)

# omp: compat/rules/auth/anthropic.kdl :: OWQxYzI1MGEtZTYxYi00NGQ5LTg4ZWQtNTk0NGQxOTYyZjVl
# omp: compat/rules/auth/anthropic.kdl :: scopes, pkce, callback, credential
_ANTHROPIC: Final = _Provider(
    # Public client id, kept as base64 in the rule so that secret scanners do not fire.
    # Decoded here because a base64 literal in the code would be worse: a divergence would
    # go unnoticed.
    client_id=base64.b64decode("OWQxYzI1MGEtZTYxYi00NGQ5LTg4ZWQtNTk0NGQxOTYyZjVl").decode(),
    authorize_url="https://claude.ai/oauth/authorize",
    token_url="https://api.anthropic.com/v1/oauth/token",
    # `user:inference` is what grants direct inference with an OAuth token. The
    # `platform.claude.com` endpoint only issues console tokens; it has to be `claude.ai`.
    scopes=(
        "org:create_api_key",
        "user:profile",
        "user:inference",
        "user:sessions:claude_code",
        "user:mcp_servers",
        "user:file_upload",
    ),
    redirect_uri="http://localhost:54545/callback",
    pkce=True,
    # `code=true` is what makes the paste viable: instead of redirecting, the page shows a
    # copyable code (in the `code#state` format). Without it the user would depend on the
    # browser reaching the local port.
    authorize_params=(("code", "true"),),
    token_body="json",
    exchange_params=(("state", "{state}"),),
    refresh_headers=(
        ("anthropic-beta", "oauth-2025-04-20"),
        (
            "User-Agent",
            f"anthropic-sdk-typescript/{CLAUDE_CODE_SDK_VERSION} userOAuthProvider",
        ),
    ),
    expiry_skew_s=300.0,
)

# The client id and the secret are base64 in the OMP rule; the anchors point at a prefix of
# each, which is what fits on one line without breaking the substring check.
# omp: compat/rules/auth/google-antigravity.kdl :: access_type, prompt, scopes, callback
# omp: compat/rules/auth/google-antigravity.kdl :: MTA3MTAwNjA2MDU5MS10bWhzc2lu
# omp: compat/rules/auth/google-antigravity.kdl :: R09DU1BYLUs1OEZXUjQ4NkxkTEoxbUxCOHNY
# omp: providers/google-auth.ts :: OAUTH_TOKEN_URL
# omp= providers/google-auth.ts :: OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
_ANTIGRAVITY: Final = _Provider(
    client_id=base64.b64decode(
        "MTA3MTAwNjA2MDU5MS10bWhzc2luMmgyMWxjcmUyMzV2dG9sb2poNGc0MDNlcC5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbQ=="
    ).decode(),
    client_secret=base64.b64decode("R09DU1BYLUs1OEZXUjQ4NkxkTEoxbUxCOHNYQzR6NnFEQWY=").decode(),
    authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
    token_url="https://oauth2.googleapis.com/token",
    scopes=(
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
        "https://www.googleapis.com/auth/cclog",
        "https://www.googleapis.com/auth/experimentsandconfigs",
    ),
    redirect_uri="http://127.0.0.1:51121/oauth-callback",
    # The OMP rule does not enable PKCE here; the client secret plays that role.
    pkce=False,
    # Without these two Google returns only an access token: `access_type=offline` is what
    # asks for the refresh token, and `prompt=consent` is what makes it be issued again on a
    # reauthorization of an account that already consented — without it the second
    # connection is left with no renewal.
    authorize_params=(("access_type", "offline"), ("prompt", "consent")),
    token_body="form",
    expiry_skew_s=300.0,
)

_PROVIDERS: Final[dict[ProviderId, _Provider]] = {
    "openai-codex": _CODEX,
    "anthropic": _ANTHROPIC,
    "google-antigravity": _ANTIGRAVITY,
}


@dataclass(frozen=True, slots=True)
class AuthRequest:
    """What stays on this side while the user authenticates in the browser.

    The ``verifier`` has to survive until the paste: without it the code exchange fails,
    because the provider only confirms the challenge against the secret that never travelled.
    """

    url: str
    """Where to send the user."""

    state: str
    """Opaque, returned on the callback; it is what ties the paste to this request."""

    verifier: str
    """PKCE. Empty for providers that do not use it (Antigravity)."""


# omp: registry/oauth/pkce.ts :: generatePKCE
def _pkce() -> tuple[str, str]:
    """``(verifier, challenge)``. 96 random bytes in base64url, S256 challenge.

    Unpadded ``base64url`` is not cosmetic: the ``=`` would have to be escaped in the
    authorization URL and some servers compare the challenge byte for byte with what they
    received.
    """
    verifier = _b64url(secrets.token_bytes(96))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


# omp: registry/oauth/callback-server.ts :: generateState
def _state() -> str:
    """16 bytes in hexadecimal, like OMP's ``generateState``."""
    return secrets.token_bytes(16).hex()


# omp: registry/engine/oauth-code.ts :: generateAuthUrl
def begin(provider: ProviderId) -> AuthRequest:
    """Builds the authorization URL and the secret the paste will need.

    It opens neither browser nor server: the caller decides how to show the ``url``.
    """
    spec = _spec(provider)
    verifier, challenge = _pkce() if spec.pkce else ("", "")
    state = _state()

    params: dict[str, str] = {
        "client_id": spec.client_id,
        "response_type": "code",
        "redirect_uri": spec.redirect_uri,
        "scope": " ".join(spec.scopes),
        "state": state,
    }
    if challenge:
        params["code_challenge"] = challenge
        params["code_challenge_method"] = "S256"
    params.update(spec.authorize_params)

    url = f"{spec.authorize_url}?{urllib.parse.urlencode(params)}"
    return AuthRequest(url=url, state=state, verifier=verifier)


def _spec(provider: ProviderId) -> _Provider:
    spec = _PROVIDERS.get(provider)
    if spec is None:  # pragma: no cover - `ProviderId` is closed; guard for loose calls
        raise OAuthError(provider, f"unknown provider: {provider!r}")
    return spec


def callback_origin(provider: ProviderId) -> str:
    """The origin (`scheme://host:port`) of the redirect registered for this provider.

    It exists so the page does not have to repeat the table in JavaScript. Two lists of the
    same value diverge silently, and the symptom would be the page probing `localhost` while
    the Antigravity server sits on `127.0.0.1` — reporting no interceptor when there is one.

    It is not cosmetic: Antigravity registers `127.0.0.1` and the other two `localhost`, and
    the callback server binds **a single family** for a literal, like the OMP source
    (`callback-server.ts :: #createServer`).
    """
    parsed = urllib.parse.urlparse(_spec(provider).redirect_uri)
    return f"{parsed.scheme}://{parsed.netloc}"


# omp: registry/oauth/callback-server.ts :: parseNativeCallback
# omp: registry/engine/oauth-code.ts :: exchangeToken
def _parse_paste(provider: ProviderId, request: AuthRequest, pasted: str) -> str:
    """Extracts the authorization code from what the user pasted.

    Three forms, all accepted because all three show up in production:

    * the whole callback URL — ``https://…/callback?code=abc&state=xyz``. It is what the
      browser shows in the address bar when the redirect fails to reach the local port;
    * ``code#state`` — what Anthropic shows on the page when the flow is requested with
      ``code=true``. The fragment is the ``state``, not part of the code;
    * the bare code, for whoever copied only that piece.

    The ``state`` is checked whenever it comes in the paste. When it does not (bare code)
    there is nothing to compare against — and refusing there would only push the user to
    paste something else. The request's ``state`` goes in the exchange anyway, so the server
    remains the authority.

    An ``error`` on the callback wins over a missing code: the provider's
    ``error_description`` says what went wrong, and turning it into "code missing" would
    erase it.
    """
    text = pasted.strip()
    if not text:
        raise OAuthError(provider, "nothing pasted: expected the callback URL or the code")

    if text.startswith(("http://", "https://")):
        return _parse_callback_url(provider, request, text)

    # `code#state`: the fragment is the authority on the state, as in OMP.
    code, separator, fragment = text.partition("#")
    if separator and fragment:
        _check_state(provider, request, fragment)
    if not code:
        raise OAuthError(provider, "the pasted text does not contain an authorization code")
    return code


def _parse_callback_url(provider: ProviderId, request: AuthRequest, text: str) -> str:
    try:
        parsed = urllib.parse.urlparse(text)
    except ValueError as exc:
        raise OAuthError(provider, f"the pasted URL is not valid: {exc}") from exc
    query = urllib.parse.parse_qs(parsed.query)

    if state := _first(query, "state"):
        _check_state(provider, request, state)

    if error := _first(query, "error"):
        description = _first(query, "error_description") or error
        raise OAuthError(provider, f"authorization denied: {description}")

    # `authCode` is the variant some native clients return instead of `code`.
    code = _first(query, "code") or _first(query, "authCode")
    if not code:
        raise OAuthError(
            provider, "the pasted URL carries no `code`; did you paste the right page?"
        )
    # Anthropic does return `code=abc#state` inside the query itself at times.
    return code.partition("#")[0]


def _first(query: Mapping[str, list[str]], key: str) -> str:
    values = query.get(key)
    return values[0] if values else ""


def _check_state(provider: ProviderId, request: AuthRequest, received: str) -> None:
    """A ``state`` different from the one issued is the CSRF signal; never ignored."""
    if received != request.state:
        raise OAuthError(
            provider,
            f"`state` does not match (expected {request.state!r}, pasted {received!r}); "
            f"the callback belongs to another authentication request — start the flow again",
        )


# omp: registry/engine/common.ts :: postTokenRequest
async def _post_token(
    provider: ProviderId,
    spec: _Provider,
    params: dict[str, str],
    *,
    client: httpx.AsyncClient,
    headers: Mapping[str, str] = {},
    what: str,
) -> dict[str, Any]:
    """POSTs to the token endpoint; returns the JSON or raises with the real body.

    The response body surfaces intact even when the status is 200 but the JSON is useless:
    some providers wrap the error in a success envelope, and the only explanation is inside
    it.
    """
    # `json` and `form` are not interchangeable: Anthropic refuses the urlencoded body and
    # Google refuses the JSON. Each provider's rule says which one it is.
    if spec.token_body == "json":
        response = await client.post(spec.token_url, headers=dict(headers), json=params)
    else:
        response = await client.post(spec.token_url, headers=dict(headers), data=params)

    body = response.text
    payload: Any = None
    if body:
        try:
            payload = response.json()
        except ValueError:
            payload = None

    if response.status_code >= 400:
        raise OAuthError(
            provider,
            f"{what} failed: HTTP {response.status_code}: {_describe(payload, body)}",
            status=response.status_code,
            body=body,
        )
    if not isinstance(payload, dict):
        raise OAuthError(
            provider,
            f"{what}: token response is not a JSON object: {body[:500]}",
            status=response.status_code,
            body=body,
        )
    return payload


def _describe(payload: Any, body: str) -> str:
    """The upstream message, preferring ``error_description`` over the raw body.

    It never substitutes: when there is no recognizable field it returns the body,
    truncated. What does not happen is fabricating a generic explanation on top of a real
    one.
    """
    if isinstance(payload, dict):
        error = payload.get("error")
        code = error.get("status") if isinstance(error, dict) else error
        description = (
            payload.get("error_description")
            or (error.get("message") if isinstance(error, dict) else None)
            or payload.get("message")
        )
        if description and code:
            return f"{code}: {description}"
        if description or code:
            return str(description or code)
    return body[:500]


# omp: registry/engine/oauth-code.ts :: exchangeToken
async def complete(
    provider: ProviderId,
    request: AuthRequest,
    pasted: str,
    *,
    client: httpx.AsyncClient,
) -> Credential:
    """Exchanges what the user pasted for a usable credential.

    ``pasted`` accepts the whole callback URL, ``code#state`` or the bare code — extracting
    and validating is this function's job, not the user's.

    On Antigravity the credential is only complete after discovering the project: the
    ``project_id`` comes neither from the token nor from an environment variable, it comes
    from ``loadCodeAssist``. Without it no inference request goes through, so discovery is
    part of connecting and not of a later step.
    """
    spec = _spec(provider)
    code = _parse_paste(provider, request, pasted)

    params: dict[str, str] = {
        "grant_type": "authorization_code",
        "client_id": spec.client_id,
        "code": code,
        "redirect_uri": spec.redirect_uri,
    }
    if spec.client_secret:
        params["client_secret"] = spec.client_secret
    if spec.pkce:
        params["code_verifier"] = request.verifier
    for key, value in spec.exchange_params:
        params[key] = value.replace("{state}", request.state)

    payload = await _post_token(
        provider, spec, params, client=client, what="the authorization code exchange"
    )
    credential = _credential(provider, spec, payload, previous=None)

    if provider == "google-antigravity":
        # The OMP rule refuses a token with no refresh here: without one the subscription
        # dies at the first expiry and the user would have to redo everything without
        # knowing why.
        if not credential.refresh_token:
            raise OAuthError(
                provider,
                "Google did not return a refresh token; repeat the authorization "
                "(it is what `access_type=offline` and `prompt=consent` ask for)",
            )
        project_id = await _discover_project(credential.access_token, client=client)
        credential = replace(credential, project_id=project_id)

    return credential


# omp: registry/engine/common.ts :: mapCredentials
def _credential(
    provider: ProviderId,
    spec: _Provider,
    payload: Mapping[str, Any],
    *,
    previous: Credential | None,
) -> Credential:
    """Projects the token response onto a ``Credential``.

    Rotation is the part that matters: if the provider returns a ``refresh_token``, that is
    the one that stays. Only in its absence — providers that do not rotate — is the previous
    one preserved. Keeping the old one over a new one is the direct route to
    ``invalid_grant`` on the next renewal.
    """
    access = payload.get("access_token")
    if not isinstance(access, str) or not access:
        excerpt = str(dict(payload))[:500]
        raise OAuthError(provider, f"token response without an access token: {excerpt}")

    rotated = payload.get("refresh_token")
    refresh_token = rotated if isinstance(rotated, str) and rotated else ""
    if not refresh_token and previous is not None:
        refresh_token = previous.refresh_token

    expires_in = payload.get("expires_in")
    expires_at = 0.0
    if isinstance(expires_in, int | float) and not isinstance(expires_in, bool):
        expires_at = time.time() + float(expires_in) - spec.expiry_skew_s

    return Credential(
        provider=provider,
        access_token=access,
        refresh_token=refresh_token,
        expires_at=expires_at,
        project_id=previous.project_id if previous is not None else "",
    )


# omp: registry/engine/refresh.ts :: createRequestRefresh
async def refresh(
    credential: Credential,
    *,
    client: httpx.AsyncClient,
    store: CredentialStore | None = None,
) -> Credential:
    """Renews the access token, returning a new ``Credential``.

    **Single owner.** If ``store`` is given and ``store.owns_refresh`` is false, it raises
    ``NotRefreshOwnerError`` without touching the network. This is not zeal: Anthropic's and
    OpenAI's refresh tokens are single-use, and two refreshers over the same credential
    invalidate each other's copy, producing ``invalid_grant`` in a loop until somebody logs
    in by hand again. A store that is not the owner reads and never exchanges — the one that
    renews is the process owning the source of truth. Without ``store``, the caller takes
    that responsibility.

    Antigravity's ``project_id`` survives the renewal: it is fixed at login and the token
    endpoint does not return it. Losing it here would break every subsequent request.
    """
    provider = credential.provider
    spec = _spec(provider)

    if store is not None and not store.owns_refresh:
        raise NotRefreshOwnerError(
            provider,
            f"{type(store).__name__} is not the refresh owner ({provider}); "
            f"refreshing from here would invalidate the owner's token. "
            f"Refresh in the process that owns it.",
        )
    if not credential.refresh_token:
        raise OAuthError(
            provider, f"{provider} has no refresh token; connect the subscription again"
        )
    if provider == "google-antigravity" and not credential.project_id:
        # The OMP rule's `require "projectId"`: a credential with no project renews but is
        # useless, and the error would show up much later, in an inference 400.
        raise OAuthError(provider, f"{provider} has no project_id; connect the subscription again")

    params: dict[str, str] = {
        "grant_type": "refresh_token",
        "client_id": spec.client_id,
        "refresh_token": credential.refresh_token,
    }
    if spec.client_secret:
        params["client_secret"] = spec.client_secret

    payload = await _post_token(
        provider,
        spec,
        params,
        client=client,
        headers=dict(spec.refresh_headers),
        what="the token refresh",
    )
    return _credential(provider, spec, payload, previous=credential)


# omp: registry/oauth/google-antigravity.ts :: discoverProject
async def _discover_project(access_token: str, *, client: httpx.AsyncClient) -> str:
    """Discovers the account's ``cloudaicompanionProject``.

    Three steps, in the same order as OMP: ask for the state, provision the free tier if the
    account does not have one yet, and ask again. The second ``loadCodeAssist`` is not
    redundant — it is the one that returns the project ``onboardUser`` just created.
    """
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": ANTIGRAVITY_USER_AGENT,
    }

    initial = await _load_code_assist(headers, client=client)
    _assert_free_tier_eligible(initial)
    if initial.get("currentTier") is None:
        await _onboard_user(headers, client=client)

    refreshed = await _load_code_assist(headers, client=client)
    if project_id := _project_of(refreshed):
        return project_id
    raise OAuthError(
        "google-antigravity", "`loadCodeAssist` did not return a `cloudaicompanionProject`"
    )


# omp: registry/oauth/google-antigravity.ts :: loadCodeAssist
async def _load_code_assist(
    headers: Mapping[str, str], *, client: httpx.AsyncClient
) -> dict[str, Any]:
    """The account's state in Cloud Code Assist.

    The second call with an explicit ``cloudaicompanionProject`` is what makes the backend
    reveal the ``paidTier``: without the project in the body it answers with the current tier
    only.
    """
    payload = await _cloud_code(
        _LOAD_CODE_ASSIST_URL,
        "POST",
        headers,
        client=client,
        body={"metadata": _ANTIGRAVITY_METADATA},
    )
    project_id = _project_of(payload)
    if payload.get("paidTier") is None and project_id:
        payload = await _cloud_code(
            _LOAD_CODE_ASSIST_URL,
            "POST",
            headers,
            client=client,
            body={
                "cloudaicompanionProject": project_id,
                "metadata": _ANTIGRAVITY_METADATA,
            },
        )
    return payload


# omp: registry/oauth/google-antigravity.ts :: extractProjectId
def _project_of(payload: Mapping[str, Any]) -> str:
    project_id = payload.get("cloudaicompanionProject")
    return project_id if isinstance(project_id, str) and project_id else ""


# omp: registry/oauth/google-antigravity.ts :: assertFreeTierEligible
def _assert_free_tier_eligible(payload: Mapping[str, Any]) -> None:
    """Refuses early an account the backend marks as ineligible.

    Silence is not ineligibility: it only raises when there is an ``ineligibleTiers`` entry
    with a message for the free tier. The ``validationUrl``, when present, is the user's next
    step and goes in the message.
    """
    allowed = payload.get("allowedTiers")
    if isinstance(allowed, list) and any(
        isinstance(tier, dict) and tier.get("id") == _FREE_TIER_ID for tier in allowed
    ):
        return

    ineligible = payload.get("ineligibleTiers")
    if not isinstance(ineligible, list):
        return
    for tier in ineligible:
        if not isinstance(tier, dict) or tier.get("tierId") != _FREE_TIER_ID:
            continue
        reason = tier.get("reasonMessage")
        if not reason:
            continue
        url = tier.get("validationUrl")
        raise OAuthError("google-antigravity", f"{reason}\n{url}" if url else str(reason))


async def _sleep(seconds: float) -> None:
    """Deliberate indirection: it is the only real wait point, and tests replace it."""
    await asyncio.sleep(seconds)


# omp: registry/oauth/google-antigravity.ts :: onboardUser
async def _onboard_user(headers: Mapping[str, str], *, client: httpx.AsyncClient) -> None:
    """Provisions the free tier, following the long-running operation to the end.

    The deadline is absolute, not per attempt: an operation that returns ``done: false``
    forever has to fail instead of pinning the connection.
    """
    deadline = time.monotonic() + _ONBOARD_TIMEOUT_S
    operation = await _cloud_code(
        _ONBOARD_USER_URL,
        "POST",
        headers,
        client=client,
        body={"tierId": _FREE_TIER_ID, "metadata": _ANTIGRAVITY_METADATA},
    )

    while True:
        if operation.get("done") is True:
            if error := operation.get("error"):
                raise OAuthError(
                    "google-antigravity", f"`onboardUser` failed: {_describe_operation(error)}"
                )
            if operation.get("response") is None:
                raise OAuthError("google-antigravity", "`onboardUser` finished without a response")
            return

        if time.monotonic() >= deadline:
            raise OAuthError(
                "google-antigravity",
                f"`onboardUser` did not finish within {_ONBOARD_TIMEOUT_S:.0f}s",
            )
        name = operation.get("name")
        if not isinstance(name, str) or not name:
            raise OAuthError(
                "google-antigravity", "`onboardUser` returned an operation without a name"
            )

        await _sleep(_ONBOARD_POLL_INTERVAL_S)
        operation = await _cloud_code(
            f"{_OPERATIONS_URL}/{name}", "GET", headers, client=client, body=None
        )


# omp: registry/oauth/google-antigravity.ts :: describeOperationError
def _describe_operation(error: Any) -> str:
    if isinstance(error, dict):
        message = error.get("message")
        code = error.get("code")
        if message:
            return f"{code}: {message}" if isinstance(code, int) else str(message)
    return str(error)


# omp: registry/oauth/google-antigravity.ts :: requestCloudCodeAssist
async def _cloud_code(
    url: str,
    method: str,
    headers: Mapping[str, str],
    *,
    client: httpx.AsyncClient,
    body: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """A control plane request. Any status other than 200 is an error, with a body."""
    response = await client.request(method, url, headers=dict(headers), json=body)
    if response.status_code != 200:
        payload: Any = None
        try:
            payload = response.json()
        except ValueError:
            payload = None
        raise OAuthError(
            "google-antigravity",
            f"{url} returned HTTP {response.status_code}: {_describe(payload, response.text)}",
            status=response.status_code,
            body=response.text,
        )
    parsed = response.json()
    if not isinstance(parsed, dict):
        raise OAuthError("google-antigravity", f"{url} did not return a JSON object")
    return parsed
