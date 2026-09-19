"""OAuth flow with paste and refresh.

What these tests defend is not that a POST returns 200. It is the set of things that, when
they fail, fail silently and only surface weeks later:

* the paste accepts the three forms the user can produce — and refuses a ``state`` that is
  not the request's, which is the only CSRF signal we have;
* rotation **replaces** the refresh token. Storing the old one over a new one gives
  ``invalid_grant`` on the next refresh, and the symptom is a manual re-login with no
  explanation;
* the Antigravity ``project_id`` comes out of ``loadCodeAssist`` and survives the refresh.
  Without it every inference request returns 400;
* the upstream's ``error_description`` reaches the user intact. It is what tells them what
  to do; replacing it with a message of ours erases it.

All of it with ``httpx.MockTransport``: no network, no real clock.

Some upstream payloads in the fixtures are deliberately in Portuguese: those tests assert
that the provider's own text propagates unrewritten, and an English fixture would not tell
a pass-through apart from a message of ours.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
import urllib.parse
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from litellm_mysubs.credentials import oauth
from litellm_mysubs.credentials.oauth import (
    AuthRequest,
    NotRefreshOwnerError,
    OAuthError,
    begin,
    complete,
    refresh,
)
from litellm_mysubs.credentials.store import Credential, CredentialStore, ProviderId

Handler = Callable[[httpx.Request], httpx.Response]


def client_of(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def request_for(provider: ProviderId = "anthropic") -> AuthRequest:
    return AuthRequest(url="https://example/authorize", state="ST4TE", verifier="v" * 43)


def query_of(url: str) -> dict[str, str]:
    return {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlparse(url).query).items()}


def form_of(request: httpx.Request) -> dict[str, str]:
    return {k: v[0] for k, v in urllib.parse.parse_qs(request.content.decode()).items()}


def json_of(request: httpx.Request) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(request.content)
    return result


def token_response(**extra: Any) -> httpx.Response:
    return httpx.Response(200, json={"access_token": "at-new", "expires_in": 3600, **extra})


class FakeStore(CredentialStore):
    """Minimal store whose only interesting property is ownership of the refresh."""

    def __init__(self, *, owns_refresh: bool) -> None:
        self.owns_refresh = owns_refresh

    def get(self, provider: ProviderId) -> Credential | None:
        return None

    def set(self, provider: ProviderId, credential: Credential) -> None:
        raise AssertionError("should not write")

    def delete(self, provider: ProviderId) -> None:
        raise AssertionError("should not delete")

    def reload(self) -> bool:
        return False


class TestPKCE:
    def test_challenge_is_s256_of_verifier(self) -> None:
        """If the challenge is not the SHA-256 of the verifier, the code exchange gives
        `invalid_grant` at the server — and the error only shows up after the user has
        already logged in."""
        request = begin("openai-codex")
        expected = (
            base64.urlsafe_b64encode(hashlib.sha256(request.verifier.encode()).digest())
            .decode()
            .rstrip("=")
        )
        assert query_of(request.url)["code_challenge"] == expected
        assert query_of(request.url)["code_challenge_method"] == "S256"

    def test_verifier_is_fresh_per_request(self) -> None:
        first, second = begin("anthropic"), begin("anthropic")
        assert first.verifier != second.verifier
        assert first.state != second.state

    def test_challenge_is_url_safe_and_unpadded(self) -> None:
        """Padding and `+`/`/` would have to be escaped; some servers compare byte for byte
        with what they received and reject the difference."""
        challenge = query_of(begin("anthropic").url)["code_challenge"]
        assert "=" not in challenge
        assert "+" not in challenge and "/" not in challenge

    def test_antigravity_has_no_pkce(self) -> None:
        """The OMP rule does not enable PKCE on Google; the client secret plays that role.
        Sending a challenge the server does not expect does not help and may be refused."""
        request = begin("google-antigravity")
        assert request.verifier == ""
        assert "code_challenge" not in query_of(request.url)


class TestAuthorizeUrl:
    def test_anthropic_asks_for_a_pasteable_code(self) -> None:
        """`code=true` is what makes the page show a copyable code instead of redirecting.
        Without it the paste stops being possible inside a container."""
        assert query_of(begin("anthropic").url)["code"] == "true"

    def test_google_asks_for_a_refresh_token(self) -> None:
        """Without `access_type=offline` Google issues no refresh token; without
        `prompt=consent` it does not reissue one to whoever already consented. Missing either
        produces a subscription that dies at the first expiry."""
        params = query_of(begin("google-antigravity").url)
        assert params["access_type"] == "offline"
        assert params["prompt"] == "consent"

    def test_codex_redirect_uri_is_the_registered_one(self) -> None:
        """OpenAI only authorises this exact URI; deriving it from a free port would be
        rejected at the server after the user had already authenticated."""
        params = query_of(begin("openai-codex").url)
        assert params["redirect_uri"] == "http://localhost:1455/auth/callback"
        assert params["client_id"] == "app_EMoamEEZ73f0CkXaXp7hrann"

    def test_state_travels_in_the_url(self) -> None:
        request = begin("anthropic")
        assert query_of(request.url)["state"] == request.state

    def test_anthropic_requests_inference_scope(self) -> None:
        """`user:inference` is what grants direct inference with a subscription token;
        without it the token is only good for account management."""
        assert "user:inference" in query_of(begin("anthropic").url)["scope"].split(" ")


class TestPaste:
    """The three forms the user can produce, and what has to be refused."""

    async def exchange(self, pasted: str, provider: ProviderId = "anthropic") -> httpx.Request:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return token_response(refresh_token="rt-1")

        async with client_of(handler) as client:
            await complete(provider, request_for(provider), pasted, client=client)
        return seen[0]

    async def test_full_callback_url(self) -> None:
        """What the browser shows in the bar when the redirect does not reach the local
        port."""
        sent = await self.exchange("https://claude.ai/callback?code=abc123&state=ST4TE")
        assert json_of(sent)["code"] == "abc123"

    async def test_bare_code(self) -> None:
        """Whoever copies only the code fragment must not be forced to rebuild a URL."""
        sent = await self.exchange("abc123")
        assert json_of(sent)["code"] == "abc123"

    async def test_code_hash_state(self) -> None:
        """The format Anthropic shows with `code=true`. The fragment is the `state`, not part
        of the code — sending it along makes the server refuse the exchange."""
        sent = await self.exchange("abc123#ST4TE")
        assert json_of(sent)["code"] == "abc123"

    async def test_surrounding_whitespace_is_tolerated(self) -> None:
        sent = await self.exchange("  abc123#ST4TE \n")
        assert json_of(sent)["code"] == "abc123"

    async def test_code_with_hash_inside_the_query(self) -> None:
        """Anthropic sometimes returns `code=abc#state` inside the query itself."""
        sent = await self.exchange("https://claude.ai/callback?code=abc123%23ST4TE&state=ST4TE")
        assert json_of(sent)["code"] == "abc123"

    async def test_alternative_auth_code_parameter(self) -> None:
        sent = await self.exchange("https://claude.ai/callback?authCode=abc123&state=ST4TE")
        assert json_of(sent)["code"] == "abc123"


class TestPasteRejection:
    """None of these ever touches the network: the handler fails if it is called."""

    def client(self) -> httpx.AsyncClient:
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError(f"there should be no request: {request.url}")

        return client_of(handler)

    async def test_state_mismatch_in_url_raises(self) -> None:
        async with self.client() as client:
            with pytest.raises(OAuthError) as excinfo:
                await complete(
                    "anthropic",
                    request_for(),
                    "https://claude.ai/callback?code=abc&state=OTHER",
                    client=client,
                )
        assert "state" in str(excinfo.value).lower()

    async def test_state_mismatch_in_fragment_raises(self) -> None:
        """The fragment is the authority on the `state`; one swapped there counts as much as
        in the query, otherwise the `code#state` format would be a CSRF hole."""
        async with self.client() as client:
            with pytest.raises(OAuthError):
                await complete("anthropic", request_for(), "abc#OTHER", client=client)

    async def test_error_in_callback_url_propagates_description(self) -> None:
        """The callback's `error_description` is what tells the user what to do."""
        async with self.client() as client:
            with pytest.raises(OAuthError) as excinfo:
                await complete(
                    "anthropic",
                    request_for(),
                    "https://claude.ai/callback?error=access_denied"
                    "&error_description=Utilizador+recusou&state=ST4TE",
                    client=client,
                )
        assert "Utilizador recusou" in str(excinfo.value)

    async def test_url_without_code_raises(self) -> None:
        async with self.client() as client:
            with pytest.raises(OAuthError):
                await complete(
                    "anthropic",
                    request_for(),
                    "https://claude.ai/callback?state=ST4TE",
                    client=client,
                )

    async def test_empty_paste_raises(self) -> None:
        async with self.client() as client:
            with pytest.raises(OAuthError):
                await complete("anthropic", request_for(), "   ", client=client)


class TestExchange:
    async def test_verifier_and_state_reach_the_token_endpoint(self) -> None:
        """The verifier is the secret that never travelled; without it the exchange fails. The
        `state` goes in the body because Anthropic's rule demands it there, not only in the
        URL."""
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return token_response(refresh_token="rt-1")

        request = request_for()
        async with client_of(handler) as client:
            await complete("anthropic", request, "abc#ST4TE", client=client)

        body = json_of(seen[0])
        assert body["code_verifier"] == request.verifier
        assert body["state"] == request.state
        assert body["grant_type"] == "authorization_code"

    async def test_anthropic_uses_json_and_codex_uses_form(self) -> None:
        """They are not interchangeable: Anthropic refuses urlencoded and Google refuses JSON.
        Swapping them gives an opaque 400 from the server."""
        content_types: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            content_types[str(request.url)] = request.headers["content-type"]
            return token_response(refresh_token="rt-1")

        async with client_of(handler) as client:
            await complete("anthropic", request_for(), "abc", client=client)
            await complete("openai-codex", request_for(), "abc", client=client)

        assert content_types["https://api.anthropic.com/v1/oauth/token"].startswith(
            "application/json"
        )
        assert content_types["https://auth.openai.com/oauth/token"].startswith(
            "application/x-www-form-urlencoded"
        )

    async def test_expiry_skew_lands_before_the_real_deadline(self) -> None:
        """Anthropic's margin is 300s. Refreshing exactly at the expiry is refreshing late:
        an in-flight request catches the token already dead."""
        async with client_of(lambda _: token_response(refresh_token="rt-1")) as client:
            credential = await complete("anthropic", request_for(), "abc", client=client)
        lead = credential.expires_at - time.time()
        assert 3200 < lead < 3310

    async def test_missing_access_token_raises_with_body(self) -> None:
        """Some providers wrap the error in a success envelope; the body is the only
        explanation and has to bubble up."""
        response = httpx.Response(200, json={"code": 4001, "msg": "conta suspensa"})
        async with client_of(lambda _: response) as client:
            with pytest.raises(OAuthError) as excinfo:
                await complete("anthropic", request_for(), "abc", client=client)
        assert "conta suspensa" in str(excinfo.value)

    async def test_provider_error_propagates_description(self) -> None:
        """Principle 3: the real `error_description`, never a generic message on top."""
        response = httpx.Response(
            400,
            json={
                "error": "invalid_grant",
                "error_description": "O código de autorização expirou; repete o login",
            },
        )
        async with client_of(lambda _: response) as client:
            with pytest.raises(OAuthError) as excinfo:
                await complete("openai-codex", request_for(), "abc", client=client)

        assert "O código de autorização expirou; repete o login" in str(excinfo.value)
        assert excinfo.value.status == 400
        assert "invalid_grant" in excinfo.value.body


class TestRefreshRotation:
    async def test_rotated_token_replaces_the_old_one(self) -> None:
        """The case that produces a loop of `invalid_grant` when it goes wrong: the provider
        rotated the token, the old one no longer works, and storing it kills the next
        refresh."""
        old = Credential("anthropic", "at-old", refresh_token="rt-old", expires_at=1.0)
        async with client_of(lambda _: token_response(refresh_token="rt-new")) as client:
            new = await refresh(old, client=client)

        assert new.refresh_token == "rt-new"
        assert new.access_token == "at-new"
        assert old.refresh_token == "rt-old", "the old credential is immutable"

    async def test_unrotated_token_is_preserved(self) -> None:
        """Not everyone rotates. A provider that omits `refresh_token` is not revoking ours —
        deleting it would force a re-login for no reason."""
        old = Credential("openai-codex", "at-old", refresh_token="rt-old")
        async with client_of(lambda _: token_response()) as client:
            new = await refresh(old, client=client)
        assert new.refresh_token == "rt-old"

    async def test_sends_the_stored_refresh_token(self) -> None:
        seen: list[dict[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(form_of(request))
            return token_response(refresh_token="rt-new")

        async with client_of(handler) as client:
            await refresh(Credential("openai-codex", "at", refresh_token="rt-old"), client=client)

        assert seen[0]["grant_type"] == "refresh_token"
        assert seen[0]["refresh_token"] == "rt-old"

    async def test_anthropic_refresh_carries_the_beta_headers(self) -> None:
        """Claude Code sends them on the refresh and not on the initial exchange; without them
        the endpoint treats the request as coming from a different client."""
        seen: list[httpx.Headers] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.headers)
            return token_response(refresh_token="rt-new")

        async with client_of(handler) as client:
            await refresh(Credential("anthropic", "at", refresh_token="rt"), client=client)

        assert seen[0]["anthropic-beta"] == "oauth-2025-04-20"
        assert "userOAuthProvider" in seen[0]["user-agent"]

    async def test_refresh_error_propagates_description(self) -> None:
        response = httpx.Response(
            400,
            json={"error": "invalid_grant", "error_description": "Refresh token expired"},
        )
        async with client_of(lambda _: response) as client:
            with pytest.raises(OAuthError) as excinfo:
                await refresh(Credential("anthropic", "at", refresh_token="rt"), client=client)
        assert "Refresh token expired" in str(excinfo.value)

    async def test_without_refresh_token_raises_before_the_network(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("there should be no request")

        async with client_of(handler) as client:
            with pytest.raises(OAuthError):
                await refresh(Credential("anthropic", "at"), client=client)


class TestRefreshOwnership:
    """Single-owner rule: two refreshers on the same single-use token produce a loop of
    `invalid_grant` and force a manual re-login."""

    async def test_non_owner_store_refuses_without_touching_the_network(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("a store that is not the owner must not exchange the token")

        async with client_of(handler) as client:
            with pytest.raises(NotRefreshOwnerError):
                await refresh(
                    Credential("anthropic", "at", refresh_token="rt"),
                    client=client,
                    store=FakeStore(owns_refresh=False),
                )

    async def test_owner_store_refreshes(self) -> None:
        async with client_of(lambda _: token_response(refresh_token="rt-new")) as client:
            new = await refresh(
                Credential("anthropic", "at", refresh_token="rt"),
                client=client,
                store=FakeStore(owns_refresh=True),
            )
        assert new.refresh_token == "rt-new"

    async def test_without_store_the_caller_owns_it(self) -> None:
        """With no store there is nobody to check; the caller takes on the responsibility."""
        async with client_of(lambda _: token_response(refresh_token="rt-new")) as client:
            new = await refresh(Credential("anthropic", "at", refresh_token="rt"), client=client)
        assert new.access_token == "at-new"


TOKEN_URL = "https://oauth2.googleapis.com/token"
LOAD_URL = "https://daily-cloudcode-pa.googleapis.com/v1internal:loadCodeAssist"
ONBOARD_URL = "https://daily-cloudcode-pa.googleapis.com/v1internal:onboardUser"


class Antigravity:
    """Scripted Antigravity backend: token, then `loadCodeAssist`/`onboardUser`."""

    def __init__(self, *steps: dict[str, Any], token: dict[str, Any] | None = None) -> None:
        self.steps = list(steps)
        self.token = token or {
            "access_token": "at-new",
            "refresh_token": "rt-1",
            "expires_in": 3600,
        }
        self.urls: list[str] = []
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.urls.append(url)
        if url == TOKEN_URL:
            return httpx.Response(200, json=self.token)
        if request.content:
            self.bodies.append(json_of(request))
        if not self.steps:
            raise AssertionError(f"one request too many: {url}")
        return httpx.Response(200, json=self.steps.pop(0))


@pytest.fixture(autouse=True)
def virtual_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """No waiting, but time that passes.

    The ``onboardUser`` polling waits 1s per round and the deadline is absolute. Replacing
    the wait with a no-op without touching the clock would make the deadline unreachable:
    the give-up test would spin forever. Here every wait advances the monotonic clock by
    exactly what it would have waited — the deadline is measured, the suite does not wait.
    """
    elapsed = 0.0

    async def sleep(seconds: float) -> None:
        nonlocal elapsed
        elapsed += seconds

    class Clock:
        """A shim instead of `monkeypatch` on the `time` module: touching the global `time`
        affects pytest and any test running alongside."""

        @staticmethod
        def monotonic() -> float:
            return time.monotonic() + elapsed

        @staticmethod
        def time() -> float:
            return time.time()

    monkeypatch.setattr(oauth, "_sleep", sleep)
    monkeypatch.setattr(oauth, "time", Clock)


class TestProjectDiscovery:
    async def test_project_id_comes_from_load_code_assist(self) -> None:
        """It comes neither from the token nor from the environment: it is the
        `cloudaicompanionProject` that `loadCodeAssist` returns. Without it every inference
        request returns 400."""
        backend = Antigravity(
            {
                "cloudaicompanionProject": "project-1",
                "currentTier": {"id": "free-tier"},
                "allowedTiers": [{"id": "free-tier"}],
                "paidTier": {"id": "paid"},
            },
            {
                "cloudaicompanionProject": "project-1",
                "currentTier": {"id": "free-tier"},
                "allowedTiers": [{"id": "free-tier"}],
                "paidTier": {"id": "paid"},
            },
        )
        async with client_of(backend) as client:
            credential = await complete("google-antigravity", request_for(), "abc", client=client)

        assert credential.project_id == "project-1"
        assert ONBOARD_URL not in backend.urls, "the account already had a tier; no provisioning"

    async def test_account_without_tier_is_onboarded_first(self) -> None:
        """Without `currentTier` the account does not exist in Cloud Code Assist yet; skipping
        `onboardUser` would leave the connection with no project at all."""
        backend = Antigravity(
            {"allowedTiers": [{"id": "free-tier"}], "paidTier": {"id": "paid"}},
            {"done": True, "response": {"@type": "OnboardUserResponse"}},
            {
                "cloudaicompanionProject": "project-new",
                "currentTier": {"id": "free-tier"},
                "allowedTiers": [{"id": "free-tier"}],
                "paidTier": {"id": "paid"},
            },
        )
        async with client_of(backend) as client:
            credential = await complete("google-antigravity", request_for(), "abc", client=client)

        assert credential.project_id == "project-new"
        assert ONBOARD_URL in backend.urls

    async def test_onboard_operation_is_polled_until_done(self) -> None:
        backend = Antigravity(
            {"allowedTiers": [{"id": "free-tier"}], "paidTier": {"id": "paid"}},
            {"name": "operations/1", "done": False},
            {"name": "operations/1", "done": True, "response": {"@type": "OnboardUserResponse"}},
            {
                "cloudaicompanionProject": "project-new",
                "currentTier": {"id": "free-tier"},
                "allowedTiers": [{"id": "free-tier"}],
                "paidTier": {"id": "paid"},
            },
        )
        async with client_of(backend) as client:
            credential = await complete("google-antigravity", request_for(), "abc", client=client)

        assert credential.project_id == "project-new"
        assert "https://daily-cloudcode-pa.googleapis.com/v1internal/operations/1" in backend.urls

    async def test_failed_onboard_operation_propagates_the_reason(self) -> None:
        backend = Antigravity(
            {"allowedTiers": [{"id": "free-tier"}], "paidTier": {"id": "paid"}},
            {"done": True, "error": {"code": 7, "message": "quota de projectos esgotada"}},
        )
        async with client_of(backend) as client:
            with pytest.raises(OAuthError) as excinfo:
                await complete("google-antigravity", request_for(), "abc", client=client)
        assert "quota de projectos esgotada" in str(excinfo.value)

    async def test_ineligible_account_reports_the_validation_url(self) -> None:
        """The account needs an action in the browser; the URL is the next step and must not
        be swallowed by a message of ours."""
        backend = Antigravity(
            {
                "allowedTiers": [{"id": "paid"}],
                "ineligibleTiers": [
                    {
                        "tierId": "free-tier",
                        "reasonMessage": "Conta precisa de validação",
                        "validationUrl": "https://valida.example/conta",
                    }
                ],
                "paidTier": {"id": "paid"},
            }
        )
        async with client_of(backend) as client:
            with pytest.raises(OAuthError) as excinfo:
                await complete("google-antigravity", request_for(), "abc", client=client)

        assert "Conta precisa de validação" in str(excinfo.value)
        assert "https://valida.example/conta" in str(excinfo.value)

    async def test_missing_project_raises_instead_of_inventing_one(self) -> None:
        """Principle 3: never fabricate a plausible value. A made-up project would give 400 on
        every request, without saying why."""
        backend = Antigravity(
            {
                "currentTier": {"id": "free-tier"},
                "allowedTiers": [{"id": "free-tier"}],
                "paidTier": {"id": "paid"},
            },
            {
                "currentTier": {"id": "free-tier"},
                "allowedTiers": [{"id": "free-tier"}],
                "paidTier": {"id": "paid"},
            },
        )
        async with client_of(backend) as client:
            with pytest.raises(OAuthError) as excinfo:
                await complete("google-antigravity", request_for(), "abc", client=client)
        assert "cloudaicompanionProject" in str(excinfo.value)

    async def test_token_without_refresh_is_rejected(self) -> None:
        """A connection with no refresh token dies at the first expiry, and the user would not
        know why. Refusing here is the only moment when it can still be retried."""
        backend = Antigravity(token={"access_token": "at", "expires_in": 3600})
        async with client_of(backend) as client:
            with pytest.raises(OAuthError) as excinfo:
                await complete("google-antigravity", request_for(), "abc", client=client)
        assert "refresh" in str(excinfo.value).lower()
        assert LOAD_URL not in backend.urls, "no project discovery for a dead connection"

    async def test_control_plane_error_propagates_the_upstream_body(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if str(request.url) == TOKEN_URL:
                return httpx.Response(
                    200,
                    json={"access_token": "at", "refresh_token": "rt", "expires_in": 3600},
                )
            return httpx.Response(
                403,
                json={"error": {"status": "PERMISSION_DENIED", "message": "API não activada"}},
            )

        async with client_of(handler) as client:
            with pytest.raises(OAuthError) as excinfo:
                await complete("google-antigravity", request_for(), "abc", client=client)

        assert "API não activada" in str(excinfo.value)
        assert excinfo.value.status == 403


class TestAntigravityRefresh:
    async def test_project_id_survives_the_refresh(self) -> None:
        """The token endpoint does not return the project; losing it here would break every
        subsequent request with a 400 that does not point back at the refresh."""
        old = Credential(
            "google-antigravity", "at-old", refresh_token="rt-old", project_id="project-1"
        )
        async with client_of(lambda _: token_response(refresh_token="rt-new")) as client:
            new = await refresh(old, client=client)

        assert new.project_id == "project-1"
        assert new.refresh_token == "rt-new"

    async def test_credential_without_project_refuses_to_refresh(self) -> None:
        """`require "projectId"` from the OMP rule: refreshing would produce a valid, useless
        token, and the error would only surface much later."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("there should be no request")

        async with client_of(handler) as client:
            with pytest.raises(OAuthError) as excinfo:
                await refresh(
                    Credential("google-antigravity", "at", refresh_token="rt"), client=client
                )
        assert "project_id" in str(excinfo.value)

    async def test_refresh_sends_the_client_secret(self) -> None:
        """Google does not accept the grant without a client secret; without it it returns
        `invalid_client` and the subscription gets stuck."""
        seen: list[dict[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(form_of(request))
            return token_response()

        async with client_of(handler) as client:
            await refresh(
                Credential("google-antigravity", "at", refresh_token="rt", project_id="project-1"),
                client=client,
            )
        assert seen[0]["client_secret"].startswith("GOCSPX-")


class TestSurvivorsClosed:
    """Cases that a first round of mutation showed to be undefended."""

    async def test_paid_tier_is_probed_with_the_project_in_the_body(self) -> None:
        """Without the `cloudaicompanionProject` in the body the backend answers with the
        current tier only, and a paid account passes for a free one. The second call is not
        redundant."""
        account = {
            "cloudaicompanionProject": "project-1",
            "currentTier": {"id": "free-tier"},
            "allowedTiers": [{"id": "free-tier"}],
        }
        complete_account = {**account, "paidTier": {"id": "paid"}}
        backend = Antigravity(account, complete_account, account, complete_account)
        async with client_of(backend) as client:
            await complete("google-antigravity", request_for(), "abc", client=client)

        with_project = [b for b in backend.bodies if "cloudaicompanionProject" in b]
        assert with_project, "the second call has to resend the discovered project"
        assert with_project[0]["cloudaicompanionProject"] == "project-1"

    async def test_onboard_that_never_finishes_gives_up(self) -> None:
        """`done: false` forever has to fail; hanging the connection would be worse than an
        error, because nothing would tell the user it had stopped."""
        pending = {"name": "operations/1", "done": False}
        backend = Antigravity(
            {"allowedTiers": [{"id": "free-tier"}], "paidTier": {"id": "paid"}},
            *[dict(pending) for _ in range(200)],
        )
        async with client_of(backend) as client:
            with pytest.raises(OAuthError) as excinfo:
                await complete("google-antigravity", request_for(), "abc", client=client)
        assert "onboardUser" in str(excinfo.value)

    async def test_onboard_done_without_response_is_a_failure(self) -> None:
        """`done: true` with no `response` is what the backend returns when the provisioning
        did not go through; treating it as success would leave the account with no project."""
        backend = Antigravity(
            {"allowedTiers": [{"id": "free-tier"}], "paidTier": {"id": "paid"}},
            {"done": True},
        )
        async with client_of(backend) as client:
            with pytest.raises(OAuthError) as excinfo:
                await complete("google-antigravity", request_for(), "abc", client=client)
        assert "without a response" in str(excinfo.value)

    async def test_google_error_object_is_not_flattened_to_raw_json(self) -> None:
        """Google wraps the explanation in `error.message`. Returning the raw JSON is
        technically honest and practically useless: the user does not find it."""
        response = httpx.Response(
            400,
            json={"error": {"status": "INVALID_ARGUMENT", "message": "redirect_uri inválido"}},
        )
        async with client_of(lambda _: response) as client:
            with pytest.raises(OAuthError) as excinfo:
                await complete("google-antigravity", request_for(), "abc", client=client)
        assert "INVALID_ARGUMENT: redirect_uri inválido" in str(excinfo.value)

    async def test_eligible_account_ignores_a_stale_ineligibility(self) -> None:
        """`allowedTiers` containing free-tier wins: a leftover entry in `ineligibleTiers` must
        not block an account the backend already authorises."""
        account = {
            "cloudaicompanionProject": "project-1",
            "currentTier": {"id": "free-tier"},
            "allowedTiers": [{"id": "free-tier"}],
            "paidTier": {"id": "paid"},
            "ineligibleTiers": [{"tierId": "free-tier", "reasonMessage": "leftover"}],
        }
        backend = Antigravity(account, account)
        async with client_of(backend) as client:
            credential = await complete("google-antigravity", request_for(), "abc", client=client)
        assert credential.project_id == "project-1"

    async def test_paste_of_only_whitespace_never_reaches_the_network(self) -> None:
        """An empty paste used as the code would produce an opaque 400 from the provider
        instead of telling the user they pasted nothing."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("there should be no request")

        async with client_of(handler) as client:
            with pytest.raises(OAuthError) as excinfo:
                await complete("anthropic", request_for(), "\n\t ", client=client)
        assert "nothing pasted" in str(excinfo.value)

    async def test_fragment_without_a_code_is_rejected(self) -> None:
        """`#ST4TE` on its own passes the `state` check but carries no code. Letting it
        through would send `code=""` to the provider."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("there should be no request")

        async with client_of(handler) as client:
            with pytest.raises(OAuthError) as excinfo:
                await complete("anthropic", request_for(), "#ST4TE", client=client)
        assert "authorization code" in str(excinfo.value)
