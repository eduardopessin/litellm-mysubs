"""Renewal ported from omp 18.4.4's `OAuthRefresher`: one exchange per credential, ever.

The providers' refresh tokens are single-use and rotating. Everything here defends one
property from different directions: a refresh token is presented to the token endpoint at
most once, and whatever the endpoint answers ends up in the store — unless someone else
(a peer worker, a login) wrote a newer credential meanwhile, in which case theirs wins.

The request-path tests go through the real proxy app, the real `litellm.Router` and the real
`Transport`; only the hosts behind them are fake. The token endpoint below behaves like the
real ones: each refresh token works once, and presenting it again is `invalid_grant`.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest

from litellm_mysubs.credentials import oauth
from litellm_mysubs.credentials import refresher as module
from litellm_mysubs.credentials.file_store import FileCredentialStore
from litellm_mysubs.credentials.lock import file_lock
from litellm_mysubs.credentials.refresher import BackgroundRefresher
from litellm_mysubs.credentials.store import Credential, ProviderId
from litellm_mysubs.ui.service import MySubsService
from tests.test_gaps_refresh import (
    CODEX,
    TOKEN_HOSTS,
    Hosts,
    OwningStore,
    ask,
    proxy,
    rejected,
    serve,
    sse,
    valid,
)
from tests.test_plugin import codex_events

__all__ = ["proxy"]  # the real proxy app, as `test_gaps_refresh` wires it (autouse)


class RotatingTokenEndpoint:
    """A token endpoint with the providers' rotation: each refresh token works once."""

    def __init__(self, refresh_token: str = "RT", *, delay_s: float = 0.0) -> None:
        self.live = refresh_token
        self.minted = 0
        self.delay_s = delay_s
        #: Refresh tokens presented, in order.
        self.presented: list[str] = []

    def answer(self, request: httpx.Request) -> httpx.Response:
        sent = _token_params(request).get("refresh_token", "")
        self.presented.append(sent)
        if sent != self.live:
            return httpx.Response(
                400, json={"error": "invalid_grant", "error_description": "already used"}
            )
        self.minted += 1
        self.live = f"RT-{self.minted}"
        return httpx.Response(
            200,
            json={
                "access_token": f"AT-{self.minted}",
                "refresh_token": self.live,
                "expires_in": 3600,
            },
        )

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.delay_s:
            # Long enough for a second request to arrive while the exchange is in the air.
            await asyncio.sleep(self.delay_s)
        return self.answer(request)


def _token_params(request: httpx.Request) -> dict[str, str]:
    body = request.content.decode()
    if body.startswith("{"):
        return {k: str(v) for k, v in json.loads(body).items()}
    return {k: v[0] for k, v in urllib.parse.parse_qs(body).items()}


class AsyncHosts(Hosts):
    """`Hosts` whose token endpoint can take time, as a real one does."""

    def __init__(self, inference: Any, endpoint: RotatingTokenEndpoint) -> None:
        super().__init__(inference)
        self.endpoint = endpoint

    async def __call__(self, request: httpx.Request) -> httpx.Response:  # type: ignore[override]
        if request.url.host in TOKEN_HOSTS:
            self.token_calls += 1
            return await self.endpoint(request)
        return super().__call__(request)


def accepting(*tokens: str) -> Any:
    """Inference that answers only these access tokens and refuses the rest with 401."""
    bearers = {f"Bearer {token}" for token in tokens}

    def inference(request: httpx.Request) -> httpx.Response:
        if request.headers.get("authorization") in bearers:
            return sse(codex_events(text="answered"))
        return rejected(request)

    return inference


def expired_codex(token: str = "AT-old") -> Credential:
    return Credential(
        provider="openai-codex",
        access_token=token,
        refresh_token="RT",
        expires_at=time.time() - 10,
    )


def provider_lock(credentials_file: Path, provider: ProviderId) -> Path:
    """The renewal lock target, spelled out: ``credentials.json.<provider>``.

    Spelled out rather than asked from the module because it is a contract with the
    processes still running 0.1.16 against the same file: they lock this very path, and a
    different one would let a mixed rollout exchange the same token twice.
    """
    return credentials_file.with_name(f"{credentials_file.name}.{provider}")


class TestRequestPath:
    async def test_concurrent_requests_share_one_exchange(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two requests find the same expired token. One exchange serves both — a second
        one would present a refresh token the first just rotated away."""
        endpoint = RotatingTokenEndpoint(delay_s=0.2)
        hosts = AsyncHosts(accepting("AT-1"), endpoint)
        store = OwningStore({"openai-codex": expired_codex()})
        serve(hosts, store, monkeypatch)

        answers = await asyncio.gather(ask(CODEX, stream=False), ask(CODEX, stream=True))

        assert answers == ["answered", "answered"]
        assert endpoint.presented == ["RT"]
        assert hosts.bearers == ["Bearer AT-1", "Bearer AT-1"]

    async def test_a_worker_renewing_is_waited_for_and_adopted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Another worker holds the renewal lock. The request waits, finds the credential
        that worker wrote, and uses it — without spending the refresh token itself."""
        endpoint = RotatingTokenEndpoint()
        hosts = AsyncHosts(accepting("AT-peer"), endpoint)
        store = OwningStore({"openai-codex": expired_codex()})
        serve(hosts, store, monkeypatch)

        with file_lock(provider_lock(module.DEFAULT_PATH, "openai-codex")):
            request = asyncio.create_task(ask(CODEX, stream=False))
            await asyncio.sleep(0.3)
            assert endpoint.presented == [], "renewed while another worker held the lock"
            store.set("openai-codex", valid("AT-peer"))
        answer = await request

        assert answer == "answered"
        assert endpoint.presented == []
        assert hosts.bearers == ["Bearer AT-peer"]

    async def test_a_401_on_a_token_the_clock_calls_valid_mints_a_new_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The provider rejected a token that has not reached its stated expiry (revoked,
        invalidated by a login elsewhere). Retrying the same token is a second certain 401;
        omp re-mints, and the retry carries the new token."""
        endpoint = RotatingTokenEndpoint()
        hosts = AsyncHosts(accepting("AT-1"), endpoint)
        store = OwningStore({"openai-codex": valid("AT-old")})
        serve(hosts, store, monkeypatch)

        assert await ask(CODEX, stream=True) == "answered"
        assert endpoint.presented == ["RT"]
        assert hosts.bearers == ["Bearer AT-old", "Bearer AT-1"]
        stored = store.get("openai-codex")
        assert stored is not None and stored.refresh_token == "RT-1"

    async def test_a_token_just_minted_here_is_not_minted_again_on_a_401(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The upstream refuses even the token this worker minted a moment ago — an outage,
        not a dead token. Re-minting on every refusal would rotate the refresh token once
        per rejected request; within the cooldown the fresh token is retried instead."""
        endpoint = RotatingTokenEndpoint()
        hosts = AsyncHosts(accepting(), endpoint)
        store = OwningStore({"openai-codex": expired_codex()})
        serve(hosts, store, monkeypatch)

        with pytest.raises(openai.APIError, match="authentication token has expired"):
            await ask(CODEX, stream=False)

        assert endpoint.presented == ["RT"]
        assert hosts.bearers == ["Bearer AT-1", "Bearer AT-1"]


def file_store(tmp_path: Path, **credentials: Credential) -> FileCredentialStore:
    store = FileCredentialStore(tmp_path / "mysubs" / "credentials.json")
    for credential in credentials.values():
        store.set(credential.provider, credential)
    return store


def expiring(provider: ProviderId = "anthropic", *, refresh_token: str = "RT") -> Credential:
    """Inside the sweep's five-minute margin."""
    return Credential(
        provider=provider,
        access_token="AT-old",
        refresh_token=refresh_token,
        expires_at=time.time() + 60,
        project_id="proj" if provider == "google-antigravity" else "",
    )


def sweeper(store: FileCredentialStore, handler: Any) -> BackgroundRefresher:
    return BackgroundRefresher(
        store,
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def report_of(reports: list[module.RefreshReport], provider: ProviderId) -> module.RefreshReport:
    return next(r for r in reports if r.provider == provider)


class TestDeadGrant:
    async def test_a_rejected_refresh_token_drops_the_credential(self, tmp_path: Path) -> None:
        """`invalid_grant` is the grant dying, not the attempt failing. omp disables the
        row; kept, the sweep would present the dead token again every minute."""
        codex = valid()
        store = file_store(tmp_path, a=expiring(), c=codex)

        def dead(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                json={"error": "invalid_grant", "error_description": "Refresh token expired"},
            )

        report = report_of(await sweeper(store, dead).sweep(), "anthropic")

        assert report.renewed is False
        assert "invalid_grant" in report.reason
        assert FileCredentialStore(store.path).get("anthropic") is None
        # The other subscription is not collateral.
        assert FileCredentialStore(store.path).get("openai-codex") == codex

    async def test_an_outage_keeps_the_credential(self, tmp_path: Path) -> None:
        """A 503 says nothing about the grant; dropping it would force a login nobody
        needed."""
        credential = expiring()
        store = file_store(tmp_path, a=credential)

        def down(_: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="upstream unavailable")

        report = report_of(await sweeper(store, down).sweep(), "anthropic")

        assert report.renewed is False
        assert "503" in report.reason
        assert FileCredentialStore(store.path).get("anthropic") == credential

    async def test_a_peer_rotation_during_the_failed_exchange_survives(
        self, tmp_path: Path
    ) -> None:
        """The exchange fails because a peer rotated the token first and wrote the result.
        What gets dropped is the credential that failed — never the peer's fresh one."""
        store = file_store(tmp_path, a=expiring())
        peer = Credential(
            provider="anthropic",
            access_token="AT-peer",
            refresh_token="RT-peer",
            expires_at=time.time() + 3600,
        )

        def rotated_elsewhere(_: httpx.Request) -> httpx.Response:
            FileCredentialStore(store.path).set("anthropic", peer)
            return httpx.Response(400, json={"error": "invalid_grant"})

        await sweeper(store, rotated_elsewhere).sweep()

        assert FileCredentialStore(store.path).get("anthropic") == peer


class TestCompareAndSet:
    async def test_a_login_during_a_renewal_is_not_overwritten(self, tmp_path: Path) -> None:
        """The user reconnects while a sweep's exchange is in the air. The login is the
        newer credential; the renewal that finishes after it must not bring the old grant's
        tokens back over it."""
        store = file_store(tmp_path, a=expiring())
        login = Credential(
            provider="anthropic",
            access_token="AT-login",
            refresh_token="RT-login",
            expires_at=time.time() + 3600,
        )

        def login_lands_meanwhile(_: httpx.Request) -> httpx.Response:
            FileCredentialStore(store.path).set("anthropic", login)
            return httpx.Response(
                200, json={"access_token": "AT-2", "refresh_token": "RT-2", "expires_in": 3600}
            )

        report = report_of(await sweeper(store, login_lands_meanwhile).sweep(), "anthropic")

        assert FileCredentialStore(store.path).get("anthropic") == login
        assert report.renewed is False


class TestBounded:
    async def test_a_hanging_token_endpoint_does_not_pin_the_lock(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A token endpoint that never answers must not hold the renewal lock — and every
        worker waiting on it — indefinitely."""
        monkeypatch.setattr(module, "REFRESH_TIMEOUT_S", 0.05, raising=False)
        credential = expiring()
        store = file_store(tmp_path, a=credential)

        async def hangs(_: httpx.Request) -> httpx.Response:
            await asyncio.sleep(2)
            return httpx.Response(200, json={"access_token": "late", "expires_in": 3600})

        async with asyncio.timeout(1.5):
            report = report_of(await sweeper(store, hangs).sweep(), "anthropic")

        assert "timed out" in report.reason
        assert FileCredentialStore(store.path).get("anthropic") == credential
        with file_lock(provider_lock(store.path, "anthropic")):
            pass  # free again: acquiring without waiting would raise `LockBusyError`


class TestManualRenewal:
    async def test_the_button_waits_for_a_renewal_in_progress(self, tmp_path: Path) -> None:
        """The renew button pressed while a worker holds the lock waits for it, instead of
        presenting the same refresh token at the same moment."""
        endpoint = RotatingTokenEndpoint()
        store = file_store(
            tmp_path,
            a=Credential(
                provider="anthropic",
                access_token="AT-old",
                refresh_token="RT",
                expires_at=time.time() + 3600,
            ),
        )
        service = MySubsService(
            store=store,
            router_source=lambda: None,
            client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(endpoint)),
        )

        with file_lock(provider_lock(store.path, "anthropic")):
            pressed = asyncio.create_task(service.refresh("anthropic"))
            await asyncio.sleep(0.3)
            assert endpoint.presented == []
        renewed = await pressed

        assert endpoint.presented == ["RT"]
        assert renewed.access_token == "AT-1"
        assert FileCredentialStore(store.path).get("anthropic") == renewed


class TestFileStore:
    def test_writers_of_different_providers_do_not_undo_each_other(
        self, tmp_path: Path
    ) -> None:
        """Two workers write different providers at the same moment. Each rewrites the whole
        file; without mutual exclusion the second brings back the first's stale entry — a
        refresh token already rotated away upstream."""
        path = tmp_path / "mysubs" / "credentials.json"
        first, second = FileCredentialStore(path), FileCredentialStore(path)
        anthropic, codex = expiring(), valid("AT-codex")
        write = first._write
        other = threading.Thread(target=second.set, args=("openai-codex", codex))

        def write_while_the_other_worker_writes() -> None:
            other.start()
            other.join(timeout=0.5)
            write()

        first._write = write_while_the_other_worker_writes  # type: ignore[method-assign]
        first.set("anthropic", anthropic)
        other.join(timeout=10)

        on_disk = FileCredentialStore(path)
        assert on_disk.get("anthropic") == anthropic
        assert on_disk.get("openai-codex") == codex

    def test_the_re_read_sees_a_write_within_the_same_mtime(self, tmp_path: Path) -> None:
        """The re-read inside the renewal lock decides whether to spend a refresh token. On
        a file system with a coarse clock a peer's write can keep the old mtime; trusting
        it, the re-read returned the stale credential and the refresh token was spent twice."""
        store = file_store(tmp_path, a=expiring())
        before = store.path.stat()
        payload = json.loads(store.path.read_text("utf-8"))
        payload["anthropic"]["refresh_token"] = "RT-peer"
        store.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.utime(store.path, ns=(before.st_atime_ns, before.st_mtime_ns))

        assert store.reload() is True
        reread = store.get("anthropic")
        assert reread is not None and reread.refresh_token == "RT-peer"


#: `credentials.json` exactly as 0.1.16 writes it: `json.dump(indent=2, sort_keys=True)` of
#: the four fields per provider. Production files have this shape, with live tokens.
V0116_FILE = """{
  "anthropic": {
    "access_token": "sk-ant-oat01-live",
    "expires_at": 1790000000.0,
    "project_id": "",
    "refresh_token": "sk-ant-ort01-live"
  },
  "google-antigravity": {
    "access_token": "ya29.live",
    "expires_at": 1790000500.5,
    "project_id": "proud-lamp-123",
    "refresh_token": "1//0g-live"
  },
  "openai-codex": {
    "access_token": "eyJ.live",
    "expires_at": 1000.0,
    "project_id": "",
    "refresh_token": "rt_live"
  }
}"""


class TestReadsWhat0116Wrote:
    async def test_a_0116_file_is_read_renewed_and_kept_in_its_shape(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "mysubs" / "credentials.json"
        path.parent.mkdir(parents=True)
        path.write_text(V0116_FILE, encoding="utf-8")
        path.chmod(0o600)
        store = FileCredentialStore(path)

        assert store.get("google-antigravity") == Credential(
            provider="google-antigravity",
            access_token="ya29.live",
            refresh_token="1//0g-live",
            expires_at=1790000500.5,
            project_id="proud-lamp-123",
        )
        endpoint = RotatingTokenEndpoint("rt_live")
        renewed = await module.fresh(
            store,
            "openai-codex",
            client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(endpoint)),
        )

        assert renewed is not None and renewed.access_token == "AT-1"
        written = json.loads(path.read_text("utf-8"))
        original = json.loads(V0116_FILE)
        assert set(written) == set(original)
        assert all(set(entry) == set(original["anthropic"]) for entry in written.values())
        assert written["anthropic"] == original["anthropic"]
        assert written["google-antigravity"] == original["google-antigravity"]
        assert written["openai-codex"]["refresh_token"] == "RT-1"


class TestTokenResponse:
    async def test_a_response_without_expires_in_is_refused(self) -> None:
        """omp requires `expires_in` for all three rules. Accepted, the credential was stored
        with an unknown expiry — which no refresher renews again until a request fails."""

        def no_expiry(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"access_token": "AT-2", "refresh_token": "RT-2"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(no_expiry)) as client:
            with pytest.raises(oauth.OAuthError, match="expires_in"):
                await oauth.refresh(expiring(), client=client)


@pytest.mark.parametrize(
    ("status", "body", "definitive"),
    [
        # Google, on a revoked or expired grant.
        (
            400,
            json.dumps(
                {
                    "error": "invalid_grant",
                    "error_description": "Token has been expired or revoked.",
                }
            ),
            True,
        ),
        # OpenAI, on a refresh token presented twice.
        (
            401,
            json.dumps(
                {
                    "error": {
                        "message": "Your refresh token has already been used",
                        "code": "refresh_token_reused",
                    }
                }
            ),
            True,
        ),
        (503, "upstream connect error", False),
        (429, '{"error": "rate_limited"}', False),
        (401, "<html>Attention Required! | Cloudflare</html>", False),
    ],
)
async def test_what_counts_as_a_dead_grant(status: int, body: str, definitive: bool) -> None:
    def answer(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
        with pytest.raises(oauth.OAuthError) as caught:
            await oauth.refresh(expiring("openai-codex"), client=client)

    assert oauth.is_definitive_failure(caught.value) is definitive
