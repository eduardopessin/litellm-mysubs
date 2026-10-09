"""The Antigravity client identity, catalog rejection and sampling support, as omp 18.8.6
has them.

The backend gates the catalog and the models on the version in the ``User-Agent``. omp
resolves it from the Antigravity update manifest (``ensureAntigravityVersion``,
``pi-catalog/src/wire/gemini-headers.ts``), before discovery and — new in 18.8.6 — before
each Antigravity request (``streamGoogleGeminiCli``), with ``2.19.1`` as the fallback. In
production the fixed ``2.8.0`` this package sent lost every Claude id from the catalog.

Discovery now tells a credential the upstream refused (401/403 on every host) from a host
that could not be asked (``fetchAntigravityDiscoveryModels``'s ``rejectedStatus``).

All of it over `httpx.MockTransport`: no network.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, Final

import httpx
import pytest

from litellm_mysubs import specs
from litellm_mysubs.catalog.discovery import DiscoveryError, discover
from litellm_mysubs.credentials.store import Credential
from litellm_mysubs.transport import antigravity_version as av
from litellm_mysubs.transport.hosts import HOSTS, MODELS_PATH
from litellm_mysubs.wire import antigravity as ag
from litellm_mysubs.wire.antigravity_models import ModelCatalog

Handler = Callable[[httpx.Request], httpx.Response]

GOOGLE = Credential(provider="google-antigravity", access_token="tok-g")

#: Shape of the electron-builder manifest the updater serves.
MANIFEST = "version: 2.21.3\nfiles:\n  - url: Antigravity-2.21.3-arm64-mac.zip\n"


def client(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def catalog_payload(*ids: str) -> dict[str, object]:
    return {"models": {model: {"displayName": model} for model in ids}, "deprecatedModelIds": []}


@pytest.fixture
def unresolved(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo conftest's pinned resolution: the lookup has not run in this process."""
    monkeypatch.setattr(av, "_lookup", av._Lookup())


class TestManifestVersion:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("version: 2.19.1\n", "2.19.1"),
            ('version: "2.20.0"\r\npath: x\n', "2.20.0"),
            ("  version : '2.20.1'  # latest\n", "2.20.1"),
            ("files: []\nversion: 3.0.0\n", "3.0.0"),
            # The first `version` line decides, even malformed: never a later one's value.
            ("version: 2.19\nversion: 2.19.1\n", None),
            ("version: v2.19.1\n", None),
            ("files: []\n", None),
        ],
    )
    def test_parse(self, text: str, expected: str | None) -> None:
        assert av.parse_manifest_version(text) == expected


class TestUserAgent:
    def test_fallback_is_omp_18_8_6_pin(self, unresolved: None) -> None:
        assert av.user_agent() == (
            "antigravity/hub/2.19.1 (aidev_client; os_type=darwin; arch=arm64; cl=963137146)"
        )

    def test_overrides(self, unresolved: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PI_AI_ANTIGRAVITY_VERSION", "9.9.9")
        monkeypatch.setenv("PI_AI_ANTIGRAVITY_CL", "1")
        monkeypatch.setenv("PI_AI_ANTIGRAVITY_OS", "linux")
        monkeypatch.setenv("PI_AI_ANTIGRAVITY_ARCH", "x64")
        assert av.user_agent() == (
            "antigravity/hub/9.9.9 (aidev_client; os_type=linux; arch=x64; cl=1)"
        )


class TestEnsureVersion:
    async def test_manifest_version_is_used_and_kept(self, unresolved: None) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, text=MANIFEST)

        async with client(handler) as http:
            await av.ensure_version(http)
            await av.ensure_version(http)

        assert len(seen) == 1
        assert str(seen[0].url) == av.MANIFEST_URL
        assert seen[0].method == "GET"
        assert seen[0].headers["User-Agent"] == "electron-builder"
        assert seen[0].headers["Cache-Control"] == "no-cache"
        assert av.user_agent().startswith("antigravity/hub/2.21.3 ")

    async def test_failure_falls_back_and_is_not_retried_at_once(
        self, unresolved: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dead manifest costs the request path one lookup per `RETRY_S`, not one per
        request."""
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(503)

        clock = [1000.0]
        monkeypatch.setattr(av.time, "monotonic", lambda: clock[0])
        async with client(handler) as http:
            await av.ensure_version(http)
            assert av.version() == av.DEFAULT_VERSION
            clock[0] += av.RETRY_S - 1
            await av.ensure_version(http)
            assert calls == 1
            clock[0] += 2
            await av.ensure_version(http)
        assert calls == 2

    async def test_override_skips_the_lookup(
        self, unresolved: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PI_AI_ANTIGRAVITY_VERSION", "2.30.0")

        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("no lookup with the override set")

        async with client(handler) as http:
            await av.ensure_version(http)
        assert av.version() == "2.30.0"

    async def test_concurrent_callers_share_one_lookup(self, unresolved: None) -> None:
        calls = 0
        release = asyncio.Event()

        async def respond(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            await release.wait()
            return httpx.Response(200, text=MANIFEST)

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            waiters = [asyncio.create_task(av.ensure_version(http)) for _ in range(3)]
            await asyncio.sleep(0)
            release.set()
            await asyncio.gather(*waiters)
        assert calls == 1
        assert av.version() == "2.21.3"

    async def test_a_cancelled_caller_does_not_cancel_the_lookup(self, unresolved: None) -> None:
        """omp 18.8.6: the caller's signal ends its own wait, not the shared lookup."""
        release = asyncio.Event()

        async def respond(request: httpx.Request) -> httpx.Response:
            await release.wait()
            return httpx.Response(200, text=MANIFEST)

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            waiter = asyncio.create_task(av.ensure_version(http))
            await asyncio.sleep(0.01)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            release.set()
            await av.ensure_version(http)
        assert av.version() == "2.21.3"


class TestRequestPath:
    async def test_spec_resolves_the_version_before_it_is_built(
        self, unresolved: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``streamGoogleGeminiCli`` 18.8.6 awaits ``ensureAntigravityVersion`` before each
        Antigravity request: a process that skipped discovery still sends the current one."""
        order: list[str] = []

        async def ensure(client: httpx.AsyncClient | None = None) -> None:
            order.append("ensure")
            av._lookup.discovered = "2.21.3"

        async def token(provider: str) -> str:
            order.append("token")
            return "tok"

        async def refresh(token: str, project_id: str) -> ModelCatalog:
            order.append("catalog")
            return ModelCatalog()

        monkeypatch.setattr(av, "ensure_version", ensure)
        monkeypatch.setattr(specs, "_access_token", token)
        monkeypatch.setattr(specs, "_refresh_catalog", refresh)
        monkeypatch.setattr(specs._state, "store", None)

        spec, _ = await specs._antigravity_spec(
            "gemini-3-flash", [{"role": "user", "content": "hi"}], {}
        )

        assert order[0] == "ensure"
        assert spec.headers["User-Agent"].startswith("antigravity/hub/2.21.3 ")


class TestDiscoveryRejection:
    async def test_discovery_resolves_the_version_on_its_own_client_first(
        self, unresolved: None
    ) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            if str(request.url) == av.MANIFEST_URL:
                return httpx.Response(200, text=MANIFEST)
            return httpx.Response(200, json=catalog_payload("claude-sonnet-4-6"))

        async with client(handler) as http:
            models = await discover(GOOGLE, client=http)

        assert [str(r.url) for r in seen] == [av.MANIFEST_URL, HOSTS[0] + MODELS_PATH]
        assert seen[1].headers["User-Agent"].startswith("antigravity/hub/2.21.3 ")
        assert [m.wire_name for m in models] == ["claude-sonnet-4-6"]

    async def test_rejected_credential_is_named_on_the_snapshot(self) -> None:
        catalog = ModelCatalog()
        catalog.update(catalog_payload("gemini-3.1-pro"), now=1000.0)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": {"code": 401}})

        async with client(handler) as http:
            models = await discover(GOOGLE, client=http, catalog=catalog, now=1060.0)

        assert [m.wire_name for m in models] == ["gemini-3.1-pro"]
        assert "60 s old; the endpoint rejected the credential (HTTP 401) now" in models[0].note

    async def test_a_transient_host_beside_a_rejection_is_not_a_rejection(self) -> None:
        """omp: ``rejectedStatus`` only when no host failed transiently — a 403 on one host
        and a 503 on the other say nothing definitive about the credential."""
        catalog = ModelCatalog()
        catalog.update(catalog_payload("gemini-3.1-pro"), now=1000.0)

        def handler(request: httpx.Request) -> httpx.Response:
            if str(request.url).startswith(HOSTS[0]):
                return httpx.Response(403)
            return httpx.Response(503)

        async with client(handler) as http:
            models = await discover(GOOGLE, client=http, catalog=catalog, now=1060.0)

        assert "the endpoint did not respond now" in models[0].note

    async def test_rejection_without_snapshot_says_so(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(403)

        async with client(handler) as http:
            with pytest.raises(DiscoveryError, match=r"rejected the credential \(HTTP 403\)"):
                await discover(GOOGLE, client=http)


class TestSamplingSupport:
    """``withSupportedSamplingParams`` (``stream.ts``): pi-catalog 18.8.6 resolves
    ``supportsSamplingParams: false`` for the Antigravity ``claude-sonnet-5-5`` and
    ``claude-opus-5-5`` ids, and omp drops every sampling field for them."""

    SAMPLING: Final[dict[str, Any]] = {
        "temperature": 0.3,
        "top_p": 0.97,
        "top_k": 40,
        "presence_penalty": 0.5,
    }

    def generation(self, model: str) -> dict[str, Any]:
        catalog = ModelCatalog(ids=(model,), info={model: {}}, fetched_at=1.0)
        body = ag.build_payload(
            model,
            [{"role": "user", "content": "x"}],
            "proj-1",
            catalog=catalog,
            extra=dict(self.SAMPLING),
        )
        generation: dict[str, Any] = body["request"]["generationConfig"]
        return generation

    def test_adaptive_claude_sends_no_sampling(self) -> None:
        generation = self.generation("claude-sonnet-5-5")
        assert not {"temperature", "topP", "topK", "presencePenalty"} & generation.keys()

    def test_claude_4_6_keeps_it(self) -> None:
        generation = self.generation("claude-sonnet-4-6")
        assert generation["temperature"] == 0.3
        assert (generation["topP"], generation["topK"]) == (0.97, 40)
        assert generation["presencePenalty"] == 0.5
