"""The `/mysubs` sub-app: what the user sees and what the buttons do."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm_mysubs.catalog.discovery import DiscoveredModel
from litellm_mysubs.credentials.store import Credential, ProviderId
from litellm_mysubs.ui import mount
from litellm_mysubs.ui.app import _UNSET as _DEFAULT_GUARD
from litellm_mysubs.ui.auth import auth_disabled, require_admin, session_user
from litellm_mysubs.ui.service import MySubsService


class FakeStore:
    owns_refresh = True

    def __init__(self, creds: dict[str, Credential] | None = None) -> None:
        self.creds = creds or {}
        self.unreadable: set[str] = set()

    def get(self, provider: ProviderId) -> Credential:
        if provider in self.unreadable:
            raise PermissionError("0644, should be 0600")
        return self.creds[provider]

    def set(self, credential: Credential) -> None:
        self.creds[credential.provider] = credential

    def delete(self, provider: ProviderId) -> None:
        self.creds.pop(provider, None)

    def reload(self) -> None: ...

    def connected(self) -> list[ProviderId]:
        return list(self.creds)  # type: ignore[arg-type]


class FakeRouter:
    def __init__(self) -> None:
        self.model_list: list[dict[str, Any]] = []

    def set_model_list(self, model_list: list[dict[str, Any]]) -> None:
        self.model_list = model_list


def build(
    store: FakeStore | None = None, router: FakeRouter | None = None
) -> tuple[TestClient, MySubsService, FakeRouter]:
    """A sub-app with no guard, to exercise the page's behaviour.

    The guard is tested separately in `TestAuth`: mixing it in here would force every
    content test to carry an identity, and what those assert is what the page does, not who
    gets to it.
    """
    used_router = router or FakeRouter()
    service = MySubsService(store=store or FakeStore(), router_source=lambda: used_router)
    app = FastAPI()
    mount(app, service, guard=None)
    return TestClient(app), service, used_router


class TestCards:
    def test_every_provider_shows_even_when_unconnected(self) -> None:
        """An unconnected provider is information: it is what tells the user what they can
        add. Hiding the unconnected ones made the page look empty for no reason.

        The labels come from `PROVIDER_LABELS`, not repeated here: a literal copy would turn
        every rename into a test failure that is not a defect.
        """
        from litellm_mysubs.ui.service import PROVIDER_LABELS

        client, _, _ = build()
        body = client.get("/mysubs/").text
        for label in PROVIDER_LABELS.values():
            assert label in body, label

    def test_an_expired_token_is_shown_as_expired_not_as_connected(self) -> None:
        """A dead token shown as "connected" sends the user to debug the wrong place."""
        store = FakeStore(
            {
                "anthropic": Credential(
                    provider="anthropic", access_token="a", expires_at=time.time() - 10
                )
            }
        )
        client, _, _ = build(store)
        body = client.get("/mysubs/").text
        assert "token expired" in body

    def test_an_unreadable_credential_does_not_take_the_page_down(self) -> None:
        """The store refuses files with loose permissions — the right call. Here that shows
        up as "not connected" with the other cards still visible, not as a 500."""
        store = FakeStore({"anthropic": Credential(provider="anthropic", access_token="a")})
        store.unreadable.add("anthropic")
        client, _, _ = build(store)
        response = client.get("/mysubs/")
        assert response.status_code == 200
        assert "Google Antigravity" in response.text


class TestConnect:
    def test_connect_sends_the_user_to_the_real_provider(self) -> None:
        client, service, _ = build()
        response = client.post("/mysubs/connect/anthropic", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"].startswith("https://claude.ai/oauth/authorize")
        assert "anthropic" in service.pending

    def test_unknown_provider_is_refused(self) -> None:
        client, _, _ = build()
        assert client.post("/mysubs/connect/made-up").status_code == 404

    def test_pasting_without_connecting_explains_instead_of_crashing(self) -> None:
        """The error has to name the missing step: a 500 here teaches nothing."""
        client, _, _ = build()
        response = client.post(
            "/mysubs/paste/anthropic", data={"pasted": "abc"}, follow_redirects=False
        )
        assert response.status_code == 303
        assert "Connect" in response.headers["location"]

    def test_upstream_error_reaches_the_user(self) -> None:
        """`invalid_grant: code expirado` says what to do; "failed" says nothing.

        The upstream description stays in Portuguese on purpose: the assertion is that the
        provider's own text reaches the user unrewritten, and an English fixture would not
        tell a pass-through apart from a message of ours.
        """

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400, json={"error": "invalid_grant", "error_description": "code expirado"}
            )

        client, service, _ = build()
        service.client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client.post("/mysubs/connect/anthropic", follow_redirects=False)
        response = client.post(
            "/mysubs/paste/anthropic", data={"pasted": "abc"}, follow_redirects=False
        )
        assert "code+expirado" in response.headers["location"].replace("%20", "+")


class TestApply:
    @pytest.fixture
    def ready(self) -> tuple[TestClient, MySubsService, FakeRouter]:
        store = FakeStore({"anthropic": Credential(provider="anthropic", access_token="a")})
        client, service, router = build(store)
        service.discovered["anthropic"] = [
            DiscoveredModel(
                wire_name="claude-opus-5", suggested_name="claude-opus-5", verified=True
            ),
            DiscoveredModel(
                wire_name="claude-haiku-4-5",
                suggested_name="claude-haiku-4-5",
                verified=False,
                note="network",
            ),
        ]
        return client, service, router

    def test_chosen_models_reach_the_router_with_both_prefixes(
        self, ready: tuple[TestClient, MySubsService, FakeRouter]
    ) -> None:
        """Without a prefix on the wire the name falls into wildcard resolution, which
        materialises phantom deployments for any requested name. The public name carries the
        subscription because `claude-sonnet-4-6` is also served by Antigravity."""
        client, _, router = ready
        client.post(
            "/mysubs/apply/anthropic", data={"chosen": ["claude-opus-5"]}, follow_redirects=False
        )
        assert len(router.model_list) == 1
        entry = router.model_list[0]
        assert entry["model_name"] == "mysubs/claudecode/claude-opus-5"
        assert entry["litellm_params"]["model"] == "anthropic/claude-opus-5"
        assert entry["model_info"]["managed_by"] == "mysubs"

    def test_unchosen_models_stay_out(
        self, ready: tuple[TestClient, MySubsService, FakeRouter]
    ) -> None:
        client, _, router = ready
        client.post(
            "/mysubs/apply/anthropic", data={"chosen": ["claude-opus-5"]}, follow_redirects=False
        )
        assert [d["model_name"] for d in router.model_list] == ["mysubs/claudecode/claude-opus-5"]

    def test_a_model_the_discovery_never_returned_is_refused(
        self, ready: tuple[TestClient, MySubsService, FakeRouter]
    ) -> None:
        """Applying a made-up name would produce exactly the 400s that motivated the package."""
        client, _, router = ready
        response = client.post(
            "/mysubs/apply/anthropic", data={"chosen": ["gpt-4-turbo"]}, follow_redirects=False
        )
        assert "gpt-4-turbo" in response.headers["location"]
        assert router.model_list == []

    def test_applying_before_discovering_explains_the_missing_step(self) -> None:
        store = FakeStore({"anthropic": Credential(provider="anthropic", access_token="a")})
        client, _, router = build(store)
        response = client.post(
            "/mysubs/apply/anthropic", data={"chosen": ["x"]}, follow_redirects=False
        )
        assert "discover" in response.headers["location"].lower()
        assert router.model_list == []


class TestState:
    def test_state_json_reports_what_the_page_shows(self) -> None:
        store = FakeStore({"anthropic": Credential(provider="anthropic", access_token="a")})
        client, service, _ = build(store)
        service.discovered["anthropic"] = [
            DiscoveredModel(wire_name="w", suggested_name="s", verified=False, note="network down")
        ]
        payload = client.get("/mysubs/api/state").json()
        connected = [p for p in payload["providers"] if p["connected"]]
        assert [p["provider"] for p in connected] == ["anthropic"]
        assert payload["discovered"]["anthropic"][0]["note"] == "network down"


class TestAuth:
    """The guard. `/mysubs` connects personal accounts and changes the models served."""

    def _app(self, guard: object) -> TestClient:
        service = MySubsService(store=FakeStore(), router_source=lambda: None)
        app = FastAPI()
        mount(app, service, guard=guard)
        return TestClient(app)

    def test_every_route_is_closed_by_default(self) -> None:
        """The default has to protect. Before this guard, `GET /mysubs/api/state` answered
        200 to anyone who could reach the proxy — and with a subscription connected it
        exposed the expiry and the project."""
        client = self._app(_DEFAULT_GUARD)
        assert client.get("/mysubs/").status_code == 403
        assert client.get("/mysubs/api/state").status_code == 403
        assert client.post("/mysubs/connect/anthropic").status_code == 403

    def test_only_a_full_admin_passes(self) -> None:
        """LiteLLM's `allowed_route_check_inside_route` accepts `proxy_admin_viewer`, which
        is right for reading lists and wrong here: a read-only role must not start an OAuth
        flow nor touch the Router."""
        for role in ("proxy_admin_viewer", "internal_user", "team", None):
            with pytest.raises(Exception, match="proxy admins only"):
                require_admin(SimpleNamespace(user_role=role))

        admin = SimpleNamespace(user_role="proxy_admin")
        assert require_admin(admin) is admin

    def test_an_enum_role_is_read_by_value(self) -> None:
        """LiteLLM hands over `LitellmUserRoles`, not a string; comparing the raw enum
        refused a legitimate administrator."""
        admin = SimpleNamespace(user_role=SimpleNamespace(value="proxy_admin"))
        assert require_admin(admin) is admin

    def test_the_opt_out_is_explicit(self) -> None:
        """Whoever runs without a key database has no `proxy_admin` at all. With no escape
        hatch, they would end up mounting the sub-app by hand — unguarded, without knowing
        it."""
        assert auth_disabled({"MYSUBS_DISABLE_AUTH": "1"}) is True
        assert auth_disabled({"MYSUBS_DISABLE_AUTH": "true"}) is True
        assert auth_disabled({}) is False
        assert auth_disabled({"MYSUBS_DISABLE_AUTH": "0"}) is False


class TestErrorPages:
    """A refusal has to be readable. Access denied already worked; it was the 500 that
    taught nothing to whoever saw it."""

    def _app(self) -> TestClient:
        service = MySubsService(store=FakeStore(), router_source=lambda: None)
        app = FastAPI()
        mount(app, service, guard=_DEFAULT_GUARD)
        return TestClient(app, raise_server_exceptions=False)

    def test_a_refusal_is_not_a_server_error(self) -> None:
        """Verified on the real proxy: without this handler the refusal arrived as

            HTTP 500  RuntimeError: Caught handled exception, but response already started

        because `user_api_key_auth` raises `ProxyException` and a mounted sub-app does not
        inherit the handler that converts it.
        """
        response = self._app().get("/mysubs/")
        assert response.status_code in (401, 403)
        assert response.status_code != 500

    def test_the_browser_gets_a_page_not_raw_json(self) -> None:
        """This is the UI: whoever opens it from the menu has to understand what to do."""
        response = self._app().get("/mysubs/", headers={"accept": "text/html"})
        assert "text/html" in response.headers["content-type"]
        assert "No access" in response.text
        assert "admin key" in response.text

    def test_a_json_client_still_gets_json(self) -> None:
        """Whoever automates does not want HTML."""
        response = self._app().get("/mysubs/api/state", headers={"accept": "application/json"})
        assert "application/json" in response.headers["content-type"]
        assert "detail" in response.json()

    def test_a_proxy_exception_is_converted_too(self) -> None:
        """`ProxyException` is not an `HTTPException`: it is a plain `Exception` that the
        proxy converts in a handler on its own app. Registering only the `HTTPException`
        handler let through exactly the exception `user_api_key_auth` raises — which is the
        one that shows up in production.
        """
        from litellm.proxy._types import ProxyException

        service = MySubsService(store=FakeStore(), router_source=lambda: None)
        app = FastAPI()
        mount(app, service, guard=None)

        sub = next(r.app for r in app.routes if getattr(r, "path", "") == "/mysubs")

        @sub.get("/explode")
        async def explode() -> None:
            raise ProxyException(
                message="Authentication Error, No api key passed in.",
                type="auth_error",
                param=None,
                code=401,
            )

        response = TestClient(app, raise_server_exceptions=False).get(
            "/mysubs/explode", headers={"accept": "text/html"}
        )
        assert response.status_code == 401
        assert "No access" in response.text


class TestSessionCookie:
    """The UI session has to work: that is how the page is reached from the menu."""

    def _token(self, role: str = "proxy_admin", key: str = "sk-master") -> str:
        import jwt

        return jwt.encode(
            {"user_id": "admin", "user_role": role, "exp": int(time.time()) + 600},
            key,
            algorithm="HS256",
        )

    def _request(self, cookies: dict[str, str]) -> Any:
        return SimpleNamespace(cookies=cookies)

    def test_a_valid_session_identifies_the_admin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`user_api_key_auth` only reads headers — verified, not one reference to cookies in
        that module. The dashboard works around it by reading the cookie via `document.cookie`
        and building the `Authorization` header in JavaScript; a page served outside the SPA
        does not do that, and without this path a valid administrator session gave 401.
        """
        import litellm.proxy.proxy_server as proxy_server

        monkeypatch.setattr(proxy_server, "master_key", "sk-master", raising=False)
        user = session_user(self._request({"token": self._token()}))
        assert user is not None
        assert user.user_role == "proxy_admin"

    def test_a_forged_cookie_is_not_a_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The JWT is HS256 signed with the `master_key`: without it, it cannot be forged."""
        import litellm.proxy.proxy_server as proxy_server

        monkeypatch.setattr(proxy_server, "master_key", "sk-master", raising=False)
        assert session_user(self._request({"token": self._token(key="other")})) is None
        assert session_user(self._request({"token": "abc.def.ghi"})) is None

    def test_an_expired_session_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import jwt
        import litellm.proxy.proxy_server as proxy_server

        monkeypatch.setattr(proxy_server, "master_key", "sk-master", raising=False)
        stale = jwt.encode(
            {"user_id": "a", "user_role": "proxy_admin", "exp": int(time.time()) - 10},
            "sk-master",
            algorithm="HS256",
        )
        assert session_user(self._request({"token": stale})) is None

    def test_no_cookie_is_not_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No session returns `None` so the key path gets its turn."""
        import litellm.proxy.proxy_server as proxy_server

        monkeypatch.setattr(proxy_server, "master_key", "sk-master", raising=False)
        assert session_user(self._request({})) is None

    def test_a_viewer_session_still_cannot_manage_subscriptions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Having a session is not having a mandate: the role still decides."""
        import litellm.proxy.proxy_server as proxy_server

        monkeypatch.setattr(proxy_server, "master_key", "sk-master", raising=False)
        user = session_user(self._request({"token": self._token(role="proxy_admin_viewer")}))
        assert user is not None
        with pytest.raises(Exception, match="proxy admins only"):
            require_admin(user)
