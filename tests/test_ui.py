"""A sub-app `/mysubs`: o que o utilizador vê e o que os botões fazem."""

from __future__ import annotations

import time
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm_mysubs.catalog.discovery import DiscoveredModel
from litellm_mysubs.credentials.store import Credential, ProviderId
from litellm_mysubs.ui import mount
from litellm_mysubs.ui.service import MySubsService


class FakeStore:
    owns_refresh = True

    def __init__(self, creds: dict[str, Credential] | None = None) -> None:
        self.creds = creds or {}
        self.unreadable: set[str] = set()

    def get(self, provider: ProviderId) -> Credential:
        if provider in self.unreadable:
            raise PermissionError("0644, devia ser 0600")
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
    used_router = router or FakeRouter()
    service = MySubsService(store=store or FakeStore(), router_source=lambda: used_router)
    app = FastAPI()
    mount(app, service)
    return TestClient(app), service, used_router


class TestCards:
    def test_every_provider_shows_even_when_unconnected(self) -> None:
        """Um provedor por ligar é informação: é o que diz ao utilizador o que pode
        acrescentar. Esconder os não-ligados fazia a página parecer vazia sem razão."""
        client, _, _ = build()
        body = client.get("/mysubs/").text
        assert "Claude Max" in body
        assert "ChatGPT Plus (Codex)" in body
        assert "Google Antigravity" in body

    def test_an_expired_token_is_shown_as_expired_not_as_connected(self) -> None:
        """ "Ligado" num token morto manda o utilizador depurar o sítio errado."""
        store = FakeStore(
            {
                "anthropic": Credential(
                    provider="anthropic", access_token="a", expires_at=time.time() - 10
                )
            }
        )
        client, _, _ = build(store)
        body = client.get("/mysubs/").text
        assert "token expirado" in body

    def test_an_unreadable_credential_does_not_take_the_page_down(self) -> None:
        """O store recusa ficheiros com permissões largas — decisão certa. Aqui traduz-se
        em "por ligar" com os outros cards ainda visíveis, não num 500."""
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
        assert client.post("/mysubs/connect/inventado").status_code == 404

    def test_pasting_without_connecting_explains_instead_of_crashing(self) -> None:
        """O erro tem de dizer o passo em falta: um 500 aqui não ensina nada."""
        client, _, _ = build()
        response = client.post(
            "/mysubs/paste/anthropic", data={"pasted": "abc"}, follow_redirects=False
        )
        assert response.status_code == 303
        assert "Conectar" in response.headers["location"]

    def test_upstream_error_reaches_the_user(self) -> None:
        """`invalid_grant: code expirado` diz o que fazer; "falhou" não diz nada."""

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
                note="rede",
            ),
        ]
        return client, service, router

    def test_chosen_models_reach_the_router_with_the_wire_prefix(
        self, ready: tuple[TestClient, MySubsService, FakeRouter]
    ) -> None:
        """Sem prefixo o nome cai na resolução por wildcard, que materializa deployments
        fantasma para qualquer nome pedido."""
        client, _, router = ready
        client.post(
            "/mysubs/apply/anthropic", data={"chosen": ["claude-opus-5"]}, follow_redirects=False
        )
        assert len(router.model_list) == 1
        entry = router.model_list[0]
        assert entry["model_name"] == "claude-opus-5"
        assert entry["litellm_params"]["model"] == "anthropic/claude-opus-5"
        assert entry["model_info"]["managed_by"] == "mysubs"

    def test_unchosen_models_stay_out(
        self, ready: tuple[TestClient, MySubsService, FakeRouter]
    ) -> None:
        client, _, router = ready
        client.post(
            "/mysubs/apply/anthropic", data={"chosen": ["claude-opus-5"]}, follow_redirects=False
        )
        assert [d["model_name"] for d in router.model_list] == ["claude-opus-5"]

    def test_a_model_the_discovery_never_returned_is_refused(
        self, ready: tuple[TestClient, MySubsService, FakeRouter]
    ) -> None:
        """Aplicar um nome inventado produziria exactamente os 400 que motivaram o pacote."""
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
        assert "descobre" in response.headers["location"].lower()
        assert router.model_list == []


class TestState:
    def test_state_json_reports_what_the_page_shows(self) -> None:
        store = FakeStore({"anthropic": Credential(provider="anthropic", access_token="a")})
        client, service, _ = build(store)
        service.discovered["anthropic"] = [
            DiscoveredModel(wire_name="w", suggested_name="s", verified=False, note="rede em baixo")
        ]
        payload = client.get("/mysubs/api/state").json()
        connected = [p for p in payload["providers"] if p["connected"]]
        assert [p["provider"] for p in connected] == ["anthropic"]
        assert payload["discovered"]["anthropic"][0]["note"] == "rede em baixo"
