"""Fluxo OAuth com paste e renovação.

O que estes testes defendem não é que um POST devolve 200. É o conjunto de coisas que,
quando falham, falham em silêncio e só aparecem semanas depois:

* o paste aceita as três formas que o utilizador consegue produzir — e recusa um ``state``
  que não é o do pedido, que é o único sinal de CSRF que temos;
* a rotação **substitui** o refresh token. Guardar o antigo por cima de um novo dá
  ``invalid_grant`` na renovação seguinte, e o sintoma é re-login manual sem explicação;
* o ``project_id`` do Antigravity sai do ``loadCodeAssist`` e sobrevive à renovação. Sem
  ele todos os pedidos de inferência dão 400;
* o ``error_description`` do upstream chega ao utilizador intacto. É o que lhe diz o que
  fazer; substituí-lo por uma mensagem nossa apaga-o.

Tudo com ``httpx.MockTransport``: sem rede, sem relógio real.
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
    return AuthRequest(url="https://exemplo/autorizar", state="ST4TE", verifier="v" * 43)


def query_of(url: str) -> dict[str, str]:
    return {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlparse(url).query).items()}


def form_of(request: httpx.Request) -> dict[str, str]:
    return {k: v[0] for k, v in urllib.parse.parse_qs(request.content.decode()).items()}


def json_of(request: httpx.Request) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(request.content)
    return result


def token_response(**extra: Any) -> httpx.Response:
    return httpx.Response(200, json={"access_token": "at-novo", "expires_in": 3600, **extra})


class FakeStore(CredentialStore):
    """Store mínimo cuja única propriedade interessante é a posse do refresh."""

    def __init__(self, *, owns_refresh: bool) -> None:
        self.owns_refresh = owns_refresh

    def get(self, provider: ProviderId) -> Credential | None:
        return None

    def set(self, provider: ProviderId, credential: Credential) -> None:
        raise AssertionError("não devia escrever")

    def delete(self, provider: ProviderId) -> None:
        raise AssertionError("não devia apagar")

    def reload(self) -> bool:
        return False


class TestPKCE:
    def test_challenge_is_s256_of_verifier(self) -> None:
        """Se o desafio não for o SHA-256 do verifier, a troca do código dá `invalid_grant`
        no servidor — e o erro só aparece depois de o utilizador já ter feito login."""
        request = begin("openai-codex")
        expected = (
            base64.urlsafe_b64encode(hashlib.sha256(request.verifier.encode()).digest())
            .decode()
            .rstrip("=")
        )
        assert query_of(request.url)["code_challenge"] == expected
        assert query_of(request.url)["code_challenge_method"] == "S256"

    def test_verifier_is_fresh_per_request(self) -> None:
        primeiro, segundo = begin("anthropic"), begin("anthropic")
        assert primeiro.verifier != segundo.verifier
        assert primeiro.state != segundo.state

    def test_challenge_is_url_safe_and_unpadded(self) -> None:
        """Padding e `+`/`/` teriam de ser escapados; há servidores que comparam byte a
        byte com o que receberam e rejeitam a diferença."""
        challenge = query_of(begin("anthropic").url)["code_challenge"]
        assert "=" not in challenge
        assert "+" not in challenge and "/" not in challenge

    def test_antigravity_has_no_pkce(self) -> None:
        """A regra do OMP não activa PKCE no Google; o segredo do cliente faz esse papel.
        Mandar um desafio que o servidor não espera não ajuda e pode ser recusado."""
        request = begin("google-antigravity")
        assert request.verifier == ""
        assert "code_challenge" not in query_of(request.url)


class TestAuthorizeUrl:
    def test_anthropic_asks_for_a_pasteable_code(self) -> None:
        """`code=true` é o que faz a página mostrar um código copiável em vez de
        redireccionar. Sem isto o paste deixa de ser possível num container."""
        assert query_of(begin("anthropic").url)["code"] == "true"

    def test_google_asks_for_a_refresh_token(self) -> None:
        """Sem `access_type=offline` o Google não emite refresh token; sem
        `prompt=consent` não o reemite a quem já consentiu. Faltar um deles produz uma
        subscrição que morre na primeira expiração."""
        params = query_of(begin("google-antigravity").url)
        assert params["access_type"] == "offline"
        assert params["prompt"] == "consent"

    def test_codex_redirect_uri_is_the_registered_one(self) -> None:
        """A OpenAI só autoriza este URI exacto; derivá-lo de um porto livre seria
        rejeitado no servidor depois de o utilizador já ter autenticado."""
        params = query_of(begin("openai-codex").url)
        assert params["redirect_uri"] == "http://localhost:1455/auth/callback"
        assert params["client_id"] == "app_EMoamEEZ73f0CkXaXp7hrann"

    def test_state_travels_in_the_url(self) -> None:
        request = begin("anthropic")
        assert query_of(request.url)["state"] == request.state

    def test_anthropic_requests_inference_scope(self) -> None:
        """`user:inference` é o que dá inferência directa com token de subscrição; sem ele
        o token só serve para gestão de conta."""
        assert "user:inference" in query_of(begin("anthropic").url)["scope"].split(" ")


class TestPaste:
    """As três formas que o utilizador consegue produzir, e o que tem de ser recusado."""

    async def exchange(self, pasted: str, provider: ProviderId = "anthropic") -> httpx.Request:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return token_response(refresh_token="rt-1")

        async with client_of(handler) as client:
            await complete(provider, request_for(provider), pasted, client=client)
        return seen[0]

    async def test_full_callback_url(self) -> None:
        """O que o browser mostra na barra quando o redirect não alcança o porto local."""
        sent = await self.exchange("https://claude.ai/callback?code=abc123&state=ST4TE")
        assert json_of(sent)["code"] == "abc123"

    async def test_bare_code(self) -> None:
        """Quem copia só o pedaço do código não deve ser obrigado a reconstruir uma URL."""
        sent = await self.exchange("abc123")
        assert json_of(sent)["code"] == "abc123"

    async def test_code_hash_state(self) -> None:
        """O formato que a Anthropic mostra com `code=true`. O fragmento é o `state`, não
        parte do código — mandá-lo junto faz o servidor recusar a troca."""
        sent = await self.exchange("abc123#ST4TE")
        assert json_of(sent)["code"] == "abc123"

    async def test_surrounding_whitespace_is_tolerated(self) -> None:
        sent = await self.exchange("  abc123#ST4TE \n")
        assert json_of(sent)["code"] == "abc123"

    async def test_code_with_hash_inside_the_query(self) -> None:
        """A Anthropic chega a devolver `code=abc#state` dentro da própria query."""
        sent = await self.exchange("https://claude.ai/callback?code=abc123%23ST4TE&state=ST4TE")
        assert json_of(sent)["code"] == "abc123"

    async def test_alternative_auth_code_parameter(self) -> None:
        sent = await self.exchange("https://claude.ai/callback?authCode=abc123&state=ST4TE")
        assert json_of(sent)["code"] == "abc123"


class TestPasteRejection:
    """Nenhum destes chega a tocar na rede: o handler falha se for chamado."""

    def client(self) -> httpx.AsyncClient:
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError(f"não devia haver pedido: {request.url}")

        return client_of(handler)

    async def test_state_mismatch_in_url_raises(self) -> None:
        async with self.client() as client:
            with pytest.raises(OAuthError) as excinfo:
                await complete(
                    "anthropic",
                    request_for(),
                    "https://claude.ai/callback?code=abc&state=OUTRO",
                    client=client,
                )
        assert "state" in str(excinfo.value).lower()

    async def test_state_mismatch_in_fragment_raises(self) -> None:
        """O fragmento é a autoridade sobre o `state`; um trocado ali conta tanto como na
        query, senão o formato `codigo#state` seria um buraco de CSRF."""
        async with self.client() as client:
            with pytest.raises(OAuthError):
                await complete("anthropic", request_for(), "abc#OUTRO", client=client)

    async def test_error_in_callback_url_propagates_description(self) -> None:
        """O `error_description` do retorno é o que diz ao utilizador o que fazer."""
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
        """O verifier é o segredo que nunca viajou; sem ele a troca falha. O `state` vai
        no corpo porque a regra da Anthropic o exige lá, não só no URL."""
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return token_response(refresh_token="rt-1")

        pedido = request_for()
        async with client_of(handler) as client:
            await complete("anthropic", pedido, "abc#ST4TE", client=client)

        corpo = json_of(seen[0])
        assert corpo["code_verifier"] == pedido.verifier
        assert corpo["state"] == pedido.state
        assert corpo["grant_type"] == "authorization_code"

    async def test_anthropic_uses_json_and_codex_uses_form(self) -> None:
        """Não são intermutáveis: a Anthropic recusa urlencoded e o Google recusa JSON.
        Trocar isto dá um 400 opaco do servidor."""
        corpos: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            corpos[str(request.url)] = request.headers["content-type"]
            return token_response(refresh_token="rt-1")

        async with client_of(handler) as client:
            await complete("anthropic", request_for(), "abc", client=client)
            await complete("openai-codex", request_for(), "abc", client=client)

        assert corpos["https://api.anthropic.com/v1/oauth/token"].startswith("application/json")
        assert corpos["https://auth.openai.com/oauth/token"].startswith(
            "application/x-www-form-urlencoded"
        )

    async def test_expiry_skew_lands_before_the_real_deadline(self) -> None:
        """A margem da Anthropic é 300s. Renovar exactamente na expiração é renovar tarde:
        um pedido em voo apanha o token já morto."""
        async with client_of(lambda _: token_response(refresh_token="rt-1")) as client:
            credencial = await complete("anthropic", request_for(), "abc", client=client)
        atraso = credencial.expires_at - time.time()
        assert 3200 < atraso < 3310

    async def test_missing_access_token_raises_with_body(self) -> None:
        """Há provedores que embrulham o erro num envelope de sucesso; o corpo é a única
        explicação e tem de subir."""
        resposta = httpx.Response(200, json={"code": 4001, "msg": "conta suspensa"})
        async with client_of(lambda _: resposta) as client:
            with pytest.raises(OAuthError) as excinfo:
                await complete("anthropic", request_for(), "abc", client=client)
        assert "conta suspensa" in str(excinfo.value)

    async def test_provider_error_propagates_description(self) -> None:
        """Princípio 3: o `error_description` real, nunca uma mensagem genérica por cima."""
        resposta = httpx.Response(
            400,
            json={
                "error": "invalid_grant",
                "error_description": "O código de autorização expirou; repete o login",
            },
        )
        async with client_of(lambda _: resposta) as client:
            with pytest.raises(OAuthError) as excinfo:
                await complete("openai-codex", request_for(), "abc", client=client)

        assert "O código de autorização expirou; repete o login" in str(excinfo.value)
        assert excinfo.value.status == 400
        assert "invalid_grant" in excinfo.value.body


class TestRefreshRotation:
    async def test_rotated_token_replaces_the_old_one(self) -> None:
        """O caso que produz `invalid_grant` em ciclo quando corre mal: o provedor rodou o
        token, o antigo já não serve, e guardá-lo mata a renovação seguinte."""
        velha = Credential("anthropic", "at-velho", refresh_token="rt-velho", expires_at=1.0)
        async with client_of(lambda _: token_response(refresh_token="rt-novo")) as client:
            nova = await refresh(velha, client=client)

        assert nova.refresh_token == "rt-novo"
        assert nova.access_token == "at-novo"
        assert velha.refresh_token == "rt-velho", "a credencial antiga é imutável"

    async def test_unrotated_token_is_preserved(self) -> None:
        """Nem todos rodam. Um provedor que omite `refresh_token` não está a revogar o
        nosso — apagá-lo forçaria re-login sem motivo."""
        velha = Credential("openai-codex", "at-velho", refresh_token="rt-velho")
        async with client_of(lambda _: token_response()) as client:
            nova = await refresh(velha, client=client)
        assert nova.refresh_token == "rt-velho"

    async def test_sends_the_stored_refresh_token(self) -> None:
        seen: list[dict[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(form_of(request))
            return token_response(refresh_token="rt-novo")

        async with client_of(handler) as client:
            await refresh(Credential("openai-codex", "at", refresh_token="rt-velho"), client=client)

        assert seen[0]["grant_type"] == "refresh_token"
        assert seen[0]["refresh_token"] == "rt-velho"

    async def test_anthropic_refresh_carries_the_beta_headers(self) -> None:
        """O Claude Code manda-os na renovação e não na troca inicial; sem eles o endpoint
        trata o pedido como vindo de outro cliente."""
        seen: list[httpx.Headers] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.headers)
            return token_response(refresh_token="rt-novo")

        async with client_of(handler) as client:
            await refresh(Credential("anthropic", "at", refresh_token="rt"), client=client)

        assert seen[0]["anthropic-beta"] == "oauth-2025-04-20"
        assert "userOAuthProvider" in seen[0]["user-agent"]

    async def test_refresh_error_propagates_description(self) -> None:
        resposta = httpx.Response(
            400,
            json={"error": "invalid_grant", "error_description": "Refresh token expired"},
        )
        async with client_of(lambda _: resposta) as client:
            with pytest.raises(OAuthError) as excinfo:
                await refresh(Credential("anthropic", "at", refresh_token="rt"), client=client)
        assert "Refresh token expired" in str(excinfo.value)

    async def test_without_refresh_token_raises_before_the_network(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("não devia haver pedido")

        async with client_of(handler) as client:
            with pytest.raises(OAuthError):
                await refresh(Credential("anthropic", "at"), client=client)


class TestRefreshOwnership:
    """Regra do dono único: dois renovadores sobre o mesmo token de uso único produzem
    `invalid_grant` em ciclo e forçam re-login manual."""

    async def test_non_owner_store_refuses_without_touching_the_network(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("um store que não é dono não pode trocar o token")

        async with client_of(handler) as client:
            with pytest.raises(NotRefreshOwnerError):
                await refresh(
                    Credential("anthropic", "at", refresh_token="rt"),
                    client=client,
                    store=FakeStore(owns_refresh=False),
                )

    async def test_owner_store_refreshes(self) -> None:
        async with client_of(lambda _: token_response(refresh_token="rt-novo")) as client:
            nova = await refresh(
                Credential("anthropic", "at", refresh_token="rt"),
                client=client,
                store=FakeStore(owns_refresh=True),
            )
        assert nova.refresh_token == "rt-novo"

    async def test_without_store_the_caller_owns_it(self) -> None:
        """Sem store não há quem verificar; quem chama assume a responsabilidade."""
        async with client_of(lambda _: token_response(refresh_token="rt-novo")) as client:
            nova = await refresh(Credential("anthropic", "at", refresh_token="rt"), client=client)
        assert nova.access_token == "at-novo"


TOKEN_URL = "https://oauth2.googleapis.com/token"
LOAD_URL = "https://daily-cloudcode-pa.googleapis.com/v1internal:loadCodeAssist"
ONBOARD_URL = "https://daily-cloudcode-pa.googleapis.com/v1internal:onboardUser"


class Antigravity:
    """Backend do Antigravity por guião: token, depois `loadCodeAssist`/`onboardUser`."""

    def __init__(self, *steps: dict[str, Any], token: dict[str, Any] | None = None) -> None:
        self.steps = list(steps)
        self.token = token or {
            "access_token": "at-novo",
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
            raise AssertionError(f"pedido a mais: {url}")
        return httpx.Response(200, json=self.steps.pop(0))


@pytest.fixture(autouse=True)
def relogio_virtual(monkeypatch: pytest.MonkeyPatch) -> None:
    """Espera nenhuma, mas tempo que passa.

    O polling do ``onboardUser`` espera 1s por volta e o prazo é absoluto. Substituir a
    espera por um no-op sem mexer no relógio tornaria o prazo inalcançável: o teste do
    desistir ficaria a girar para sempre. Aqui cada espera adianta o relógio monotónico
    exactamente o que teria esperado — o prazo é medido, a suite não espera.
    """
    decorrido = 0.0

    async def dormir(seconds: float) -> None:
        nonlocal decorrido
        decorrido += seconds

    class Relogio:
        """Shim em vez de `monkeypatch` no módulo `time`: mexer no `time` global afecta
        o pytest e qualquer teste a correr ao lado."""

        @staticmethod
        def monotonic() -> float:
            return time.monotonic() + decorrido

        @staticmethod
        def time() -> float:
            return time.time()

    monkeypatch.setattr(oauth, "_sleep", dormir)
    monkeypatch.setattr(oauth, "time", Relogio)


class TestProjectDiscovery:
    async def test_project_id_comes_from_load_code_assist(self) -> None:
        """Não vem do token nem do ambiente: é o `cloudaicompanionProject` que o
        `loadCodeAssist` devolve. Sem ele todos os pedidos de inferência dão 400."""
        backend = Antigravity(
            {
                "cloudaicompanionProject": "projecto-1",
                "currentTier": {"id": "free-tier"},
                "allowedTiers": [{"id": "free-tier"}],
                "paidTier": {"id": "pago"},
            },
            {
                "cloudaicompanionProject": "projecto-1",
                "currentTier": {"id": "free-tier"},
                "allowedTiers": [{"id": "free-tier"}],
                "paidTier": {"id": "pago"},
            },
        )
        async with client_of(backend) as client:
            credencial = await complete("google-antigravity", request_for(), "abc", client=client)

        assert credencial.project_id == "projecto-1"
        assert ONBOARD_URL not in backend.urls, "a conta já tinha tier; não se provisiona"

    async def test_account_without_tier_is_onboarded_first(self) -> None:
        """Sem `currentTier` a conta ainda não existe no Cloud Code Assist; saltar o
        `onboardUser` deixaria a ligação sem projecto nenhum."""
        backend = Antigravity(
            {"allowedTiers": [{"id": "free-tier"}], "paidTier": {"id": "pago"}},
            {"done": True, "response": {"@type": "OnboardUserResponse"}},
            {
                "cloudaicompanionProject": "projecto-novo",
                "currentTier": {"id": "free-tier"},
                "allowedTiers": [{"id": "free-tier"}],
                "paidTier": {"id": "pago"},
            },
        )
        async with client_of(backend) as client:
            credencial = await complete("google-antigravity", request_for(), "abc", client=client)

        assert credencial.project_id == "projecto-novo"
        assert ONBOARD_URL in backend.urls

    async def test_onboard_operation_is_polled_until_done(self) -> None:
        backend = Antigravity(
            {"allowedTiers": [{"id": "free-tier"}], "paidTier": {"id": "pago"}},
            {"name": "operations/1", "done": False},
            {"name": "operations/1", "done": True, "response": {"@type": "OnboardUserResponse"}},
            {
                "cloudaicompanionProject": "projecto-novo",
                "currentTier": {"id": "free-tier"},
                "allowedTiers": [{"id": "free-tier"}],
                "paidTier": {"id": "pago"},
            },
        )
        async with client_of(backend) as client:
            credencial = await complete("google-antigravity", request_for(), "abc", client=client)

        assert credencial.project_id == "projecto-novo"
        assert "https://daily-cloudcode-pa.googleapis.com/v1internal/operations/1" in backend.urls

    async def test_failed_onboard_operation_propagates_the_reason(self) -> None:
        backend = Antigravity(
            {"allowedTiers": [{"id": "free-tier"}], "paidTier": {"id": "pago"}},
            {"done": True, "error": {"code": 7, "message": "quota de projectos esgotada"}},
        )
        async with client_of(backend) as client:
            with pytest.raises(OAuthError) as excinfo:
                await complete("google-antigravity", request_for(), "abc", client=client)
        assert "quota de projectos esgotada" in str(excinfo.value)

    async def test_ineligible_account_reports_the_validation_url(self) -> None:
        """A conta precisa de uma acção no browser; a URL é o passo seguinte e não pode
        ser engolida por uma mensagem nossa."""
        backend = Antigravity(
            {
                "allowedTiers": [{"id": "pago"}],
                "ineligibleTiers": [
                    {
                        "tierId": "free-tier",
                        "reasonMessage": "Conta precisa de validação",
                        "validationUrl": "https://valida.example/conta",
                    }
                ],
                "paidTier": {"id": "pago"},
            }
        )
        async with client_of(backend) as client:
            with pytest.raises(OAuthError) as excinfo:
                await complete("google-antigravity", request_for(), "abc", client=client)

        assert "Conta precisa de validação" in str(excinfo.value)
        assert "https://valida.example/conta" in str(excinfo.value)

    async def test_missing_project_raises_instead_of_inventing_one(self) -> None:
        """Princípio 3: nunca fabricar um valor plausível. Um projecto inventado daria 400
        em cada pedido, sem dizer porquê."""
        backend = Antigravity(
            {
                "currentTier": {"id": "free-tier"},
                "allowedTiers": [{"id": "free-tier"}],
                "paidTier": {"id": "pago"},
            },
            {
                "currentTier": {"id": "free-tier"},
                "allowedTiers": [{"id": "free-tier"}],
                "paidTier": {"id": "pago"},
            },
        )
        async with client_of(backend) as client:
            with pytest.raises(OAuthError) as excinfo:
                await complete("google-antigravity", request_for(), "abc", client=client)
        assert "cloudaicompanionProject" in str(excinfo.value)

    async def test_token_without_refresh_is_rejected(self) -> None:
        """Uma ligação sem refresh token morre na primeira expiração, e o utilizador não
        saberia porquê. Recusar aqui é o único momento em que ainda dá para repetir."""
        backend = Antigravity(token={"access_token": "at", "expires_in": 3600})
        async with client_of(backend) as client:
            with pytest.raises(OAuthError) as excinfo:
                await complete("google-antigravity", request_for(), "abc", client=client)
        assert "refresh" in str(excinfo.value).lower()
        assert LOAD_URL not in backend.urls, "não se descobre projecto para uma ligação morta"

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
        """O endpoint de token não devolve o projecto; perdê-lo aqui partiria todos os
        pedidos seguintes com um 400 que não aponta para a renovação."""
        velha = Credential(
            "google-antigravity", "at-velho", refresh_token="rt-velho", project_id="projecto-1"
        )
        async with client_of(lambda _: token_response(refresh_token="rt-novo")) as client:
            nova = await refresh(velha, client=client)

        assert nova.project_id == "projecto-1"
        assert nova.refresh_token == "rt-novo"

    async def test_credential_without_project_refuses_to_refresh(self) -> None:
        """`require "projectId"` da regra do OMP: renovar produziria um token válido e
        inútil, e o erro só apareceria muito depois."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("não devia haver pedido")

        async with client_of(handler) as client:
            with pytest.raises(OAuthError) as excinfo:
                await refresh(
                    Credential("google-antigravity", "at", refresh_token="rt"), client=client
                )
        assert "project_id" in str(excinfo.value)

    async def test_refresh_sends_the_client_secret(self) -> None:
        """O Google não aceita o grant sem segredo do cliente; sem ele devolve
        `invalid_client` e a subscrição fica presa."""
        seen: list[dict[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(form_of(request))
            return token_response()

        async with client_of(handler) as client:
            await refresh(
                Credential("google-antigravity", "at", refresh_token="rt", project_id="projecto-1"),
                client=client,
            )
        assert seen[0]["client_secret"].startswith("GOCSPX-")


class TestSurvivorsClosed:
    """Casos que uma primeira volta de mutação mostrou estarem por defender."""

    async def test_paid_tier_is_probed_with_the_project_in_the_body(self) -> None:
        """Sem o `cloudaicompanionProject` no corpo o backend responde só com o tier
        corrente, e a conta paga passa por gratuita. A segunda chamada não é redundante."""
        conta = {
            "cloudaicompanionProject": "projecto-1",
            "currentTier": {"id": "free-tier"},
            "allowedTiers": [{"id": "free-tier"}],
        }
        completa = {**conta, "paidTier": {"id": "pago"}}
        backend = Antigravity(conta, completa, conta, completa)
        async with client_of(backend) as client:
            await complete("google-antigravity", request_for(), "abc", client=client)

        com_projecto = [b for b in backend.bodies if "cloudaicompanionProject" in b]
        assert com_projecto, "a segunda chamada tem de reenviar o projecto descoberto"
        assert com_projecto[0]["cloudaicompanionProject"] == "projecto-1"

    async def test_onboard_that_never_finishes_gives_up(self) -> None:
        """`done: false` para sempre tem de falhar; prender a ligação seria pior que um
        erro, porque nada indicaria ao utilizador que parou."""
        pendente = {"name": "operations/1", "done": False}
        backend = Antigravity(
            {"allowedTiers": [{"id": "free-tier"}], "paidTier": {"id": "pago"}},
            *[dict(pendente) for _ in range(200)],
        )
        async with client_of(backend) as client:
            with pytest.raises(OAuthError) as excinfo:
                await complete("google-antigravity", request_for(), "abc", client=client)
        assert "onboardUser" in str(excinfo.value)

    async def test_onboard_done_without_response_is_a_failure(self) -> None:
        """`done: true` sem `response` é o que o backend devolve quando o provisionamento
        não se concretizou; tratá-lo como sucesso deixaria a conta sem projecto."""
        backend = Antigravity(
            {"allowedTiers": [{"id": "free-tier"}], "paidTier": {"id": "pago"}},
            {"done": True},
        )
        async with client_of(backend) as client:
            with pytest.raises(OAuthError) as excinfo:
                await complete("google-antigravity", request_for(), "abc", client=client)
        assert "sem resposta" in str(excinfo.value)

    async def test_google_error_object_is_not_flattened_to_raw_json(self) -> None:
        """O Google embrulha a explicação em `error.message`. Devolver o JSON cru é
        tecnicamente honesto e praticamente inútil: o utilizador não a encontra."""
        resposta = httpx.Response(
            400,
            json={"error": {"status": "INVALID_ARGUMENT", "message": "redirect_uri inválido"}},
        )
        async with client_of(lambda _: resposta) as client:
            with pytest.raises(OAuthError) as excinfo:
                await complete("google-antigravity", request_for(), "abc", client=client)
        assert "INVALID_ARGUMENT: redirect_uri inválido" in str(excinfo.value)

    async def test_eligible_account_ignores_a_stale_ineligibility(self) -> None:
        """`allowedTiers` com free-tier ganha: uma entrada residual em `ineligibleTiers`
        não pode bloquear uma conta que o backend já autoriza."""
        conta = {
            "cloudaicompanionProject": "projecto-1",
            "currentTier": {"id": "free-tier"},
            "allowedTiers": [{"id": "free-tier"}],
            "paidTier": {"id": "pago"},
            "ineligibleTiers": [{"tierId": "free-tier", "reasonMessage": "residual"}],
        }
        backend = Antigravity(conta, conta)
        async with client_of(backend) as client:
            credencial = await complete("google-antigravity", request_for(), "abc", client=client)
        assert credencial.project_id == "projecto-1"

    async def test_paste_of_only_whitespace_never_reaches_the_network(self) -> None:
        """Um paste vazio como código produziria um 400 opaco do provedor em vez de dizer
        ao utilizador que não colou nada."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("não devia haver pedido")

        async with client_of(handler) as client:
            with pytest.raises(OAuthError) as excinfo:
                await complete("anthropic", request_for(), "\n\t ", client=client)
        assert "nada colado" in str(excinfo.value)

    async def test_fragment_without_a_code_is_rejected(self) -> None:
        """`#ST4TE` sozinho passa a verificação do `state` mas não tem código. Deixá-lo
        seguir mandaria `code=""` ao provedor."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("não devia haver pedido")

        async with client_of(handler) as client:
            with pytest.raises(OAuthError) as excinfo:
                await complete("anthropic", request_for(), "#ST4TE", client=client)
        assert "código" in str(excinfo.value)
