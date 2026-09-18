"""Fluxo OAuth com paste e renovação de token, para os três provedores.

Porque é que o paste é a via principal, e não o plano B: o fluxo nativo destes clientes
prende um servidor de callback em ``localhost:1455`` (Codex), ``:54545`` (Anthropic) ou
``:51121`` (Antigravity). Num LiteLLM em container ou em cluster o browser do utilizador
não alcança nenhum desses portos. O que ele alcança é a caixa de texto da página. Portanto
o redirect URI continua a ser o que o provedor tem registado — muda-se onde o código é
lido, não para onde ele é enviado.

O que este módulo **não** faz: não abre sockets à espera de callbacks, não abre browsers e
não decide quando renovar. Recebe o que o utilizador colou, ou uma credencial a expirar, e
devolve uma ``Credential``.

Duas regras herdadas de incidentes medidos:

* **Um só dono do refresh** (ver ``store.py``). ``refresh()`` aceita um ``CredentialStore``
  opcional; se ele existir e não for dono, recusa-se a trocar em vez de correr a corrida.
* **Rotação substitui.** Anthropic e OpenAI emitem refresh tokens de uso único. Guardar o
  antigo depois de o provedor devolver um novo é garantir ``invalid_grant`` na renovação
  seguinte — por isso o token novo substitui, e só quando o provedor não roda é que o
  anterior é preservado.

Erros do provedor sobem com o corpo real. Um ``error_description`` do upstream é a única
coisa que diz ao utilizador o que fazer a seguir; trocá-lo por uma mensagem genérica
apaga-a.
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
    "complete",
    "refresh",
]


class OAuthError(RuntimeError):
    """Falha do fluxo OAuth, com o estado e o corpo **reais** do upstream.

    ``status`` é zero quando a falha é local (paste inválido, ``state`` trocado) e não
    houve resposta do provedor.
    """

    __slots__ = ("body", "provider", "status")

    def __init__(self, provider: str, message: str, *, status: int = 0, body: str = "") -> None:
        super().__init__(message)
        self.provider = provider
        self.status = status
        self.body = body


class NotRefreshOwnerError(OAuthError):
    """Renovação pedida a partir de um store que não é dono do refresh."""


# omp: registry/oauth/anthropic-constants.ts :: ANTHROPIC_OAUTH_GRANT_TTL_MS
#: Vida absoluta do grant da Anthropic, ancorada no login interactivo. A rotação **não** a
#: estende: ~30 dias depois o endpoint devolve ``invalid_grant`` para o token mais recente
#: e só um login novo recupera a conta. É heurística de aviso, não contrato de fio.
ANTHROPIC_GRANT_TTL_S: Final = 30 * 24 * 60 * 60.0

# omp: providers/claude-code-fingerprint.ts :: claudeCodeSdkVersion
#: Vai no ``User-Agent`` da renovação da Anthropic. O Claude Code manda estes cabeçalhos na
#: renovação mas não na troca inicial do código.
CLAUDE_CODE_SDK_VERSION: Final = "0.112.1"

# omp: wire/gemini-headers.ts :: getAntigravityUserAgent
#: ``User-Agent`` do plano de controlo do Antigravity (``loadCodeAssist``/``onboardUser``).
#: O backend não valida o ``cl``; só a versão faz gating. Repetido aqui em vez de importado
#: de ``plugin.py`` porque esse módulo arrasta o LiteLLM inteiro, e descobrir um projecto
#: não precisa dele.
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
_FREE_TIER_ID: Final = "free-tier"

# omp: registry/oauth/google-antigravity.ts :: ONBOARD_TIMEOUT_MS, ONBOARD_POLL_INTERVAL_MS
_ONBOARD_TIMEOUT_S: Final = 30.0
_ONBOARD_POLL_INTERVAL_S: Final = 1.0


@dataclass(frozen=True, slots=True)
class _Provider:
    """O que distingue um provedor do outro no fluxo de código de autorização."""

    client_id: str
    authorize_url: str
    token_url: str
    scopes: tuple[str, ...]
    redirect_uri: str
    pkce: bool
    authorize_params: tuple[tuple[str, str], ...]
    token_body: Literal["json", "form"]
    #: Parâmetros extra na troca do código. ``{state}`` é substituído pelo state validado.
    exchange_params: tuple[tuple[str, str], ...] = ()
    refresh_headers: tuple[tuple[str, str], ...] = ()
    client_secret: str = ""
    #: Margem subtraída ao ``expires_in``, para renovar antes de o token morrer.
    expiry_skew_s: float = 0.0


# As três fichas vêm das regras declarativas do OMP (`compat/rules/auth/<provedor>.kdl`),
# que é onde os client ids, URLs e parâmetros extra vivem de facto. Os nós KDL citados nos
# comentários — `authorize-url`, `token url`, `authorize-params`, `callback` — não cabem
# numa âncora (o verificador não aceita hífens em símbolos), por isso a âncora aponta para
# um símbolo literal do mesmo ficheiro e o nó fica nomeado aqui.

# omp: registry/oauth/openai-codex.ts :: CLIENT_ID, AUTHORIZE_URL, TOKEN_URL, SCOPE
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
    # A OpenAI só autoriza este URI exacto. Um porto ocupado tem de falhar, não cair para
    # outro: o `port-fallback=#false` da regra diz isso, e no paste traduz-se em o URI ser
    # fixo em vez de derivado do servidor que não chegámos a abrir.
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
    # Client id público, guardado em base64 na regra para os scanners de segredos não
    # dispararem. Descodificado aqui porque um literal base64 no código seria pior: uma
    # divergência passaria despercebida.
    client_id=base64.b64decode("OWQxYzI1MGEtZTYxYi00NGQ5LTg4ZWQtNTk0NGQxOTYyZjVl").decode(),
    authorize_url="https://claude.ai/oauth/authorize",
    token_url="https://api.anthropic.com/v1/oauth/token",
    # `user:inference` é o que dá inferência directa com token OAuth. O endpoint da
    # `platform.claude.com` só emite tokens de consola; tem de ser o `claude.ai`.
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
    # `code=true` é o que torna o paste viável: em vez de redireccionar, a página mostra um
    # código copiável (no formato `codigo#state`). Sem isto o utilizador ficava dependente
    # de o browser alcançar o porto local.
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

# O client id e o segredo estão em base64 na regra do OMP; as âncoras apontam para um
# prefixo de cada um, que é o que cabe numa linha sem partir a verificação por substring.
# omp: compat/rules/auth/google-antigravity.kdl :: access_type, prompt, scopes, callback
# omp: compat/rules/auth/google-antigravity.kdl :: MTA3MTAwNjA2MDU5MS10bWhzc2lu
# omp: compat/rules/auth/google-antigravity.kdl :: R09DU1BYLUs1OEZXUjQ4NkxkTEoxbUxCOHNY
# omp: providers/google-auth.ts :: OAUTH_TOKEN_URL
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
    # A regra do OMP não activa PKCE aqui; o segredo do cliente faz esse papel.
    pkce=False,
    # Sem estes dois o Google devolve só um access token: `access_type=offline` é o que
    # pede o refresh token, e `prompt=consent` é o que o volta a emitir numa reautorização
    # de uma conta que já consentiu — sem ele a segunda ligação fica sem renovação.
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
    """O que fica do lado de cá enquanto o utilizador autentica no browser.

    O ``verifier`` tem de sobreviver até ao paste: sem ele a troca do código falha, porque
    o provedor só confirma o desafio contra o segredo que nunca viajou.
    """

    url: str
    """Para onde mandar o utilizador."""

    state: str
    """Opaco, devolvido no retorno; é o que liga o paste a este pedido."""

    verifier: str
    """PKCE. Vazio nos provedores que não o usam (Antigravity)."""


# omp: registry/oauth/pkce.ts :: generatePKCE
def _pkce() -> tuple[str, str]:
    """``(verifier, challenge)``. 96 bytes aleatórios em base64url, desafio S256.

    O ``base64url`` sem padding não é cosmético: o ``=`` teria de ser escapado no URL de
    autorização e há servidores que comparam o desafio byte a byte com o que receberam.
    """
    verifier = _b64url(secrets.token_bytes(96))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


# omp: registry/oauth/callback-server.ts :: generateState
def _state() -> str:
    """16 bytes em hexadecimal, como o ``generateState`` do OMP."""
    return secrets.token_bytes(16).hex()


# omp: registry/engine/oauth-code.ts :: generateAuthUrl
def begin(provider: ProviderId) -> AuthRequest:
    """Monta o URL de autorização e o segredo que o paste vai precisar.

    Não abre browser nem servidor: quem chama decide como mostrar o ``url``.
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
    if spec is None:  # pragma: no cover - `ProviderId` é fechado; defesa para chamadas soltas
        raise OAuthError(provider, f"provedor desconhecido: {provider!r}")
    return spec


# omp: registry/oauth/callback-server.ts :: parseNativeCallback
# omp: registry/engine/oauth-code.ts :: exchangeToken
def _parse_paste(provider: ProviderId, request: AuthRequest, pasted: str) -> str:
    """Extrai o código de autorização do que o utilizador colou.

    Três formas, todas aceites porque as três aparecem em produção:

    * a URL de retorno inteira — ``https://…/callback?code=abc&state=xyz``. É o que o
      browser mostra na barra quando o redirect falha por não alcançar o porto local;
    * ``codigo#state`` — o que a Anthropic mostra na página quando o fluxo é pedido com
      ``code=true``. O fragmento é o ``state``, não parte do código;
    * o código nu, para quem só copiou esse pedaço.

    O ``state`` é verificado sempre que vem no paste. Quando não vem (código nu) não há
    nada para comparar — e recusar aí só empurraria o utilizador a colar outra coisa. O
    ``state`` do pedido segue na troca de qualquer modo, portanto o servidor continua a ser
    a autoridade.

    Um ``error`` no retorno ganha à ausência de código: o ``error_description`` do provedor
    diz o que correu mal, e transformá-lo em "código em falta" apagava-o.
    """
    text = pasted.strip()
    if not text:
        raise OAuthError(provider, "nada colado: esperava a URL de retorno ou o código")

    if text.startswith(("http://", "https://")):
        return _parse_callback_url(provider, request, text)

    # `codigo#state`: o fragmento é a autoridade sobre o state, como no OMP.
    code, separator, fragment = text.partition("#")
    if separator and fragment:
        _check_state(provider, request, fragment)
    if not code:
        raise OAuthError(provider, "o texto colado não contém um código de autorização")
    return code


def _parse_callback_url(provider: ProviderId, request: AuthRequest, text: str) -> str:
    try:
        parsed = urllib.parse.urlparse(text)
    except ValueError as exc:
        raise OAuthError(provider, f"a URL colada não é válida: {exc}") from exc
    query = urllib.parse.parse_qs(parsed.query)

    if state := _first(query, "state"):
        _check_state(provider, request, state)

    if error := _first(query, "error"):
        description = _first(query, "error_description") or error
        raise OAuthError(provider, f"autorização recusada: {description}")

    # `authCode` é a variante que alguns clientes nativos devolvem em vez de `code`.
    code = _first(query, "code") or _first(query, "authCode")
    if not code:
        raise OAuthError(provider, "a URL colada não traz `code`; colaste a página certa?")
    # A Anthropic chega a devolver `code=abc#state` dentro da própria query.
    return code.partition("#")[0]


def _first(query: Mapping[str, list[str]], key: str) -> str:
    values = query.get(key)
    return values[0] if values else ""


def _check_state(provider: ProviderId, request: AuthRequest, received: str) -> None:
    """Um ``state`` diferente do emitido é o sinal de CSRF; nunca se ignora."""
    if received != request.state:
        raise OAuthError(
            provider,
            f"`state` não corresponde (esperado {request.state!r}, colado {received!r}); "
            f"o retorno pertence a outro pedido de autenticação — recomeça o fluxo",
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
    """POST ao endpoint de token; devolve o JSON ou levanta com o corpo real.

    O corpo da resposta sobe intacto mesmo quando o estado é 200 mas o JSON não presta:
    há provedores que embrulham o erro num envelope de sucesso, e é lá dentro que está a
    única explicação.
    """
    # `json` e `form` não são intermutáveis: a Anthropic recusa o corpo urlencoded e o
    # Google recusa o JSON. A regra de cada provedor diz qual é.
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
            f"{what} falhou: HTTP {response.status_code}: {_describe(payload, body)}",
            status=response.status_code,
            body=body,
        )
    if not isinstance(payload, dict):
        raise OAuthError(
            provider,
            f"{what}: resposta do token não é um objecto JSON: {body[:500]}",
            status=response.status_code,
            body=body,
        )
    return payload


def _describe(payload: Any, body: str) -> str:
    """Mensagem do upstream, preferindo o ``error_description`` ao corpo cru.

    Nunca substitui: quando não há campo reconhecível devolve o corpo, truncado. O que não
    acontece é fabricar uma explicação genérica por cima de uma real.
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
    """Troca o que o utilizador colou por uma credencial utilizável.

    ``pasted`` aceita a URL de retorno inteira, ``codigo#state`` ou o código nu — extrair e
    validar é responsabilidade desta função, não do utilizador.

    No Antigravity a credencial só fica completa depois de descobrir o projecto: o
    ``project_id`` não vem do token nem de uma variável de ambiente, vem do
    ``loadCodeAssist``. Sem ele nenhum pedido de inferência passa, por isso a descoberta
    faz parte da ligação e não de um passo posterior.
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
        provider, spec, params, client=client, what="a troca do código de autorização"
    )
    credential = _credential(provider, spec, payload, previous=None)

    if provider == "google-antigravity":
        # A regra do OMP recusa aqui um token sem refresh: sem ele a subscrição morre na
        # primeira expiração e o utilizador teria de refazer tudo sem saber porquê.
        if not credential.refresh_token:
            raise OAuthError(
                provider,
                "o Google não devolveu refresh token; repete a autorização "
                "(é o que `access_type=offline` e `prompt=consent` pedem)",
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
    """Projecta a resposta do token numa ``Credential``.

    A rotação é a parte que importa: se o provedor devolve ``refresh_token``, é esse que
    fica. Só na ausência dele — provedores que não rodam — se preserva o anterior. Guardar
    o antigo por cima de um novo é o caminho directo para ``invalid_grant`` na renovação
    seguinte.
    """
    access = payload.get("access_token")
    if not isinstance(access, str) or not access:
        excerpt = str(dict(payload))[:500]
        raise OAuthError(provider, f"resposta do token sem access token: {excerpt}")

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
    """Renova o access token, devolvendo uma ``Credential`` nova.

    **Dono único.** Se ``store`` for dado e ``store.owns_refresh`` for falso, levanta
    ``NotRefreshOwnerError`` sem tocar na rede. Não é zelo: os refresh tokens da Anthropic e
    da OpenAI são de uso único, e dois renovadores sobre a mesma credencial invalidam a
    cópia um do outro, produzindo ``invalid_grant`` em ciclo até alguém voltar a fazer
    login à mão. Um store que não é dono lê e nunca troca — quem renova é o processo que
    possui a fonte de verdade. Sem ``store``, quem chama assume essa responsabilidade.

    O ``project_id`` do Antigravity sobrevive à renovação: é fixado no login e o endpoint de
    token não o devolve. Perdê-lo aqui partiria todos os pedidos seguintes.
    """
    provider = credential.provider
    spec = _spec(provider)

    if store is not None and not store.owns_refresh:
        raise NotRefreshOwnerError(
            provider,
            f"{type(store).__name__} não é dono do refresh ({provider}); "
            f"renovar daqui invalidaria o token do dono. Renova no processo que o possui.",
        )
    if not credential.refresh_token:
        raise OAuthError(provider, f"{provider} não tem refresh token; liga a subscrição de novo")
    if provider == "google-antigravity" and not credential.project_id:
        # O `require "projectId"` da regra do OMP: uma credencial sem projecto renova mas
        # não serve para nada, e o erro apareceria muito depois, num 400 de inferência.
        raise OAuthError(provider, f"{provider} não tem project_id; liga a subscrição de novo")

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
        what="a renovação do token",
    )
    return _credential(provider, spec, payload, previous=credential)


# omp: registry/oauth/google-antigravity.ts :: discoverProject
async def _discover_project(access_token: str, *, client: httpx.AsyncClient) -> str:
    """Descobre o ``cloudaicompanionProject`` da conta.

    Três passos, pela mesma ordem do OMP: perguntar o estado, provisionar o free tier se a
    conta ainda não tem um, e voltar a perguntar. O segundo ``loadCodeAssist`` não é
    redundante — é ele que devolve o projecto que o ``onboardUser`` acabou de criar.
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
        "google-antigravity", "o `loadCodeAssist` não devolveu um `cloudaicompanionProject`"
    )


# omp: registry/oauth/google-antigravity.ts :: loadCodeAssist
async def _load_code_assist(
    headers: Mapping[str, str], *, client: httpx.AsyncClient
) -> dict[str, Any]:
    """Estado da conta no Cloud Code Assist.

    A segunda chamada com ``cloudaicompanionProject`` explícito é o que faz o backend
    revelar o ``paidTier``: sem o projecto no corpo ele responde só com o tier corrente.
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
    """Recusa cedo uma conta que o backend marca como inelegível.

    Silêncio não é inelegibilidade: só quando há um ``ineligibleTiers`` com mensagem para o
    free tier é que se levanta. O ``validationUrl``, quando vem, é o passo seguinte do
    utilizador e segue na mensagem.
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
    """Indirecção deliberada: é o único ponto de espera real, e os testes substituem-no."""
    await asyncio.sleep(seconds)


# omp: registry/oauth/google-antigravity.ts :: onboardUser
async def _onboard_user(headers: Mapping[str, str], *, client: httpx.AsyncClient) -> None:
    """Provisiona o free tier, seguindo a operação de longa duração até ao fim.

    O prazo é absoluto, não por tentativa: uma operação que devolve ``done: false`` para
    sempre tem de falhar em vez de prender a ligação.
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
                    "google-antigravity", f"o `onboardUser` falhou: {_describe_operation(error)}"
                )
            if operation.get("response") is None:
                raise OAuthError("google-antigravity", "o `onboardUser` terminou sem resposta")
            return

        if time.monotonic() >= deadline:
            raise OAuthError(
                "google-antigravity",
                f"o `onboardUser` não terminou em {_ONBOARD_TIMEOUT_S:.0f}s",
            )
        name = operation.get("name")
        if not isinstance(name, str) or not name:
            raise OAuthError("google-antigravity", "o `onboardUser` devolveu uma operação sem nome")

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
    """Pedido ao plano de controlo. Qualquer estado que não seja 200 é erro, com corpo."""
    response = await client.request(method, url, headers=dict(headers), json=body)
    if response.status_code != 200:
        payload: Any = None
        try:
            payload = response.json()
        except ValueError:
            payload = None
        raise OAuthError(
            "google-antigravity",
            f"{url} devolveu HTTP {response.status_code}: {_describe(payload, response.text)}",
            status=response.status_code,
            body=response.text,
        )
    parsed = response.json()
    if not isinstance(parsed, dict):
        raise OAuthError("google-antigravity", f"{url} não devolveu um objecto JSON")
    return parsed
