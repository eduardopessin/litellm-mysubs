"""O que a UI faz, sem saber que é uma UI.

Separado de `app.py` de propósito: aqui está o fluxo — começar a ligação, trocar o código
colado, descobrir modelos, aplicar — e lá está o HTTP. Testar isto não precisa de cliente
web, e o dia em que houver um CLI ele chama estas funções sem passar por FastAPI.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from ..catalog.deployments import to_deployments
from ..catalog.discovery import DiscoveredModel, discover
from ..credentials import oauth
from ..credentials.store import PROVIDER_IDS, Credential, CredentialStore, ProviderId
from ..registry import ModelRegistry, RouterLike

#: Nome legível de cada provedor, para os cards.
PROVIDER_LABELS: dict[ProviderId, str] = {
    "anthropic": "Claude Max",
    "openai-codex": "ChatGPT Plus (Codex)",
    "google-antigravity": "Google Antigravity",
}


class RouterSource(Protocol):
    """De onde vem o Router em runtime.

    O proxy preenche o seu `llm_router` no arranque, depois de a sub-app estar montada.
    Capturá-lo na montagem guardaria `None` para sempre; por isso pede-se quando é preciso.
    """

    def __call__(self) -> RouterLike | None: ...


@dataclass(frozen=True, slots=True)
class ProviderCard:
    """Estado de um provedor, como o utilizador o vê."""

    provider: ProviderId
    label: str
    connected: bool
    expires_in_s: float | None
    #: `True` quando há credencial mas o access token já passou da validade.
    stale: bool
    project_id: str = ""
    applied: int = 0


@dataclass(slots=True)
class PendingAuth:
    """Uma ligação começada e à espera do paste.

    Guarda-se o `verifier` do PKCE e o `state`: sem eles a troca não fecha. Vive em memória
    porque é efémero — um reinício do proxy a meio de uma ligação obriga a recomeçar, que é
    melhor do que persistir um segredo de curta duração em disco.
    """

    request: oauth.AuthRequest
    started_at: float


@dataclass(slots=True)
class MySubsService:
    """O estado e as operações da página."""

    store: CredentialStore
    router_source: RouterSource
    client_factory: Any = httpx.AsyncClient
    pending: dict[ProviderId, PendingAuth] = field(default_factory=dict)
    discovered: dict[ProviderId, list[DiscoveredModel]] = field(default_factory=dict)
    #: Escolha do utilizador por provedor, retida entre aplicações: o `apply` do registry
    #: substitui **todas** as entradas do plugin, por isso aplicar um provedor tem de
    #: reinjectar o que já estava escolhido nos outros.
    selected: dict[ProviderId, list[str]] = field(default_factory=dict)

    # -- cards -----------------------------------------------------------------

    def cards(self, *, now: float | None = None) -> list[ProviderCard]:
        """Um card por provedor, ligado ou não.

        Mostra-se sempre os três: um provedor por ligar é informação, não ausência dela —
        é o que diz ao utilizador o que pode acrescentar.
        """
        moment = time.time() if now is None else now
        applied = self._applied_counts()
        cards: list[ProviderCard] = []
        for provider in PROVIDER_IDS:
            credential = self._safe_get(provider)
            cards.append(
                ProviderCard(
                    provider=provider,
                    label=PROVIDER_LABELS[provider],
                    connected=credential is not None,
                    expires_in_s=(
                        None
                        if credential is None or credential.expires_at <= 0
                        else credential.expires_at - moment
                    ),
                    stale=credential is not None and credential.is_expired(now=moment),
                    project_id="" if credential is None else credential.project_id,
                    applied=applied.get(provider, 0),
                )
            )
        return cards

    def _applied_counts(self) -> dict[ProviderId, int]:
        router = self.router_source()
        if router is None:
            return {}
        counts: dict[ProviderId, int] = {}
        for deployment in ModelRegistry(router=router).managed():
            info = deployment.get("model_info") or {}
            provider = info.get("mysubs_provider")
            if provider in PROVIDER_IDS:
                counts[provider] = counts.get(provider, 0) + 1
        return counts

    def _safe_get(self, provider: ProviderId) -> Credential | None:
        """Uma credencial ilegível não pode derrubar a página inteira.

        `FileCredentialStore` recusa-se a ler um ficheiro com permissões largas — é a
        decisão certa, mas aqui traduz-se em "não ligado" com o card ainda visível, em vez
        de um 500 que esconde os outros provedores.
        """
        try:
            return self.store.get(provider)
        except Exception:
            return None

    def _require(self, provider: ProviderId) -> Credential:
        """A credencial, ou um erro que nomeia o passo em falta.

        `store.get` devolve `None` para "não ligado"; deixá-lo seguir dava um
        `AttributeError` longe da causa.
        """
        credential = self.store.get(provider)
        if credential is None:
            raise LookupError(f"{PROVIDER_LABELS[provider]} não está ligado")
        return credential

    # -- ligação ---------------------------------------------------------------

    def begin(self, provider: ProviderId) -> oauth.AuthRequest:
        """Passo 4: devolve o URL para onde mandar o utilizador."""
        request = oauth.begin(provider)
        self.pending[provider] = PendingAuth(request=request, started_at=time.time())
        return request

    async def complete(self, provider: ProviderId, pasted: str) -> Credential:
        """Passo 4: fecha a ligação com o que o utilizador colou.

        Aceita a URL de retorno inteira, o código nu ou `código#state` — a normalização é
        do `oauth.complete`, não do utilizador.
        """
        waiting = self.pending.get(provider)
        if waiting is None:
            raise LookupError(
                f"não há ligação a decorrer para {PROVIDER_LABELS[provider]}: "
                "carrega em Conectar primeiro"
            )
        async with self.client_factory() as client:
            credential = await oauth.complete(provider, waiting.request, pasted, client=client)
        self.store.set(provider, credential)
        del self.pending[provider]
        return credential

    async def refresh(self, provider: ProviderId) -> Credential:
        """Passo 5: renova e guarda a credencial rodada."""
        credential = self._require(provider)
        async with self.client_factory() as client:
            renewed = await oauth.refresh(credential, client=client, store=self.store)
        self.store.set(provider, renewed)
        return renewed

    # -- modelos ---------------------------------------------------------------

    async def discover(self, provider: ProviderId) -> list[DiscoveredModel]:
        """Passo 6: lista o que a subscrição serve."""
        credential = self._require(provider)
        async with self.client_factory() as client:
            found = await discover(credential, client=client)
        self.discovered[provider] = found
        return found

    def apply(self, provider: ProviderId, chosen: list[str]) -> int:
        """Passo 6: injecta no Router os modelos escolhidos.

        `chosen` são `suggested_name`s. Um nome que não veio da descoberta é recusado em vez
        de ser inventado — aplicar um modelo que a subscrição não serve produziria
        exactamente os 400 que motivaram este pacote.

        Aplica-se por provedor mas injecta-se tudo o que está seleccionado: o `apply` do
        registry substitui as entradas do plugin, e passar só as de um provedor apagaria as
        dos outros.
        """
        router = self.router_source()
        if router is None:
            raise RuntimeError(
                "o Router do LiteLLM ainda não está pronto — o proxy só o cria no arranque"
            )
        found = self.discovered.get(provider)
        if not found:
            raise LookupError("descobre os modelos antes de aplicar")

        by_name = {model.suggested_name: model for model in found}
        unknown = [name for name in chosen if name not in by_name]
        if unknown:
            raise LookupError(f"modelos que a descoberta não devolveu: {', '.join(unknown)}")

        self.selected[provider] = list(chosen)
        deployments: list[dict[str, Any]] = []
        for other, names in self.selected.items():
            models = self.discovered.get(other) or []
            wanted = {name for name in names}
            deployments += to_deployments(
                [model for model in models if model.suggested_name in wanted], other
            )
        return ModelRegistry(router=router).apply(deployments)
