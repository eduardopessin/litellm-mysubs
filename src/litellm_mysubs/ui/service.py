"""What the UI does, without knowing it is a UI.

Separate from `app.py` on purpose: the flow is here — start the connection, exchange the
pasted code, discover models, apply — and the HTTP is there. Testing this needs no web
client, and the day there is a CLI it calls these functions without going through FastAPI.
"""

from __future__ import annotations

import time
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from ..catalog.deployments import to_deployments
from ..catalog.discovery import DiscoveredModel, discover
from ..catalog.selection import SelectionStore
from ..catalog.usage import (
    QUOTA_SUMMARY_PATH,
    UsageSnapshot,
    from_antigravity_models,
    from_antigravity_summary,
    from_headers,
)
from ..credentials import oauth
from ..credentials.store import (
    PROVIDER_IDS,
    Credential,
    CredentialStore,
    ProviderId,
    from_payload,
)
from ..registry import ModelRegistry, RouterLike
from .pairing import Pairing, PairingRegistry

#: Age past which usage is requested again. The quota moves in minutes; asking on every
#: page load would be noise against the upstream for no gain at all.
USAGE_TTL_S = 120.0

#: Readable name of each provider, for the cards.
PROVIDER_LABELS: dict[ProviderId, str] = {
    "anthropic": "Anthropic Claude Code Subscription",
    "openai-codex": "OpenAI Codex Subscription",
    "google-antigravity": "Google Antigravity Subscription",
}


class RouterSource(Protocol):
    """Where the Router comes from at runtime.

    The proxy fills in its `llm_router` at startup, after the sub-app is mounted. Capturing
    it at mount time would store `None` forever; so it is asked for when it is needed.
    """

    def __call__(self) -> RouterLike | None: ...


@dataclass(frozen=True, slots=True)
class ProviderCard:
    """A provider's state, as the user sees it."""

    provider: ProviderId
    label: str
    connected: bool
    expires_in_s: float | None
    #: `True` when there is a credential but the access token is past its validity.
    stale: bool
    project_id: str = ""
    applied: int = 0
    #: Empty when the provider publishes no usage — `known` set to `False`. See
    #: `catalog/usage.py`.
    usage: UsageSnapshot = field(default_factory=UsageSnapshot)
    #: Whether the background refresher covers this token. False when the validity is
    #: unknown (`expires_at <= 0`) or the store does not own the refresh — in those two
    #: cases the loop **never** refreshes, and the manual refresh is the only way out. It
    #: is what decides whether the page shows the button: always offering it is noise,
    #: always hiding it leaves a card with no action at all exactly where action is
    #: needed.
    auto_renews: bool = True


@dataclass(frozen=True, slots=True)
class ApplyResult:
    """What changed in the Router after applying.

    `added` and `removed` are public names (already carrying the subscription prefix),
    because those are the ones the user will request the models by.
    """

    provider: ProviderId
    added: list[str]
    removed: list[str]
    total: int

    @property
    def changed(self) -> bool:
        return bool(self.added or self.removed)


@dataclass(slots=True)
class PendingAuth:
    """A connection started and waiting for the paste.

    The PKCE `verifier` and the `state` are kept: without them the exchange does not close.
    It lives in memory because it is ephemeral — restarting the proxy halfway through a
    connection forces a restart of the flow, which is better than persisting a short-lived
    secret to disk.
    """

    request: oauth.AuthRequest
    started_at: float


@dataclass(slots=True)
class MySubsService:
    """The page's state and operations."""

    store: CredentialStore
    router_source: RouterSource
    client_factory: Any = httpx.AsyncClient
    pending: dict[ProviderId, PendingAuth] = field(default_factory=dict)
    discovered: dict[ProviderId, list[DiscoveredModel]] = field(default_factory=dict)
    #: Last known usage per provider, updated on every response that carries headers.
    usage: dict[ProviderId, UsageSnapshot] = field(default_factory=dict)
    #: The user's choice per provider, retained between applications: the registry's
    #: `apply` replaces **every** plugin entry, so applying one provider has to reinject
    #: what was already chosen in the others.
    selected: dict[ProviderId, list[str]] = field(default_factory=dict)
    #: Pairing codes yet to be redeemed. It lives on the service and not in a global
    #: module because their validity is tied to this mounting: a restarted proxy must not
    #: honour a code issued by the previous instance.
    pairings: PairingRegistry = field(default_factory=PairingRegistry)
    #: The background refresher. Created late (in `start_refresher`) because it needs an
    #: event loop, and the service is built outside one.
    refresher: Any = None
    #: The callback server opened in this process, when the proxy runs on the same machine
    #: as the browser. Created late: it needs an event loop.
    local_flow: Any = None
    #: Model selection on disk. It is what survives a proxy restart: the Router is rebuilt
    #: from scratch at every startup, and without this the user would find the page again
    #: with the subscriptions connected and no model served.
    selections: SelectionStore = field(default_factory=lambda: SelectionStore())

    # -- background refresh ----------------------------------------------------

    def start_refresher(self) -> None:
        """Starts the periodic refresh. Idempotent.

        Without this the tokens only refreshed on a request path: a proxy with no traffic
        overnight woke up with every subscription expired, and the card said so correctly —
        but nobody had done anything to prevent it.
        """
        from ..credentials.refresher import BackgroundRefresher

        if self.refresher is None:
            self.refresher = BackgroundRefresher(self.store)
        self.refresher.start()

    async def stop_refresher(self) -> None:
        """Stops the periodic refresh. Silent if it never started."""
        if self.refresher is not None:
            await self.refresher.stop()

    # -- cards -----------------------------------------------------------------

    def cards(self, *, now: float | None = None) -> list[ProviderCard]:
        """One card per provider, connected or not.

        All three are always shown: a provider yet to be connected is information, not the
        absence of it — it is what tells the user what they can add.
        """
        moment = time.time() if now is None else now
        applied = self._applied_counts()
        # The same two conditions the `refresher` uses to give up. Read from the store and
        # from `Credential`, not copied as a constant: if the loop changes its criterion,
        # this card would stop telling the truth with nothing failing.
        owns = bool(getattr(self.store, "owns_refresh", False))
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
                    usage=self.usage.get(provider) or UsageSnapshot(),
                    auto_renews=(
                        owns and credential is not None and credential.expires_at > 0
                    ),
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
        """An unreadable credential cannot bring down the whole page.

        `FileCredentialStore` refuses to read a file with loose permissions — that is the
        right decision, but here it translates into "not connected" with the card still
        visible, instead of a 500 that hides the other providers.
        """
        try:
            return self.store.get(provider)
        except Exception:
            return None

    def _require(self, provider: ProviderId) -> Credential:
        """The credential, or an error that names the missing step.

        `store.get` returns `None` for "not connected"; letting that through gave an
        `AttributeError` far from the cause.
        """
        credential = self.store.get(provider)
        if credential is None:
            raise LookupError(f"{PROVIDER_LABELS[provider]} is not connected")
        return credential

    def observe(self, provider: ProviderId, headers: Any) -> UsageSnapshot:
        """Absorbs the headers of an upstream response.

        A subscription has no quota endpoint: the state only travels in the response
        headers. An empty snapshot does **not** replace the previous one — a response with
        no headers is no proof that usage changed, and clearing it would leave the card
        blinking between "no data" and the real value depending on the kind of request.
        """
        snapshot = from_headers(provider, headers)
        if snapshot.known:
            self.usage[provider] = snapshot
        return self.usage.get(provider) or UsageSnapshot()

    # -- connection ------------------------------------------------------------

    def begin(self, provider: ProviderId) -> oauth.AuthRequest:
        """Step 4: returns the URL to send the user to."""
        request = oauth.begin(provider)
        self.pending[provider] = PendingAuth(request=request, started_at=time.time())
        return request

    async def complete(self, provider: ProviderId, pasted: str) -> Credential:
        """Step 4: closes the connection with what the user pasted.

        It accepts the whole return URL, the bare code, or `code#state` — the normalisation
        is `oauth.complete`'s job, not the user's.
        """
        waiting = self.pending.get(provider)
        if waiting is None:
            raise LookupError(
                f"no connection in progress for {PROVIDER_LABELS[provider]}: "
                "press Connect first"
            )
        async with self.client_factory() as client:
            credential = await oauth.complete(provider, waiting.request, pasted, client=client)
        self.store.set(provider, credential)
        del self.pending[provider]
        return credential

    async def refresh(self, provider: ProviderId) -> Credential:
        """Step 5: refreshes and stores the rotated credential."""
        credential = self._require(provider)
        async with self.client_factory() as client:
            renewed = await oauth.refresh(credential, client=client, store=self.store)
        self.store.set(provider, renewed)
        return renewed

    # -- interceptor -----------------------------------------------------------

    def start_local_callback(self, provider: ProviderId, request: Any) -> bool:
        """Tries to open the callback port in **this** process. Returns whether it worked.

        It only serves when LiteLLM runs on the same machine as the browser — there the
        redirect to `localhost` reaches here, and the user installs and runs nothing. On a
        remote proxy the port opens in the wrong place and nobody knocks on it; so the
        failure is silent and the paste stays in sight.
        """
        from .local_flow import LocalCallbackFlow

        if self.local_flow is None:
            self.local_flow = LocalCallbackFlow()
        return bool(self.local_flow.start(provider, request).started)

    async def collect_local_callback(self, provider: ProviderId) -> Credential | None:
        """Closes the connection if the local port has already caught the code. `None` if
        not yet.

        It does not block: it is called on every page load, and waiting here would hold the
        HTTP request for however long the user took to authenticate.
        """
        if self.local_flow is None:
            return None
        result = await self.local_flow.result(provider)
        if result is None:
            return None
        waiting = self.pending.get(provider)
        if waiting is None:
            return None
        async with self.client_factory() as client:
            credential = await oauth.complete(provider, waiting.request, result.code, client=client)
        self.store.set(provider, credential)
        self.pending.pop(provider, None)
        self.discovered.pop(provider, None)
        return credential

    def pair(self, provider: ProviderId) -> Pairing:
        """Issues the code that authorises a `mysubs-login` to deposit here.

        It exists because the callback server has to open on the machine the browser runs
        on, and that machine is not this one when the proxy sits in a container or a
        cluster: the `http://localhost:54545/callback` redirect resolves on the user's
        loopback. A port opened here would be listening in the wrong place.

        The code is what avoids asking the user for the administrator key for an action
        that only needs to deposit a credential. Ten minutes of life, one use, one
        provider — compared with the key that administers the whole proxy, it is acceptable
        material to carry around a terminal.
        """
        return self.pairings.issue(provider)

    def deposit(self, code: str, payload: Any) -> ProviderId:
        """Stores a credential handed over by `mysubs-login`.

        The provider comes from the **code**, not from the body: whoever redeems gets what
        was already assigned to them. Letting the client name it would turn a code issued to
        connect Codex into a write over the Anthropic credential.
        """
        pairing = self.pairings.redeem(code)
        credential = from_payload(pairing.provider, payload)
        if credential is None:
            raise ValueError("the deposit carries no `access_token`")
        self.store.set(pairing.provider, credential)
        # A reconnected provider serves other models: the previous discovery stops being
        # proof of anything, and keeping it would show the user a list that was not measured
        # against this credential.
        self.discovered.pop(pairing.provider, None)
        return pairing.provider

    # -- models ----------------------------------------------------------------

    async def fetch_usage(self, provider: ProviderId) -> UsageSnapshot:
        """Fetches usage from the provider's quota endpoint.

        All three have one. Measured against the real tokens: `/api/oauth/usage` at
        Anthropic, `/backend-api/wham/usage` at Codex, `:quotaSummary` at Antigravity — all
        200 without a single inference request. Reading the headers (`observe`) continues,
        and it is what brings the freshest value when there is traffic; this is what fills
        the card of whoever has just connected the subscription.

        An empty snapshot does **not** replace the previous one: an endpoint that is down is
        no proof the quota changed, and clearing it would leave the card blinking.
        """
        credential = self._require(provider)

        if provider != "google-antigravity":
            from ..catalog.usage_probe import probe

            async with self.client_factory() as client:
                snapshot = await probe(credential, client=client)
            if snapshot.known:
                self.usage[provider] = snapshot
                return snapshot
            return self.usage.get(provider) or UsageSnapshot()

        from ..catalog.discovery import ANTIGRAVITY_USER_AGENT
        from ..transport.hosts import HOSTS, MODELS_PATH

        headers = {
            "Authorization": f"Bearer {credential.access_token}",
            "Content-Type": "application/json",
            "User-Agent": ANTIGRAVITY_USER_AGENT,
        }
        body = {"project": credential.project_id}

        # `:quotaSummary` goes first: it is what Antigravity's own UI consults. When it
        # brings back nothing, the model catalogue still has per-model `quotaInfo` —
        # measured, it gives the three families (Anthropic, Google, OpenAI) the Quota
        # Dashboard shows, and it is their only source.
        async with self.client_factory() as client:
            for path, parse in (
                (QUOTA_SUMMARY_PATH, from_antigravity_summary),
                (MODELS_PATH, from_antigravity_models),
            ):
                for host in HOSTS:
                    try:
                        response = await client.post(
                            f"{host}{path}",
                            headers=headers,
                            json=body,
                            timeout=30.0,
                        )
                    except Exception:
                        continue
                    if response.status_code != 200:
                        continue
                    snapshot = parse(response.json())
                    if snapshot.known:
                        self.usage[provider] = snapshot
                        return snapshot
                    # A valid but empty response: move on to the next endpoint instead of
                    # giving up, which was what left the card without the families.
                    break
        # A host being down is no proof that usage changed: what was known is kept.
        return self.usage.get(provider) or UsageSnapshot()

    async def refresh_usage(self) -> None:
        """Updates the usage of **every** connected provider, without letting the page
        fail.

        Previously only Antigravity was probed, because Anthropic and Codex were assumed to
        publish quota only in the response headers. Measured against the real tokens: both
        have a queryable endpoint and answer 200 with no traffic at all. Under the old rule,
        whoever connected the subscription and made no request saw the card empty forever.

        A short TTL avoids asking on every load: the quota moves in minutes, not
        milliseconds, and one request per page refresh would be noise against the upstream.
        """
        for provider in PROVIDER_IDS:
            if self._safe_get(provider) is None:
                continue
            known = self.usage.get(provider)
            if known is not None and known.age_s() < USAGE_TTL_S:
                continue
            with suppress(Exception):
                await self.fetch_usage(provider)

    async def discover(self, provider: ProviderId) -> list[DiscoveredModel]:
        """Step 6: lists what the subscription serves."""
        credential = self._require(provider)
        async with self.client_factory() as client:
            found = await discover(credential, client=client)
        self.discovered[provider] = found
        return found

    def apply(self, provider: ProviderId, chosen: list[str]) -> ApplyResult:
        """Step 6: injects the chosen models into the Router.

        `chosen` are `suggested_name`s. A name that did not come from discovery is refused
        rather than invented — applying a model the subscription does not serve would
        produce exactly the 400s that motivated this package.

        It applies per provider but injects everything that is selected: the registry's
        `apply` replaces the plugin entries, and passing only one provider's would erase the
        others'.

        It returns what **changed**, not just the total. A "12 in the Router" after applying
        does not tell the user whether what they just unticked actually went, and that is
        the only question they have at that moment.
        """
        router = self.router_source()
        if router is None:
            raise RuntimeError(
                "the LiteLLM Router is not ready yet — the proxy only creates it at startup"
            )
        found = self.discovered.get(provider)
        if not found:
            raise LookupError("discover the models before applying")

        by_name = {model.suggested_name: model for model in found}
        unknown = [name for name in chosen if name not in by_name]
        if unknown:
            raise LookupError(f"models that discovery did not return: {', '.join(unknown)}")

        before = {
            str(d.get("model_name") or "")
            for saved in self.selections.all()
            if saved.provider == provider
            for d in saved.deployments
        }

        self.selected[provider] = list(chosen)
        mine = to_deployments([by_name[name] for name in chosen], provider)
        # Save before injecting. `registry.apply` replaces the plugin entries, and if the
        # process died between the injection and the write the Router would be left with
        # models the next startup would not know how to restore — exactly the state this
        # exists to prevent.
        self.selections.save(provider, mine)

        # **Everything** stored is injected, not just this provider: the registry's `apply`
        # replaces the plugin entries in one go, and passing only one provider's would erase
        # the others'.
        deployments: list[dict[str, Any]] = []
        for saved in self.selections.all():
            deployments += saved.deployments
        total = ModelRegistry(router=router).apply(deployments)

        after = {str(d.get("model_name") or "") for d in mine}
        return ApplyResult(
            provider=provider,
            added=sorted(after - before),
            removed=sorted(before - after),
            total=total,
        )

    def disconnect(self, provider: ProviderId) -> int:
        """Disconnects a subscription: deletes the credential and removes its models from
        the Router.

        Both things together, and in this order, because splitting them leaves the worst
        possible state: models advertised in `/v1/models` with no credential to serve them.
        Every request to one of them would fail with a 401 from the upstream, and the user
        would have no way to know the cause was pressing Disconnect.

        It returns how many deployments were left in the Router, so the page can say it.

        What is **not** done: revoking the token at the provider. None of the three
        publishes a revocation endpoint for these clients, and pretending to have revoked
        would be worse than telling the truth — the session stays valid on their side until
        it expires.
        """
        self.store.delete(provider)
        self.discovered.pop(provider, None)
        self.selected.pop(provider, None)
        self.usage.pop(provider, None)
        self.pending.pop(provider, None)
        # From disk too: without this the next startup would reinject models from a
        # subscription the user has just disconnected, and the card would say "not
        # connected" with its models showing up in `/v1/models`.
        self.selections.drop(provider)

        router = self.router_source()
        if router is None:
            # With no Router there is nothing injected that could be orphaned: the state
            # has already been cleared, and it is what the next startup will reapply.
            return 0
        deployments: list[dict[str, Any]] = []
        for saved in self.selections.all():
            deployments += saved.deployments
        return ModelRegistry(router=router).apply(deployments)

    def reapply(self) -> int:
        """Reinjects into the Router what was stored. Called at proxy startup.

        It is the missing half of the promise at the top of `registry.py` — "injects
        directly and persists on its own". Measured before this: apply `gpt-5.5`, restart
        the proxy, and `/v1/models` was back to just `eco`, with the cards saying
        `connected=True applied=0`. The credentials survived; the model choice did not.

        It runs no discovery: the stored deployments are self-sufficient. Going to the
        network at startup would leave the user without their models whenever the upstream
        was down — and what they applied does not depend on the catalogue being reachable
        now.
        """
        router = self.router_source()
        if router is None:
            return 0
        deployments: list[dict[str, Any]] = []
        for saved in self.selections.all():
            deployments += saved.deployments
            self.selected[saved.provider] = [
                str(d.get("model_name") or "") for d in saved.deployments
            ]
        if not deployments:
            return 0
        return ModelRegistry(router=router).apply(deployments)
