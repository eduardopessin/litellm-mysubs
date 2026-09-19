"""The FastAPI sub-app served at `/mysubs`.

HTTP and HTML only: the flow lives in `service.py`. The page is served with no build step
and no frontend dependencies — a wheel that needed `npm` to show four cards would be worse
to install than the problem it solves.
"""

from __future__ import annotations

import contextlib
import html
import json
import os
import urllib.parse
from typing import Any

from fastapi import APIRouter, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from ..credentials.store import PROVIDER_IDS, ProviderId
from .auth import admin_dependency
from .service import PROVIDER_LABELS, MySubsService, ProviderCard
from .throttle import Throttle, client_origin, retry_after

#: `guard` sentinel: tells "I passed nothing" apart from "I passed `None` on purpose".
_UNSET: Any = object()


def _resolve_guard(guard: Any) -> Any:
    """The dependency to apply. `_UNSET` means "use the proxy's"."""
    return admin_dependency() if guard is _UNSET else guard


#: Variable that changes the mount prefix.
#:
#: It exists because `/mysubs` does not always reach the proxy: a port forwarding — VS
#: Code's is the measured case — routes some paths to itself, and the page is never
#: served. Changing the prefix is the way out, and it has to be a single decision: the
#: injected menu, the redirects and the "already mounted" guard have to agree, or the
#: button points where there is nothing.
MOUNT_PATH_ENV = "MYSUBS_PATH"


def mount_path() -> str:
    """The prefix where the page lives, normalised.

    Without a leading slash, or with a trailing one, `app.mount()` mounts somewhere other
    than the one asked for, and the symptom is a 404 that looks like the plugin's.
    """
    raw = os.environ.get(MOUNT_PATH_ENV, "").strip().rstrip("/")
    if not raw:
        return "/mysubs"
    return raw if raw.startswith("/") else "/" + raw


#: The mount prefix. The LiteLLM menu item points here.
MOUNT_PATH = mount_path()

#: Form fields as singletons: `Form(...)` as a default is evaluated at import time and
#: ruff refuses it in signatures (B008).
_PASTED: Any = Form(...)
_CHOSEN: Any = Form(default=[])
_CONFIRM: Any = Form(default="")


def _provider_or_404(raw: str) -> ProviderId:
    if raw not in PROVIDER_IDS:
        raise HTTPException(status_code=404, detail=f"unknown provider: {raw}")
    return raw


def _duration(seconds: float) -> str:
    """The duration only: ``3 min``, ``2 h``, ``4 days``.

    Separate from the sentence on purpose. The previous version embedded "expires in" and
    was reused for quota resets, which produced "resets expires in 1 h" on screen.
    """
    if seconds < 3600:
        return f"{int(seconds // 60)} min"
    if seconds < 86400:
        return f"{int(seconds // 3600)} h"
    return f"{int(seconds // 86400)} days"


def _age(seconds: float | None) -> str:
    """Token validity, as text.

    A snapshot with no age lies by omission: whoever sees "connected" assumes "working". So
    the validity shows whenever it is known, and its absence is stated, not hidden.
    """
    if seconds is None:
        return "unknown validity"
    if seconds <= 0:
        return "expired"
    return f"expires in {_duration(seconds)}"


def build_app(service: MySubsService, *, guard: Any | None = _UNSET) -> FastAPI:
    """The sub-app. It takes the service instead of building it: that is what makes it
    testable.

    `guard` is the authentication dependency, applied to **every** route. The default is not
    `None` — it is a sentinel that says to ask `auth`: a default with no guard would make
    "I forgot to pass it" indistinguishable from "I decided not to protect it", and the
    first is the mistake that exposes the page.
    """
    # The refresher does **not** start here. Measured: Starlette does not propagate the
    # `lifespan` of a sub-app mounted with `app.mount()` — it would never run, and the
    # symptom would be everything green with the tokens expiring all the same. What starts
    # it is `ui/install.py`, in the host app's life cycle.
    dependencies = [] if guard is None else [_resolve_guard(guard)]
    app = FastAPI(title="MySubs", docs_url=None, redoc_url=None)
    _install_error_pages(app)

    # The guard lives on the router, not on the app: `/api/deposit` has to stay outside
    # it, and a dependency declared on the app would catch **every** route, including that
    # one. A deposit refused for want of an administrator key would be an interceptor that
    # can never hand over what it went to fetch.
    guarded = APIRouter(dependencies=[d for d in dependencies if d is not None])

    @guarded.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        # If the local port has already caught a code, the connection is closed here. It
        # is what makes the flow automatic: the user authenticates, the page reloads, and
        # the card already shows as connected — with nothing pasted.
        for provider in PROVIDER_IDS:
            with contextlib.suppress(Exception):
                await service.collect_local_callback(provider)
        # Antigravity does not publish usage in the headers: it has to be asked for. A
        # failure here cannot hide the page — the other providers' cards are still valid,
        # and its own card shows the last known snapshot with its age.
        await service.refresh_usage()
        return HTMLResponse(_page(service.cards(), service))

    @guarded.post("/connect/{provider}")
    async def connect(provider: str, url: str = "") -> Response:
        """Step 4: starts the connection and sends the user to the provider.

        With `?url=1` it returns the destination as JSON instead of redirecting. It is what
        the page uses to open the provider **in a new window** and stay alive probing the
        local port — a 303 would take the browser away from here and the probe would die
        with the page.

        Without the parameter it redirects as always: it is the path for whoever has
        JavaScript turned off, and what keeps the button working without any of this.
        """
        target = _provider_or_404(provider)
        request = service.begin(target)
        started = service.start_local_callback(target, request)
        if url:
            return JSONResponse({"url": request.url, "local": started})
        return RedirectResponse(request.url, status_code=303)

    @guarded.post("/paste/{provider}")
    async def paste(provider: str, pasted: str = _PASTED) -> RedirectResponse:
        """Step 4: closes the connection with the pasted return URL."""
        target = _provider_or_404(provider)
        try:
            await service.complete(target, pasted.strip())
        except Exception as error:
            return _back(f"{target}: {error}", provider=target)
        return _to_card(target)

    @guarded.post("/refresh/{provider}")
    async def refresh(provider: str) -> RedirectResponse:
        """Step 5: refreshes the token."""
        target = _provider_or_404(provider)
        try:
            await service.refresh(target)
        except Exception as error:
            return _back(f"{target}: {error}", provider=target)
        return _to_card(target)

    @guarded.post("/discover/{provider}")
    async def discover(provider: str) -> RedirectResponse:
        """Step 6: lists what the subscription serves."""
        target = _provider_or_404(provider)
        try:
            await service.discover(target)
        except Exception as error:
            return _back(f"{target}: {error}", provider=target)
        return _to_card(target)

    @guarded.post("/apply/{provider}")
    async def apply(provider: str, chosen: list[str] = _CHOSEN) -> RedirectResponse:
        """Step 6: injects into the Router what the user chose."""
        target = _provider_or_404(provider)
        try:
            result = service.apply(target, chosen)
        except Exception as error:
            return _back(f"{target}: {error}", provider=target)

        extra = ""
        if result.added:
            extra += "&added=" + urllib.parse.quote(",".join(result.added))
        if result.removed:
            extra += "&removed=" + urllib.parse.quote(",".join(result.removed))
        return _to_card(target, f"applied={target}{extra}")

    @guarded.post("/disconnect/{provider}")
    async def disconnect(provider: str, confirm: str = _CONFIRM) -> RedirectResponse:
        """Disconnects the subscription: deletes the credential and removes its models
        from the Router.

        It requires `confirm=<provider>` in the body. It is the second time the user names
        what they are deleting, and it exists because a stray POST — a double click, a
        refresh that resends the form — would delete a refresh token that only a new login
        restores. The browser's `confirm()` is not enough: it does not survive a resend of
        the request.
        """
        target = _provider_or_404(provider)
        if confirm != target:
            return _back(
                f"{target}: confirmation does not match; nothing was deleted", provider=target
            )
        try:
            service.disconnect(target)
        except Exception as error:
            return _back(f"{target}: {error}", provider=target)
        return RedirectResponse(MOUNT_PATH + "/", status_code=303)

    @guarded.post("/pair/{provider}", response_class=HTMLResponse)
    async def pair(provider: str, request: Request) -> HTMLResponse:
        """Issues the code `mysubs-login` will use to deposit the credential.

        It returns the page with the command already assembled instead of redirecting with
        the code in the query string: an authorisation code in a URL ends up in the browser
        history, in the logs of any proxy in front, and in the `Referer` of the next
        request. None of those places is where it should stay, even with ten minutes of
        life.
        """
        target = _provider_or_404(provider)
        pairing = service.pair(target)
        await service.refresh_usage()
        return HTMLResponse(
            _page(service.cards(), service, pairing=(target, pairing.code, _base_url(request)))
        )

    @guarded.get("/api/state")
    async def state() -> JSONResponse:
        """The same state as JSON, for whoever prefers to automate."""
        return JSONResponse(
            {
                "providers": [
                    {
                        "provider": card.provider,
                        "label": card.label,
                        "connected": card.connected,
                        "expires_in_s": card.expires_in_s,
                        "stale": card.stale,
                        "applied": card.applied,
                    }
                    for card in service.cards()
                ],
                "discovered": {
                    provider: [
                        {
                            "wire_name": model.wire_name,
                            "suggested_name": model.suggested_name,
                            "verified": model.verified,
                            "note": model.note,
                        }
                        for model in models
                    ]
                    for provider, models in service.discovered.items()
                },
            }
        )

    # One limiter per app: the deposit is the only route outside the guard, and the only
    # one that consults it. It lives in `build_app` so each proxy (and each test) has its
    # own.
    throttle = Throttle()

    @app.post("/api/deposit")
    async def api_deposit(request: Request, payload: dict[str, Any]) -> JSONResponse:
        """Takes the credential from the `mysubs-login` that ran on the user's machine.

        It stays **outside** the administrator guard because the pairing code is the
        authorisation: forcing the command to carry the proxy key would be putting the
        installation's most dangerous secret on a command line to spare this route.

        Any failure is a 403 with the message: a spent, expired or invented code cannot be
        told apart from the outside, and telling them apart would tell a guesser which
        codes ever existed.

        The 429 comes out **before** the code is looked at. Two reasons: not to spend a
        registry scan per request from whoever is hammering, and not to let the response
        body say whether the code was good — a different 429 for a good code and a bad one
        would give a guesser exactly the oracle the single 403 refuses to give.
        """
        origin = client_origin(request)
        remaining = throttle.blocked(origin)
        if remaining > 0:
            return JSONResponse(
                {"detail": "Too many attempts. Try again later."},
                status_code=429,
                headers={"Retry-After": retry_after(remaining)},
            )
        code = str(payload.get("pairing_code") or "")
        try:
            provider = service.deposit(code, payload.get("credential"))
        except Exception as error:
            throttle.record_failure(origin)
            return JSONResponse({"detail": str(error)}, status_code=403)
        throttle.record_success(origin)
        return JSONResponse({"provider": provider})

    app.include_router(guarded)
    return app


def _install_error_pages(app: FastAPI) -> None:
    """Turns authentication failures into a readable page.

    `user_api_key_auth` raises `ProxyException`, which is a plain `Exception` — the proxy
    handles it in a handler registered on its own app. A mounted sub-app does **not
    inherit** that handler, and without this the refusal reached the browser as::

        HTTP 500  Internal Server Error
        RuntimeError: Caught handled exception, but response already started.

    Access stayed denied, which is what matters, but whoever saw the 500 had no way to know
    they were missing a key. An error that does not teach is an error left unsolved.
    """
    from fastapi import HTTPException

    async def as_page(request: Request, error: Exception) -> Response:
        status = int(getattr(error, "code", 0) or getattr(error, "status_code", 0) or 403)
        detail = str(getattr(error, "message", "") or getattr(error, "detail", "") or error)
        if "json" in str(request.headers.get("accept", "")):
            return JSONResponse({"detail": detail}, status_code=status)
        return HTMLResponse(_error_page(status, detail), status_code=status)

    app.add_exception_handler(HTTPException, as_page)
    try:
        from litellm.proxy._types import ProxyException

        app.add_exception_handler(ProxyException, as_page)
    except ImportError:  # pragma: no cover - without LiteLLM there is nothing to convert
        pass


def _error_page(status: int, detail: str) -> str:
    """The refusal page. For 401/403, with a way out — not just a diagnosis.

    "Sign in to the UI with an administrator key" was advice impossible to follow on a
    proxy with no key database: without a DB there is no UI login, so there is never a
    session cookie, and the refusal was permanent with nothing to be done. Measured against
    a test proxy (`db: Not connected`).

    So the page takes the key here. It stays in the `sessionStorage` of the origin you are
    accessing from — which also settles the case where the UI was opened on another origin
    (tunnel, port forwarding), where the login cookie does not travel.
    """
    if status not in (401, 403):
        return _shell(
            '<section class="card"><header><h2>No access</h2>'
            f'<span class="chip warn">HTTP {status}</span></header>'
            f'<p class="muted">{html.escape(detail)}</p></section>'
        )
    return _shell(_DENIED.format(status=status, detail=html.escape(detail)))


def _shell(body: str) -> str:
    """The full page. The only place that fills in `_SHELL`.

    The callback endpoints come from `oauth.callback_origin`, which reads the same
    `redirect_uri` that goes in the authorization request — one table only, instead of a
    JavaScript copy that would diverge with nothing failing.
    """
    from ..credentials import oauth

    endpoints = {p: oauth.callback_origin(p) for p in PROVIDER_IDS}
    return _SHELL.format(body=body, endpoints=json.dumps(endpoints))


def _to_card(provider: str, extra: str = "") -> RedirectResponse:
    """Back to the page, at the card it came from.

    The destination carries **`?card=` and `#anchor`**, both. The anchor reaches the browser
    on a normal navigation; the parameter is what survives a `fetch`, and measured: `fetch`
    with `redirect: "follow"` discards the fragment of `Location` — `r.url` arrived without
    it, and the key replay reloaded the page at the top. The page uses whichever is there.
    """
    target = f"{MOUNT_PATH}/?card={provider}"
    if extra:
        target += "&" + extra
    return RedirectResponse(f"{target}#{provider}", status_code=303)


def _back(message: str, *, provider: str = "") -> RedirectResponse:
    """Back to the page with the error in sight, at the card it came from.

    The error goes in the query string instead of being swallowed: the provider's
    `error_description` is what tells the user what to do next.

    The anchor is what avoids the jump to the top. A `303` without it reloads the page at
    position zero, and the user who pressed a button on the third card loses sight of both
    the card and the message they just triggered.
    """
    target = f"{MOUNT_PATH}/?error={html.escape(message)}"
    if not provider:
        return RedirectResponse(target, status_code=303)
    return RedirectResponse(f"{target}&card={provider}#{provider}", status_code=303)

def mount(
    app: Any, service: MySubsService, *, path: str = MOUNT_PATH, guard: Any | None = _UNSET
) -> None:
    """Mounts the sub-app on the proxy.

    `app.mount()` is the route LiteLLM itself uses for `/ui` and `/swagger`.
    """
    app.mount(path, build_app(service, guard=guard))


# -- HTML ----------------------------------------------------------------------


def _base_url(request: Any) -> str:
    """The public URL of this proxy, as the browser reached it.

    It comes from the request and not from configuration because it is the only place the
    right value exists: behind an ingress the proxy does not know what name it was called
    by, and a command pointing at the internal name fails for exactly the user who needs
    the interceptor. `X-Forwarded-*` is honoured by Starlette when the proxy in front sends
    it.
    """
    return str(request.base_url).rstrip("/").removesuffix(MOUNT_PATH)


def _page(
    cards: list[ProviderCard],
    service: MySubsService,
    *,
    pairing: tuple[ProviderId, str, str] | None = None,
) -> str:
    banner = "" if pairing is None else _pairing_panel(*pairing)
    return _shell(banner + "\n".join(_card(card, service) for card in cards))


def _pairing_panel(provider: ProviderId, code: str, base_url: str) -> str:
    """The command to run, with the code and this proxy's URL already inside.

    Assembled here and not left to the user because every piece they have to put together
    by hand — the right proxy URL, the provider's internal name — is a piece they can get
    wrong, and the mistake only shows up after they have already logged in.
    """
    command = f"mysubs-login {provider} --url {base_url} --code {code}"
    # The internal id goes in the command, which is what the machine reads; the prose
    # carries the name the card shows. Swapping them would force the user to connect two
    # names to the same card.
    return _PAIRING.format(
        command=html.escape(command), provider=html.escape(PROVIDER_LABELS[provider])
    )


def _usage(card: ProviderCard) -> str:
    """The usage bars, or an honest sentence about why there are none.

    A bar at zero for a provider with no data would read as "unused" — the opposite of what
    is known, which is nothing.

    The previous sentence — "this provider does not publish usage" — was **false** for all
    three. Measured against the real tokens, all of them answer 200 to a quota probe:
    `api.anthropic.com/api/oauth/usage`, `chatgpt.com/backend-api/wham/usage`, and
    Antigravity's `:quotaSummary`. What is missing when there are no bars is the probe
    having arrived, not the provider's capability — and making the user believe otherwise
    sends them looking for the problem in the wrong place.
    """
    if not card.usage.known:
        return (
            '<p class="nodata">No quota reading yet — the probe did not answer or the '
            "token has just been connected.</p>"
        )

    bars = []
    for window in card.usage.windows:
        pct = max(0.0, min(100.0, window.used_percent))
        tone = "hot" if pct >= 90 else "warm" if pct >= 70 else "cool"
        resets = window.resets_in_s()
        bars.append(
            _BAR.format(
                label=html.escape(window.label),
                pct=f"{pct:.0f}",
                tone=tone,
                resets=html.escape(f"resets in {_duration(resets)}") if resets else "",
            )
        )
    meta = []
    if card.usage.plan:
        meta.append(f"plan {html.escape(card.usage.plan)}")
    if card.usage.credits_balance:
        meta.append(f"{html.escape(card.usage.credits_balance)} credits")
    age = card.usage.age_s()
    meta.append("now" if age < 60 else f"{int(age // 60)} min ago")
    return _USAGE.format(bars="".join(bars), meta=html.escape(" · ".join(meta)))


def _card(card: ProviderCard, service: MySubsService) -> str:
    if not card.connected:
        return _CARD.format(
            label=html.escape(card.label),
            badge='<span class="chip off">not connected</span>',
            usage="",
            body=_CONNECT.format(provider=card.provider, base=MOUNT_PATH),
            logo=PROVIDER_LOGO[card.provider],
            provider=card.provider,
        )

    badge = (
        '<span class="chip warn">token expired</span>'
        if card.stale
        else '<span class="chip on">connected</span>'
    )
    detail = _age(card.expires_in_s)
    if card.project_id:
        detail += f" · project {card.project_id}"
    if card.applied:
        detail += f" · {card.applied} in Router"

    # The refresh button only shows where the background refresher does not reach:
    # unknown validity, or a store that does not own the refresh. On the normal path it
    # would be an action that does nothing the loop will not do by itself in 60 seconds —
    # and a redundant button teaches the user to watch something already taken care of.
    if card.auto_renews:
        detail += " · renews automatically"
        renew = ""
    else:
        renew = _RENEW.format(provider=card.provider, base=MOUNT_PATH)

    disconnect = _DISCONNECT.format(provider=card.provider, base=MOUNT_PATH)
    models = service.discovered.get(card.provider)
    if models is None:
        body = _DISCOVER.format(
            provider=card.provider,
            detail=html.escape(detail),
            base=MOUNT_PATH,
            renew=renew,
            disconnect=disconnect,
        )
    else:
        chosen = set(service.selected.get(card.provider) or [])
        rows = "".join(
            _ROW.format(
                name=html.escape(model.suggested_name),
                wire=html.escape(model.wire_name),
                checked=" checked" if not chosen or model.suggested_name in chosen else "",
                mark=(
                    '<span class="chip on">verified</span>'
                    if model.verified
                    else f'<span class="chip unk" title="{html.escape(model.note)}">'
                    "unverified</span>"
                ),
            )
            for model in models
        )
        body = _APPLY.format(
            provider=card.provider,
            detail=html.escape(detail),
            rows=rows,
            base=MOUNT_PATH,
            renew=renew,
            disconnect=disconnect,
        )

    return _CARD.format(
        label=html.escape(card.label),
        badge=badge,
        usage=_usage(card),
        body=body,
        logo=PROVIDER_LOGO[card.provider],
        provider=card.provider,
    )


_BAR = """<div class="bar">
  <div class="bar-head"><span>{label}</span><span class="pct">{pct}%</span></div>
  <div class="track"><div class="fill {tone}" style="width:{pct}%"></div></div>
  <div class="resets">{resets}</div>
</div>"""

_USAGE = """<div class="usage">{bars}</div><p class="stamp">{meta}</p>"""

#: Refusal page with a way out. `sessionStorage` (and not `localStorage`) is deliberate:
#: the proxy key dies when the tab closes, instead of sitting on disk in the browser
#: profile. And the resend goes through `fetch` + `Authorization`, not a cookie, because
#: that header is what LiteLLM's `user_api_key_auth` actually reads.
_DENIED = """<section class="card">
  <header><h2>No access</h2><span class="chip warn">HTTP {status}</span></header>
  <p class="muted">{detail}</p>
  <p class="muted">Paste a proxy admin key here. It stays in this tab only.</p>
  <form class="paste" onsubmit="return mysubsKey(event)">
    <input id="k" type="password" placeholder="sk-…" autocomplete="off">
    <button class="primary">Sign in</button>
  </form>
</section>
<script>
 function mysubsKey(e) {{
   e.preventDefault();
   sessionStorage.setItem("mysubs_key", document.getElementById("k").value.trim());
   location.reload();
   return false;
 }}
</script>"""

#: The `action`s are **absolute**, with `{base}` replaced by the mount prefix.
#:
#: Relative ones broke the pairing page: it is served at `/mysubs/pair/<p>`, and an
#: `action="pair/x"` resolves against `/mysubs/pair/` — giving `/mysubs/pair/pair/x`, which
#: is a 404. The bug only showed after pressing «Issue code», because that is the only 200
#: response served outside the root.
#:
#: The connect card. The paste is in plain sight, not hidden: measured that **no** browser
#: mechanism allows capturing the return URL from another origin — 16 attempts across
#: `window.open` (11 variants), `iframe` (3), clipboard and the Performance API, all with
#: `SecurityError`. The three providers further refuse to be framed
#: (`X-Frame-Options: SAMEORIGIN`/`DENY`). The only automatic route requires a receiver on
#: the loopback of the browser's machine, and that is not the proxy's on a remote install.
#:
#: So the paste is the main path again. The automatic one still works when the port
#: answers — but whoever does not have it cannot be left hunting for where to paste.
_CONNECT = """
<p class="muted">Connect your subscription in two steps.</p>
<ol class="steps">
  <li>
    <form method="post" action="{base}/connect/{provider}" data-connect="{provider}">
      <button class="primary">Connect</button>
    </form>
    <span class="muted">sign in on the window that opens</span>
  </li>
  <li>
    <p class="muted">The return page will fail with <em>&laquo;can&rsquo;t reach this
    page&raquo;</em> — that is expected. Copy the URL from the address bar and paste it
    here:</p>
    <form method="post" action="{base}/paste/{provider}" class="paste">
      <input name="pasted" placeholder="http://localhost:…?code=…" autocomplete="off">
      <button>Finish</button>
    </form>
  </li>
</ol>
<div class="waiting" data-waiting="{provider}" hidden>
  <span class="spin"></span>
  <span>If the interceptor is running, this closes by itself…</span>
</div>
<details class="alt" data-manual="{provider}">
  <summary>Skip the paste step</summary>
  <p class="muted">The redirect goes to <code>localhost</code>, which is the loopback of
  <strong>your</strong> machine — not this proxy&rsquo;s. With a receiver there, the loop
  closes by itself. Two ways:</p>
  <p class="muted"><strong>1.</strong> Forward ports
  <code>54545</code>, <code>1455</code> and <code>51121</code> from this server to your
  machine (VS Code: <em>Ports</em> tab; or <code>ssh -L</code>).</p>
  <p class="muted"><strong>2.</strong> Run the interceptor on your machine:</p>
  <form method="post" action="{base}/pair/{provider}"><button>Issue code</button></form>
</details>
"""

_PAIRING = """<section class="card pair">
  <header><h2>Token interceptor</h2>
    <span class="chip unk">valid for 10 minutes</span></header>
  <p class="muted">Run this <strong>on the machine where your browser is</strong>. It
  opens the port {provider} requires there, catches the code and hands the credential to
  this proxy.</p>
  <pre class="cmd">{command}</pre>
  <p class="muted">You need the package on your machine:
  <code>pip install litellm-mysubs</code>. The code works once only; if it expires, press
  Issue code again.</p>
</section>"""

#: Disconnect lives in a closed `details`, away from the everyday buttons: it is the only
#: action on the page that destroys something only a new login restores.
_DISCONNECT = """
<details class="alt danger">
  <summary>Disconnect this subscription</summary>
  <p class="muted">Deletes the credential from this proxy and removes from the Router the
  models it served. The tokens are not revoked at the provider — none of the three allows
  it for these clients — but they are no longer here, and coming back requires a new
  login.</p>
  <form method="post" action="{base}/disconnect/{provider}"
        onsubmit="return confirm('Disconnect {provider}? Coming back needs a new login.')">
    <input type="hidden" name="confirm" value="{provider}">
    <button class="danger">Disconnect {provider}</button>
  </form>
</details>
"""

#: The manual refresh. Only rendered where the background refresher does not reach — see
#: `_card`.
_RENEW = """
<form method="post" action="{base}/refresh/{provider}"><button>Refresh token</button></form>
"""

_DISCOVER = """
<p class="muted">{detail}</p>
<form method="post" action="{base}/discover/{provider}">
  <button class="primary">See models</button></form>
{renew}
{disconnect}
"""

_APPLY = """
<p class="muted">{detail}</p>
<form method="post" action="{base}/apply/{provider}">
  <table>{rows}</table>
  <div class="actions">
    <button class="primary">Apply</button>
    <button formaction="{base}/discover/{provider}" formnovalidate>Rediscover</button>
  </div>
</form>
{renew}
{disconnect}
"""

_ROW = """<tr>
  <td><label><input type="checkbox" name="chosen" value="{name}"{checked}>
    <span>{name}</span></label></td>
  <td class="wire">{wire}</td><td class="mark">{mark}</td>
</tr>"""

#: Provider logos, served by **LiteLLM itself** at `/ui/assets/logos/`. Pointing there
#: instead of embedding them keeps the page aligned with the host UI without duplicating
#: files: if it swaps the Anthropic logo, the card swaps with it. Verified that all three
#: answer 200.
PROVIDER_LOGO: dict[ProviderId, str] = {
    "anthropic": "/ui/assets/logos/anthropic.svg",
    "openai-codex": "/ui/assets/logos/openai_small.svg",
    "google-antigravity": "/ui/assets/logos/google.svg",
}

_CARD = """<section class="card" id="{provider}">
  <header><img class="logo" src="{logo}" alt="" aria-hidden="true">
    <h2>{label}</h2>{badge}</header>
  {usage}
  {body}
</section>"""

#: Palette extracted from the LiteLLM UI bundle
#: (`_experimental/out/_next/static/chunks/*.css`): the same
#: `--background`/`--foreground`/`--border`/`--primary` tokens, the Inter font and the
#: light/dark pair it defines. Copying the values instead of importing the stylesheet is
#: deliberate: the file name is a build hash and changes with every version, and a page
#: that depends on it goes blank after a `pip install -U litellm`.
_SHELL = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MySubs</title>
<script>
 // Before the CSS, on purpose: applying the class after paint gives a white flash on
 // every load inside a dark UI.
 //
 // `dark` is only applied when LiteLLM wrote it. Measured in the same state — no
 // `localStorage.theme` and the system in dark — its UI stays **light**: the default is
 // `light`, not the system's. The previous version fell back to `prefers-color-scheme`
 // and served a dark page to whoever had LiteLLM in light.
 //
 // Any other value (`system`, or a future one we do not know) falls back to the light
 // theme of `:root`, which is the same one it shows by default. Guessing here is how you
 // end up with two pages side by side in different themes.
 (function () {{
   try {{
     if (localStorage.getItem("theme") === "dark") {{
       document.documentElement.className = "dark";
     }}
   }} catch (e) {{ /* localStorage blocked: the light theme stays, like LiteLLM */ }}
 }})();
</script>
<style>
 :root{{
   --background:#fff; --foreground:#030712; --card:#fff; --border:#e5e7eb;
   --muted:#6a7282; --primary:#101828; --primary-foreground:#f9fafb;
   --accent:#f3f4f6; --ring:#99a1af; --radius:.5rem;
   --on-bg:#e6f4ea; --on-fg:#137333; --warn-bg:#fce8e6; --warn-fg:#c5221f;
   --unk-bg:#fffbeb; --unk-fg:#b75000; --cool:#155dfc; --warm:#f99c00; --hot:#c5221f;
 }}
 /* LiteLLM does not follow the system theme: it has its own switch, which writes
    `localStorage.theme` and the `light`/`dark` class on `<html>`. Using
    `prefers-color-scheme` here made the page ignore that choice — light inside a dark UI,
    which is worse than having no theme at all. We read the same place it uses. */
 html.dark{{
   /* Values measured from the real UI's `getComputedStyle` (which serves them in `lab()`)
      and converted to sRGB: `--background` and `--card` are the **same** #212121 there, not
      two shades. Having a #181818 background darker than the card gave the page a frame the
      LiteLLM UI does not have, and the effect was looking like an alien window inside it. */
   --background:#212121; --foreground:#f3f3f3; --card:#212121; --border:#303030;
   --muted:#afafaf; --primary:#e7e7e7; --primary-foreground:#181818;
   --accent:#303030; --ring:#777;
   --on-bg:#12261a; --on-fg:#6ee7a0; --warn-bg:#2a1512; --warn-fg:#ff9e94;
   --unk-bg:#2a2010; --unk-fg:#fcbb00;
 }}
 *{{box-sizing:border-box}}
 body{{margin:0;padding:2rem 1.5rem;background:var(--background);color:var(--foreground);
   font:14px/1.5 Inter,"Inter Fallback",system-ui,sans-serif;
   -webkit-font-smoothing:antialiased}}
 .wrap{{max-width:780px;margin:0 auto}}
 h1{{font-size:1.5rem;font-weight:600;letter-spacing:-.02em;margin:0 0 .25rem;
   display:flex;align-items:center;gap:.6rem}}
 .lead{{color:var(--muted);margin:0 0 1.75rem}}
 /* The title icon is the same as the menu item's (`KeyRound`), only bigger: it is what
    ties the page to the button you arrive from. */
 .mark{{width:1.5rem;height:1.5rem;flex:none;color:var(--muted)}}
 /* The logos come from LiteLLM, each with its own box and background. `object-fit:contain`
    and a fixed height give them the same presence without distorting them. */
 .logo{{width:1.15rem;height:1.15rem;object-fit:contain;flex:none}}
 /* Each provider's frame. It was lost when the icon rules landed on top of the base
    rule — and the symptom was subtle: the cards still existed in the HTML and were still
    separated by whitespace, they had just lost any outline.

    `--card` and `--background` are the same shade in LiteLLM's dark, so it is the border
    that does the separating, not a surface contrast. */
 .card{{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);
   padding:1.25rem 1.5rem;margin-bottom:1rem}}
 .card header{{display:flex;align-items:center;gap:.6rem;margin-bottom:.9rem}}
 h2{{font-size:1rem;font-weight:600;margin:0}}
 .chip{{font-size:.7rem;font-weight:500;padding:.15rem .5rem;border-radius:9999px;
   border:1px solid transparent;white-space:nowrap}}
 .chip.on{{background:var(--on-bg);color:var(--on-fg)}}
 .chip.off{{background:var(--accent);color:var(--muted)}}
 .chip.warn{{background:var(--warn-bg);color:var(--warn-fg)}}
 .chip.unk{{background:var(--unk-bg);color:var(--unk-fg);cursor:help}}
 .muted{{color:var(--muted);margin:0 0 .9rem}}
 .usage{{display:flex;gap:1.25rem;margin-bottom:.5rem;flex-wrap:wrap}}
 .bar{{flex:1;min-width:160px}}
 .bar-head{{display:flex;justify-content:space-between;font-size:.75rem;
   color:var(--muted);margin-bottom:.3rem}}
 .pct{{font-variant-numeric:tabular-nums;font-weight:600;color:var(--foreground)}}
 .track{{height:6px;background:var(--accent);border-radius:9999px;overflow:hidden}}
 .fill{{height:100%;border-radius:9999px;transition:width .3s}}
 .fill.cool{{background:var(--cool)}} .fill.warm{{background:var(--warm)}}
 .fill.hot{{background:var(--hot)}}
 .resets{{font-size:.7rem;color:var(--muted);margin-top:.25rem}}
 .stamp{{font-size:.72rem;color:var(--muted);margin:0 0 1rem}}
 .nodata{{font-size:.78rem;color:var(--muted);margin:0 0 1rem;font-style:italic}}
 form{{display:inline-block;margin:0 .5rem .5rem 0}}
 .paste{{display:block;margin-top:.25rem}}
 .paste input{{width:min(430px,68%);padding:.45rem .7rem;border:1px solid var(--border);
   border-radius:var(--radius);background:var(--background);color:var(--foreground);
   font:inherit}}
 .paste input:focus{{outline:2px solid var(--ring);outline-offset:-1px}}
 button{{padding:.45rem .9rem;border:1px solid var(--border);background:var(--card);
   color:var(--foreground);border-radius:var(--radius);cursor:pointer;font:inherit;
   font-weight:500}}
 button:hover{{background:var(--accent)}}
 button.primary{{background:var(--primary);color:var(--primary-foreground);
   border-color:var(--primary)}}
 button.primary:hover{{opacity:.9}}
 table{{border-collapse:collapse;width:100%;margin:.25rem 0 .9rem}}
 td{{padding:.4rem .5rem .4rem 0;border-bottom:1px solid var(--border)}}
 tr:last-child td{{border-bottom:0}}
 label{{display:flex;align-items:center;gap:.5rem;cursor:pointer}}
 .wire{{color:var(--muted);font-family:ui-monospace,SFMono-Regular,monospace;
   font-size:.76rem}}
 .mark{{text-align:right}}
 /* The two numbered steps. The number on the left is what tells the user there is an
    order — the previous version put both buttons side by side and the second looked like
    an alternative to the first, not its continuation. */
 .steps{{list-style:none;counter-reset:s;margin:0;padding:0}}
 .steps li{{counter-increment:s;position:relative;padding:0 0 .85rem 1.9rem}}
 .steps li:last-child{{padding-bottom:0}}
 .steps li::before{{content:counter(s);position:absolute;left:0;top:.15rem;
   width:1.25rem;height:1.25rem;border-radius:50%;background:var(--accent);
   color:var(--muted);font-size:.72rem;font-weight:600;display:flex;
   align-items:center;justify-content:center}}
 .steps form{{margin:0 .5rem .35rem 0}}
 .steps .paste{{display:block;margin-top:.35rem}}
 .steps p{{margin:.15rem 0 .5rem}}
 .auto-ok{{font-size:.78rem;margin:.5rem 0 0}}
 .waiting{{display:flex;align-items:center;gap:.6rem;color:var(--muted);
   font-size:.82rem;margin:.6rem 0}}
 /* After the base rule, on purpose: `display:flex` on a class beats the `hidden`
    attribute (which the browser applies in the user-agent sheet, of lower specificity).
    Without this the spinner showed as soon as the page loaded, claiming to wait for an
    authentication nobody had started. */
 .waiting[hidden]{{display:none}}
 .spin{{width:13px;height:13px;border:2px solid var(--border);
   border-top-color:var(--muted);border-radius:50%;animation:sp .7s linear infinite;
   flex:none}}
 /* After `.spin`, on purpose: it inherits the box and overrides the colour. Inside a
    button the spinner has to follow the button's text, otherwise it is invisible on
    `button.primary`, which inverts the background/text pair. */
 .spin-btn{{display:inline-block;vertical-align:-2px;margin-right:.45rem;
   border-color:currentColor;border-top-color:transparent;opacity:.7}}
 button:disabled{{opacity:.75;cursor:progress}}
 @keyframes sp{{to{{transform:rotate(360deg)}}}}
 /* The two buttons of the same decision, side by side. In separate `form`s the browser
    put them on different lines, and «Rediscover» looked like it belonged to the next
    block. */
 .actions{{display:flex;gap:.5rem;align-items:center}}
 .actions form{{margin:0}}
 /* The dialog uses the card's own tokens: it is the LiteLLM surface, not a system window.
    `::backdrop` darkens the rest without hiding where you were. */
 dialog{{border:1px solid var(--border);border-radius:var(--radius);
   background:var(--card);color:var(--foreground);padding:1.25rem 1.5rem;
   max-width:min(460px,90vw);box-shadow:0 10px 30px rgba(0,0,0,.18)}}
 dialog::backdrop{{background:rgba(0,0,0,.45)}}
 dialog h3{{font-size:1rem;font-weight:600;margin:0 0 .75rem}}
 dialog form{{margin:.75rem 0 0}}
 .diff{{margin:.25rem 0 .9rem;padding-left:1.1rem}}
 .diff li{{margin:.15rem 0}}
 .diff code{{font-size:.78rem}}
 .added{{color:var(--on-fg);font-size:.8rem;font-weight:500;margin:0}}
 .removed{{color:var(--warn-fg);font-size:.8rem;font-weight:500;margin:0}}
 .error{{background:var(--warn-bg);color:var(--warn-fg);padding:.75rem 1rem;
   border-radius:var(--radius);margin-bottom:1rem}}
 /* Disconnect is the only destructive action on the page: it is set apart by colour, but
    only in the outline and the text. A solid red button next to «Apply» competed for the
    eye with the action the user actually wants to take. */
 .danger summary{{color:var(--warn-fg)}}
 button.danger{{border-color:var(--warn-fg);color:var(--warn-fg);background:transparent}}
 button.danger:hover{{background:var(--warn-bg)}}
 .alt{{display:block;margin-top:.75rem;border-top:1px solid var(--border);
   padding-top:.75rem}}
 .alt summary{{cursor:pointer;color:var(--muted);font-size:.8rem;
   list-style:revert}}
 .alt summary:hover{{color:var(--foreground)}}
 .alt p{{margin:.6rem 0}}
 .pair{{border-color:var(--ring)}}
 /* `pre` with `overflow-x` instead of wrapping: the command carries a code the user will
    copy whole, and a visual wrap ends up pasted into the terminal as two lines. */
 .cmd{{background:var(--accent);border:1px solid var(--border);
   border-radius:var(--radius);padding:.7rem .9rem;margin:0 0 .9rem;overflow-x:auto;
   font-family:ui-monospace,SFMono-Regular,monospace;font-size:.8rem;
   user-select:all}}
 code{{background:var(--accent);border-radius:.25rem;padding:.1rem .3rem;
   font-family:ui-monospace,SFMono-Regular,monospace;font-size:.8rem}}
</style></head>
<body><div class="wrap">
<h1><svg class="mark" viewBox="0 0 24 24" fill="none" stroke="currentColor"
  stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path
  d="M2.586 17.414A2 2 0 0 0 2 18.828V21a1 1 0 0 0 1 1h3a1 1 0 0 0 1-1v-1a1 1 0 0 1 1-1h1a1
  1 0 0 0 1-1v-1a1 1 0 0 1 1-1h.172a2 2 0 0 0 1.414-.586l.814-.814a6.5 6.5 0 1 0-4-4z"/><circle
  cx="16.5" cy="7.5" r=".5" fill="currentColor"/></svg>MySubs</h1>
<p class="lead">Connect your subscriptions and serve them as LiteLLM models.</p>
<div id="error"></div>
{body}
</div>
<dialog id="done">
  <h3>Models applied</h3>
  <div id="done-body"></div>
  <form method="dialog"><button class="primary">Close</button></form>
</dialog>
<script>
 const q = new URLSearchParams(location.search);
 const esc = {{"<": "&lt;", ">": "&gt;", "&": "&amp;"}};
 const clean = s => s.replace(/[<>&]/g, c => esc[c]);
 const e = q.get("error");
 if (e) document.getElementById("error").innerHTML =
   '<div class="error">' + clean(e) + '</div>';

 // What changed in the Router, in a dialog. A silent redirect left the user comparing
 // checkboxes to find out whether what they unticked actually went — and the answer to
 // that question is the only thing they want at that moment.
 //
 // A function and not a loose block: with a stored key the page arrives by `fetch` and is
 // rewritten with `document.write`, and there this script runs with the **old** URL — the
 // `q` at the top has already been read. The `fetch` handler calls this again with the
 // right URL.
 function mysubsDone(search) {{
   var p = new URLSearchParams(search);
   if (!p.has("applied")) return false;
   var list = function (title, csv, cls) {{
     if (!csv) return "";
     var items = csv.split(",").filter(Boolean)
       .map(n => '<li><code>' + clean(n) + '</code></li>').join("");
     return '<p class="' + cls + '">' + title + '</p><ul class="diff">' + items + '</ul>';
   }};
   var body = list("Added", p.get("added"), "added")
            + list("Removed", p.get("removed"), "removed");
   var box = document.getElementById("done-body");
   if (!box) return false;
   var empty = '<p class="muted">Nothing changed: that was already the selection.</p>';
   box.innerHTML = body || empty;
   var d = document.getElementById("done");
   if (d && d.showModal) d.showModal();
   return true;
 }}

 if (mysubsDone(location.search)) {{
   // Clears the query string so an F5 does not repeat the dialog, but keeps the anchor —
   // it is what holds the page at the card it came from.
   history.replaceState(null, "", location.pathname + location.hash);
 }}

 // Restores the position at the card it came from. `?card=` is the route that survives a
 // `fetch` — measured: `fetch` with `redirect: "follow"` discards the fragment of
 // `Location`, and without this the page reappeared at the top on every button pressed.
 (function () {{
   var target = q.get("card") || location.hash.slice(1);
   if (!target) return;
   var el = document.getElementById(target);
   if (el) el.scrollIntoView({{block: "start"}});
   if (q.has("card") && !q.has("applied")) {{
     history.replaceState(null, "", location.pathname + "#" + target);
   }}
 }})();

 // Resend of the stored key. LiteLLM's `user_api_key_auth` only reads headers — a
 // `location.reload()` does not carry them, and without this the pasted key was good for
 // nothing. Top-level navigations do not allow headers, so the page is fetched with
 // `fetch` and the document replaced. It only runs when there is a key **and** the current
 // page is a refusal: otherwise every load would make a doubled request.
 (function () {{
   var k = sessionStorage.getItem("mysubs_key");
   if (!k || !document.querySelector(".chip.warn")) return;
   fetch(location.href, {{headers: {{Authorization: "Bearer " + k}}}})
     .then(r => r.status === 200 ? r.text() : null)
     .then(html => {{
       if (!html) {{ sessionStorage.removeItem("mysubs_key"); return; }}
       document.open(); document.write(html); document.close();
     }})
     .catch(() => {{}});
 }})();

 // A waiting state on any button that submits. Antigravity discovery is measured in
 // seconds — it probes the catalogue and verifies each model — and with no signal the user
 // presses again, generating a second discovery that only delays the first.
 //
 // `capture: true` so it runs before the `fetch` handler below: if that one cancels the
 // navigation, the button is already marked.
 document.addEventListener("submit", function (ev) {{
   var f = ev.target;
   if (!f || f.tagName !== "FORM") return;
   var b = f.querySelector("button");
   if (!b || b.dataset.busy) return;
   b.dataset.busy = "1";
   b.dataset.label = b.textContent;
   b.disabled = true;
   b.innerHTML = '<span class="spin spin-btn"></span>' + b.dataset.label;
 }}, true);

 // The buttons are POST `form`s, which also carry no headers. With a stored key, they are
 // submitted by `fetch` and the redirect is followed by hand.
 //
 // `data-connect` stays out: that one has its own handling further down, and catching it
 // here first — this handler is on the `document`, so it runs earlier — would swap the new
 // window for a navigation that kills the probe.
 document.addEventListener("submit", function (ev) {{
   var k = sessionStorage.getItem("mysubs_key");
   var f = ev.target;
   if (f.hasAttribute("data-connect")) return;
   if (!k || !f.method || f.method.toLowerCase() !== "post") return;
   ev.preventDefault();
  // The button's `formaction` beats the form's `action` — it is what lets «Rediscover»
  // live inside the «Apply» form without separating them visually.
  var target = (ev.submitter && ev.submitter.getAttribute("formaction")) || f.action;
  fetch(target, {{
    method: "POST",
    headers: {{Authorization: "Bearer " + k}},
    body: new FormData(f),
    redirect: "follow",
  }})
    .then(function (r) {{
      // `document.write` replaces the document and **discards the URL**, including the
      // anchor the server sent — measured: `location.hash` came out empty and the page
      // reappeared at the top. Restoring the response's final URL is what holds the
      // position.
      return r.text().then(function (html) {{ return {{html: html, url: r.url}}; }});
    }})
    .then(function (out) {{
      document.open(); document.write(out.html); document.close();
      if (out.url) history.replaceState(null, "", out.url);
      // `?card=` and not the hash: `fetch` discards the fragment of `Location`, and
      // `scrollIntoView` has to run **after** the `document.write` — the new page's script
      // reads the URL before this `replaceState` restores it.
      var search = out.url ? out.url.split("?")[1] || "" : location.search;
      var card = new URLSearchParams(search).get("card") || location.hash.slice(1);
      if (card) {{
        var el = document.getElementById(card);
        if (el) el.scrollIntoView({{block: "start"}});
      }}
      // For the same reason: the new page's script has already run with the old URL, so
      // the result dialog has to be opened from here.
      if (window.mysubsDone && mysubsDone(search)) {{
        history.replaceState(null, "", location.pathname + (card ? "#" + card : ""));
      }}
    }})
    .catch(function () {{}});
 }});

 // -- automatic connection ----------------------------------------------------
 //
 // What makes this possible, and what does not. The URL of the window that failed **cannot**
 // be read: different origin, `SecurityError`, and no trick gets around it — it is the
 // guarantee that stops a site from seeing where you are in another tab.
 //
 // What can be done, measured: this page, served by the proxy, does a `fetch` to the
 // `localhost` **of the browser's machine** and reads the response, as long as the callback
 // server sends `Access-Control-Allow-Origin`. That is how we know the login has finished.
 //
 // So: the provider is opened in a new window, the local port is probed until it says
 // `done`, and the page reloads. If the port never answers, none of this happens and the
 // user has the paste where it always was.
 (function () {{
  // Filled in by the server from `oauth.callback_origin`, which reads the same
  // `redirect_uri` that goes in the authorization request. Repeating the table here would
  // diverge silently: Antigravity registers `127.0.0.1` and the others `localhost`, and the
  // callback server binds a single family for a literal — probing the wrong host would
  // never find it, and the page would say there is no interceptor when there is.
  var ENDPOINTS = {endpoints};

  function statusUrl(provider) {{
    return ENDPOINTS[provider] + "/mysubs-status";
  }}

   // A short probe: the goal is to know whether anyone is listening, not to wait on the
   // network.
   function probe(provider, ms) {{
     var ctl = new AbortController();
     var timer = setTimeout(function () {{ ctl.abort(); }}, ms || 1200);
     return fetch(statusUrl(provider), {{signal: ctl.signal, cache: "no-store"}})
       .then(function (r) {{ return r.ok ? r.json() : null; }})
       .catch(function () {{ return null; }})
       .finally(function () {{ clearTimeout(timer); }});
   }}

   document.querySelectorAll("form[data-connect]").forEach(function (form) {{
     var provider = form.getAttribute("data-connect");
    form.addEventListener("submit", function (ev) {{
      // The top-level navigation is replaced by a new window **plus** a probe. A 303
      // would take the browser away from this page and the probe would die with it.
      ev.preventDefault();
      var key = sessionStorage.getItem("mysubs_key");
      var opts = {{method: "POST"}};
      if (key) opts.headers = {{Authorization: "Bearer " + key}};
      fetch(form.action + "?url=1", opts)
        .then(function (r) {{ return r.json(); }})
        .then(function (data) {{
          if (!data || !data.url) {{ location.reload(); return; }}
          window.open(data.url, "_blank", "noopener");
          var wait = document.querySelector('[data-waiting="' + provider + '"]');
          if (wait) wait.hidden = false;
          watch(provider);
        }})
        .catch(function () {{ form.submit(); }});
    }});
  }});

   // Probes until the callback arrives. The ceiling exists so the page does not keep
   // knocking on a port forever when the user gives up halfway through the login.
   function watch(provider) {{
     var deadline = Date.now() + 300000;
     (function tick() {{
       if (Date.now() > deadline) return;
       probe(provider, 1500).then(function (s) {{
         if (s && s.done) {{ location.reload(); return; }}
         setTimeout(tick, 1500);
       }});
     }})();
   }}

  // On load: if there is a receiver listening, we say the paste step will not be needed.
  // The box is **not** hidden: the probe can be right and the flow fail afterwards (port
  // forwarded to the wrong proxy, receiver from another session), and then whoever had
  // lost the box would be left with no way out in the middle of the login.
  document.querySelectorAll("details[data-manual]").forEach(function (box) {{
    var provider = box.getAttribute("data-manual");
    probe(provider, 1000).then(function (s) {{
      if (!s || !s.mysubs) return;
      box.open = false;
      var mark = document.createElement("p");
      mark.className = "muted auto-ok";
      mark.textContent = "Receiver detected: this login should close by itself.";
      box.parentNode.insertBefore(mark, box);
    }});
  }});
 }})();
</script>
</body></html>"""
