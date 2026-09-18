"""A sub-app FastAPI servida em `/mysubs`.

Só HTTP e HTML: o fluxo vive em `service.py`. A página é servida sem build step nem
dependências de frontend — um wheel que precisasse de `npm` para mostrar quatro cards
seria pior de instalar do que o problema que resolve.
"""

from __future__ import annotations

import html
from typing import Any

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from ..credentials.store import PROVIDER_IDS, ProviderId
from .service import MySubsService, ProviderCard

#: Prefixo da montagem. O item de menu do LiteLLM aponta para aqui.
MOUNT_PATH = "/mysubs"

#: Campos de formulário como singletons: o `Form(...)` em default é avaliado no import e
#: o ruff recusa-o em assinaturas (B008).
_PASTED: Any = Form(...)
_CHOSEN: Any = Form(default=[])


def _provider_or_404(raw: str) -> ProviderId:
    if raw not in PROVIDER_IDS:
        raise HTTPException(status_code=404, detail=f"provedor desconhecido: {raw}")
    return raw


def _age(seconds: float | None) -> str:
    """Validade em texto.

    Um instantâneo sem idade mente por omissão: quem vê "ligado" assume "a funcionar". Por
    isso a validade aparece sempre que é conhecida, e a ausência dela é dita, não escondida.
    """
    if seconds is None:
        return "validade desconhecida"
    if seconds <= 0:
        return "expirado"
    if seconds < 3600:
        return f"expira em {int(seconds // 60)} min"
    if seconds < 86400:
        return f"expira em {int(seconds // 3600)} h"
    return f"expira em {int(seconds // 86400)} dias"


def build_app(service: MySubsService) -> FastAPI:
    """A sub-app. Recebe o serviço em vez de o construir: é o que a torna testável."""
    app = FastAPI(title="MySubs", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(_page(service.cards(), service))

    @app.post("/connect/{provider}")
    async def connect(provider: str) -> RedirectResponse:
        """Passo 4: começa a ligação e manda o utilizador ao provedor."""
        request = service.begin(_provider_or_404(provider))
        return RedirectResponse(request.url, status_code=303)

    @app.post("/paste/{provider}")
    async def paste(provider: str, pasted: str = _PASTED) -> RedirectResponse:
        """Passo 4: fecha a ligação com a URL de retorno colada."""
        target = _provider_or_404(provider)
        try:
            await service.complete(target, pasted.strip())
        except Exception as error:
            return _back(f"{target}: {error}")
        return RedirectResponse(MOUNT_PATH + "/", status_code=303)

    @app.post("/refresh/{provider}")
    async def refresh(provider: str) -> RedirectResponse:
        """Passo 5: renova o token."""
        target = _provider_or_404(provider)
        try:
            await service.refresh(target)
        except Exception as error:
            return _back(f"{target}: {error}")
        return RedirectResponse(MOUNT_PATH + "/", status_code=303)

    @app.post("/discover/{provider}")
    async def discover(provider: str) -> RedirectResponse:
        """Passo 6: lista o que a subscrição serve."""
        target = _provider_or_404(provider)
        try:
            await service.discover(target)
        except Exception as error:
            return _back(f"{target}: {error}")
        return RedirectResponse(MOUNT_PATH + "/", status_code=303)

    @app.post("/apply/{provider}")
    async def apply(provider: str, chosen: list[str] = _CHOSEN) -> RedirectResponse:
        """Passo 6: injecta no Router o que o utilizador escolheu."""
        target = _provider_or_404(provider)
        try:
            service.apply(target, chosen)
        except Exception as error:
            return _back(f"{target}: {error}")
        return RedirectResponse(MOUNT_PATH + "/", status_code=303)

    @app.get("/api/state")
    async def state() -> JSONResponse:
        """O mesmo estado em JSON, para quem preferir automatizar."""
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

    return app


def _back(message: str) -> RedirectResponse:
    """Volta à página com o erro à vista.

    O erro vai no query string em vez de ser engolido: o `error_description` do provedor é
    o que diz ao utilizador o que fazer a seguir.
    """
    return RedirectResponse(f"{MOUNT_PATH}/?erro={html.escape(message)}", status_code=303)


def mount(app: Any, service: MySubsService, *, path: str = MOUNT_PATH) -> None:
    """Monta a sub-app no proxy.

    `app.mount()` é a via que o próprio LiteLLM usa para `/ui` e `/swagger`.
    """
    app.mount(path, build_app(service))


# -- HTML ----------------------------------------------------------------------


def _page(cards: list[ProviderCard], service: MySubsService) -> str:
    return _SHELL.format(body="\n".join(_card(card, service) for card in cards))


def _card(card: ProviderCard, service: MySubsService) -> str:
    if not card.connected:
        return _CARD.format(
            label=html.escape(card.label),
            badge='<span class="off">por ligar</span>',
            body=_CONNECT.format(provider=card.provider),
        )

    state = (
        '<span class="warn">token expirado</span>'
        if card.stale
        else '<span class="on">ligado</span>'
    )
    detail = _age(card.expires_in_s)
    if card.project_id:
        detail += f" · projecto {html.escape(card.project_id)}"
    if card.applied:
        detail += f" · {card.applied} modelo(s) no Router"

    models = service.discovered.get(card.provider)
    if models is None:
        body = _DISCOVER.format(provider=card.provider, detail=html.escape(detail))
    else:
        chosen = set(service.selected.get(card.provider) or [])
        rows = "\n".join(
            _ROW.format(
                name=html.escape(model.suggested_name),
                wire=html.escape(model.wire_name),
                checked=" checked" if not chosen or model.suggested_name in chosen else "",
                mark=(
                    '<span class="ok">verificado</span>'
                    if model.verified
                    else f'<span class="unk" title="{html.escape(model.note)}">por verificar</span>'
                ),
            )
            for model in models
        )
        body = _APPLY.format(provider=card.provider, detail=html.escape(detail), rows=rows)

    return _CARD.format(label=html.escape(card.label), badge=state, body=body)


_CONNECT = """
<p class="muted">Liga a tua subscrição. O provedor devolve um código — cola-o abaixo.</p>
<form method="post" action="connect/{provider}">
  <button class="primary">Conectar</button>
</form>
<form method="post" action="paste/{provider}" class="paste">
  <input name="pasted" placeholder="cola aqui a URL de retorno ou o código" autocomplete="off">
  <button>Concluir</button>
</form>
"""

_DISCOVER = """
<p class="muted">{detail}</p>
<form method="post" action="discover/{provider}">
  <button class="primary">Ver modelos</button>
</form>
<form method="post" action="refresh/{provider}"><button>Renovar token</button></form>
"""

_APPLY = """
<p class="muted">{detail}</p>
<form method="post" action="apply/{provider}">
  <table>{rows}</table>
  <button class="primary">Aplicar</button>
</form>
<form method="post" action="discover/{provider}"><button>Redescobrir</button></form>
"""

_ROW = """<tr>
  <td><label><input type="checkbox" name="chosen" value="{name}"{checked}> {name}</label></td>
  <td class="wire">{wire}</td><td>{mark}</td>
</tr>"""

_CARD = """<section class="card">
  <h2>{label} {badge}</h2>
  {body}
</section>"""

_SHELL = """<!doctype html>
<html lang="pt"><head><meta charset="utf-8">
<title>MySubs</title>
<style>
 body{{font:14px/1.5 system-ui,sans-serif;margin:0;padding:2rem;background:#f6f7f9;color:#1a1a1a}}
 h1{{font-size:1.4rem;margin:0 0 .25rem}}
 .lead{{color:#666;margin:0 0 1.5rem}}
 .card{{background:#fff;border:1px solid #e3e5e8;border-radius:10px;
        padding:1.25rem;margin-bottom:1rem;max-width:760px}}
 h2{{font-size:1.05rem;margin:0 0 .75rem;display:flex;gap:.5rem;align-items:center}}
 .on,.off,.warn,.ok,.unk{{font-size:.72rem;font-weight:600;padding:.1rem .5rem;border-radius:99px}}
 .on{{background:#e6f4ea;color:#137333}} .off{{background:#eceff1;color:#5f6368}}
 .warn{{background:#fce8e6;color:#c5221f}} .ok{{background:#e6f4ea;color:#137333}}
 .unk{{background:#fef7e0;color:#b06000;cursor:help}}
 .muted{{color:#666;margin:.25rem 0 .75rem}}
 form{{display:inline-block;margin:0 .5rem .5rem 0}}
 .paste{{display:block}}
 .paste input{{width:min(460px,70%);padding:.45rem .6rem;
               border:1px solid #d0d4d9;border-radius:6px}}
 button{{padding:.45rem .9rem;border:1px solid #d0d4d9;background:#fff;
         border-radius:6px;cursor:pointer}}
 button.primary{{background:#1a73e8;border-color:#1a73e8;color:#fff}}
 table{{border-collapse:collapse;margin:.5rem 0 .75rem;width:100%}}
 td{{padding:.25rem .5rem .25rem 0;border-bottom:1px solid #f0f1f3}}
 .wire{{color:#777;font-family:ui-monospace,monospace;font-size:.8rem}}
 .erro{{background:#fce8e6;color:#c5221f;padding:.75rem 1rem;border-radius:8px;
        max-width:760px;margin-bottom:1rem}}
</style></head>
<body>
<h1>MySubs</h1>
<p class="lead">Liga as tuas subscrições e serve-as como modelos do LiteLLM.</p>
<div id="erro"></div>
{body}
<script>
 const e = new URLSearchParams(location.search).get("erro");
 const esc = {{"<": "&lt;", ">": "&gt;", "&": "&amp;"}};
 if (e) document.getElementById("erro").innerHTML =
   '<div class="erro">' + e.replace(/[<>&]/g, c => esc[c]) + '</div>';
</script>
</body></html>"""
