"""A sub-app FastAPI servida em `/mysubs`.

Só HTTP e HTML: o fluxo vive em `service.py`. A página é servida sem build step nem
dependências de frontend — um wheel que precisasse de `npm` para mostrar quatro cards
seria pior de instalar do que o problema que resolve.
"""

from __future__ import annotations

import html
from typing import Any

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from ..credentials.store import PROVIDER_IDS, ProviderId
from .auth import admin_dependency
from .service import MySubsService, ProviderCard

#: Sentinela do `guard`: distingue "não passei nada" de "passei `None` de propósito".
_UNSET: Any = object()


def _resolve_guard(guard: Any) -> Any:
    """A dependência a aplicar. `_UNSET` significa "usa a do proxy"."""
    return admin_dependency() if guard is _UNSET else guard


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


def _duration(seconds: float) -> str:
    """Só a duração: ``3 min``, ``2 h``, ``4 dias``.

    Separada da frase de propósito. A versão anterior embutia "expira em" e era reutilizada
    para as reposições de quota, o que produzia "repõe expira em 1 h" no ecrã.
    """
    if seconds < 3600:
        return f"{int(seconds // 60)} min"
    if seconds < 86400:
        return f"{int(seconds // 3600)} h"
    return f"{int(seconds // 86400)} dias"


def _age(seconds: float | None) -> str:
    """Validade do token, em texto.

    Um instantâneo sem idade mente por omissão: quem vê "ligado" assume "a funcionar". Por
    isso a validade aparece sempre que é conhecida, e a ausência dela é dita, não escondida.
    """
    if seconds is None:
        return "validade desconhecida"
    if seconds <= 0:
        return "expirado"
    return f"expira em {_duration(seconds)}"


def build_app(service: MySubsService, *, guard: Any | None = _UNSET) -> FastAPI:
    """A sub-app. Recebe o serviço em vez de o construir: é o que a torna testável.

    `guard` é a dependência de autenticação, aplicada a **todas** as rotas. O default não é
    `None` — é um sentinela que manda perguntar ao `auth`: um default sem guarda tornaria
    "esqueci-me de passar" indistinguível de "decidi não proteger", e o primeiro é o erro
    que expõe a página.
    """
    dependencies = [] if guard is None else [_resolve_guard(guard)]
    app = FastAPI(
        title="MySubs",
        docs_url=None,
        redoc_url=None,
        dependencies=[d for d in dependencies if d is not None],
    )
    _install_error_pages(app)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        # O Antigravity não publica uso nos cabeçalhos: tem de ser pedido. Uma falha aqui
        # não pode esconder a página — os cards dos outros provedores continuam válidos, e
        # o card dele mostra o último instantâneo conhecido com a idade.
        await service.refresh_usage()
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


def _install_error_pages(app: FastAPI) -> None:
    """Converte falhas de autenticação numa página legível.

    O `user_api_key_auth` levanta `ProxyException`, que é uma `Exception` simples — o proxy
    trata-a num handler registado na app dele. Uma sub-app montada **não herda** esse
    handler, e sem isto a recusa chegava ao browser como::

        HTTP 500  Internal Server Error
        RuntimeError: Caught handled exception, but response already started.

    O acesso ficava negado, que é o que importa, mas quem via o 500 não tinha como saber
    que lhe faltava uma chave. Um erro que não ensina é um erro por resolver.
    """
    from fastapi import HTTPException
    from fastapi.requests import Request

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
    except ImportError:  # pragma: no cover - sem LiteLLM não há o que converter
        pass


def _error_page(status: int, detail: str) -> str:
    hint = (
        "Entra na UI do LiteLLM com uma chave de administrador e volta a abrir o MySubs."
        if status in (401, 403)
        else ""
    )
    return _SHELL.format(
        body=(
            '<section class="card"><header><h2>Sem acesso</h2>'
            f'<span class="chip warn">HTTP {status}</span></header>'
            f'<p class="muted">{html.escape(detail)}</p>'
            + (f'<p class="muted">{html.escape(hint)}</p>' if hint else "")
            + "</section>"
        )
    )


def _back(message: str) -> RedirectResponse:
    """Volta à página com o erro à vista.

    O erro vai no query string em vez de ser engolido: o `error_description` do provedor é
    o que diz ao utilizador o que fazer a seguir.
    """
    return RedirectResponse(f"{MOUNT_PATH}/?erro={html.escape(message)}", status_code=303)


def mount(
    app: Any, service: MySubsService, *, path: str = MOUNT_PATH, guard: Any | None = _UNSET
) -> None:
    """Monta a sub-app no proxy.

    `app.mount()` é a via que o próprio LiteLLM usa para `/ui` e `/swagger`.
    """
    app.mount(path, build_app(service, guard=guard))


# -- HTML ----------------------------------------------------------------------


def _page(cards: list[ProviderCard], service: MySubsService) -> str:
    return _SHELL.format(body="\n".join(_card(card, service) for card in cards))


def _usage(card: ProviderCard) -> str:
    """As barras de uso, ou uma frase honesta quando o provedor não publica nada.

    Uma barra a zero num provedor sem dados seria lida como "por usar" — o oposto do que se
    sabe, que é nada.
    """
    if not card.usage.known:
        return '<p class="nodata">Este provedor não publica uso.</p>'

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
                resets=html.escape(f"repõe em {_duration(resets)}") if resets else "",
            )
        )
    meta = []
    if card.usage.plan:
        meta.append(f"plano {html.escape(card.usage.plan)}")
    if card.usage.credits_balance:
        meta.append(f"{html.escape(card.usage.credits_balance)} créditos")
    age = card.usage.age_s()
    meta.append("agora" if age < 60 else f"há {int(age // 60)} min")
    return _USAGE.format(bars="".join(bars), meta=html.escape(" · ".join(meta)))


def _card(card: ProviderCard, service: MySubsService) -> str:
    if not card.connected:
        return _CARD.format(
            label=html.escape(card.label),
            badge='<span class="chip off">por ligar</span>',
            usage="",
            body=_CONNECT.format(provider=card.provider),
        )

    badge = (
        '<span class="chip warn">token expirado</span>'
        if card.stale
        else '<span class="chip on">ligado</span>'
    )
    detail = _age(card.expires_in_s)
    if card.project_id:
        detail += f" · projecto {card.project_id}"
    if card.applied:
        detail += f" · {card.applied} no Router"

    models = service.discovered.get(card.provider)
    if models is None:
        body = _DISCOVER.format(provider=card.provider, detail=html.escape(detail))
    else:
        chosen = set(service.selected.get(card.provider) or [])
        rows = "".join(
            _ROW.format(
                name=html.escape(model.suggested_name),
                wire=html.escape(model.wire_name),
                checked=" checked" if not chosen or model.suggested_name in chosen else "",
                mark=(
                    '<span class="chip on">verificado</span>'
                    if model.verified
                    else f'<span class="chip unk" title="{html.escape(model.note)}">'
                    "por verificar</span>"
                ),
            )
            for model in models
        )
        body = _APPLY.format(provider=card.provider, detail=html.escape(detail), rows=rows)

    return _CARD.format(label=html.escape(card.label), badge=badge, usage=_usage(card), body=body)


_BAR = """<div class="bar">
  <div class="bar-head"><span>{label}</span><span class="pct">{pct}%</span></div>
  <div class="track"><div class="fill {tone}" style="width:{pct}%"></div></div>
  <div class="resets">{resets}</div>
</div>"""

_USAGE = """<div class="usage">{bars}</div><p class="stamp">{meta}</p>"""

_CONNECT = """
<p class="muted">Liga a tua subscrição. O provedor devolve um código — cola-o abaixo.</p>
<form method="post" action="connect/{provider}"><button class="primary">Conectar</button></form>
<form method="post" action="paste/{provider}" class="paste">
  <input name="pasted" placeholder="cola aqui a URL de retorno ou o código" autocomplete="off">
  <button>Concluir</button>
</form>
"""

_DISCOVER = """
<p class="muted">{detail}</p>
<form method="post" action="discover/{provider}"><button class="primary">Ver modelos</button></form>
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
  <td><label><input type="checkbox" name="chosen" value="{name}"{checked}>
    <span>{name}</span></label></td>
  <td class="wire">{wire}</td><td class="mark">{mark}</td>
</tr>"""

_CARD = """<section class="card">
  <header><h2>{label}</h2>{badge}</header>
  {usage}
  {body}
</section>"""

#: Paleta extraída do bundle da UI do LiteLLM (`_experimental/out/_next/static/chunks/*.css`):
#: os mesmos tokens `--background`/`--foreground`/`--border`/`--primary`, a fonte Inter e o
#: par claro/escuro que ele define. Copiar os valores em vez de importar a folha de estilo é
#: deliberado: o nome do ficheiro é um hash de build e muda a cada versão, e uma página que
#: depende dele fica em branco depois de um `pip install -U litellm`.
_SHELL = """<!doctype html>
<html lang="pt"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MySubs</title>
<script>
 // Antes do CSS, de propósito: aplicar a classe depois da pintura dá um clarão branco a
 // cada carregamento dentro de uma UI escura. O LiteLLM guarda a escolha em
 // `localStorage.theme`; sem nada guardado, segue-se o sistema, que é o que ele faz.
 (function () {{
   try {{
     var t = localStorage.getItem("theme");
     if (!t) t = matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
     document.documentElement.className = t;
   }} catch (e) {{ /* localStorage bloqueado: fica o tema claro */ }}
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
 /* O LiteLLM não segue o tema do sistema: tem um interruptor próprio, que escreve
    `localStorage.theme` e a classe `light`/`dark` no `<html>`. Usar
    `prefers-color-scheme` aqui fazia a página ignorar essa escolha — claro dentro de uma
    UI escura, que é pior do que não ter tema nenhum. Lê-se o mesmo sítio que ele usa. */
 html.dark{{
   --background:#181818; --foreground:#f3f3f3; --card:#212121; --border:#303030;
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
 h1{{font-size:1.5rem;font-weight:600;letter-spacing:-.02em;margin:0 0 .25rem}}
 .lead{{color:var(--muted);margin:0 0 1.75rem}}
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
 .erro{{background:var(--warn-bg);color:var(--warn-fg);padding:.75rem 1rem;
   border-radius:var(--radius);margin-bottom:1rem}}
</style></head>
<body><div class="wrap">
<h1>MySubs</h1>
<p class="lead">Liga as tuas subscrições e serve-as como modelos do LiteLLM.</p>
<div id="erro"></div>
{body}
</div>
<script>
 const e = new URLSearchParams(location.search).get("erro");
 const esc = {{"<": "&lt;", ">": "&gt;", "&": "&amp;"}};
 if (e) document.getElementById("erro").innerHTML =
   '<div class="erro">' + e.replace(/[<>&]/g, c => esc[c]) + '</div>';
</script>
</body></html>"""
