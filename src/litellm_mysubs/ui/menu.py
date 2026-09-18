"""O item "MySubs" no menu Experimental da UI do LiteLLM.

A UI do proxy é Next.js **pré-compilado**, servida de `_experimental/out/` por
`StaticFiles`. Não há ponto de extensão: o menu é um literal dentro de um chunk minificado.

    {key:"experimental",label:"Experimental",icon:…,children:[{key:"prompts",…}]}

Duas coisas tornam a injecção possível sem recompilar o frontend:

1. O item `experimental` tem **`children`** — acrescentar uma entrada é acrescentar um
   elemento a essa lista.
2. Outras entradas usam **`external_url`** (o `learning-resources` aponta para
   `models.litellm.ai/cookbook`), e o renderer trata-as como `<a href target="_blank">`.
   Há precedente para um item que sai da SPA, que é o que `/mysubs` precisa — a página é
   servida por uma sub-app FastAPI, não por uma rota do Next.

## Porque isto é best-effort, e tem de ser

O nome do ficheiro é um hash de build (`0c63y7umyjwi-.js`) e muda a cada versão do LiteLLM.
Um patch que assuma o contrário parte em silêncio no `pip install -U litellm` seguinte.

Portanto: **`/mysubs` funciona sempre por URL directo**, e a injecção é um extra. Quando o
padrão não é encontrado, não se falha — regista-se que esta versão da UI não foi
reconhecida, e o utilizador continua a ter a página.

## E porque não se reescreve o ficheiro

O chunk vive no `site-packages`. Editá-lo lá deixaria um `pip install --force-reinstall` a
restaurar o original sem aviso, e um ficheiro modificado a confundir quem investigue a
instalação. Serve-se uma cópia alterada **em memória**, a partir de uma rota montada antes
do `StaticFiles` do proxy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

#: Onde a página vive. Tem de bater com `ui/app.py :: MOUNT_PATH`.
UI_PATH: Final = "/mysubs"

#: O item a injectar. `external_url` porque a página não é uma rota do Next: sem ele o
#: router da SPA tentava resolver `mysubs` como página interna e mostrava o 404 dela.
MENU_ENTRY: Final = (
    '{key:"mysubs",page:"mysubs",label:"MySubs",'
    "icon:(0,a.jsx)($.FlaskConical,{...e0}),"
    f'external_url:"{UI_PATH}/"}},'
)

#: O ponto de inserção: o `children:[` do item `experimental`.
#:
#: `[^}]*?` não serve — entre a chave e o `children` está
#: `icon:(0,a.jsx)($.FlaskConical,{...e0})`, que tem chavetas, e a classe negada parava aí.
#: `.{0,400}?` atravessa-as e o limite impede que um `children:[` de outro item, mais
#: abaixo no ficheiro, seja apanhado por engano.
_ANCHOR = re.compile(r'(\{key:"experimental".{0,400}?children:\[)', re.DOTALL)

#: O ficheiro do menu, entre os chunks. Identificado pelo conteúdo, não pelo nome: o nome é
#: um hash de build.
_MENU_MARKER = '{key:"experimental"'


@dataclass(frozen=True, slots=True)
class Injection:
    """O resultado de tentar injectar."""

    ok: bool
    reason: str = ""
    chunk: str = ""


def find_ui_root() -> Path | None:
    """A pasta da UI compilada do LiteLLM instalado."""
    try:
        import litellm
    except ImportError:
        return None
    root = Path(litellm.__file__).parent / "proxy" / "_experimental" / "out"
    return root if root.is_dir() else None


def find_menu_chunk(root: Path) -> Path | None:
    """O chunk que contém o menu, procurado pelo conteúdo.

    Percorrer os ficheiros é mais lento que abrir um nome fixo, e é o que sobrevive a uma
    actualização do LiteLLM. Corre uma vez por arranque.
    """
    chunks = root / "_next" / "static" / "chunks"
    if not chunks.is_dir():
        return None
    for path in sorted(chunks.glob("*.js")):
        try:
            if _MENU_MARKER in path.read_text("utf-8", errors="ignore"):
                return path
        except OSError:
            continue
    return None


def inject(source: str) -> Injection:
    """Acrescenta o item ao menu. Não levanta.

    Idempotente: um chunk já modificado é devolvido como está. A servir-se de memória, isto
    protege contra dois arranques na mesma instalação — e contra o caso em que o LiteLLM
    passe a trazer a entrada de origem.
    """
    if '{key:"mysubs"' in source:
        return Injection(ok=True, reason="já presente")
    match = _ANCHOR.search(source)
    if match is None:
        return Injection(
            ok=False,
            reason="o menu desta versão do LiteLLM não foi reconhecido; "
            f"a página continua em {UI_PATH}/",
        )
    end = match.end(1)
    return Injection(ok=True, chunk=source[:end] + MENU_ENTRY + source[end:])


def patched_chunk() -> Injection:
    """O chunk do menu já com o item, pronto a servir.

    Devolve `ok=False` com a razão quando não dá — nunca levanta. Um botão em falta é uma
    inconveniência; um arranque falhado é uma avaria.
    """
    root = find_ui_root()
    if root is None:
        return Injection(ok=False, reason="UI do LiteLLM não encontrada")
    chunk = find_menu_chunk(root)
    if chunk is None:
        return Injection(ok=False, reason="chunk do menu não encontrado")
    try:
        source = chunk.read_text("utf-8")
    except OSError as error:
        return Injection(ok=False, reason=f"não consegui ler o chunk: {error}")
    result = inject(source)
    return Injection(ok=result.ok, reason=result.reason, chunk=result.chunk or source)


#: Todos os prefixos por onde o mesmo ficheiro é servido.
#:
#: O proxy monta o `_next` **três** vezes (`proxy_server.py:2048-2060`), e a UI carrega os
#: chunks por `/litellm-asset-prefix/_next` — não por `/ui/_next`. Patchar só um caminho
#: entregava o chunk modificado a quem o pedisse à mão e o original ao browser: verificado,
#: o `curl` recebia 29650 bytes com o item e a página 29533 sem ele.
_ASSET_PREFIXES: Final = ("/ui/_next", "/litellm-asset-prefix/_next", "/_next")


def install_menu(app: Any) -> Injection:
    """Monta a rota que serve o chunk modificado.

    O FastAPI resolve por **ordem de registo**: a primeira rota que casa ganha. Quando este
    código corre, o `StaticFiles` do proxy já está montado em `/ui` — uma rota acrescentada
    a seguir nunca é alcançada.

    Medido::

        rota acrescentada depois do mount -> ORIGINAL   (o StaticFiles ganha)
        rota movida para o início         -> PATCHED

    Por isso regista-se e move-se para a frente. Mexer na ordem das rotas do proxy é
    intrusivo, e limita-se ao mínimo: **uma** entrada, para **um** caminho exacto de
    ficheiro. Nenhuma outra rota muda de posição relativa entre si.
    """
    result = patched_chunk()
    if not result.ok or not result.chunk:
        return result

    root = find_ui_root()
    chunk = find_menu_chunk(root) if root else None
    if root is None or chunk is None:  # pragma: no cover - já coberto por patched_chunk
        return Injection(ok=False, reason="UI não encontrada")

    # O caminho dentro de `_next`, comum aos três prefixos.
    inside_next = chunk.relative_to(root / "_next").as_posix()
    body = result.chunk

    from fastapi.responses import Response

    async def serve_patched_chunk() -> Response:
        # Sem cache: o browser guarda os chunks por hash de nome, e o nome não muda quando
        # o conteúdo passa a ser o nosso. Uma resposta cacheável deixava a página a servir
        # o original até o utilizador limpar a cache à mão — foi o que aconteceu no teste.
        return Response(
            content=body,
            media_type="application/javascript",
            headers={"Cache-Control": "no-store"},
        )

    routes = app.router.routes
    added = []
    for prefix in _ASSET_PREFIXES:
        route_path = f"{prefix}/{inside_next}"
        app.add_api_route(
            route_path,
            serve_patched_chunk,
            methods=["GET"],
            include_in_schema=False,
        )
        routes.insert(0, routes.pop())
        added.append(route_path)
    return Injection(ok=True, reason=f"menu injectado em {len(added)} caminhos")
