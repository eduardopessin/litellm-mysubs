"""The "MySubs" item in the Experimental menu of the LiteLLM UI.

The proxy UI is **pre-compiled** Next.js, served from `_experimental/out/` by
`StaticFiles`. There is no extension point: the menu is a literal inside a minified chunk.

    {key:"experimental",label:"Experimental",icon:…,children:[{key:"prompts",…}]}

Two things make the injection possible without recompiling the frontend:

1. The `experimental` item has **`children`** — adding an entry is adding an element to
   that list.
2. Other entries use **`external_url`** (`learning-resources` points at
   `models.litellm.ai/cookbook`), and the renderer treats them as `<a href target="_blank">`.
   There is precedent for an item that leaves the SPA, which is what `/mysubs` needs — the
   page is served by a FastAPI sub-app, not by a Next route.

## Why this is best-effort, and has to be

The file name is a build hash (`0c63y7umyjwi-.js`) and changes with every LiteLLM version.
A patch that assumes otherwise breaks silently on the next `pip install -U litellm`.

So: **`/mysubs` always works by direct URL**, and the injection is a bonus. When the pattern
is not found, nothing fails — it is recorded that this UI version was not recognised, and
the user still has the page.

## And why the file is not rewritten

The chunk lives in `site-packages`. Editing it there would leave a `pip install
--force-reinstall` restoring the original without warning, and a modified file confusing
whoever investigates the installation. A modified copy is served **from memory**, from a
route mounted ahead of the proxy's `StaticFiles`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

#: Where the page lives. Read from `ui/app.py` instead of repeated: a configurable prefix
#: with two sources diverged silently, and the symptom would be a button pointing at a 404.
from .app import MOUNT_PATH as UI_PATH

#: The item to inject. `external_url` because the page is not a Next route: without it the
#: SPA router tried to resolve `mysubs` as an internal page and showed its 404.
#:
#: `V.KeyRound` and not the parent item's `$.FlaskConical`: a subscription is a credential,
#: and that is the icon LiteLLM itself uses in «Virtual Keys». The `V` namespace is not our
#: choice — each icon in the bundle lives in its own, and using one that is not in **that
#: chunk** gives a menu item that blows up on render. Verified: `V.KeyRound` is in
#: `0c63y7umyjwi-.js`, the same file the entry is injected into.
MENU_ENTRY: Final = (
    '{key:"mysubs",page:"mysubs",label:"MySubs",'
    "icon:(0,a.jsx)(V.KeyRound,{...e0}),"
    f'external_url:"{UI_PATH}/"}},'
)

#: The insertion point: the `children:[` of the `experimental` item.
#:
#: `[^}]*?` does not work — between the key and `children` sits
#: `icon:(0,a.jsx)($.FlaskConical,{...e0})`, which has braces, and the negated class stopped
#: there. `.{0,400}?` crosses them and the bound stops a `children:[` from another item,
#: further down the file, from being caught by mistake.
_ANCHOR = re.compile(r'(\{key:"experimental".{0,400}?children:\[)', re.DOTALL)

#: The menu file, among the chunks. Identified by content, not by name: the name is a build
#: hash.
_MENU_MARKER = '{key:"experimental"'


@dataclass(frozen=True, slots=True)
class Injection:
    """The result of attempting the injection."""

    ok: bool
    reason: str = ""
    chunk: str = ""


def find_ui_root() -> Path | None:
    """The compiled UI folder of the installed LiteLLM."""
    try:
        import litellm
    except ImportError:
        return None
    root = Path(litellm.__file__).parent / "proxy" / "_experimental" / "out"
    return root if root.is_dir() else None


def find_menu_chunk(root: Path) -> Path | None:
    """The chunk that contains the menu, looked up by content.

    Walking the files is slower than opening a fixed name, and it is what survives a LiteLLM
    upgrade. It runs once per startup.
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
    """Adds the item to the menu. Does not raise.

    Idempotent: an already-modified chunk is returned as is. Serving from memory, this
    protects against two startups on the same installation — and against the case where
    LiteLLM starts shipping the entry itself.
    """
    if '{key:"mysubs"' in source:
        return Injection(ok=True, reason="already present")
    match = _ANCHOR.search(source)
    if match is None:
        return Injection(
            ok=False,
            reason="the menu of this LiteLLM version was not recognised; "
            f"the page is still at {UI_PATH}/",
        )
    end = match.end(1)
    return Injection(ok=True, chunk=source[:end] + MENU_ENTRY + source[end:])


def patched_chunk() -> Injection:
    """The menu chunk with the item already in, ready to serve.

    Returns `ok=False` with the reason when it cannot be done — it never raises. A missing
    button is an inconvenience; a failed startup is a breakdown.
    """
    root = find_ui_root()
    if root is None:
        return Injection(ok=False, reason="LiteLLM UI not found")
    chunk = find_menu_chunk(root)
    if chunk is None:
        return Injection(ok=False, reason="menu chunk not found")
    try:
        source = chunk.read_text("utf-8")
    except OSError as error:
        return Injection(ok=False, reason=f"could not read the chunk: {error}")
    result = inject(source)
    return Injection(ok=result.ok, reason=result.reason, chunk=result.chunk or source)


#: Every prefix the same file is served under.
#:
#: The proxy mounts `_next` **three** times (`proxy_server.py:2048-2060`), and the UI loads
#: the chunks through `/litellm-asset-prefix/_next` — not through `/ui/_next`. Patching only
#: one path delivered the modified chunk to whoever asked for it by hand and the original to
#: the browser: verified, `curl` received 29650 bytes with the item and the page 29533
#: without it.
_ASSET_PREFIXES: Final = ("/ui/_next", "/litellm-asset-prefix/_next", "/_next")


def install_menu(app: Any) -> Injection:
    """Mounts the route that serves the modified chunk.

    FastAPI resolves by **registration order**: the first matching route wins. By the time
    this code runs, the proxy's `StaticFiles` is already mounted at `/ui` — a route added
    afterwards is never reached.

    Measured::

        route added after the mount -> ORIGINAL   (StaticFiles wins)
        route moved to the front    -> PATCHED

    So it is registered and moved to the front. Touching the order of the proxy's routes is
    intrusive, and is kept to the minimum: **one** entry, for **one** exact file path. No
    other route changes position relative to any other.
    """
    result = patched_chunk()
    if not result.ok or not result.chunk:
        return result

    root = find_ui_root()
    chunk = find_menu_chunk(root) if root else None
    if root is None or chunk is None:  # pragma: no cover - already covered by patched_chunk
        return Injection(ok=False, reason="UI not found")

    # The path inside `_next`, common to all three prefixes.
    inside_next = chunk.relative_to(root / "_next").as_posix()
    body = result.chunk

    from fastapi.responses import Response

    async def serve_patched_chunk() -> Response:
        # No cache: the browser keys chunks by the hash in the name, and the name does not
        # change when the content becomes ours. A cacheable response left the page serving
        # the original until the user cleared the cache by hand — that is what happened in
        # testing.
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
    return Injection(ok=True, reason=f"menu injected at {len(added)} paths")
