"""Injection of the "MySubs" item into the LiteLLM menu.

The tests in this layer exist because the real work was done against the running proxy,
and three defects only showed up there. Each one is pinned below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

from litellm_mysubs.ui.menu import (
    MENU_ENTRY,
    UI_PATH,
    find_menu_chunk,
    find_ui_root,
    inject,
    install_menu,
)

#: The real shape of the menu, as it appears in the minified chunk of LiteLLM 1.101.0. The
#: `icon:(0,a.jsx)($.FlaskConical,{...e0})` is here on purpose: it is what broke the first
#: regex.
REAL_MENU = (
    'e2=[{groupLabel:"AI Gateway",items:[{key:"api-keys",page:"api-keys",label:"Virtual Keys"},'
    '{key:"experimental",page:"experimental",label:"Experimental",'
    "icon:(0,a.jsx)($.FlaskConical,{...e0}),children:["
    '{key:"prompts",page:"prompts",label:"Prompts",icon:(0,a.jsx)(H.FileText,{...e0})},'
    '{key:"tag-management",page:"tag-management",label:"Tag Management"}]}]}]'
)


class TestAnchor:
    def test_the_entry_lands_as_the_first_child(self) -> None:
        patched = inject(REAL_MENU)
        assert patched.ok
        assert 'children:[{key:"mysubs"' in patched.chunk

    def test_braces_inside_the_icon_do_not_stop_the_match(self) -> None:
        """The first version used `[^}]*?` between the key and the `children`. Between them
        sits `icon:(0,a.jsx)($.FlaskConical,{...e0})`, which has braces — the negated class
        stopped there and the injection failed silently on the real bundle."""
        assert "{...e0}" in REAL_MENU
        assert inject(REAL_MENU).ok

    def test_existing_children_survive(self) -> None:
        patched = inject(REAL_MENU).chunk
        assert '{key:"prompts"' in patched
        assert '{key:"tag-management"' in patched

    def test_other_menu_groups_are_untouched(self) -> None:
        """Injecting in the wrong place would move entries of another group."""
        patched = inject(REAL_MENU).chunk
        assert '{key:"api-keys",page:"api-keys",label:"Virtual Keys"}' in patched

    def test_the_entry_uses_external_url(self) -> None:
        """`/mysubs` is served by a FastAPI sub-app, not by a Next route: without
        `external_url` the SPA router tried to resolve it internally and showed its own
        404."""
        assert f'external_url:"{UI_PATH}/"' in MENU_ENTRY

    def test_injecting_twice_does_not_duplicate(self) -> None:
        once = inject(REAL_MENU).chunk
        assert inject(once).chunk == ""  # returns "already present" without touching it

    def test_an_unrecognised_bundle_fails_soft(self) -> None:
        """A missing button is an inconvenience; a failed startup is a breakdown. The
        reason has to name the alternative — the direct URL."""
        result = inject('e2=[{groupLabel:"something else",items:[]}]')
        assert result.ok is False
        assert UI_PATH in result.reason


class TestRouteWiring:
    def _app_with_static(self, tmp_path: Path) -> tuple[FastAPI, Path]:
        chunk_dir = tmp_path / "_next" / "static" / "chunks"
        chunk_dir.mkdir(parents=True)
        (chunk_dir / "menu.js").write_text(REAL_MENU, "utf-8")
        app = FastAPI()
        # Reproduces the proxy's three mounts (`proxy_server.py:2048-2060`).
        for prefix in ("/_next", "/litellm-asset-prefix/_next"):
            app.mount(prefix, StaticFiles(directory=str(tmp_path / "_next")), name=prefix)
        app.mount("/ui", StaticFiles(directory=str(tmp_path)), name="ui")
        return app, tmp_path

    def test_a_route_added_after_the_mount_never_wins(self, tmp_path: Path) -> None:
        """FastAPI resolves by registration order. This was the first defect: the route was
        added and `StaticFiles` kept serving the original."""
        app, _ = self._app_with_static(tmp_path)

        async def patched() -> Any:
            from fastapi.responses import Response

            return Response("PATCHED", media_type="application/javascript")

        app.add_api_route("/ui/_next/static/chunks/menu.js", patched, methods=["GET"])
        assert TestTextOf(app, "/ui/_next/static/chunks/menu.js") != "PATCHED"

    def test_every_asset_prefix_serves_the_patched_chunk(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """The second defect, and the most expensive one: the UI loads the chunks through
        `/litellm-asset-prefix/_next`, not through `/ui/_next`. Patching a single path gave
        the modified chunk to whoever requested it by hand and the original to the browser
        — 29650 bytes against 29533, with the same apparent URL.
        """
        app, root = self._app_with_static(tmp_path)
        monkeypatch.setattr("litellm_mysubs.ui.menu.find_ui_root", lambda: root)
        assert install_menu(app).ok

        for prefix in ("/ui/_next", "/litellm-asset-prefix/_next", "/_next"):
            body = TestTextOf(app, f"{prefix}/static/chunks/menu.js")
            assert '{key:"mysubs"' in body, prefix

    def test_the_patched_chunk_is_not_cacheable(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Third defect: the browser caches the chunks by name hash, and the name does not
        change when the content becomes ours. Without `no-store`, the page served the
        original until somebody cleared the cache by hand."""
        app, root = self._app_with_static(tmp_path)
        monkeypatch.setattr("litellm_mysubs.ui.menu.find_ui_root", lambda: root)
        install_menu(app)
        response = TestClient(app).get("/ui/_next/static/chunks/menu.js")
        assert response.headers["cache-control"] == "no-store"

    def test_nothing_is_written_to_site_packages(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Editing the chunk on disk would leave a `pip install --force-reinstall`
        restoring the original without warning, and a modified file confusing whoever
        investigates."""
        app, root = self._app_with_static(tmp_path)
        chunk = root / "_next" / "static" / "chunks" / "menu.js"
        before = chunk.read_text("utf-8")
        monkeypatch.setattr("litellm_mysubs.ui.menu.find_ui_root", lambda: root)
        install_menu(app)
        assert chunk.read_text("utf-8") == before


class TestRealBundle:
    def test_the_installed_litellm_bundle_is_recognised(self) -> None:
        """If this fails on a new LiteLLM version, the button stopped appearing — and the
        page keeps working by URL. It is the warning, not a breakdown."""
        root = find_ui_root()
        if root is None:
            return  # LiteLLM without a compiled UI
        chunk = find_menu_chunk(root)
        assert chunk is not None, "the menu chunk was not found"
        assert inject(chunk.read_text("utf-8")).ok


def TestTextOf(app: FastAPI, path: str) -> str:  # noqa: N802 - helper, not a test
    return TestClient(app).get(path).text
