"""Injecção do item "MySubs" no menu do LiteLLM.

Os testes desta camada existem porque o trabalho real foi feito contra o proxy a correr, e
três defeitos só apareceram lá. Cada um está fixado abaixo.
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

#: A forma real do menu, tal como aparece no chunk minificado do LiteLLM 1.101.0. O
#: `icon:(0,a.jsx)($.FlaskConical,{...e0})` está aqui de propósito: foi o que partiu a
#: primeira regex.
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
        """A primeira versão usava `[^}]*?` entre a chave e o `children`. Entre eles está
        `icon:(0,a.jsx)($.FlaskConical,{...e0})`, que tem chavetas — a classe negada parava
        aí e a injecção falhava em silêncio no bundle real."""
        assert "{...e0}" in REAL_MENU
        assert inject(REAL_MENU).ok

    def test_existing_children_survive(self) -> None:
        patched = inject(REAL_MENU).chunk
        assert '{key:"prompts"' in patched
        assert '{key:"tag-management"' in patched

    def test_other_menu_groups_are_untouched(self) -> None:
        """Injectar no sítio errado moveria entradas de outro grupo."""
        patched = inject(REAL_MENU).chunk
        assert '{key:"api-keys",page:"api-keys",label:"Virtual Keys"}' in patched

    def test_the_entry_uses_external_url(self) -> None:
        """`/mysubs` é servida por uma sub-app FastAPI, não por uma rota do Next: sem
        `external_url` o router da SPA tentava resolvê-la internamente e mostrava o 404
        dela."""
        assert f'external_url:"{UI_PATH}/"' in MENU_ENTRY

    def test_injecting_twice_does_not_duplicate(self) -> None:
        once = inject(REAL_MENU).chunk
        assert inject(once).chunk == ""  # devolve "já presente" sem mexer

    def test_an_unrecognised_bundle_fails_soft(self) -> None:
        """Um botão em falta é uma inconveniência; um arranque falhado é uma avaria. A
        razão tem de nomear a alternativa — o URL directo."""
        result = inject('e2=[{groupLabel:"outra coisa",items:[]}]')
        assert result.ok is False
        assert UI_PATH in result.reason


class TestRouteWiring:
    def _app_with_static(self, tmp_path: Path) -> tuple[FastAPI, Path]:
        chunk_dir = tmp_path / "_next" / "static" / "chunks"
        chunk_dir.mkdir(parents=True)
        (chunk_dir / "menu.js").write_text(REAL_MENU, "utf-8")
        app = FastAPI()
        # Reproduz as três montagens do proxy (`proxy_server.py:2048-2060`).
        for prefix in ("/_next", "/litellm-asset-prefix/_next"):
            app.mount(prefix, StaticFiles(directory=str(tmp_path / "_next")), name=prefix)
        app.mount("/ui", StaticFiles(directory=str(tmp_path)), name="ui")
        return app, tmp_path

    def test_a_route_added_after_the_mount_never_wins(self, tmp_path: Path) -> None:
        """O FastAPI resolve por ordem de registo. Foi o primeiro defeito: a rota era
        acrescentada e o `StaticFiles` continuava a servir o original."""
        app, _ = self._app_with_static(tmp_path)

        async def patched() -> Any:
            from fastapi.responses import Response

            return Response("PATCHED", media_type="application/javascript")

        app.add_api_route("/ui/_next/static/chunks/menu.js", patched, methods=["GET"])
        assert TestTextOf(app, "/ui/_next/static/chunks/menu.js") != "PATCHED"

    def test_every_asset_prefix_serves_the_patched_chunk(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """O segundo defeito, e o mais caro: a UI carrega os chunks por
        `/litellm-asset-prefix/_next`, não por `/ui/_next`. Patchar um só caminho dava o
        chunk modificado a quem o pedisse à mão e o original ao browser — 29650 bytes
        contra 29533, com o mesmo URL aparente.
        """
        app, root = self._app_with_static(tmp_path)
        monkeypatch.setattr("litellm_mysubs.ui.menu.find_ui_root", lambda: root)
        assert install_menu(app).ok

        for prefix in ("/ui/_next", "/litellm-asset-prefix/_next", "/_next"):
            body = TestTextOf(app, f"{prefix}/static/chunks/menu.js")
            assert '{key:"mysubs"' in body, prefix

    def test_the_patched_chunk_is_not_cacheable(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Terceiro defeito: o browser guarda os chunks por hash de nome, e o nome não muda
        quando o conteúdo passa a ser o nosso. Sem `no-store`, a página servia o original
        até alguém limpar a cache à mão."""
        app, root = self._app_with_static(tmp_path)
        monkeypatch.setattr("litellm_mysubs.ui.menu.find_ui_root", lambda: root)
        install_menu(app)
        response = TestClient(app).get("/ui/_next/static/chunks/menu.js")
        assert response.headers["cache-control"] == "no-store"

    def test_nothing_is_written_to_site_packages(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Editar o chunk no disco deixaria um `pip install --force-reinstall` a restaurar o
        original sem aviso, e um ficheiro modificado a confundir quem investigue."""
        app, root = self._app_with_static(tmp_path)
        chunk = root / "_next" / "static" / "chunks" / "menu.js"
        before = chunk.read_text("utf-8")
        monkeypatch.setattr("litellm_mysubs.ui.menu.find_ui_root", lambda: root)
        install_menu(app)
        assert chunk.read_text("utf-8") == before


class TestRealBundle:
    def test_the_installed_litellm_bundle_is_recognised(self) -> None:
        """Se esta falhar numa versão nova do LiteLLM, o botão deixou de aparecer — e a
        página continua a funcionar por URL. É o aviso, não uma avaria."""
        root = find_ui_root()
        if root is None:
            return  # LiteLLM sem UI compilada
        chunk = find_menu_chunk(root)
        assert chunk is not None, "o chunk do menu não foi encontrado"
        assert inject(chunk.read_text("utf-8")).ok


def TestTextOf(app: FastAPI, path: str) -> str:  # noqa: N802 - auxiliar, não é um teste
    return TestClient(app).get(path).text
