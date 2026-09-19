"""Arranque do plugin, por um `CustomLogger` que o `config.yaml` carrega.

## Porque não um `.pth` com hook de importação

A primeira versão deste módulo registava um finder em `sys.meta_path` a partir de um
ficheiro `.pth`, para reagir à importação do proxy. Foi medido e abandonado:

    finder que levanta em meta_path[0]  ->  `import secrets` rebenta com RuntimeError

Um `.pth` corre em **todos** os processos Python do ambiente, e um finder em `meta_path[0]`
vê **todos** os imports. Um defeito nele não parte o plugin — parte o interpretador, para
`pip`, `pytest` e qualquer script no mesmo ambiente, com o erro a aparecer antes de existir
qualquer log. Para um pacote cujo objectivo é acrescentar modelos, é risco desproporcionado.

## O que se faz em vez disso

`MySubs` é um `litellm.integrations.custom_logger.CustomLogger`, a extensão que o LiteLLM
documenta. Entra por uma linha no `config.yaml`:

    litellm_settings:
      callbacks: ["litellm_mysubs.MySubs"]

O proxy instancia-o durante o arranque, dentro do fluxo dele. Consequências:

- **Não se toca no roteamento existente.** Os deployments do `config.yaml` não são lidos,
  reordenados nem substituídos. `ModelRegistry.apply` preserva tudo o que não tenha
  `model_info.managed_by == "mysubs"`, e esta é a única marca que o plugin escreve.
- **O patch só se aplica se houver o que servir.** Sem credenciais ligadas e sem modelos
  aplicados, `install()` não corre: o pacote fica presente e inerte.
- **Desinstalar é apagar a linha.** Sem ficheiros no `site-packages` a caçar.

## E a UI

Montada em `/mysubs` no primeiro pedido, por `app.mount()` — a mesma via que o proxy usa
para `/ui` e `/swagger`. Montar é acrescentar um prefixo novo; nenhuma rota existente muda
de destino.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Final

#: Desliga tudo sem editar o `config.yaml`. Existe para quando o plugin é o suspeito de um
#: problema: uma variável de ambiente é mais rápida e reversível do que desinstalar.
DISABLE_ENV: Final = "MYSUBS_DISABLE"

#: Prefixo onde a UI é montada.
UI_PATH: Final = "/mysubs"


def disabled(environ: dict[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return str(env.get(DISABLE_ENV, "")).strip().lower() in ("1", "true", "yes")


class Bootstrap:
    """Aplica o plugin uma só vez, e regista o que correu.

    Separado do `CustomLogger` para ser testável sem o LiteLLM: o que aqui está é a decisão
    de *se* e *o quê*, não o encaixe no proxy.
    """

    def __init__(self) -> None:
        self.patched = False
        self.mounted = False
        self.menu = ""
        self.error = ""
        self._lock = threading.Lock()

    # -- decisão ---------------------------------------------------------------

    def should_patch(self, store: Any) -> bool:
        """Se vale a pena mexer no caminho dos pedidos.

        Sem nenhuma subscrição ligada não há modelo de subscrição para servir, e o patch só
        acrescentaria um wrapper que delega sempre no original. Um plugin instalado e sem
        credenciais tem de ser indistinguível de um plugin ausente.
        """
        try:
            return any(store.get(provider) is not None for provider in _providers())
        except Exception:
            # Um store ilegível não é prova de que há credenciais.
            return False

    # -- aplicação -------------------------------------------------------------

    def _inject_menu(self, app: Any) -> None:
        """Acrescenta o item ao menu do LiteLLM. Falhar aqui não é falhar.

        A página funciona por URL directo; o botão é conveniência. O chunk da UI tem um
        nome que é hash de build, e uma versão nova do LiteLLM pode não ser reconhecida —
        nesse caso regista-se a razão e segue-se.
        """
        try:
            self.menu = _inject_menu_impl(app)
        except Exception as error:
            self.menu = f"não injectado: {type(error).__name__}: {error}"

    def patch(self, store: Any) -> bool:
        """Aplica o monkey-patch se houver subscrição ligada. Idempotente."""
        with self._lock:
            if self.patched or disabled() or not self.should_patch(store):
                return False
            try:
                from . import plugin

                # Sem isto o plugin não tem de onde tirar a credencial: `_access_token`
                # devolve "" e o pedido sai com `Authorization: Bearer `, que o httpx
                # recusa com `Illegal header value b'Bearer '`. O `install()` sozinho põe
                # o patch e deixa-o inútil — e o sintoma aparece longe da causa, no cliente
                # do provedor.
                plugin.configure(store=store)
                plugin.install()
                self.patched = True
                return True
            except Exception as error:
                self.error = f"patch: {type(error).__name__}: {error}"
                return False

    def mount(self, app: Any, store: Any) -> bool:
        """Monta a UI. Idempotente, e nunca sobre um prefixo já ocupado."""
        with self._lock:
            if self.mounted or disabled():
                return False
            try:
                if _already_mounted(app, UI_PATH):
                    # Outro processo ou uma montagem manual chegaram primeiro. Montar por
                    # cima criaria duas sub-apps no mesmo prefixo, com a segunda a apanhar
                    # os pedidos e a primeira a ficar inalcançável.
                    self.mounted = True
                    return False
                from .ui.install import install as install_ui

                install_ui(app, store=store)
                self.mounted = True
                self._inject_menu(app)
                return True
            except Exception as error:
                self.error = f"mount: {type(error).__name__}: {error}"
                return False


def _inject_menu_impl(app: Any) -> str:
    from .ui.menu import install_menu

    return install_menu(app).reason


def _providers() -> tuple[str, ...]:
    from .credentials.store import PROVIDER_IDS

    return PROVIDER_IDS


def _already_mounted(app: Any, path: str) -> bool:
    return any(getattr(route, "path", None) == path for route in getattr(app, "routes", []))
