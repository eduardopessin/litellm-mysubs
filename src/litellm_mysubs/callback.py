"""O `CustomLogger` que o `config.yaml` carrega.

    litellm_settings:
      callbacks: ["litellm_mysubs.MySubs"]

É o único ponto de entrada do pacote numa instalação normal. Tudo o resto — o patch, a UI,
o registry — é accionado a partir daqui.

## A regra que não se quebra

**Este callback nunca altera o pedido.** O `async_pre_call_hook` do LiteLLM pode devolver um
`data` modificado, e é assim que guardrails e injectores de prompt funcionam. Aqui devolve-se
sempre `None`: o hook serve só de gatilho de arranque, e um plugin cujo objectivo é
*acrescentar* modelos não tem negócio nenhum a mexer nos pedidos dos que já existiam.

Pela mesma razão não se implementa `async_filter_deployments`: filtrar deployments é mexer
no roteamento de modelos que não são nossos.
"""

from __future__ import annotations

import contextlib
from typing import Any

from .bootstrap import Bootstrap, disabled
from .credentials.store import ProviderId


def _base() -> type:
    """A classe base: `CustomLogger` se o LiteLLM estiver presente, `object` se não.

    O proxy **exige** `isinstance(loaded, CustomLogger)` — medido: um objecto com os hooks
    certos mas sem a herança faz `load_config` levantar e o arranque falha por completo. Não
    é opcional.

    A resolução é tardia para `litellm_mysubs` continuar importável sem o LiteLLM: o
    `mysubs-setup` corre antes de haver proxy configurado, e os testes desta camada não
    devem arrastar o pacote inteiro.
    """
    try:
        from litellm.integrations.custom_logger import CustomLogger

        return CustomLogger
    except ImportError:
        return object


class MySubs(_base()):  # type: ignore[misc]
    """Liga as subscrições ao proxy."""

    def __init__(self, store: Any = None) -> None:
        self._bootstrap = Bootstrap()
        self._store = store

    # -- estado ----------------------------------------------------------------

    @property
    def store(self) -> Any:
        """O store de credenciais, resolvido à primeira utilização.

        Tardio porque a escolha depende de `litellm.secret_manager_client`, que o proxy só
        preenche depois de processar `key_management_system` — e este objecto é construído
        antes disso.
        """
        if self._store is None:
            from .ui.install import default_store

            self._store = default_store()
        return self._store

    @property
    def status(self) -> dict[str, Any]:
        """Para diagnóstico: `mysubs-setup --estado` mostra isto."""
        return {
            "disabled": disabled(),
            "patched": self._bootstrap.patched,
            "mounted": self._bootstrap.mounted,
            "error": self._bootstrap.error,
        }

    # -- arranque --------------------------------------------------------------

    def setup(self, app: Any = None) -> dict[str, Any]:
        """Monta a UI e aplica o patch se houver subscrição ligada.

        Chamado pelos hooks e pelo `mysubs-setup`. Idempotente.
        """
        if disabled():
            return self.status
        target = app if app is not None else _proxy_app()
        if target is not None:
            self._bootstrap.mount(target, self.store)
        self._bootstrap.patch(self.store)
        return self.status

    # -- hooks do LiteLLM ------------------------------------------------------

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any = None,
        cache: Any = None,
        data: dict[str, Any] | None = None,
        call_type: str = "",
    ) -> None:
        """Gatilho de arranque. **Devolve sempre `None`.**

        `None` significa "não modifiquei nada" e é o que garante que o pedido segue
        exactamente como chegou. Devolver `data` aqui — mesmo intacto — poria este plugin no
        caminho de escrita de todos os pedidos da instalação, incluindo os dos modelos que
        já lá estavam.
        """
        self.setup()
        return None

    async def async_post_call_success_hook(
        self,
        data: dict[str, Any] | None = None,
        user_api_key_dict: Any = None,
        response: Any = None,
    ) -> None:
        """Absorve o uso que vier nos cabeçalhos da resposta.

        É como os cards sabem a quota: uma subscrição não tem endpoint de uso, o estado só
        viaja nas respostas. Um erro aqui não pode afectar a resposta que o cliente recebe.
        """
        with contextlib.suppress(Exception):
            self._observe(data, response)
        return None

    def _observe(self, data: dict[str, Any] | None, response: Any) -> None:
        from .ui.install import shared_service

        service = shared_service()
        if service is None:
            return
        headers = getattr(response, "_hidden_params", {}) or {}
        headers = headers.get("additional_headers") or {}
        if not headers:
            return
        model = str((data or {}).get("model", ""))
        provider = _provider_of(model)
        if provider:
            service.observe(provider, headers)


def _provider_of(model: str) -> ProviderId | None:
    from .wire import codex
    from .wire.anthropic import is_anthropic_model

    lowered = model.lower()
    if "gemini" in lowered or "antigravity" in lowered:
        return "google-antigravity"
    if codex.is_codex_model(model):
        return "openai-codex"
    if is_anthropic_model(model):
        return "anthropic"
    return None


def _proxy_app() -> Any:
    try:
        from litellm.proxy import proxy_server

        return getattr(proxy_server, "app", None)
    except Exception:
        return None
