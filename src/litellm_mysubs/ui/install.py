"""Montagem da UI no proxy do LiteLLM."""

from __future__ import annotations

from typing import Any

from ..credentials.file_store import FileCredentialStore
from ..credentials.store import CredentialStore
from .app import MOUNT_PATH, mount
from .service import MySubsService


def _live_router() -> Any:
    """O Router do proxy, lido **no momento** em vez de capturado.

    O proxy só cria o `llm_router` no arranque, depois de a sub-app estar montada. Guardar
    a referência na montagem fixava `None` para sempre e o botão de aplicar nunca faria
    nada — o modo de falha que este pacote existe para evitar.
    """
    from litellm.proxy import proxy_server

    return getattr(proxy_server, "llm_router", None)


def install(app: Any | None = None, *, store: CredentialStore | None = None) -> str:
    """Monta `/mysubs`. Devolve o caminho montado.

    Sem `app`, usa o do proxy. O store por omissão é o de ficheiro, que é o que funciona
    numa instalação qualquer — o do Kubernetes exige um ServiceAccount montado.
    """
    if app is None:
        from litellm.proxy import proxy_server

        app = proxy_server.app

    service = MySubsService(
        store=store or FileCredentialStore(),
        router_source=_live_router,
    )
    mount(app, service)
    return MOUNT_PATH
