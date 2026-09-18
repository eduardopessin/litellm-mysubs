"""Montagem da UI no proxy do LiteLLM."""

from __future__ import annotations

from typing import Any

from ..credentials.file_store import FileCredentialStore
from ..credentials.secret_store import SecretManagerCredentialStore, is_available
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


def default_store() -> CredentialStore:
    """Onde guardar as credenciais, por ordem de preferência.

    O cofre do LiteLLM primeiro: quem configurou `key_management_system` já decidiu onde os
    segredos da instalação vivem, e escrever os das subscrições noutro sítio deixaria
    tokens em disco fora da política dele.

    Sem cofre, o ficheiro em `~/.litellm/mysubs/` com permissões `0600` — que é o que
    funciona numa instalação qualquer, sem infraestrutura.

    A escolha é feita no arranque e não muda: um store que trocasse de sítio a meio
    espalharia metade das credenciais em cada um.
    """
    if is_available():
        return SecretManagerCredentialStore()
    return FileCredentialStore()


#: O serviço montado, para quem precise de lhe falar depois — o callback usa-o para
#: entregar os cabeçalhos de uso das respostas. Um único por processo: dois serviços
#: teriam estados de descoberta diferentes e a página mostraria o de quem montou primeiro.
_SERVICE: MySubsService | None = None


def shared_service() -> MySubsService | None:
    """O serviço da UI montada, ou `None` se ainda não foi montada."""
    return _SERVICE


def install(app: Any | None = None, *, store: CredentialStore | None = None) -> str:
    """Monta `/mysubs`. Devolve o caminho montado.

    Sem `app`, usa o do proxy. Sem `store`, escolhe-se o melhor disponível.
    """
    if app is None:
        from litellm.proxy import proxy_server

        app = proxy_server.app

    global _SERVICE
    service = MySubsService(
        store=store or default_store(),
        router_source=_live_router,
    )
    mount(app, service)
    _SERVICE = service
    return MOUNT_PATH
