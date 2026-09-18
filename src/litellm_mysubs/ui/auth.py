"""Quem pode mexer nas subscrições.

A página liga contas pessoais e injecta modelos no Router: é administração, não consulta.
Usa-se a autenticação que o proxy já tem — `user_api_key_auth`, a mesma dependência de
`/v1/models` e das rotas de gestão — em vez de inventar uma.

Exige-se `proxy_admin` estrito. O `allowed_route_check_inside_route` do LiteLLM aceita
também `proxy_admin_viewer`, o que está certo para ler listas e errado aqui: um papel de
leitura não deve poder iniciar um fluxo OAuth que associa a subscrição pessoal de alguém à
instalação, nem alterar que modelos o Router serve.
"""

from __future__ import annotations

import os
from typing import Any

#: Desliga a guarda. Existe porque a alternativa é pior: sem escape, quem corre o proxy
#: sem base de dados de chaves — e portanto sem forma de ter um `proxy_admin` — ficaria de
#: fora e acabaria a montar a sub-app à mão, sem guarda nenhuma e sem o saber.
#:
#: O nome diz o que faz. Ninguém escreve isto por engano.
OPT_OUT_ENV = "MYSUBS_DISABLE_AUTH"

#: Papel exigido. Escrito à mão em vez de importado: `LitellmUserRoles` vive em
#: `litellm.proxy._types`, um módulo privado, e o valor é estável na API pública do proxy.
ADMIN_ROLE = "proxy_admin"


class AuthUnavailableError(RuntimeError):
    """O proxy não expõe a dependência de autenticação."""


def auth_disabled(environ: dict[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return str(env.get(OPT_OUT_ENV, "")).strip().lower() in ("1", "true", "yes")


def _forbidden(detail: str) -> Exception:
    from fastapi import HTTPException

    return HTTPException(status_code=403, detail=detail)


def require_admin(user: Any) -> Any:
    """Deixa passar só um administrador do proxy.

    Recebe o que o `user_api_key_auth` devolve. Um papel em falta é recusa, não omissão
    tolerada: uma chave sem papel atribuído não é prova de privilégio.
    """
    role = getattr(user, "user_role", None)
    value = getattr(role, "value", role)
    if value != ADMIN_ROLE:
        raise _forbidden(
            "MySubs é só para administradores do proxy: liga subscrições pessoais e "
            f"altera os modelos servidos. Papel actual: {value or 'nenhum'}."
        )
    return user


def admin_dependency() -> Any:
    """A dependência a pôr na sub-app, ou `None` quando a guarda está desligada.

    Levanta se o proxy não tiver `user_api_key_auth` — montar sem guarda e sem avisar seria
    exactamente o modo de falha que esta função existe para evitar.
    """
    if auth_disabled():
        return None

    try:
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    except ImportError as error:  # pragma: no cover - depende da versão do proxy
        raise AuthUnavailableError(
            "o LiteLLM instalado não expõe `user_api_key_auth`: actualiza o proxy ou "
            f"define {OPT_OUT_ENV}=1 se a instalação for de confiança"
        ) from error

    from fastapi import Depends

    #: Singleton de módulo: `Depends(...)` avaliado em assinatura é recusado pelo ruff
    #: (B008), e com razão — um default avaliado no import esconde a ordem de arranque.
    authenticated: Any = Depends(user_api_key_auth)

    async def guard(user: Any = authenticated) -> Any:
        return require_admin(user)

    return Depends(guard)
