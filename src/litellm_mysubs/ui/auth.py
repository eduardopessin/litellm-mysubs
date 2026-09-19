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


#: Nome do cookie de sessão da UI do LiteLLM.
SESSION_COOKIE = "token"


class AuthUnavailableError(RuntimeError):
    """O proxy não expõe a dependência de autenticação."""


def session_user(request: Any) -> Any | None:
    """O utilizador a partir do cookie de sessão da UI, ou `None`.

    O `user_api_key_auth` só lê cabeçalhos — verificado, não há uma única referência a
    cookies nesse módulo. A dashboard contorna isso lendo o cookie por `document.cookie` e
    construindo o `Authorization` em JavaScript (o cookie é deliberadamente **não**
    HttpOnly por essa razão).

    Uma página servida fora da SPA não faz nada disso: o browser envia o cookie e mais
    nada, e o pedido era recusado com 401 mesmo vindo de uma sessão de administrador
    válida. Era o que acontecia ao abrir o MySubs pelo menu.

    O JWT é HS256 assinado com a `master_key` (`auth/login_utils.py ::
    encode_ui_session_jwt`), portanto pode ser validado aqui sem tocar na base de dados.
    """
    raw = None
    cookies = getattr(request, "cookies", None)
    if isinstance(cookies, dict):
        raw = cookies.get(SESSION_COOKIE)
    if not raw:
        return None

    try:
        import jwt
        from litellm.proxy.proxy_server import master_key
    except ImportError:  # pragma: no cover - depende do ambiente do proxy
        return None
    if not master_key:
        return None

    try:
        claims = jwt.decode(raw, master_key, algorithms=["HS256"])
    except Exception:
        # Assinatura inválida ou expirada é ausência de sessão, não erro a propagar: o
        # caminho normal segue para o `user_api_key_auth`, que dá a mensagem certa.
        return None

    from types import SimpleNamespace

    return SimpleNamespace(
        user_role=claims.get("user_role"),
        user_id=claims.get("user_id"),
        user_email=claims.get("user_email"),
    )


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
        import litellm.proxy.auth.user_api_key_auth  # noqa: F401
    except ImportError as error:  # pragma: no cover - depende da versão do proxy
        raise AuthUnavailableError(
            "o LiteLLM instalado não expõe `user_api_key_auth`: actualiza o proxy ou "
            f"define {OPT_OUT_ENV}=1 se a instalação for de confiança"
        ) from error

    from fastapi import Depends, Request

    async def cookie_first(request: Request) -> Any:
        """Tenta o cookie; só pede a chave se não houver sessão.

        O `user_api_key_auth` **só lê cabeçalhos** — verificado, nem uma referência a
        cookies nesse módulo. A dashboard contorna-o lendo o cookie por `document.cookie` e
        construindo o `Authorization` em JavaScript (por isso o cookie não é HttpOnly). Uma
        página servida fora da SPA não faz nada disso: o browser envia o cookie e mais
        nada, e uma sessão de administrador válida era recusada com 401.

        Quando há sessão, a dependência da chave **não** é resolvida: declará-la sempre
        faria o `user_api_key_auth` correr e levantar em quem só traz cookie.
        """
        user = session_user(request)
        if user is not None:
            return require_admin(user)
        return await _resolve_key_user(request)

    cookie_first.__annotations__["request"] = Request

    # `from __future__ import annotations` torna as anotações strings, e o FastAPI não as
    # resolve numa função aninhada: sem repor, ele lia `request: "Request"` como parâmetro
    # de query e respondia 422 `Field required` a todos os pedidos.
    return Depends(cookie_first)


#: Mensagem única da recusa por ausência de credencial.
_NO_CREDENTIALS = (
    "MySubs precisa de sessão de administrador na UI do LiteLLM, "
    "ou de uma chave de administrador no cabeçalho Authorization."
)


async def _resolve_key_user(request: Any) -> Any:
    """O utilizador da chave, resolvendo as dependências do proxy como o FastAPI faz.

    `solve_dependencies` é o que preenche os `Security(...)`; sem ele a chamada directa
    rebenta. Não é elegante, mas é o que permite tratar a ausência de chave como um caso
    normal em vez de uma excepção — e o cookie precisa disso para ter a sua vez.
    """
    from contextlib import AsyncExitStack

    from fastapi import HTTPException
    from fastapi.dependencies.utils import get_dependant, solve_dependencies
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    dependant = get_dependant(path=request.url.path, call=user_api_key_auth)
    async with AsyncExitStack() as stack:
        solved = await solve_dependencies(
            request=request,
            dependant=dependant,
            async_exit_stack=stack,
            embed_body_fields=False,
        )
        if solved.errors:
            raise HTTPException(status_code=401, detail=_NO_CREDENTIALS)
        try:
            user = await user_api_key_auth(**solved.values)
        except HTTPException:
            raise
        except Exception as error:
            # "No api key passed in." é o caso normal de quem abre a página sem
            # credencial; o proxy levanta-o como `Exception` simples, e deixá-lo subir dava
            # 500 numa recusa perfeitamente esperada.
            raise HTTPException(status_code=401, detail=str(error) or _NO_CREDENTIALS) from error
    return require_admin(user)
