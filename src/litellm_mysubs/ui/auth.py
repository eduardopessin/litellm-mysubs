"""Who is allowed to touch the subscriptions.

The page connects personal accounts and injects models into the Router: that is
administration, not consultation. It uses the authentication the proxy already has —
`user_api_key_auth`, the same dependency as `/v1/models` and the management routes —
instead of inventing one.

Strict `proxy_admin` is required. LiteLLM's `allowed_route_check_inside_route` also accepts
`proxy_admin_viewer`, which is right for reading lists and wrong here: a read-only role must
not be able to start an OAuth flow that ties somebody's personal subscription to the
installation, nor change which models the Router serves.
"""

from __future__ import annotations

import os
from typing import Any

#: Turns the guard off. It exists because the alternative is worse: with no escape hatch,
#: whoever runs the proxy without a key database — and therefore with no way to have a
#: `proxy_admin` — would be locked out and would end up mounting the sub-app by hand, with
#: no guard at all and without knowing it.
#:
#: The name says what it does. Nobody writes this by accident.
OPT_OUT_ENV = "MYSUBS_DISABLE_AUTH"

#: The required role. Written out by hand instead of imported: `LitellmUserRoles` lives in
#: `litellm.proxy._types`, a private module, and the value is stable in the proxy's public
#: API.
ADMIN_ROLE = "proxy_admin"


#: Name of the LiteLLM UI session cookie.
SESSION_COOKIE = "token"


class AuthUnavailableError(RuntimeError):
    """The proxy does not expose the authentication dependency."""


def session_user(request: Any) -> Any | None:
    """The user from the UI session cookie, or `None`.

    `user_api_key_auth` only reads headers — verified, there is not a single reference to
    cookies in that module. The dashboard works around it by reading the cookie through
    `document.cookie` and building the `Authorization` in JavaScript (the cookie is
    deliberately **not** HttpOnly for that reason).

    A page served outside the SPA does none of that: the browser sends the cookie and
    nothing else, and the request was refused with 401 even coming from a valid
    administrator session. That is what happened when opening MySubs from the menu.

    The JWT is HS256 signed with the `master_key` (`auth/login_utils.py ::
    encode_ui_session_jwt`), so it can be validated here without touching the database.
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
    except ImportError:  # pragma: no cover - depends on the proxy environment
        return None
    if not master_key:
        return None

    try:
        claims = jwt.decode(raw, master_key, algorithms=["HS256"])
    except Exception:
        # An invalid or expired signature is an absent session, not an error to propagate:
        # the normal path goes on to `user_api_key_auth`, which gives the right message.
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
    """Lets only a proxy administrator through.

    It takes what `user_api_key_auth` returns. A missing role is a refusal, not a tolerated
    omission: a key with no role assigned is no proof of privilege.
    """
    role = getattr(user, "user_role", None)
    value = getattr(role, "value", role)
    if value != ADMIN_ROLE:
        raise _forbidden(
            "MySubs is for proxy admins only: it connects personal subscriptions and "
            f"changes the models being served. Current role: {value or 'none'}."
        )
    return user


def admin_dependency() -> Any:
    """The dependency to put on the sub-app, or `None` when the guard is turned off.

    It raises if the proxy has no `user_api_key_auth` — mounting with no guard and no
    warning would be exactly the failure mode this function exists to prevent.
    """
    if auth_disabled():
        return None

    try:
        import litellm.proxy.auth.user_api_key_auth  # noqa: F401
    except ImportError as error:  # pragma: no cover - depends on the proxy version
        raise AuthUnavailableError(
            "the installed LiteLLM does not expose `user_api_key_auth`: upgrade the "
            f"proxy, or set {OPT_OUT_ENV}=1 if the installation is trusted"
        ) from error

    from fastapi import Depends, Request

    async def cookie_first(request: Request) -> Any:
        """Tries the cookie; only asks for the key when there is no session.

        `user_api_key_auth` **only reads headers** — verified, not one reference to cookies
        in that module. The dashboard works around it by reading the cookie through
        `document.cookie` and building the `Authorization` in JavaScript (which is why the
        cookie is not HttpOnly). A page served outside the SPA does none of that: the
        browser sends the cookie and nothing else, and a valid administrator session was
        refused with 401.

        When there is a session, the key dependency is **not** resolved: declaring it
        unconditionally would make `user_api_key_auth` run and raise for whoever brings only
        a cookie.
        """
        user = session_user(request)
        if user is not None:
            return require_admin(user)
        return await _resolve_key_user(request)

    cookie_first.__annotations__["request"] = Request

    # `from __future__ import annotations` turns the annotations into strings, and FastAPI
    # does not resolve them in a nested function: without restoring it, it read
    # `request: "Request"` as a query parameter and answered 422 `Field required` to every
    # request.
    return Depends(cookie_first)


#: The single refusal message for a missing credential.
_NO_CREDENTIALS = (
    "MySubs needs an admin session in the LiteLLM UI, "
    "or an admin key in the Authorization header."
)


async def _resolve_key_user(request: Any) -> Any:
    """The key's user, resolving the proxy's dependencies the way FastAPI does.

    `solve_dependencies` is what fills in the `Security(...)`; without it the direct call
    blows up. It is not elegant, but it is what allows treating an absent key as a normal
    case instead of an exception — and the cookie needs that to get its turn.
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
            # "No api key passed in." is the normal case for whoever opens the page with no
            # credential; the proxy raises it as a plain `Exception`, and letting it climb
            # gave a 500 on a perfectly expected refusal.
            raise HTTPException(status_code=401, detail=str(error) or _NO_CREDENTIALS) from error
    return require_admin(user)
