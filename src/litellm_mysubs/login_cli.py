"""`mysubs-login` — the token interceptor, run on the user's machine.

Why this is a separate command and not a proxy route: these clients' OAuth redirect is
``http://localhost:54545/callback`` (and ``:1455``, ``:51121``), and ``localhost`` resolves
in the user's **browser**. A callback server opened inside the LiteLLM pod would be
listening on the wrong machine: the browser would hit the workstation's loopback and find
nothing. It is the same reason Quota Desktop exists as a native application instead of a
dashboard page.

This command closes that gap without forcing an app install: it runs where the browser runs,
catches the code, exchanges it for a credential, and deposits the result in the proxy —
local or remote. The proxy authorizes the deposit with a single-use pairing code issued on
the page, not with the administrator key: a code that lives for ten minutes and only works
for one provider is far less dangerous material to leave in a terminal scrollback than the
key that administers the whole proxy.

The paste still exists and is still the path that survives everything — occupied port, SSH
without a tunnel, machine without a browser. This command is the transparent path for
whoever has a browser on the same machine; it does not replace the other one.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import urllib.parse
import webbrowser

import httpx

from .credentials import callback_server
from .credentials.oauth import AuthRequest, begin, complete
from .credentials.store import PROVIDER_IDS, Credential, ProviderId, to_payload

#: Where to deposit the credential on the proxy. Relative to the `/mysubs` mount.
DEPOSIT_PATH = "/mysubs/api/deposit"

#: Names the user types, mapped to the internal ids. `claude` and `chatgpt` exist because
#: that is what people call the subscriptions — refusing the name they use in order to
#: demand the internal id would be pedantry with a cost.
ALIASES: dict[str, ProviderId] = {
    "anthropic": "anthropic",
    "claude": "anthropic",
    "codex": "openai-codex",
    "openai": "openai-codex",
    "openai-codex": "openai-codex",
    "chatgpt": "openai-codex",
    "antigravity": "google-antigravity",
    "google": "google-antigravity",
    "google-antigravity": "google-antigravity",
    "gemini": "google-antigravity",
}


def resolve_provider(name: str) -> ProviderId:
    """The internal id from what the user typed."""
    key = name.strip().lower()
    resolved = ALIASES.get(key)
    if resolved is None:
        known = ", ".join(sorted(set(ALIASES)))
        raise SystemExit(f"unknown provider: {name!r}\nknown: {known}")
    return resolved


def callback_target(request: AuthRequest) -> tuple[int, str, str]:
    """`(port, path, host)` of the redirect the provider has registered.

    Derived from the authorization URL itself instead of duplicated in a table: the redirect
    the callback server opens has to be, byte for byte, the one that goes in the request.
    Two sources for the same value diverge, and the divergence shows up as an opaque
    provider error after the user has already signed in.
    """
    query = urllib.parse.parse_qs(urllib.parse.urlparse(request.url).query)
    values = query.get("redirect_uri") or []
    if not values:
        raise SystemExit("the authorization URL carries no `redirect_uri`")
    parsed = urllib.parse.urlparse(values[0])
    if parsed.port is None:
        raise SystemExit(f"the redirect {values[0]!r} has no explicit port")
    return parsed.port, parsed.path or "/callback", parsed.hostname or "localhost"


async def intercept(provider: ProviderId, *, timeout_s: float, open_browser: bool) -> Credential:
    """Runs the whole flow: opens the port, hands off to the browser, exchanges the code.

    The port opens **before** the browser is launched. The other way around, a fast redirect
    — already authenticated account, cached consent — hits a port that is still closed and
    the user sees a connection error page without realizing the flow was correct.
    """
    request = begin(provider)
    port, path, host = callback_target(request)

    waiter = asyncio.ensure_future(
        callback_server.serve_once(
            port=port,
            path=path,
            expected_state=request.state,
            host=host,
            timeout_s=timeout_s,
        )
    )
    # An immediate bind failure (occupied port) has to surface before the browser opens:
    # sending the user off to authenticate only to tell them afterwards that the port is
    # taken would spend a login for nothing.
    await asyncio.sleep(0)
    if waiter.done():
        await waiter

    print(f"Listening on http://{host}:{port}{path}")
    if open_browser and webbrowser.open(request.url):
        print("Opened the browser. Sign in on the window that appeared.")
    else:
        print("Open this URL in your browser:\n")
        print(f"  {request.url}\n")

    try:
        result = await waiter
    except BaseException:
        waiter.cancel()
        raise
    print("Code received. Exchanging it for a credential…")

    async with httpx.AsyncClient() as client:
        return await complete(provider, request, result.code, client=client)


async def deposit(
    credential: Credential, *, base_url: str, pairing_code: str, verify: bool = True
) -> None:
    """Hands the credential to the proxy, authorized by the pairing code.

    The provider does **not** travel in the body: what decides which provider the credential
    belongs to is the code, on the proxy's side. Letting the client choose would turn a code
    issued to connect Codex into a write over the Anthropic credential.
    """
    url = base_url.rstrip("/") + DEPOSIT_PATH
    async with httpx.AsyncClient(verify=verify, timeout=30.0) as client:
        response = await client.post(
            url,
            json={"pairing_code": pairing_code, "credential": to_payload(credential)},
        )
    if response.status_code != 200:
        raise SystemExit(
            f"the proxy refused the deposit (HTTP {response.status_code}):\n{response.text[:500]}"
        )


def save_local(credential: Credential) -> str:
    """Saves to the local store, for whoever runs LiteLLM on their own machine."""
    from .credentials.file_store import FileCredentialStore

    store = FileCredentialStore()
    store.set(credential.provider, credential)
    return str(store.path)


def _report(credential: Credential) -> None:
    print(f"\nConnected: {credential.provider}")
    if credential.project_id:
        print(f"  project {credential.project_id}")


async def _run(args: argparse.Namespace) -> int:
    provider = resolve_provider(args.provider)
    credential = await intercept(
        provider, timeout_s=args.timeout, open_browser=not args.no_browser
    )
    _report(credential)

    if args.json:
        print(json.dumps({"provider": provider, **to_payload(credential)}, indent=2))
        return 0

    if args.url:
        if not args.code:
            raise SystemExit("--url requires --code (the pairing code from the page)")
        await deposit(
            credential,
            base_url=args.url,
            pairing_code=args.code,
            verify=not args.insecure,
        )
        print(f"  deposited at {args.url.rstrip('/')}/mysubs")
        return 0

    print(f"  saved at {save_local(credential)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mysubs-login",
        description=(
            "Opens the callback port on this machine, signs in to the subscription and "
            "hands the credential to LiteLLM."
        ),
    )
    parser.add_argument(
        "provider", help=f"one of: {', '.join(PROVIDER_IDS)} (or claude/chatgpt/gemini)"
    )
    parser.add_argument("--url", help="remote proxy URL; without it, saves to the local store")
    parser.add_argument("--code", help="pairing code issued by the /mysubs page")
    parser.add_argument(
        "--timeout",
        type=float,
        default=callback_server.DEFAULT_TIMEOUT_S,
        help="seconds to wait for the callback (default: %(default)s)",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="do not open the browser; print the URL"
    )
    parser.add_argument("--json", action="store_true", help="print the credential and do not save")
    parser.add_argument(
        "--insecure", action="store_true", help="do not verify the proxy's TLS certificate"
    )
    args = parser.parse_args(argv)

    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130
    except callback_server.CallbackTimeoutError as error:
        print(f"\n{error}", file=sys.stderr)
        print("The paste on the /mysubs page still works.", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"\n{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
