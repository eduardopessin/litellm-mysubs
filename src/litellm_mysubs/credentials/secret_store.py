"""Credentials in the secret manager LiteLLM already has configured.

Anyone running LiteLLM seriously configures `general_settings.key_management_system` —
Vault, AWS Secrets Manager, Azure Key Vault, GCP. Writing the subscription tokens somewhere
else would ignore the decision the operator already made, and leave secrets on disk in a
place their policy does not cover.

The client lives in `litellm.secret_manager_client`, filled in at proxy startup. It is not
captured at construction: the store is created when the UI mounts, and at that point the
proxy has not yet read the configuration. It is asked for when needed.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Final

from .store import (
    PROVIDER_IDS,
    Credential,
    CredentialStore,
    ProviderId,
    from_payload,
    to_payload,
)

#: Secret name prefix. A predictable name is what lets the operator write the access policy
#: — `litellm-mysubs-*` in a Vault, a tag in AWS — without having to guess.
SECRET_PREFIX: Final = "litellm-mysubs"

#: Description stored with the secret where the manager supports it. An anonymous secret in
#: a shared vault is a secret nobody dares delete.
SECRET_DESCRIPTION: Final = "Subscription OAuth credential, managed by litellm-mysubs"


def secret_name(provider: ProviderId) -> str:
    return f"{SECRET_PREFIX}-{provider}"


class SecretManagerUnavailableError(RuntimeError):
    """There is no secret manager configured in LiteLLM."""


def active_client() -> Any | None:
    """The configured manager, or ``None``.

    Read on the spot — `litellm.secret_manager_client` is only filled in when the proxy
    processes `key_management_system`, after this module has been imported.
    """
    try:
        import litellm
    except ImportError:  # pragma: no cover - the package is a dependency of the proxy
        return None
    return getattr(litellm, "secret_manager_client", None)


def is_available() -> bool:
    """Whether there is somewhere to store. This is what decides which store the UI uses."""
    return active_client() is not None


class SecretManagerCredentialStore(CredentialStore):
    """Credentials in the LiteLLM vault.

    It owns the refresh: a vault is the shared source of truth, and the serialization point
    the single-owner rule demands. Two proxies against the same vault still need only one of
    them to renew — that is configuration, not something this store can enforce.
    """

    owns_refresh = True

    def __init__(self, *, client: Any | None = None) -> None:
        #: Injectable for tests; `None` means "ask LiteLLM when you need it".
        self._client = client
        self._cache: dict[ProviderId, Credential] = {}
        self._loaded = False
        self._lock = threading.Lock()

    # -- client access ---------------------------------------------------------

    def _require_client(self) -> Any:
        client = self._client if self._client is not None else active_client()
        if client is None:
            raise SecretManagerUnavailableError(
                "no `key_management_system` configured in LiteLLM: "
                "set it in general_settings or use the file store"
            )
        return client

    # -- reading ---------------------------------------------------------------

    def reload(self) -> bool:
        """Re-reads the vault. Returns ``True`` if anything changed.

        An unreadable secret does **not** clear what was cached: a network failure against
        the vault is no proof that the credential disappeared, and treating it as such would
        disconnect every subscription in the middle of a network incident.
        """
        client = self._require_client()
        with self._lock:
            fresh: dict[ProviderId, Credential] = {}
            for provider in PROVIDER_IDS:
                raw = self._read_one(client, provider)
                if raw is None:
                    # "The vault said it does not exist" is distinguished from "I could not
                    # ask": only the former removes.
                    continue
                credential = from_payload(provider, raw)
                if credential is not None:
                    fresh[provider] = credential
            changed = fresh != self._cache
            self._cache = fresh
            self._loaded = True
            return changed

    def _read_one(self, client: Any, provider: ProviderId) -> dict[str, Any] | None:
        try:
            value = client.sync_read_secret(secret_name(provider))
        except Exception:
            # Absent and unreachable are indistinguishable in LiteLLM's interface
            # (`sync_read_secret` returns `None` for one and raises for the other, but not
            # every backend honours that). Conservative: keep what was there.
            return self._cached_payload(provider)
        if not value:
            return None
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _cached_payload(self, provider: ProviderId) -> dict[str, Any] | None:
        existing = self._cache.get(provider)
        return None if existing is None else dict(to_payload(existing))

    def get(self, provider: ProviderId) -> Credential | None:
        if not self._loaded:
            self.reload()
        return self._cache.get(provider)

    # -- writing ---------------------------------------------------------------

    def set(self, provider: ProviderId, credential: Credential) -> None:
        """Writes to the vault.

        LiteLLM's interface only has an asynchronous write — `sync_write_secret` does not
        exist on `BaseSecretManager`. The *coroutine* is run here instead of making
        `CredentialStore` async: the contract is shared with the file and environment
        stores, which are synchronous by nature, and an `async def set` would force every
        caller to change because of one implementation.
        """
        client = self._require_client()
        _run(
            client.async_write_secret(
                secret_name=secret_name(provider),
                secret_value=json.dumps(to_payload(credential), sort_keys=True),
                description=SECRET_DESCRIPTION,
            )
        )
        with self._lock:
            self._cache[provider] = credential

    def delete(self, provider: ProviderId) -> None:
        client = self._require_client()
        _run(client.async_delete_secret(secret_name=secret_name(provider)))
        with self._lock:
            self._cache.pop(provider, None)


def _run(coro: Any) -> Any:
    """Runs a *coroutine* from synchronous code.

    Inside a running loop — which is the case in the UI, served by FastAPI — `asyncio.run`
    raises. So a thread with its own loop is used: it blocks the handler for the duration of
    the write, which is the right behaviour here. Saving a credential and replying before it
    is in the vault would show "connected" to someone who, after a restart, would not be.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: list[Any] = []
    error: list[BaseException] = []

    def worker() -> None:
        try:
            result.append(asyncio.run(coro))
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result[0] if result else None
