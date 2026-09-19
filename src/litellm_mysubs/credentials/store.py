"""Storage for the subscriptions' OAuth credentials.

Two rules that come from measured incidents, not from preference:

1. **A single refresh owner.** Anthropic and OpenAI issue rotating single-use refresh
   tokens. Two independent refreshers running read-modify-write over the same token
   invalidate each other's copy and produce ``invalid_grant`` in a loop, forcing a manual
   re-login. A store with ``owns_refresh=False`` reads and never exchanges.

2. **Rotation has to stay alive without a restart.** Environment variables are a snapshot
   of process startup; a credential renewed elsewhere would never reach here. Hence
   ``reload()`` and the short cache TTL.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Final, Literal

ProviderId = Literal["anthropic", "openai-codex", "google-antigravity"]

PROVIDER_IDS: Final[tuple[ProviderId, ...]] = (
    "anthropic",
    "openai-codex",
    "google-antigravity",
)


@dataclass(frozen=True, slots=True)
class Credential:
    """A provider's credential.

    Immutable on purpose: a rotation produces a new object instead of mutating what
    another thread may be reading.
    """

    provider: ProviderId
    access_token: str
    refresh_token: str = ""
    expires_at: float = 0.0
    project_id: str = ""
    """Only Google Antigravity uses it; it is discovered at the end of the OAuth flow."""

    def is_expired(self, *, now: float | None = None, leeway_s: float = 60.0) -> bool:
        """Whether the access token is no longer usable.

        ``expires_at`` at zero means "unknown", not "expired": a token pasted by hand
        carries no validity and has to be tried, not discarded.
        """
        if self.expires_at <= 0:
            return False
        return (now if now is not None else time.time()) >= self.expires_at - leeway_s

    def with_access_token(self, token: str, expires_at: float = 0.0) -> Credential:
        return replace(self, access_token=token, expires_at=expires_at)


class CredentialStore(ABC):
    """Minimal interface. Implementations: file, Kubernetes Secret, environment."""

    #: Whether this store may exchange refresh tokens. See rule 1 at the top of the module.
    owns_refresh: bool = False

    @abstractmethod
    def get(self, provider: ProviderId) -> Credential | None:
        """Current credential, or ``None`` if the provider is not connected."""

    @abstractmethod
    def set(self, provider: ProviderId, credential: Credential) -> None:
        """Persists a credential. Raises ``ReadOnlyStoreError`` if the store is read-only."""

    @abstractmethod
    def delete(self, provider: ProviderId) -> None:
        """Removes a provider's credential."""

    @abstractmethod
    def reload(self) -> bool:
        """Re-reads the source. Returns ``True`` if anything changed since the last read."""

    def connected(self) -> tuple[ProviderId, ...]:
        """Providers with a credential present."""
        return tuple(p for p in PROVIDER_IDS if self.get(p) is not None)


class ReadOnlyStoreError(RuntimeError):
    """A write attempted on a read-only store (typically the environment one)."""


def to_payload(credential: Credential) -> dict[str, object]:
    """The credential as plain data, for whoever has to serialize it."""
    return {
        "access_token": credential.access_token,
        "refresh_token": credential.refresh_token,
        "expires_at": credential.expires_at,
        "project_id": credential.project_id,
    }


def from_payload(provider: ProviderId, raw: object) -> Credential | None:
    """A credential from plain data, or ``None`` if there is nothing usable.

    An entry without `access_token` is not a degraded credential — it is the absence of one.
    Returning an empty object would make the rest of the code treat "not connected" as
    "connected and broken".
    """
    if not isinstance(raw, dict) or not raw.get("access_token"):
        return None
    return Credential(
        provider=provider,
        access_token=str(raw.get("access_token", "")),
        refresh_token=str(raw.get("refresh_token", "")),
        expires_at=float(raw.get("expires_at", 0) or 0),
        project_id=str(raw.get("project_id", "")),
    )
