"""File store — the default for any installation.

Stores in ``~/.litellm/mysubs/credentials.json`` with ``0600`` permissions. The file holds
refresh tokens for personal subscriptions: loose permissions are refused instead of
silently corrected, because a file that was readable by others may already have been read,
and tightening the bits afterwards undoes nothing.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from .store import Credential, CredentialStore, ProviderId

DEFAULT_PATH = Path.home() / ".litellm" / "mysubs" / "credentials.json"

#: Group/other bits. Any one of them makes the file suspect.
_UNSAFE_BITS = stat.S_IRWXG | stat.S_IRWXO


class InsecurePermissionsError(RuntimeError):
    """The credentials file is readable by someone other than its owner."""


class FileCredentialStore(CredentialStore):
    owns_refresh = True

    def __init__(self, path: Path | str = DEFAULT_PATH) -> None:
        self.path = Path(path)
        self._cache: dict[str, Credential] = {}
        self._mtime: float = -1.0
        self.reload()

    # -- reading ---------------------------------------------------------------

    def _check_permissions(self) -> None:
        mode = self.path.stat().st_mode
        if mode & _UNSAFE_BITS:
            raise InsecurePermissionsError(
                f"{self.path} has permissions {stat.filemode(mode)}; "
                f"it holds refresh tokens and must be 0600. "
                f"Fix it with: chmod 600 {self.path}"
            )

    def reload(self) -> bool:
        if not self.path.exists():
            changed = bool(self._cache)
            self._cache, self._mtime = {}, -1.0
            return changed

        self._check_permissions()
        mtime = self.path.stat().st_mtime
        if mtime == self._mtime:
            return False

        raw: dict[str, Any] = json.loads(self.path.read_text("utf-8") or "{}")
        self._cache = {
            provider: Credential(
                provider=provider,  # type: ignore[arg-type]
                access_token=str(entry.get("access_token", "")),
                refresh_token=str(entry.get("refresh_token", "")),
                expires_at=float(entry.get("expires_at", 0) or 0),
                project_id=str(entry.get("project_id", "")),
            )
            for provider, entry in raw.items()
            if isinstance(entry, dict) and entry.get("access_token")
        }
        self._mtime = mtime
        return True

    def get(self, provider: ProviderId) -> Credential | None:
        self.reload()
        return self._cache.get(provider)

    # -- writing ---------------------------------------------------------------

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        payload = {
            provider: {
                "access_token": c.access_token,
                "refresh_token": c.refresh_token,
                "expires_at": c.expires_at,
                "project_id": c.project_id,
            }
            for provider, c in self._cache.items()
        }
        # Atomic write: a crash halfway would leave the file truncated, and a truncated
        # credentials file disconnects every subscription at once.
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".credentials-")
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        self._mtime = self.path.stat().st_mtime

    def set(self, provider: ProviderId, credential: Credential) -> None:
        self.reload()
        self._cache[provider] = credential
        self._write()

    def delete(self, provider: ProviderId) -> None:
        self.reload()
        if self._cache.pop(provider, None) is not None:
            self._write()
