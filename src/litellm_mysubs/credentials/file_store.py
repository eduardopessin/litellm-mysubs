"""File store — the default for any installation.

Stores in ``~/.litellm/mysubs/credentials.json`` with ``0600`` permissions. The file holds
refresh tokens for personal subscriptions: loose permissions are refused instead of
silently corrected, because a file that was readable by others may already have been read,
and tightening the bits afterwards undoes nothing.

Every write is a read-modify-write of the **whole** file under ``credentials.json.lock``.
The per-provider refresh locks do not cover this: two workers renewing *different*
providers at the same moment each read the file, change their own entry and replace it —
and without the file lock the second replace brings back the first one's old entry, with a
refresh token the provider has already rotated away.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Final

from .lock import file_lock
from .store import Credential, CredentialStore, ProviderId

DEFAULT_PATH = Path.home() / ".litellm" / "mysubs" / "credentials.json"

#: Group/other bits. Any one of them makes the file suspect.
_UNSAFE_BITS = stat.S_IRWXG | stat.S_IRWXO

#: How long a write waits for another process's write. A write holds the lock for one read
#: and one replace of a few hundred bytes, and a dead holder releases it with its process;
#: running out of this means a stuck file system, which is worth an error.
_WRITE_LOCK_TIMEOUT_S: Final = 5.0


class InsecurePermissionsError(RuntimeError):
    """The credentials file is readable by someone other than its owner."""


class FileCredentialStore(CredentialStore):
    owns_refresh = True

    def __init__(self, path: Path | str = DEFAULT_PATH) -> None:
        self.path = Path(path)
        self._cache: dict[str, Credential] = {}
        #: `(inode, mtime_ns, size)` of what `_cache` holds. The inode is what catches a
        #: write another process made within the same mtime tick: every write replaces the
        #: file, so it always lands on a new inode.
        self._seen: tuple[int, int, int] | None = None
        self.reload()

    # -- reading ---------------------------------------------------------------

    def _check(self, info: os.stat_result) -> tuple[int, int, int]:
        """Refuses loose permissions; returns the identity of what was looked at."""
        if info.st_mode & _UNSAFE_BITS:
            raise InsecurePermissionsError(
                f"{self.path} has permissions {stat.filemode(info.st_mode)}; "
                f"it holds refresh tokens and must be 0600. "
                f"Fix it with: chmod 600 {self.path}"
            )
        return info.st_ino, info.st_mtime_ns, info.st_size

    def _load(self) -> None:
        """Reads the file into the cache, unconditionally.

        The identity comes from the descriptor that was read, not from a second look at
        the path: a replace landing in between would otherwise pair new contents with the
        old identity, or the other way round.
        """
        try:
            with open(self.path, encoding="utf-8") as handle:
                seen = self._check(os.fstat(handle.fileno()))
                text = handle.read()
        except FileNotFoundError:
            self._cache, self._seen = {}, None
            return
        raw: dict[str, Any] = json.loads(text or "{}")
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
        self._seen = seen

    def reload(self) -> bool:
        """Re-reads the file, always.

        This is what the refresh path calls inside its lock, just before deciding whether to
        spend a refresh token, so it cannot trust the metadata: on a shared volume the
        attributes a ``stat`` returns may be cached, while opening the file revalidates it.
        """
        before = dict(self._cache)
        self._load()
        return self._cache != before

    def _sync(self) -> None:
        """The cheap re-read of the request path: only when the file looks different."""
        try:
            info = self.path.stat()
        except FileNotFoundError:
            self._cache, self._seen = {}, None
            return
        if self._check(info) != self._seen:
            self._load()

    def get(self, provider: ProviderId) -> Credential | None:
        self._sync()
        return self._cache.get(provider)

    # -- writing ---------------------------------------------------------------

    def _write(self) -> None:
        """Replaces the file with the cache. Call only while holding the file lock."""
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
        # credentials file disconnects every subscription at once. The `fsync` comes before
        # the replace so the name never points at contents still sitting in a cache — the
        # file carries a rotated refresh token, and the old one is already dead upstream.
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".credentials-")
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
                # Taken from the descriptor: `os.replace` keeps inode and mtime, and a stat
                # of the path afterwards could already be describing someone else's write.
                seen = self._check(os.fstat(handle.fileno()))
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        self._seen = seen

    def set(self, provider: ProviderId, credential: Credential) -> None:
        with file_lock(self.path, timeout_s=_WRITE_LOCK_TIMEOUT_S):
            self._load()
            self._cache[provider] = credential
            self._write()

    def delete(self, provider: ProviderId) -> None:
        with file_lock(self.path, timeout_s=_WRITE_LOCK_TIMEOUT_S):
            self._load()
            if self._cache.pop(provider, None) is not None:
                self._write()

    # omp: auth/sqlite-credential-store.ts :: tryUpdateAuthCredentialIfMatches
    def update_if_matches(
        self, provider: ProviderId, expected: Credential, credential: Credential
    ) -> bool:
        """Writes ``credential`` only while the file still holds ``expected``.

        A renewal that finishes after a login, or after a peer's renewal, must not bring
        the credential it started from back over theirs. ``False`` means someone else wrote
        first; nothing was written.

        A ``credential`` equal to what the file holds matches but is not written: the same
        bytes again would only move the file's mtime and inode, and every other worker's
        `_sync` would re-read it for nothing.
        """
        with file_lock(self.path, timeout_s=_WRITE_LOCK_TIMEOUT_S):
            self._load()
            if self._cache.get(provider) != expected:
                return False
            if credential == expected:
                return True
            self._cache[provider] = credential
            self._write()
            return True

    # omp: auth/sqlite-credential-store.ts :: tryDisableAuthCredentialIfMatches
    def delete_if_matches(self, provider: ProviderId, expected: Credential) -> bool:
        """Removes the credential only while the file still holds ``expected``.

        It is how a dead grant is dropped without dropping the fresh one a peer or a login
        wrote while the failing exchange was in the air.
        """
        with file_lock(self.path, timeout_s=_WRITE_LOCK_TIMEOUT_S):
            self._load()
            if self._cache.get(provider) != expected:
                return False
            del self._cache[provider]
            self._write()
            return True
