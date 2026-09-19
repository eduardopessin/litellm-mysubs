"""Environment variable store — read-only.

It covers two cases: a token pasted by hand to try things out without running the OAuth
flow, and compatibility with installations that already inject the tokens through ``env``
(which is how the original plugin reads them today).

Read-only on purpose. A process's variables are a snapshot of startup: writing to them
persists nothing and would give the illusion of having saved a credential.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Final

from .store import Credential, CredentialStore, ProviderId, ReadOnlyStoreError

#: Names inherited from the original plugin, for whoever already has them set.
ENV_NAMES: Final[dict[ProviderId, tuple[str, str, str]]] = {
    "anthropic": (
        "ANTHROPIC_OAUTH_TOKEN",
        "ANTHROPIC_REFRESH_TOKEN",
        "",
    ),
    "openai-codex": (
        "OPENAI_CODEX_OAUTH_TOKEN",
        "OPENAI_CODEX_REFRESH_TOKEN",
        "",
    ),
    "google-antigravity": (
        "GOOGLE_ANTIGRAVITY_OAUTH_TOKEN",
        "GOOGLE_ANTIGRAVITY_REFRESH_TOKEN",
        "GOOGLE_ANTIGRAVITY_PROJECT_ID",
    ),
}


class EnvCredentialStore(CredentialStore):
    owns_refresh = False

    def __init__(self, environ: Mapping[str, str] | None = None) -> None:
        self._environ: Mapping[str, str] = os.environ if environ is None else environ

    def get(self, provider: ProviderId) -> Credential | None:
        access_name, refresh_name, project_name = ENV_NAMES[provider]
        access = self._environ.get(access_name, "")
        if not access:
            return None
        return Credential(
            provider=provider,
            access_token=access,
            refresh_token=self._environ.get(refresh_name, ""),
            project_id=self._environ.get(project_name, "") if project_name else "",
        )

    def set(self, provider: ProviderId, credential: Credential) -> None:
        raise ReadOnlyStoreError(
            "EnvCredentialStore is read-only: writing to os.environ does not persist "
            "anything and the value is lost on the next start. Use FileCredentialStore."
        )

    def delete(self, provider: ProviderId) -> None:
        raise ReadOnlyStoreError("EnvCredentialStore is read-only.")

    def reload(self) -> bool:
        # The process environment does not change on its own.
        return False
