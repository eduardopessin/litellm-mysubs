"""Store de variáveis de ambiente — só leitura.

Serve dois casos: um token colado à mão para experimentar sem correr o fluxo OAuth, e a
compatibilidade com instalações que já injectam os tokens por ``env`` (é assim que o
plugin original lê hoje).

Só de leitura de propósito. As variáveis de um processo são um instantâneo do arranque:
escrever nelas não persiste nada e daria a ilusão de ter guardado uma credencial.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Final

from .store import Credential, CredentialStore, ProviderId, ReadOnlyStoreError

#: Nomes herdados do plugin original, para quem já os tem definidos.
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
            "EnvCredentialStore é só de leitura: escrever em os.environ não persiste "
            "nada e o valor perde-se no próximo arranque. Usa FileCredentialStore."
        )

    def delete(self, provider: ProviderId) -> None:
        raise ReadOnlyStoreError("EnvCredentialStore é só de leitura.")

    def reload(self) -> bool:
        # O ambiente do processo não muda sozinho.
        return False
