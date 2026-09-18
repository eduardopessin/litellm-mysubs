"""Armazenamento de credenciais OAuth das subscrições.

Duas regras que vêm de incidentes medidos, não de preferência:

1. **Um só dono do refresh.** Anthropic e OpenAI emitem refresh tokens rotativos de uso
   único. Dois renovadores independentes correndo read-modify-write sobre o mesmo token
   invalidam a cópia um do outro e produzem ``invalid_grant`` em ciclo, forçando
   re-login manual. Um store com ``owns_refresh=False`` lê e nunca troca.

2. **Rotação tem de ficar viva sem restart.** Variáveis de ambiente são um instantâneo do
   arranque do processo; uma credencial renovada por outro agente nunca chegaria cá. Daí
   ``reload()`` e o TTL curto de cache.
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
    """Credencial de um provedor.

    Imutável de propósito: uma rotação produz um objecto novo em vez de mutar o que
    outra thread possa estar a ler.
    """

    provider: ProviderId
    access_token: str
    refresh_token: str = ""
    expires_at: float = 0.0
    project_id: str = ""
    """Só o Google Antigravity o usa; é descoberto no fim do fluxo OAuth."""

    def is_expired(self, *, now: float | None = None, leeway_s: float = 60.0) -> bool:
        """Se o access token já não serve.

        ``expires_at`` a zero significa "desconhecido", não "expirado": um token manual
        colado à mão não traz validade e tem de ser tentado, não descartado.
        """
        if self.expires_at <= 0:
            return False
        return (now if now is not None else time.time()) >= self.expires_at - leeway_s

    def with_access_token(self, token: str, expires_at: float = 0.0) -> Credential:
        return replace(self, access_token=token, expires_at=expires_at)


class CredentialStore(ABC):
    """Interface mínima. Implementações: ficheiro, Kubernetes Secret, ambiente."""

    #: Se este store pode trocar refresh tokens. Ver regra 1 no topo do módulo.
    owns_refresh: bool = False

    @abstractmethod
    def get(self, provider: ProviderId) -> Credential | None:
        """Credencial actual, ou ``None`` se o provedor não estiver ligado."""

    @abstractmethod
    def set(self, provider: ProviderId, credential: Credential) -> None:
        """Persiste uma credencial. Levanta ``ReadOnlyStoreError`` se for só de leitura."""

    @abstractmethod
    def delete(self, provider: ProviderId) -> None:
        """Remove a credencial de um provedor."""

    @abstractmethod
    def reload(self) -> bool:
        """Relê a fonte. Devolve ``True`` se algo mudou desde a última leitura."""

    def connected(self) -> tuple[ProviderId, ...]:
        """Provedores com credencial presente."""
        return tuple(p for p in PROVIDER_IDS if self.get(p) is not None)


class ReadOnlyStoreError(RuntimeError):
    """Escrita tentada num store de leitura (tipicamente o de variáveis de ambiente)."""


def to_payload(credential: Credential) -> dict[str, object]:
    """Credencial em dados simples, para quem a tiver de serializar."""
    return {
        "access_token": credential.access_token,
        "refresh_token": credential.refresh_token,
        "expires_at": credential.expires_at,
        "project_id": credential.project_id,
    }


def from_payload(provider: ProviderId, raw: object) -> Credential | None:
    """Credencial a partir de dados simples, ou ``None`` se não houver nada de útil.

    Uma entrada sem `access_token` não é uma credencial degradada — é ausência dela. Devolver
    um objecto vazio faria o resto do código tratar "não ligado" como "ligado e partido".
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
