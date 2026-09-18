"""Credenciais no gestor de segredos que o LiteLLM já tem configurado.

Quem corre o LiteLLM a sério configura `general_settings.key_management_system` — Vault,
AWS Secrets Manager, Azure Key Vault, GCP. Escrever os tokens das subscrições noutro sítio
seria ignorar a decisão que o operador já tomou, e deixar segredos em disco num sítio que a
política dele não cobre.

O cliente vive em `litellm.secret_manager_client`, preenchido no arranque do proxy. Não é
capturado na construção: o store é criado quando a UI monta, e nessa altura o proxy ainda
não leu a configuração. Pergunta-se quando é preciso.
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

#: Prefixo dos segredos. Um nome previsível é o que permite ao operador escrever a política
#: de acesso — `litellm-mysubs-*` num Vault, uma tag no AWS — sem ter de adivinhar.
SECRET_PREFIX: Final = "litellm-mysubs"

#: Descrição gravada com o segredo onde o gestor a suporta. Um segredo anónimo num cofre
#: partilhado é um segredo que ninguém se atreve a apagar.
SECRET_DESCRIPTION: Final = "Credencial OAuth de subscrição, gerida pelo litellm-mysubs"


def secret_name(provider: ProviderId) -> str:
    return f"{SECRET_PREFIX}-{provider}"


class SecretManagerUnavailableError(RuntimeError):
    """Não há gestor de segredos configurado no LiteLLM."""


def active_client() -> Any | None:
    """O gestor configurado, ou ``None``.

    Lido no momento — `litellm.secret_manager_client` só é preenchido quando o proxy
    processa `key_management_system`, depois de este módulo ser importado.
    """
    try:
        import litellm
    except ImportError:  # pragma: no cover - o pacote é dependência do proxy
        return None
    return getattr(litellm, "secret_manager_client", None)


def is_available() -> bool:
    """Se há onde guardar. É o que decide qual store a UI usa."""
    return active_client() is not None


class SecretManagerCredentialStore(CredentialStore):
    """Credenciais no cofre do LiteLLM.

    É dono do refresh: um cofre é a fonte de verdade partilhada, e o ponto de serialização
    que a regra do dono único exige. Dois proxies contra o mesmo cofre continuam a precisar
    de que só um renove — isso é configuração, não algo que este store possa impor.
    """

    owns_refresh = True

    def __init__(self, *, client: Any | None = None) -> None:
        #: Injectável para teste; `None` significa "pergunta ao LiteLLM quando precisares".
        self._client = client
        self._cache: dict[ProviderId, Credential] = {}
        self._loaded = False
        self._lock = threading.Lock()

    # -- acesso ao cliente -----------------------------------------------------

    def _require_client(self) -> Any:
        client = self._client if self._client is not None else active_client()
        if client is None:
            raise SecretManagerUnavailableError(
                "não há `key_management_system` configurado no LiteLLM: "
                "define-o em general_settings ou usa o store de ficheiro"
            )
        return client

    # -- leitura ---------------------------------------------------------------

    def reload(self) -> bool:
        """Relê o cofre. Devolve ``True`` se algo mudou.

        Um segredo ilegível **não** apaga o que estava em cache: uma falha de rede contra o
        cofre não é prova de que a credencial desapareceu, e tratá-la como tal desligaria
        todas as subscrições a meio de um incidente de rede.
        """
        client = self._require_client()
        with self._lock:
            fresh: dict[ProviderId, Credential] = {}
            for provider in PROVIDER_IDS:
                raw = self._read_one(client, provider)
                if raw is None:
                    # Distingue-se "o cofre disse que não existe" de "não consegui
                    # perguntar": só o primeiro remove.
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
            # Ausente e inalcançável são indistinguíveis na interface do LiteLLM
            # (`sync_read_secret` devolve `None` para um e levanta para o outro, mas nem
            # todos os backends respeitam isso). Conservador: mantém-se o que havia.
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

    # -- escrita ---------------------------------------------------------------

    def set(self, provider: ProviderId, credential: Credential) -> None:
        """Grava no cofre.

        A interface do LiteLLM só tem escrita assíncrona — `sync_write_secret` não existe no
        `BaseSecretManager`. Corre-se o *coroutine* aqui em vez de tornar `CredentialStore`
        async: o contrato é partilhado com os stores de ficheiro e de ambiente, que são
        síncronos por natureza, e um `async def set` obrigaria todos os chamadores a mudar
        por causa de uma implementação.
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
    """Corre um *coroutine* a partir de código síncrono.

    Dentro de um loop a correr — que é o caso na UI, servida por FastAPI — `asyncio.run`
    levanta. Usa-se então uma thread com loop próprio: bloqueia o handler o tempo da
    escrita, que é o comportamento certo aqui. Guardar uma credencial e responder antes de
    ela estar no cofre mostraria "ligado" a quem, depois de um reinício, não estaria.
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
