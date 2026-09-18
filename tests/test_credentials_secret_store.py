"""Credenciais no gestor de segredos do LiteLLM."""

from __future__ import annotations

import json
from typing import Any

import pytest

from litellm_mysubs.credentials.secret_store import (
    SecretManagerCredentialStore,
    SecretManagerUnavailableError,
    secret_name,
)
from litellm_mysubs.credentials.store import Credential


class FakeVault:
    """Um cofre em memória com a interface do `BaseSecretManager`."""

    def __init__(self, initial: dict[str, str] | None = None) -> None:
        self.data = dict(initial or {})
        self.reads = 0
        self.unreachable = False

    def sync_read_secret(self, secret_name: str, **_: Any) -> str | None:
        self.reads += 1
        if self.unreachable:
            raise ConnectionError("cofre inalcançável")
        return self.data.get(secret_name)

    async def async_write_secret(
        self, secret_name: str, secret_value: str, **_: Any
    ) -> dict[str, Any]:
        self.data[secret_name] = secret_value
        return {"ok": True}

    async def async_delete_secret(self, secret_name: str, **_: Any) -> dict[str, Any]:
        self.data.pop(secret_name, None)
        return {"ok": True}


def stored(credential: Credential) -> str:
    return json.dumps(
        {
            "access_token": credential.access_token,
            "refresh_token": credential.refresh_token,
            "expires_at": credential.expires_at,
            "project_id": credential.project_id,
        },
        sort_keys=True,
    )


class TestRoundtrip:
    def test_a_written_credential_comes_back_whole(self) -> None:
        vault = FakeVault()
        store = SecretManagerCredentialStore(client=vault)
        credential = Credential(
            provider="anthropic",
            access_token="AT",
            refresh_token="RT",
            expires_at=1789790331.0,
            project_id="p",
        )
        store.set("anthropic", credential)
        store.reload()
        assert store.get("anthropic") == credential

    def test_the_secret_name_is_predictable(self) -> None:
        """Um nome estável é o que permite ao operador escrever a política de acesso sem
        adivinhar."""
        assert secret_name("anthropic") == "litellm-mysubs-anthropic"

    def test_deleting_removes_it_from_the_vault(self) -> None:
        vault = FakeVault()
        store = SecretManagerCredentialStore(client=vault)
        store.set("anthropic", Credential(provider="anthropic", access_token="AT"))
        store.delete("anthropic")
        assert vault.data == {}
        assert store.get("anthropic") is None


class TestResilience:
    def test_an_unreachable_vault_does_not_wipe_what_was_known(self) -> None:
        """Uma falha de rede contra o cofre não é prova de que a credencial desapareceu.

        Tratá-la como tal desligaria todas as subscrições a meio de um incidente de rede —
        exactamente quando o operador menos quer descobrir que perdeu as credenciais.
        """
        vault = FakeVault(
            {secret_name("anthropic"): stored(Credential(provider="anthropic", access_token="AT"))}
        )
        store = SecretManagerCredentialStore(client=vault)
        assert store.get("anthropic") is not None

        vault.unreachable = True
        store.reload()
        assert store.get("anthropic") is not None

    def test_a_secret_the_vault_says_is_gone_is_dropped(self) -> None:
        """Ausência confirmada é diferente de falha: esta remove."""
        vault = FakeVault(
            {secret_name("anthropic"): stored(Credential(provider="anthropic", access_token="AT"))}
        )
        store = SecretManagerCredentialStore(client=vault)
        assert store.get("anthropic") is not None

        vault.data.clear()
        store.reload()
        assert store.get("anthropic") is None

    def test_corrupt_json_is_not_a_credential(self) -> None:
        vault = FakeVault({secret_name("anthropic"): "isto não é json"})
        store = SecretManagerCredentialStore(client=vault)
        assert store.get("anthropic") is None

    def test_reload_reports_whether_anything_changed(self) -> None:
        vault = FakeVault()
        store = SecretManagerCredentialStore(client=vault)
        store.reload()
        assert store.reload() is False

        vault.data[secret_name("anthropic")] = stored(
            Credential(provider="anthropic", access_token="AT")
        )
        assert store.reload() is True


class TestAvailability:
    def test_without_a_configured_manager_it_says_so(self) -> None:
        """A mensagem tem de nomear a configuração em falta: quem a vê precisa de saber o
        que escrever no `config.yaml`."""
        store = SecretManagerCredentialStore(client=None)
        with pytest.raises(SecretManagerUnavailableError, match="key_management_system"):
            store.reload()

    def test_it_owns_the_refresh(self) -> None:
        """Um cofre é o ponto de serialização que a regra do dono único exige."""
        assert SecretManagerCredentialStore(client=FakeVault()).owns_refresh is True


class TestInsideAnEventLoop:
    @pytest.mark.asyncio
    async def test_writing_works_while_a_loop_is_running(self) -> None:
        """A UI corre dentro do loop do FastAPI e a escrita do LiteLLM é async: um
        `asyncio.run` directo levantaria `RuntimeError` no primeiro Conectar.
        """
        vault = FakeVault()
        store = SecretManagerCredentialStore(client=vault)
        store.set("anthropic", Credential(provider="anthropic", access_token="AT"))
        assert secret_name("anthropic") in vault.data
