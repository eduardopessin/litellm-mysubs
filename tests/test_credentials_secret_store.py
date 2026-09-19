"""Credentials in the LiteLLM secret manager."""

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
    """An in-memory vault with the `BaseSecretManager` interface."""

    def __init__(self, initial: dict[str, str] | None = None) -> None:
        self.data = dict(initial or {})
        self.reads = 0
        self.unreachable = False

    def sync_read_secret(self, secret_name: str, **_: Any) -> str | None:
        self.reads += 1
        if self.unreachable:
            raise ConnectionError("vault unreachable")
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
        """A stable name is what lets the operator write the access policy without
        guessing."""
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
        """A network failure against the vault is no proof that the credential is gone.

        Treating it as such would disconnect every subscription in the middle of a network
        incident — exactly when the operator least wants to find out the credentials are
        lost.
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
        """A confirmed absence is different from a failure: this one removes."""
        vault = FakeVault(
            {secret_name("anthropic"): stored(Credential(provider="anthropic", access_token="AT"))}
        )
        store = SecretManagerCredentialStore(client=vault)
        assert store.get("anthropic") is not None

        vault.data.clear()
        store.reload()
        assert store.get("anthropic") is None

    def test_corrupt_json_is_not_a_credential(self) -> None:
        vault = FakeVault({secret_name("anthropic"): "this is not json"})
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
        """The message has to name the missing configuration: whoever sees it needs to know
        what to write in `config.yaml`."""
        store = SecretManagerCredentialStore(client=None)
        with pytest.raises(SecretManagerUnavailableError, match="key_management_system"):
            store.reload()

    def test_it_owns_the_refresh(self) -> None:
        """A vault is the serialization point that the single-owner rule requires."""
        assert SecretManagerCredentialStore(client=FakeVault()).owns_refresh is True


class TestInsideAnEventLoop:
    @pytest.mark.asyncio
    async def test_writing_works_while_a_loop_is_running(self) -> None:
        """The UI runs inside the FastAPI loop and the LiteLLM write is async: a direct
        `asyncio.run` would raise `RuntimeError` on the first Connect.
        """
        vault = FakeVault()
        store = SecretManagerCredentialStore(client=vault)
        store.set("anthropic", Credential(provider="anthropic", access_token="AT"))
        assert secret_name("anthropic") in vault.data
