"""Armazenamento de credenciais.

O ficheiro guarda refresh tokens de subscrições pessoais. Os testes de permissões e de
escrita atómica defendem isso; os de ``owns_refresh`` defendem a regra do dono único,
cuja violação produz ``invalid_grant`` em ciclo e obriga a re-login manual.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from litellm_mysubs.credentials.env_store import EnvCredentialStore
from litellm_mysubs.credentials.file_store import FileCredentialStore, InsecurePermissionsError
from litellm_mysubs.credentials.store import Credential, ReadOnlyStoreError


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "mysubs" / "credentials.json"


def cred(**kw: object) -> Credential:
    base = {"provider": "anthropic", "access_token": "at-1"}
    return Credential(**{**base, **kw})  # type: ignore[arg-type]


class TestRoundTrip:
    def test_persists_across_instances(self, store_path: Path) -> None:
        FileCredentialStore(store_path).set("anthropic", cred(refresh_token="rt-1"))
        reloaded = FileCredentialStore(store_path).get("anthropic")
        assert reloaded is not None
        assert (reloaded.access_token, reloaded.refresh_token) == ("at-1", "rt-1")

    def test_missing_provider_is_none(self, store_path: Path) -> None:
        assert FileCredentialStore(store_path).get("openai-codex") is None

    def test_connected_lists_only_present(self, store_path: Path) -> None:
        store = FileCredentialStore(store_path)
        store.set("anthropic", cred())
        store.set("google-antigravity", cred(provider="google-antigravity", project_id="p-1"))
        assert set(store.connected()) == {"anthropic", "google-antigravity"}

    def test_project_id_survives(self, store_path: Path) -> None:
        """Descoberto no fluxo OAuth do Google; perdê-lo obriga a religar a subscrição."""
        store = FileCredentialStore(store_path)
        store.set("google-antigravity", cred(provider="google-antigravity", project_id="proj-9"))
        assert FileCredentialStore(store_path).get("google-antigravity").project_id == "proj-9"  # type: ignore[union-attr]

    def test_delete_removes(self, store_path: Path) -> None:
        store = FileCredentialStore(store_path)
        store.set("anthropic", cred())
        store.delete("anthropic")
        assert FileCredentialStore(store_path).get("anthropic") is None


class TestPermissions:
    def test_written_file_is_0600(self, store_path: Path) -> None:
        FileCredentialStore(store_path).set("anthropic", cred())
        assert stat.S_IMODE(store_path.stat().st_mode) == 0o600

    def test_rejects_world_readable_file(self, store_path: Path) -> None:
        """Apertar os bits em silêncio não desfaz uma leitura que já possa ter ocorrido."""
        store_path.parent.mkdir(parents=True, exist_ok=True)
        store_path.write_text("{}", encoding="utf-8")
        os.chmod(store_path, 0o644)
        with pytest.raises(InsecurePermissionsError):
            FileCredentialStore(store_path)


class TestReload:
    def test_detects_external_change(self, store_path: Path) -> None:
        """Uma rotação escrita por outro agente tem de ficar viva sem restart."""
        store = FileCredentialStore(store_path)
        store.set("anthropic", cred(access_token="old"))

        payload = json.loads(store_path.read_text("utf-8"))
        payload["anthropic"]["access_token"] = "rotated"
        store_path.write_text(json.dumps(payload), encoding="utf-8")
        os.chmod(store_path, 0o600)
        os.utime(store_path, (0, 0))  # mtime distinto do gravado em cache

        assert store.get("anthropic").access_token == "rotated"  # type: ignore[union-attr]

    def test_unchanged_file_reports_no_change(self, store_path: Path) -> None:
        store = FileCredentialStore(store_path)
        store.set("anthropic", cred())
        assert store.reload() is False


class TestExpiry:
    def test_unknown_expiry_is_not_expired(self) -> None:
        """Um token colado à mão não traz validade e tem de ser tentado, não descartado."""
        assert cred(expires_at=0).is_expired(now=1_000_000) is False

    def test_leeway_expires_early(self) -> None:
        """Renovar à tangente entrega um token que morre a meio do pedido.

        Com validade em 1_000_000 e leeway de 60s, o corte está em 999_940.
        """
        c = cred(expires_at=1_000_000)
        assert c.is_expired(now=1_000_001, leeway_s=60) is True   # já passou
        assert c.is_expired(now=999_950, leeway_s=60) is True     # dentro da leeway
        assert c.is_expired(now=999_940, leeway_s=60) is True     # no limite exacto
        assert c.is_expired(now=999_939, leeway_s=60) is False    # um segundo antes


class TestEnvStore:
    def test_reads_legacy_names(self) -> None:
        store = EnvCredentialStore({"ANTHROPIC_OAUTH_TOKEN": "at", "ANTHROPIC_REFRESH_TOKEN": "rt"})
        got = store.get("anthropic")
        assert got is not None and got.refresh_token == "rt"

    def test_absent_provider_is_none(self) -> None:
        assert EnvCredentialStore({}).get("anthropic") is None

    def test_is_not_refresh_owner(self) -> None:
        """Dois renovadores sobre um refresh token rotativo dão invalid_grant em ciclo."""
        assert EnvCredentialStore({}).owns_refresh is False
        assert FileCredentialStore.owns_refresh is True

    def test_write_fails_loudly(self) -> None:
        """Escrever em os.environ não persiste; falhar alto evita a ilusão de ter guardado."""
        with pytest.raises(ReadOnlyStoreError):
            EnvCredentialStore({}).set("anthropic", cred())
