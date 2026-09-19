"""Model selection persistence.

These defend the measured incident: after a restart the proxy listed only the models from
`config.yaml` and the cards tab said ``connected=True applied=0``. The corruption tests
defend startup (a damaged file must not block the proxy) and the wide-permissions one pins
the deliberate difference from the credential store.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from litellm_mysubs.catalog.selection import SelectionStore


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "mysubs" / "models.json"


def dep(name: str, model: str = "openai/gpt-5.5") -> dict[str, object]:
    return {
        "model_name": name,
        "litellm_params": {"model": model, "api_base": "https://example.invalid"},
        "model_info": {"managed_by": "mysubs", "id": f"id-{name}"},
    }


class TestRoundTrip:
    def test_deployments_survive_intact(self, store_path: Path) -> None:
        saved = [dep("mysubs/codex/gpt-5.5"), dep("mysubs/codex/gpt-5.5-mini")]
        SelectionStore(store_path).save("openai-codex", saved)

        got = SelectionStore(store_path).all()
        assert [s.provider for s in got] == ["openai-codex"]
        assert got[0].deployments == saved

    def test_two_providers_both_kept(self, store_path: Path) -> None:
        store = SelectionStore(store_path)
        store.save("anthropic", [dep("mysubs/claude/sonnet")])
        store.save("openai-codex", [dep("mysubs/codex/gpt-5.5")])

        got = SelectionStore(store_path).all()
        assert [s.provider for s in got] == ["anthropic", "openai-codex"]

    def test_resave_replaces_without_duplicating(self, store_path: Path) -> None:
        store = SelectionStore(store_path)
        store.save("anthropic", [dep("mysubs/claude/sonnet"), dep("mysubs/claude/opus")])
        store.save("anthropic", [dep("mysubs/claude/haiku")])

        got = SelectionStore(store_path).all()
        assert len(got) == 1
        assert [d["model_name"] for d in got[0].deployments] == ["mysubs/claude/haiku"]

    def test_all_is_ordered_by_provider_ids(self, store_path: Path) -> None:
        store = SelectionStore(store_path)
        store.save("google-antigravity", [dep("mysubs/gemini/pro")])
        store.save("anthropic", [dep("mysubs/claude/sonnet")])
        assert [s.provider for s in store.all()] == ["anthropic", "google-antigravity"]

    def test_drop_removes_one_and_keeps_the_rest(self, store_path: Path) -> None:
        store = SelectionStore(store_path)
        store.save("anthropic", [dep("mysubs/claude/sonnet")])
        store.save("openai-codex", [dep("mysubs/codex/gpt-5.5")])
        store.drop("anthropic")

        assert [s.provider for s in SelectionStore(store_path).all()] == ["openai-codex"]

    def test_drop_of_absent_provider_is_quiet(self, store_path: Path) -> None:
        SelectionStore(store_path).drop("anthropic")
        assert SelectionStore(store_path).all() == []


class TestTolerance:
    """Nothing here may raise: the proxy has to start up regardless."""

    def test_missing_file_is_empty(self, store_path: Path) -> None:
        assert SelectionStore(store_path).all() == []

    def test_invalid_json_is_empty(self, store_path: Path) -> None:
        store_path.parent.mkdir(parents=True)
        store_path.write_text("{not json", "utf-8")
        assert SelectionStore(store_path).all() == []

    def test_unknown_version_is_empty(self, store_path: Path) -> None:
        store_path.parent.mkdir(parents=True)
        payload = {"version": 99, "providers": {"anthropic": [dep("mysubs/claude/sonnet")]}}
        store_path.write_text(json.dumps(payload), "utf-8")
        assert SelectionStore(store_path).all() == []

    def test_unknown_provider_skipped_valid_ones_kept(self, store_path: Path) -> None:
        store_path.parent.mkdir(parents=True)
        payload = {
            "version": 1,
            "providers": {
                "provider-that-does-not-exist": [dep("mysubs/x/y")],
                "anthropic": [dep("mysubs/claude/sonnet")],
            },
        }
        store_path.write_text(json.dumps(payload), "utf-8")

        got = SelectionStore(store_path).all()
        assert [s.provider for s in got] == ["anthropic"]

    def test_deployment_without_model_name_is_skipped(self, store_path: Path) -> None:
        store_path.parent.mkdir(parents=True)
        nameless = {"litellm_params": {"model": "openai/gpt-5.5"}}
        payload = {
            "version": 1,
            "providers": {"anthropic": [nameless, dep("mysubs/claude/sonnet")]},
        }
        store_path.write_text(json.dumps(payload), "utf-8")

        got = SelectionStore(store_path).all()
        assert [d["model_name"] for d in got[0].deployments] == ["mysubs/claude/sonnet"]

    def test_non_dict_entry_is_skipped(self, store_path: Path) -> None:
        store_path.parent.mkdir(parents=True)
        payload = {"version": 1, "providers": {"anthropic": ["junk", dep("mysubs/claude/x")]}}
        store_path.write_text(json.dumps(payload), "utf-8")

        got = SelectionStore(store_path).all()
        assert [d["model_name"] for d in got[0].deployments] == ["mysubs/claude/x"]

    def test_save_over_corrupt_file_works(self, store_path: Path) -> None:
        store_path.parent.mkdir(parents=True)
        store_path.write_text("]]]", "utf-8")
        SelectionStore(store_path).save("anthropic", [dep("mysubs/claude/sonnet")])
        assert [s.provider for s in SelectionStore(store_path).all()] == ["anthropic"]


class TestOnDisk:
    def test_permissions_are_tight(self, store_path: Path) -> None:
        SelectionStore(store_path).save("anthropic", [dep("mysubs/claude/sonnet")])
        assert stat.S_IMODE(store_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(store_path.parent.stat().st_mode) == 0o700

    def test_wide_permissions_still_load(self, store_path: Path) -> None:
        # Deliberate difference from the credential store: there are no secrets here, and
        # refusing over one bit would leave the proxy without the user's models.
        SelectionStore(store_path).save("anthropic", [dep("mysubs/claude/sonnet")])
        store_path.chmod(0o644)
        assert [s.provider for s in SelectionStore(store_path).all()] == ["anthropic"]

    def test_write_leaves_no_temporary(self, store_path: Path) -> None:
        SelectionStore(store_path).save("anthropic", [dep("mysubs/claude/sonnet")])
        assert [p.name for p in store_path.parent.iterdir()] == ["models.json"]

    def test_file_shape_is_versioned(self, store_path: Path) -> None:
        SelectionStore(store_path).save("anthropic", [dep("mysubs/claude/sonnet")])
        raw = json.loads(store_path.read_text("utf-8"))
        assert raw["version"] == 1
        assert list(raw["providers"]) == ["anthropic"]
