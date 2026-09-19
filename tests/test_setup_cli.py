"""`mysubs-setup`: wire the plugin in without wrecking the installer's configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from litellm_mysubs.setup_cli import CALLBACK_PATH, find_config, main, patch_text

REAL_CONFIG = """# My production configuration — DO NOT TOUCH without talking to me

model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
  - model_name: my-llama                # basement GPU
    litellm_params:
      model: openai/llama-3
      api_base: http://gpu.lan:8000

general_settings:
  master_key: sk-secret
  supported_db_objects: ["agents"]

litellm_settings:
  drop_params: true
  callbacks: ["prometheus"]
"""


def callbacks_of(text: str) -> list[str]:
    loaded = yaml.safe_load(text) or {}
    return list((loaded.get("litellm_settings") or {}).get("callbacks") or [])


class TestPatchShapes:
    @pytest.mark.parametrize(
        ("name", "source"),
        [
            ("block", "litellm_settings:\n  callbacks:\n    - prometheus\n"),
            ("inline", 'litellm_settings:\n  callbacks: ["prometheus"]\n'),
            ("empty inline", "litellm_settings:\n  callbacks: []\n"),
            ("settings without callbacks", "litellm_settings:\n  drop_params: true\n"),
            ("settings as an inline empty mapping", "litellm_settings: {}\n"),
            ("settings as a null key", "litellm_settings:\n"),
            ("no settings", "model_list:\n  - model_name: x\n"),
        ],
    )
    def test_the_callback_lands_whatever_the_shape(self, name: str, source: str) -> None:
        """A real `config.yaml` shows up in any of these shapes. Failing on one of them
        means a user editing YAML by hand — which is what this command avoids."""
        assert CALLBACK_PATH in callbacks_of(patch_text(source)), name

    def test_existing_callbacks_are_kept(self) -> None:
        """Replacing the list would disable prometheus for whoever already had it."""
        patched = patch_text("litellm_settings:\n  callbacks:\n    - prometheus\n    - langfuse\n")
        assert callbacks_of(patched) == ["prometheus", "langfuse", CALLBACK_PATH]


class TestPreservation:
    def test_comments_and_formatting_survive(self) -> None:
        """Regenerating the YAML with `safe_dump` produces an equivalent and unrecognisable
        file: measured, 20 lines changed to add one. Whoever opens it next does not
        recognise what was theirs, and the `git diff` of the configuration repository
        becomes unreadable.
        """
        patched = patch_text(REAL_CONFIG)
        assert "# My production configuration" in patched
        assert "# basement GPU" in patched
        assert 'supported_db_objects: ["agents"]' in patched

    def test_exactly_one_line_changes(self) -> None:
        before = REAL_CONFIG.splitlines()
        after = patch_text(REAL_CONFIG).splitlines()
        differing = [i for i, (a, b) in enumerate(zip(before, after, strict=False)) if a != b]
        assert len(differing) + (len(after) - len(before)) == 1

    def test_the_model_list_is_never_touched(self) -> None:
        """The routing that already exists is none of this installer's business."""
        before = yaml.safe_load(REAL_CONFIG)
        after = yaml.safe_load(patch_text(REAL_CONFIG))
        assert after["model_list"] == before["model_list"]
        assert after["general_settings"] == before["general_settings"]


class TestSafety:
    def test_a_broken_result_is_refused_before_it_reaches_the_disk(self, tmp_path: Path) -> None:
        """Editing YAML as text is fragile; this check is what makes the fragility
        acceptable. A file that no longer loads means a proxy that does not start."""
        config = tmp_path / "config.yaml"
        config.write_text(REAL_CONFIG, "utf-8")
        original = config.read_text("utf-8")

        import litellm_mysubs.setup_cli as cli

        monkey = cli.patch_text
        cli.patch_text = lambda _: "this: [does not\n  close"  # type: ignore[assignment]
        try:
            assert main(["--config", str(config), "--yes"]) == 1
        finally:
            cli.patch_text = monkey  # type: ignore[assignment]
        assert config.read_text("utf-8") == original

    def test_a_backup_is_left_next_to_the_original(self, tmp_path: Path) -> None:
        config = tmp_path / "config.yaml"
        config.write_text(REAL_CONFIG, "utf-8")
        assert main(["--config", str(config), "--yes"]) == 0
        backup = tmp_path / "config.yaml.mysubs-bak"
        assert backup.read_text("utf-8") == REAL_CONFIG

    def test_running_twice_does_not_duplicate_the_callback(self, tmp_path: Path) -> None:
        config = tmp_path / "config.yaml"
        config.write_text(REAL_CONFIG, "utf-8")
        main(["--config", str(config), "--yes"])
        main(["--config", str(config), "--yes"])
        assert callbacks_of(config.read_text("utf-8")).count(CALLBACK_PATH) == 1


class TestDiscovery:
    def test_an_explicit_path_that_does_not_exist_is_not_invented(self, tmp_path: Path) -> None:
        """Writing to the wrong file is worse than asking."""
        assert find_config(str(tmp_path / "does-not-exist.yaml")) is None

    def test_the_environment_wins_over_the_working_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """It is what the container defines, and it is what the proxy inside it is really
        using."""
        chosen = tmp_path / "from-container.yaml"
        chosen.write_text("litellm_settings: {}\n", "utf-8")
        monkeypatch.setenv("LITELLM_CONFIG", str(chosen))
        assert find_config() == chosen


class TestRealWorldConfigs:
    """Configurations that show up in production, not the ones that are easy to patch.

    The two failures below were measured in a real deployment test, not imagined.
    """

    def test_an_inline_list_with_a_trailing_comment(self) -> None:
        """`callbacks: ["langfuse"]  # comment` is valid YAML and showed up in a real
        configuration. The parser required the line to **end** at the `]`, fell into the
        block branch and produced a duplicated `callbacks:` — which `_verify` caught, but
        only after refusing to write, leaving the user with no installation and no clue."""
        import yaml

        from litellm_mysubs.setup_cli import CALLBACK_PATH, patch_text

        original = (
            "litellm_settings:\n"
            "  drop_params: true\n"
            '  callbacks: ["langfuse"]     # we already had a callback\n'
        )
        out = patch_text(original)
        loaded = yaml.safe_load(out)
        assert loaded["litellm_settings"]["callbacks"] == ["langfuse", CALLBACK_PATH]
        assert "# we already had a callback" in out, "the comment was erased"
        assert out.count("callbacks:") == 1, f"duplicated key:\n{out}"

    def test_the_rest_of_the_file_is_untouched(self) -> None:
        """A `safe_dump` rewrote the whole file. The promise is one line."""
        from litellm_mysubs.setup_cli import patch_text

        original = (
            "# Production configuration\n"
            "model_list:\n"
            "  - model_name: gpt-4o\n"
            "    litellm_params:\n"
            "      model: openai/gpt-4o\n"
            "general_settings:\n"
            "  master_key: sk-production\n"
            "  # important comment\n"
            '  supported_db_objects: ["agents"]\n'
            "litellm_settings:\n"
            '  callbacks: ["langfuse"]\n'
        )
        out = patch_text(original)
        before, after = original.splitlines(), out.splitlines()
        changed = [b for b, a in zip(before, after, strict=True) if b != a]
        assert len(changed) == 1, f"touched {len(changed)} lines: {changed}"
        assert "# important comment" in out
        assert 'supported_db_objects: ["agents"]' in out

    def test_a_fresh_install_gets_a_config_created(self, tmp_path: Path) -> None:
        """On an installation without a `config.yaml`, telling the user to write YAML by
        hand before running the command that exists to spare them exactly that would be
        trading one step for two."""
        import yaml

        from litellm_mysubs.setup_cli import CALLBACK_PATH, _create_config

        target = tmp_path / "config.yaml"
        assert _create_config(target) == target
        loaded = yaml.safe_load(target.read_text())
        assert loaded["litellm_settings"]["callbacks"] == [CALLBACK_PATH]
        assert "model_list" not in loaded, "invented models the user did not ask for"
        assert "general_settings" not in loaded, "invented a master_key"

    def test_creating_never_overwrites(self, tmp_path: Path) -> None:
        """A file that is present is edited by `patch_text`, never replaced."""
        from litellm_mysubs.setup_cli import _create_config

        target = tmp_path / "config.yaml"
        target.write_text("model_list: []\n")
        assert _create_config(target) == target
        assert target.read_text() == "model_list: []\n", "replaced an existing file"


class TestWhatTheUserIsTold:
    """What `mysubs-setup` prints is the whole interface at that moment.

    Found by installing from a clean clone into an empty `HOME`: the run ended with
    "Already connected. Nothing to do." on a machine with no credential at all. The line
    was about the callback being in `config.yaml`, but nobody reads it that way — it sends
    someone with a fresh install looking for a subscription they never connected.
    """

    def test_a_second_run_does_not_claim_a_subscription_is_connected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config = tmp_path / "config.yaml"
        config.write_text(patch_text("litellm_settings: {}\n"), "utf-8")
        monkeypatch.setenv("LITELLM_CONFIG", str(config))

        assert main(["--yes"]) == 0

        out = capsys.readouterr().out
        assert "callback is already in" in out
        assert "connected" not in out.lower().split("connect a subscription")[0], out
        assert "/mysubs" in out, "the user is still told where to go next"
