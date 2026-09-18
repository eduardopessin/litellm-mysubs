"""`mysubs-setup`: ligar o plugin sem estragar a configuração de quem o instala."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from litellm_mysubs.setup_cli import CALLBACK_PATH, find_config, main, patch_text

REAL_CONFIG = """# A minha configuração de produção — NÃO MEXER sem falar comigo

model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
  - model_name: meu-llama            # GPU da cave
    litellm_params:
      model: openai/llama-3
      api_base: http://gpu.lan:8000

general_settings:
  master_key: sk-segredo
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
            ("bloco", "litellm_settings:\n  callbacks:\n    - prometheus\n"),
            ("inline", 'litellm_settings:\n  callbacks: ["prometheus"]\n'),
            ("inline vazia", "litellm_settings:\n  callbacks: []\n"),
            ("settings sem callbacks", "litellm_settings:\n  drop_params: true\n"),
            ("sem settings", "model_list:\n  - model_name: x\n"),
        ],
    )
    def test_the_callback_lands_whatever_the_shape(self, name: str, source: str) -> None:
        """Um `config.yaml` real aparece em qualquer destas formas. Falhar numa delas
        significa um utilizador a editar YAML à mão — que é o que este comando evita."""
        assert CALLBACK_PATH in callbacks_of(patch_text(source)), name

    def test_existing_callbacks_are_kept(self) -> None:
        """Substituir a lista desligaria o prometheus de quem já o tinha."""
        patched = patch_text("litellm_settings:\n  callbacks:\n    - prometheus\n    - langfuse\n")
        assert callbacks_of(patched) == ["prometheus", "langfuse", CALLBACK_PATH]


class TestPreservation:
    def test_comments_and_formatting_survive(self) -> None:
        """Regenerar o YAML com `safe_dump` produz um ficheiro equivalente e irreconhecível:
        medido, 20 linhas alteradas para acrescentar uma. Quem abrir a seguir não reconhece
        o que era seu, e o `git diff` do repositório de configuração fica ilegível.
        """
        patched = patch_text(REAL_CONFIG)
        assert "# A minha configuração de produção" in patched
        assert "# GPU da cave" in patched
        assert 'supported_db_objects: ["agents"]' in patched

    def test_exactly_one_line_changes(self) -> None:
        before = REAL_CONFIG.splitlines()
        after = patch_text(REAL_CONFIG).splitlines()
        differing = [i for i, (a, b) in enumerate(zip(before, after, strict=False)) if a != b]
        assert len(differing) + (len(after) - len(before)) == 1

    def test_the_model_list_is_never_touched(self) -> None:
        """O roteamento que já existe não é negócio deste instalador."""
        before = yaml.safe_load(REAL_CONFIG)
        after = yaml.safe_load(patch_text(REAL_CONFIG))
        assert after["model_list"] == before["model_list"]
        assert after["general_settings"] == before["general_settings"]


class TestSafety:
    def test_a_broken_result_is_refused_before_it_reaches_the_disk(self, tmp_path: Path) -> None:
        """Editar YAML por texto é frágil; esta verificação é o que torna a fragilidade
        aceitável. Um ficheiro que já não carrega significa um proxy que não arranca."""
        config = tmp_path / "config.yaml"
        config.write_text(REAL_CONFIG, "utf-8")
        original = config.read_text("utf-8")

        import litellm_mysubs.setup_cli as cli

        monkey = cli.patch_text
        cli.patch_text = lambda _: "isto: [não\n  fecha"  # type: ignore[assignment]
        try:
            assert main(["--config", str(config), "--sim"]) == 1
        finally:
            cli.patch_text = monkey  # type: ignore[assignment]
        assert config.read_text("utf-8") == original

    def test_a_backup_is_left_next_to_the_original(self, tmp_path: Path) -> None:
        config = tmp_path / "config.yaml"
        config.write_text(REAL_CONFIG, "utf-8")
        assert main(["--config", str(config), "--sim"]) == 0
        backup = tmp_path / "config.yaml.mysubs-bak"
        assert backup.read_text("utf-8") == REAL_CONFIG

    def test_running_twice_does_not_duplicate_the_callback(self, tmp_path: Path) -> None:
        config = tmp_path / "config.yaml"
        config.write_text(REAL_CONFIG, "utf-8")
        main(["--config", str(config), "--sim"])
        main(["--config", str(config), "--sim"])
        assert callbacks_of(config.read_text("utf-8")).count(CALLBACK_PATH) == 1


class TestDiscovery:
    def test_an_explicit_path_that_does_not_exist_is_not_invented(self, tmp_path: Path) -> None:
        """Escrever no ficheiro errado é pior do que perguntar."""
        assert find_config(str(tmp_path / "não-existe.yaml")) is None

    def test_the_environment_wins_over_the_working_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """É o que o container define, e é o que o proxy lá dentro está mesmo a usar."""
        chosen = tmp_path / "do-container.yaml"
        chosen.write_text("litellm_settings: {}\n", "utf-8")
        monkeypatch.setenv("LITELLM_CONFIG", str(chosen))
        assert find_config() == chosen
