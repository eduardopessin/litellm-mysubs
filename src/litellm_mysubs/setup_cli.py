"""`mysubs-setup` — liga o plugin ao LiteLLM que já está instalado.

Uma execução, sem editar YAML à mão. O que faz:

1. encontra o `litellm` do ambiente e o `config.yaml` que ele usa;
2. acrescenta `litellm_mysubs.MySubs` a `litellm_settings.callbacks`;
3. diz o URL da página.

O que **não** faz, de propósito:

- não toca em `model_list`, `router_settings` nem `general_settings` — o roteamento que já
  existe não é negócio deste instalador;
- não reescreve o ficheiro sem uma cópia de segurança ao lado;
- não instala nada no `site-packages` (nem `.pth` nem `sitecustomize.py`): desinstalar é
  apagar uma linha, e um plugin que se enxerta no interpretador é difícil de remover e
  fácil de culpar quando outra coisa parte.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any

#: A instância, não a classe: o proxy recusa uma classe com `ValueError` no arranque.
CALLBACK_PATH = "litellm_mysubs.proxy_handler_instance"

#: Sítios onde um `config.yaml` costuma estar, por ordem de probabilidade. A variável de
#: ambiente ganha porque é o que o container define.
CANDIDATES: tuple[str, ...] = (
    "config.yaml",
    "config.yml",
    "litellm_config.yaml",
    "/app/config.yaml",
    "/etc/litellm/config.yaml",
)


def find_litellm() -> tuple[str, str] | None:
    """`(versão, caminho)` do LiteLLM instalado, ou `None`."""
    try:
        import litellm
    except ImportError:
        return None
    try:
        from importlib.metadata import version

        installed = version("litellm")
    except Exception:
        installed = "desconhecida"
    return installed, str(Path(litellm.__file__).parent)


def find_config(explicit: str | None = None) -> Path | None:
    """O `config.yaml` em uso.

    Não se adivinha quando há dúvida: sem candidato encontrado devolve-se `None` e o
    utilizador indica o caminho. Escrever no ficheiro errado é pior do que perguntar.
    """
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.exists() else None
    env = os.environ.get("LITELLM_CONFIG") or os.environ.get("CONFIG_FILE_PATH")
    if env and Path(env).exists():
        return Path(env)
    for candidate in CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return path
    return None


def already_installed(config: dict[str, Any]) -> bool:
    callbacks = (config.get("litellm_settings") or {}).get("callbacks") or []
    return CALLBACK_PATH in callbacks


def add_callback(config: dict[str, Any]) -> dict[str, Any]:
    """Acrescenta o callback preservando tudo o resto.

    Modifica a estrutura carregada em vez de reescrever o ficheiro de raiz: um `config.yaml`
    real tem comentários, ordem e chaves que este instalador não conhece, e regenerá-lo
    perderia-os.
    """
    settings = dict(config.get("litellm_settings") or {})
    callbacks = list(settings.get("callbacks") or [])
    if CALLBACK_PATH not in callbacks:
        callbacks.append(CALLBACK_PATH)
    settings["callbacks"] = callbacks
    updated = dict(config)
    updated["litellm_settings"] = settings
    return updated


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    loaded = yaml.safe_load(path.read_text("utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} não contém um mapeamento YAML")
    return loaded


def patch_text(original: str) -> str:
    """Acrescenta o callback **editando o texto**, não regenerando o YAML.

    Um `safe_dump` da estrutura carregada produz um ficheiro equivalente e ilegível: perde
    comentários, reindenta as listas todas e reescreve `["a"]` como bloco. Medido num
    `config.yaml` de exemplo: 20 linhas alteradas para acrescentar uma. Quem abrir o
    ficheiro a seguir não reconhece o que era seu, e um `git diff` do repositório de
    configuração fica ilegível.

    Três casos, por ordem:

    1. já existe `callbacks:` em lista de bloco -> acrescenta-se um item com a mesma
       indentação do primeiro;
    2. existe `callbacks: [...]` em linha -> insere-se antes do fecho;
    3. existe `litellm_settings:` sem `callbacks` -> cria-se a chave lá dentro;
    4. não existe nada -> acrescenta-se o bloco no fim.
    """
    lines = original.splitlines()
    entry = CALLBACK_PATH

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("callbacks:"):
            continue
        value = stripped[len("callbacks:") :].strip()
        indent = len(line) - len(line.lstrip())

        if value.startswith("["):
            if value.rstrip().endswith("]"):
                closing = line.rindex("]")
                inner = line[line.index("[") + 1 : closing].strip()
                joined = f'{inner}, "{entry}"' if inner else f'"{entry}"'
                lines[index] = f"{line[: line.index('[')]}[{joined}]"
                return "\n".join(lines) + ("\n" if original.endswith("\n") else "")
            continue

        # Lista de bloco: usa-se a indentação do primeiro item, não uma inventada.
        item_indent = indent + 2
        insert_at = index + 1
        for offset in range(index + 1, len(lines)):
            candidate = lines[offset]
            if candidate.strip().startswith("-"):
                item_indent = len(candidate) - len(candidate.lstrip())
                insert_at = offset + 1
            elif candidate.strip():
                break
        lines.insert(insert_at, f"{' ' * item_indent}- {entry}")
        return "\n".join(lines) + ("\n" if original.endswith("\n") else "")

    for index, line in enumerate(lines):
        if line.strip().startswith("litellm_settings:"):
            indent = len(line) - len(line.lstrip())
            lines.insert(index + 1, f"{' ' * (indent + 2)}callbacks:")
            lines.insert(index + 2, f"{' ' * (indent + 4)}- {entry}")
            return "\n".join(lines) + ("\n" if original.endswith("\n") else "")

    tail = "" if original.endswith("\n") or not original else "\n"
    return original + tail + f"\nlitellm_settings:\n  callbacks:\n    - {entry}\n"


def _write_yaml(path: Path, original: str) -> Path:
    """Escreve com cópia de segurança. Devolve o caminho da cópia."""
    patched = patch_text(original)
    _verify(patched)
    backup = path.with_suffix(path.suffix + ".mysubs-bak")
    shutil.copy2(path, backup)
    path.write_text(patched, "utf-8")
    return backup


def _verify(text: str) -> None:
    """Recusa-se a escrever um ficheiro que já não carrega.

    Editar YAML por texto é rápido e frágil. Esta verificação é o que torna a fragilidade
    aceitável: um erro de indentação é apanhado **antes** de o ficheiro chegar ao disco, em
    vez de o proxy não arrancar no reinício seguinte.
    """
    import yaml

    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError("o resultado não é um mapeamento YAML")
    callbacks = (loaded.get("litellm_settings") or {}).get("callbacks") or []
    if CALLBACK_PATH not in callbacks:
        raise ValueError("o callback não ficou na configuração")


def _ask(question: str, *, default: bool = True) -> bool:
    suffix = "[S/n]" if default else "[s/N]"
    try:
        answer = input(f"{question} {suffix} ").strip().lower()
    except EOFError:
        return default
    if not answer:
        return default
    return answer in ("s", "sim", "y", "yes")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mysubs-setup",
        description="Liga o litellm-mysubs ao LiteLLM instalado neste ambiente.",
    )
    parser.add_argument("--config", help="caminho do config.yaml (senão, procura-se)")
    parser.add_argument("--sim", action="store_true", help="não perguntar nada")
    parser.add_argument("--estado", action="store_true", help="mostrar o estado e sair")
    args = parser.parse_args(argv)

    found = find_litellm()
    if found is None:
        print("LiteLLM não encontrado neste ambiente.", file=sys.stderr)
        print("Instala-o primeiro, ou corre o mysubs-setup no ambiente onde ele vive.")
        return 1
    installed, location = found
    print(f"LiteLLM {installed}")
    print(f"  em {location}")

    if args.estado:
        from . import MySubs

        for key, value in MySubs().status.items():
            print(f"  {key}: {value or '—'}")
        return 0

    config_path = find_config(args.config)
    if config_path is None:
        print("\nNão encontrei o config.yaml.", file=sys.stderr)
        print("Indica-o com --config /caminho/para/config.yaml")
        print("\nOu acrescenta à mão:")
        print("  litellm_settings:")
        print(f'    callbacks: ["{CALLBACK_PATH}"]')
        return 1
    print(f"  config {config_path}")

    try:
        config = _load_yaml(config_path)
    except Exception as error:
        print(f"\nNão consegui ler {config_path}: {error}", file=sys.stderr)
        return 1

    if already_installed(config):
        print("\nJá está ligado. Nada a fazer.")
        print("A página fica em  <url-do-proxy>/mysubs")
        return 0

    models = len(config.get("model_list") or [])
    print(f"\nVou acrescentar o callback. Os teus {models} modelos não são tocados.")
    if not args.sim and not _ask("Continuar?"):
        print("Cancelado.")
        return 1

    try:
        backup = _write_yaml(config_path, config_path.read_text("utf-8"))
    except Exception as error:
        print(f"\nNão consegui escrever: {error}", file=sys.stderr)
        return 1

    print(f"\nFeito. Cópia do original em {backup.name}")
    print("Reinicia o proxy e abre  <url-do-proxy>/mysubs")
    print("\nA página exige uma chave de administrador (proxy_admin).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
