"""`mysubs-setup` — connects the plugin to the LiteLLM that is already installed.

One run, with no YAML edited by hand. What it does:

1. finds the environment's `litellm` and the `config.yaml` it uses;
2. adds `litellm_mysubs.MySubs` to `litellm_settings.callbacks`;
3. prints the page's URL.

What it does **not** do, on purpose:

- it does not touch `model_list`, `router_settings` or `general_settings` — the routing that
  already exists is none of this installer's business;
- it does not rewrite the file without leaving a backup copy beside it;
- it installs nothing into `site-packages` (no `.pth`, no `sitecustomize.py`): uninstalling
  is deleting one line, and a plugin grafted into the interpreter is hard to remove and easy
  to blame when something else breaks.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any

#: The instance, not the class: the proxy refuses a class with `ValueError` at startup.
CALLBACK_PATH = "litellm_mysubs.proxy_handler_instance"

#: The menu item is injected into a chunk that Next.js serves with a one-year `max-age` and
#: `immutable`. An already cached entry never goes back to the server, so the new response's
#: `no-store` is never read: whoever used the UI before installing keeps getting the old
#: bundle, without the button.
#:
#: `index.html` could be patched too, to change the script's URL, but that is one more
#: generated file to track on every LiteLLM version. One cache clear, once per installation,
#: costs less than that maintenance.
CACHE_HINT = (
    "If the item does not show up in the menu, clear the browser cache (Ctrl+Shift+R):\n"
    "  the UI keeps chunks for a year and yours is still the one from before the install.\n"
    "  The page works either way at <proxy-url>/mysubs."
)

#: Places a `config.yaml` usually lives, in order of likelihood. The environment variable
#: wins because it is what the container sets.
CANDIDATES: tuple[str, ...] = (
    "config.yaml",
    "config.yml",
    "litellm_config.yaml",
    "/app/config.yaml",
    "/etc/litellm/config.yaml",
)


def find_litellm() -> tuple[str, str] | None:
    """`(version, path)` of the installed LiteLLM, or `None`."""
    try:
        import litellm
    except ImportError:
        return None
    try:
        from importlib.metadata import version

        installed = version("litellm")
    except Exception:
        installed = "unknown"
    return installed, str(Path(litellm.__file__).parent)


def find_config(explicit: str | None = None) -> Path | None:
    """The `config.yaml` in use.

    No guessing when in doubt: with no candidate found, `None` is returned and the user
    gives the path. Writing to the wrong file is worse than asking.
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
    """Adds the callback while preserving everything else.

    Modifies the loaded structure instead of rewriting the file from scratch: a real
    `config.yaml` has comments, ordering and keys this installer does not know about, and
    regenerating it would lose them.
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
        raise ValueError(f"{path} does not contain a YAML mapping")
    return loaded


def patch_text(original: str) -> str:
    """Adds the callback by **editing the text**, not by regenerating the YAML.

    A `safe_dump` of the loaded structure produces an equivalent, unreadable file: it loses
    comments, reindents every list and rewrites `["a"]` as a block. Measured on an example
    `config.yaml`: 20 lines changed to add one. Whoever opens the file next does not
    recognize what was theirs, and a `git diff` of the configuration repository becomes
    unreadable.

    Four cases, in order:

    1. `callbacks:` already exists as a block list -> an item is added with the same
       indentation as the first;
    2. `callbacks: [...]` exists inline -> it is inserted before the closing bracket;
    3. `litellm_settings:` exists without `callbacks` -> the key is created inside it;
    4. nothing exists -> the block is appended at the end.
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
            # The `]` may not be the end of the line: `callbacks: ["langfuse"]  # comment`
            # is valid YAML and showed up in a real configuration. Looking for the closing
            # bracket instead of requiring the line to end with it is what keeps the patch
            # from falling into the block branch and producing a duplicate `callbacks:` —
            # which `_verify` caught, but only after refusing to write, leaving the user
            # with no installation and no idea why.
            if "]" in line:
                closing = line.rindex("]")
                inner = line[line.index("[") + 1 : closing].strip()
                joined = f'{inner}, "{entry}"' if inner else f'"{entry}"'
                lines[index] = f"{line[: line.index('[')]}[{joined}]{line[closing + 1 :]}"
                return "\n".join(lines) + ("\n" if original.endswith("\n") else "")
            continue

        # Block list: the first item's indentation is used, not an invented one.
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
    """Writes with a backup copy. Returns the copy's path."""
    patched = patch_text(original)
    _verify(patched)
    backup = path.with_suffix(path.suffix + ".mysubs-bak")
    shutil.copy2(path, backup)
    path.write_text(patched, "utf-8")
    return backup


def _verify(text: str) -> None:
    """Refuses to write a file that no longer loads.

    Editing YAML as text is fast and fragile. This check is what makes the fragility
    acceptable: an indentation error is caught **before** the file reaches the disk, instead
    of the proxy failing to start on the next restart.
    """
    import yaml

    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError("the result is not a YAML mapping")
    callbacks = (loaded.get("litellm_settings") or {}).get("callbacks") or []
    if CALLBACK_PATH not in callbacks:
        raise ValueError("the callback did not end up in the configuration")


def _ask(question: str, *, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        answer = input(f"{question} {suffix} ").strip().lower()
    except EOFError:
        return default
    if not answer:
        return default
    return answer in ("y", "yes")


#: The `config.yaml` for an installation that does not have one yet.
#:
#: Minimal on purpose: only what the plugin needs. **No `model_list`** — the subscription
#: models come in through the page, and inventing entries here would give the user models
#: they did not choose. No `master_key` either: the key is their decision, and a default
#: value in a configuration file is the kind of thing that survives into production.
_TEMPLATE = f"""# Created by mysubs-setup.
# The subscription models are added by the /mysubs page, not here.
litellm_settings:
  callbacks:
    - {CALLBACK_PATH}
"""


def _create_config(path: Path) -> Path | None:
    """Creates a minimal `config.yaml`. `None` if that is not possible.

    It exists because a fresh installation has no file at all, and telling the user to write
    YAML by hand before they can run the command that exists to spare them exactly that
    would trade one step for two.

    **It only creates what does not exist.** A file that is present is always edited by
    `patch_text`, which preserves comments and indentation — never replaced.
    """
    if path.exists():
        return path
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_TEMPLATE, encoding="utf-8")
    except OSError as error:
        print(f"\nCould not create {path}: {error}", file=sys.stderr)
        print("Give a path with --config, or add this by hand:")
        print("  litellm_settings:")
        print(f'    callbacks: ["{CALLBACK_PATH}"]')
        return None
    print(f"\nThere was no config.yaml; created one at {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mysubs-setup",
        description="Connects litellm-mysubs to the LiteLLM installed in this environment.",
    )
    parser.add_argument("--config", help="path to config.yaml (otherwise it is searched for)")
    parser.add_argument("--yes", action="store_true", help="do not ask anything")
    parser.add_argument("--status", action="store_true", help="show the status and exit")
    args = parser.parse_args(argv)

    found = find_litellm()
    if found is None:
        print("LiteLLM not found in this environment.", file=sys.stderr)
        print("Install it first, or run mysubs-setup in the environment where it lives.")
        return 1
    installed, location = found
    print(f"LiteLLM {installed}")
    print(f"  at {location}")

    if args.status:
        from . import MySubs

        for key, value in MySubs().status.items():
            print(f"  {key}: {value or '—'}")
        return 0

    config_path = find_config(args.config)
    if config_path is None:
        config_path = _create_config(Path(args.config) if args.config else Path("config.yaml"))
        if config_path is None:
            return 1
    print(f"  config {config_path}")

    try:
        config = _load_yaml(config_path)
    except Exception as error:
        print(f"\nCould not read {config_path}: {error}", file=sys.stderr)
        return 1

    if already_installed(config):
        print("\nAlready connected. Nothing to do.")
        print("The page is at  <proxy-url>/mysubs")
        return 0

    models = len(config.get("model_list") or [])
    print(f"\nAdding the callback. Your existing models ({models}) are not touched.")
    if not args.yes and not _ask("Continue?"):
        print("Cancelled.")
        return 1

    try:
        backup = _write_yaml(config_path, config_path.read_text("utf-8"))
    except Exception as error:
        print(f"\nCould not write: {error}", file=sys.stderr)
        return 1

    print(f"\nDone. Copy of the original at {backup.name}")
    print("\nNext:")
    print("  1. restart the proxy")
    print("  2. open the UI and sign in as an administrator")
    print("  3. Experimental -> MySubs")
    print(f"\n{CACHE_HINT}")
    print("\nThe page accepts the UI session; from outside, it needs a proxy_admin key.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
