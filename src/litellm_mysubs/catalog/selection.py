"""Persistence of the model selection applied by the user.

Measured against the real proxy: before a restart `/model/info` listed ``eco`` and
``mysubs/codex/gpt-5.5``; after the restart it listed only ``eco``, with the card tab at
``connected=True applied=0``. The credentials survived (they have their own file), the
selection had no file at all - `apply` injected into the Router and nothing else. This
module is the second half that was missing from the "injects directly and persists on its
own" promised at the top of `registry.py`.

What is stored is the **already built deployment**, not the chosen names. Rebuilding from
the names would mean running discovery at startup, that is, a network round trip before
the proxy serves traffic: with the upstream down the user would again be left without the
models they had already applied. The stored dict is self-sufficient and reinjects exactly
what was chosen - never a guess.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..credentials.store import PROVIDER_IDS, ProviderId

DEFAULT_PATH = Path.home() / ".litellm" / "mysubs" / "models.json"

#: On-disk format version. It exists so that a future format is *recognised* instead of
#: guessed: a file from another version is ignored whole, not read on a hunch.
_VERSION = 1


@dataclass(frozen=True, slots=True)
class Selection:
    """What a provider had applied, ready to hand back to the Router."""

    provider: ProviderId
    deployments: list[dict[str, Any]]


def _upgraded(entry: dict[str, Any]) -> dict[str, Any]:
    """A stored deployment brought up to what `to_deployment` builds today.

    What is persisted is the **built deployment**, not the model name — see the note in
    `service.reapply`. So a field added to `to_deployment` reaches new selections only,
    and an installation that applied its models before the change keeps the old shape
    across every restart, forever. Measured on a live gateway: after deploying the release
    that declares `custom_llm_provider`, all 55 stored deployments still came back without
    it, because `reapply` reinjects what is on disk verbatim.

    Upgrading on read rather than rewriting the file keeps this recoverable: the file is
    only rewritten when the user applies a selection, so a downgrade still finds what it
    wrote.

    `custom_llm_provider` is derived from the wire prefix already in the entry, which is
    the one with a price table behind it — the same rule `to_deployment` follows, so an
    upgraded entry and a freshly built one agree.
    """
    params = entry.get("litellm_params")
    if not isinstance(params, dict) or params.get("custom_llm_provider"):
        return entry
    wire = params.get("model")
    if not isinstance(wire, str) or "/" not in wire:
        return entry
    upgraded = dict(entry)
    upgraded["litellm_params"] = {**params, "custom_llm_provider": wire.split("/", 1)[0]}
    return upgraded


class SelectionStore:
    """``models.json`` file next to ``credentials.json``.

    Unlike `credentials/file_store.py`, reading a file with loose permissions is **not**
    refused. There the refusal protects a refresh token that others may already have read;
    here the content is model names, and refusing to load them over a permission bit left
    the proxy starting without the user's models - the very failure this module exists to
    fix. Writing still tightens to 0600/0700, because inheriting the neighbour's posture
    costs one line.
    """

    def __init__(self, path: Path | str = DEFAULT_PATH) -> None:
        self.path = Path(path)

    # -- reading ---------------------------------------------------------------

    def _load(self) -> dict[str, list[dict[str, Any]]]:
        """Provider -> deployments map, or empty. Never raises.

        A corrupt file must not stop the proxy from starting: we prefer starting without
        managed models (recoverable with an `apply` in the UI) to not starting at all.
        """
        try:
            raw = json.loads(self.path.read_text("utf-8") or "{}")
        except (OSError, ValueError):
            return {}
        if not isinstance(raw, dict) or raw.get("version") != _VERSION:
            return {}
        providers = raw.get("providers")
        if not isinstance(providers, dict):
            return {}

        out: dict[str, list[dict[str, Any]]] = {}
        for provider, entries in providers.items():
            if provider not in PROVIDER_IDS or not isinstance(entries, list):
                continue
            # A deployment without `model_name` is skipped: injecting it into the Router
            # blows up later, somewhere that no longer points at the file that caused it.
            out[provider] = [
                _upgraded(entry)
                for entry in entries
                if isinstance(entry, dict) and entry.get("model_name")
            ]
        return out



    def all(self) -> list[Selection]:
        """Stored selections, in ``PROVIDER_IDS`` order so the result is deterministic."""
        data = self._load()
        return [Selection(provider=p, deployments=data[p]) for p in PROVIDER_IDS if data.get(p)]

    # -- writing ---------------------------------------------------------------

    def _write(self, data: dict[str, list[dict[str, Any]]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        payload = {"version": _VERSION, "providers": data}
        # Atomic write: a file truncated by a crash mid-restart switched off every model
        # at once, which is the original incident all over again.
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".models-")
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def save(self, provider: ProviderId, deployments: list[dict[str, Any]]) -> None:
        """Replaces whatever exists for ``provider``, preserving the others."""
        data = self._load()
        data[provider] = [dict(d) for d in deployments]
        self._write(data)

    def drop(self, provider: ProviderId) -> None:
        data = self._load()
        if data.pop(provider, None) is not None:
            self._write(data)
