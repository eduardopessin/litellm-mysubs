"""Google Antigravity (Cloud Code API) model name resolution.

This is the only provider with a queryable catalog: ``:fetchAvailableModels`` returns what
the account serves, including ``deprecatedModelIds``. Subtracting that list from the
catalog itself beats maintaining a static one, because the catalog advertises variants
that ``streamGenerateContent`` refuses.

The static map is the fallback, used only when the catalog did not answer. It exists so a
network failure does not render the account unusable, never to guess: a name that does not
match raises instead of being silently served by another model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Final

#: The catalog advertises these variants, but streamGenerateContent returns
#: 400 INVALID_ARGUMENT for them. omp documents the same and routes to gemini-pro-agent.
BROKEN_WIRE: Final[tuple[str, ...]] = ("gemini-3.1-pro-high", "gemini-3-pro-high")

#: Effort -> variant suffix, in order of preference.
EFFORT_SUFFIXES: Final[dict[str, tuple[str, ...]]] = {
    "none": ("-extra-low", "-low", "", "-tiered"),
    "minimal": ("-extra-low", "-low", "", "-tiered"),
    "low": ("-low", "-extra-low", "", "-tiered"),
    "medium": ("-medium", "-low", "", "-tiered"),
    "high": ("-high", "-medium", "-low", ""),
    "xhigh": ("-high", "-medium", "-low", ""),
    "max": ("-high", "-medium", "-low", ""),
}

#: Families with no usable suffix variant at the top of the scale.
EFFORT_OVERRIDES: Final[dict[tuple[str, str], str]] = {
    ("gemini-3.1-pro", "high"): "gemini-pro-agent",
    ("gemini-3.1-pro", "xhigh"): "gemini-pro-agent",
    ("gemini-3.1-pro", "max"): "gemini-pro-agent",
    ("gemini-3-pro", "high"): "gemini-pro-agent",
    ("gemini-3.5-flash", "high"): "gemini-3-flash-agent",
    ("gemini-3.5-flash", "xhigh"): "gemini-3-flash-agent",
    ("gemini-3.5-flash", "max"): "gemini-3-flash-agent",
}

# Variant suffixes that get stripped to reach the family.
#
# `-thinking` was left out on purpose: `gemini-2.5-flash-thinking` exists in the catalog
# and matches by exact name, while `gemini-3.8-flash-thinking` does not — stripping it made
# an invented name be silently served by `-low`, exactly what removing those entries from
# the config was meant to prevent.
SUFFIXES: Final[tuple[str, ...]] = (
    "-tiered",
    "-extra-low",
    "-low",
    "-medium",
    "-high",
    "-agent",
)

# Fallback for when the catalog does not answer. The `-thinking` entries were dropped: they
# do not exist upstream. The `-tiered` ones do exist and map to themselves, because
# stripping an explicit request is lying about what was served.
STATIC_MAP: Final[dict[str, str]] = {
    "gemini-3.8-flash-tiered": "gemini-3.8-flash-tiered",
    "gemini-3.8-flash": "gemini-3.8-flash-low",
    "gemini-3.7-flash-tiered": "gemini-3.7-flash-tiered",
    "gemini-3.7-flash": "gemini-3.7-flash-low",
    "gemini-3.6-flash": "gemini-3.6-flash-low",
    "gemini-3.5-flash": "gemini-3.5-flash-extra-low",
    "gemini-3.1-flash-lite": "gemini-3.1-flash-lite",
    "gemini-3.1-pro": "gemini-3.1-pro-low",
    "gemini-3-flash": "gemini-3-flash",
    "gemini-3-pro": "gemini-3-pro-low",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
    "gemini-2.5-flash": "gemini-2.5-flash",
}

CATALOG_TTL_S: Final = 600.0


class ModelNotServedError(Exception):
    """The requested name matches nothing the account serves.

    Raising is deliberate: a ``gemini-*`` wildcard would make any invented name answer as
    ``gemini-2.5-flash``, with the ``model`` field echoing the requested name — and
    billing, comparisons and reproducibility would start to lie.
    """


@dataclass(slots=True)
class ModelCatalog:
    """The account's catalog, with a TTL.

    An instance rather than a global: two proxies in the same process would have different
    accounts, and a shared cache would serve one's catalog to the other.
    """

    ids: tuple[str, ...] = ()
    info: dict[str, Any] = field(default_factory=dict)
    fetched_at: float = 0.0

    def is_fresh(self, *, now: float | None = None) -> bool:
        if not self.ids:
            return False
        return ((now if now is not None else time.time()) - self.fetched_at) < CATALOG_TTL_S

    def update(self, payload: dict[str, Any], *, now: float | None = None) -> tuple[str, ...]:
        """Absorb a ``:fetchAvailableModels`` response.

        The catalog lists variants that no longer answer and flags them in
        ``deprecatedModelIds`` — that is how ``gemini-3.1-pro-high`` shows up as served and
        returns 400.
        """
        models = payload.get("models") or {}
        deprecated = {str(x).lower() for x in (payload.get("deprecatedModelIds") or [])}
        ids = tuple(key for key in models if str(key).lower() not in deprecated)
        if ids:
            self.ids = ids
            self.info = models
            self.fetched_at = now if now is not None else time.time()
        return self.ids


def base_family(model: str) -> str:
    """Name without the variant suffix: ``gemini-3.8-flash-low`` -> ``gemini-3.8-flash``."""
    base = str(model).split("/")[-1].lower()
    for suffix in SUFFIXES:
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def supports_function_ids(model: str) -> bool:
    return str(model).split("/")[-1].lower().startswith("gemini-3")


def _from_catalog(raw: str, effort: str, available: tuple[str, ...]) -> str | None:
    # If the requested name *is* a served variant, honour it: asking for
    # `gemini-3.8-flash-tiered` (real in the catalog) must not end up at `-low` just because
    # effort said so. The suffix used to be stripped unconditionally and the request was lost.
    if raw in available and raw not in BROKEN_WIRE:
        return raw

    base = base_family(raw)
    candidates: list[str] = []
    if override := EFFORT_OVERRIDES.get((base, effort)):
        candidates.append(override)
    candidates.extend(base + suffix for suffix in EFFORT_SUFFIXES.get(effort, ("-low", "")))

    for candidate in candidates:
        if candidate in BROKEN_WIRE:
            continue
        if candidate in available:
            return candidate
    return None


def _from_static_map(raw: str) -> str | None:
    if raw in STATIC_MAP:
        return STATIC_MAP[raw]
    # Partial match only when what is left over is a known variant suffix. With a raw
    # `if k in raw`, `gemini-3.8-flash` matched inside `gemini-3.8-flash-thinking` and
    # served `-low` for a name that does not exist.
    for known, wire in STATIC_MAP.items():
        if not raw.startswith(known):
            continue
        rest = raw[len(known) :]
        if not rest or rest in SUFFIXES:
            return wire
    return None


# omp: providers/google-gemini-cli.ts :: lastGoodEndpoint
def map_model(model: str, effort: str | None = None, catalog: ModelCatalog | None = None) -> str:
    """Name that goes on the wire. Raises ``ModelNotServedError`` if nothing matches."""
    raw = str(model).split("/")[-1].lower()
    normalized_effort = str(effort or "medium").strip().lower() or "medium"

    if catalog is not None and catalog.ids:
        resolved = _from_catalog(raw, normalized_effort, catalog.ids)
        if resolved is not None:
            return resolved

    if resolved := _from_static_map(raw):
        return resolved

    raise ModelNotServedError(
        f"Google Antigravity: model '{raw}' is not served by this account "
        f"(no variant matches in the catalog or in the static map)"
    )
