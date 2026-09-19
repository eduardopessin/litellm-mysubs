"""Management of the deployments the plugin adds to LiteLLM's Router.

Why ``POST /model/new`` is not used: the endpoint returns 200 and writes to Postgres, but
the model never reaches the Router when ``general_settings.supported_db_objects`` does not
include ``"models"`` — and not including it is the correct configuration on installations
that serve A2A agents. Measured: ``/model/new`` → 200, then ``/v1/models`` without the
model and the call failing with 400 ``no healthy deployments``. An "apply" that reports
success and does nothing is the worst possible failure mode, so the plugin injects
directly and persists on its own.

The guard against phantom deployments comes from a defect measured in production: a
``claude-*`` wildcard makes the native provider materialize a deployment for **any**
requested name, before it even talks to the upstream. The 404 that follows is correct, but
the entry stays glued to the Router forever — 25 accumulated, and with ``simple-shuffle``
the copies join the draw against the legitimate model of the same name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

#: Marks the entries created by this plugin. What came from config.yaml is never touched.
MANAGED_BY = "mysubs"


class RouterLike(Protocol):
    """The part of ``litellm.Router`` we depend on."""

    model_list: list[dict[str, Any]]

    def set_model_list(self, model_list: list[dict[str, Any]]) -> None: ...


def is_declared(deployment: dict[str, Any]) -> bool:
    """Whether the deployment came from ``config.yaml``.

    LiteLLM gives config entries the ``model_info.id`` they declare — and our config
    declares ``id`` equal to ``model_name``. The ones created by wildcard resolution get a
    hashed id. That difference is what tells a real entry from a shadow, and it is what
    stops the cleanup from deleting a declared model.
    """
    info = deployment.get("model_info") or {}
    return str(info.get("id") or "") == deployment.get("model_name")


def is_managed(deployment: dict[str, Any]) -> bool:
    """Whether the deployment was added by this plugin."""
    info = deployment.get("model_info") or {}
    return info.get("managed_by") == MANAGED_BY


@dataclass(slots=True)
class ModelRegistry:
    """Injects, removes and reapplies the plugin's deployments."""

    router: RouterLike
    #: Names the upstream refused with "does not exist". See ``remember_not_found``.
    not_found: set[str] = field(default_factory=set)

    # -- guards ----------------------------------------------------------------

    def remember_not_found(self, wire_name: str) -> bool:
        """Remembers a name the upstream said it does not serve.

        Only a "does not exist" error gets here. A 429 or a 500 are transient: treating
        them as non-existence would disable a good model until the next restart.
        """
        wire = wire_name.strip().lower()
        if not wire or wire in self.not_found:
            return False
        self.not_found.add(wire)
        return True

    def is_known_bad(self, wire_name: str) -> bool:
        return wire_name.strip().lower() in self.not_found

    def evict_shadow(self, wire_name: str, *, only_if_declared: bool = False) -> int:
        """Removes deployments that wildcard resolution materialized for ``wire_name``.

        ``only_if_declared=True`` is the success path: a request to a declared model
        matches the config entry **and** the wildcard, and the wildcard copy ends up
        competing in the draw without the real entry's ``model_info``. It is only removed
        when a declared twin exists, that is, when the copy is redundant by construction.

        ``only_if_declared=False`` is the error path: the name has already been refused by
        the upstream, so it is never a served model.

        A name with no twin in the config survives the success path on purpose — it is the
        new model of a family, served on day one without editing configuration.
        """
        wire = wire_name.strip().lower()
        model_list = list(self.router.model_list or [])
        declared = {d.get("model_name") for d in model_list if is_declared(d)}

        keep: list[dict[str, Any]] = []
        dropped = 0
        for deployment in model_list:
            name = str(deployment.get("model_name") or "").lower()
            shadow = name == wire and not is_declared(deployment) and not is_managed(deployment)
            if shadow and (not only_if_declared or deployment.get("model_name") in declared):
                dropped += 1
                continue
            keep.append(deployment)

        if dropped:
            self.router.set_model_list(keep)
        return dropped

    # -- injection --------------------------------------------------------------

    def apply(self, deployments: list[dict[str, Any]]) -> int:
        """Replaces the plugin's entries with the given ones, preserving the config's."""
        kept = [d for d in (self.router.model_list or []) if not is_managed(d)]
        marked = [self._mark(d) for d in deployments]
        self.router.set_model_list(kept + marked)
        return len(marked)

    def managed(self) -> list[dict[str, Any]]:
        return [d for d in (self.router.model_list or []) if is_managed(d)]

    @staticmethod
    def _mark(deployment: dict[str, Any]) -> dict[str, Any]:
        out = dict(deployment)
        info = dict(out.get("model_info") or {})
        info["managed_by"] = MANAGED_BY
        info.setdefault("id", out.get("model_name"))
        out["model_info"] = info
        return out
