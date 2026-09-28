"""Management of the deployments the plugin adds to LiteLLM's Router.

Why ``POST /model/new`` is not used: the endpoint returns 200 and writes to Postgres, but
the model never reaches the Router when ``general_settings.supported_db_objects`` does not
include ``"models"`` — and not including it is the correct configuration on installations
that serve A2A agents. Measured: ``/model/new`` → 200, then ``/v1/models`` without the
model and the call failing with 400 ``no healthy deployments``. An "apply" that reports
success and does nothing is the worst possible failure mode, so the plugin injects
directly and persists on its own.

The phantom-deployment defect was measured in production: a ``claude-*`` wildcard makes
the native provider materialize a deployment for **any** requested name, before it even
talks to the upstream. The 404 that follows is correct, but the entry stays glued to the
Router forever — 25 accumulated, and with ``simple-shuffle`` the copies join the draw
against the legitimate model of the same name. What keeps this plugin's entries out of it
is the provider prefix every injected ``litellm_params.model`` carries
(`catalog/deployments.py :: wire_prefix`), not a cleanup here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

#: Marks the entries created by this plugin. What came from config.yaml is never touched.
MANAGED_BY = "mysubs"


class RouterLike(Protocol):
    """The part of ``litellm.Router`` we depend on."""

    model_list: list[dict[str, Any]]

    def set_model_list(self, model_list: list[dict[str, Any]]) -> None: ...


def is_managed(deployment: dict[str, Any]) -> bool:
    """Whether the deployment was added by this plugin."""
    info = deployment.get("model_info") or {}
    return info.get("managed_by") == MANAGED_BY


@dataclass(slots=True)
class ModelRegistry:
    """Injects, removes and reapplies the plugin's deployments."""

    router: RouterLike

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
