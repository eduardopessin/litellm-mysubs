"""Demo da UI com duplos: serve /mysubs sem tocar em nenhum provedor real."""

from __future__ import annotations

import sys
import time
from typing import Any

sys.path.insert(0, "src")

import uvicorn
from fastapi import FastAPI

from litellm_mysubs.catalog.discovery import DiscoveredModel
from litellm_mysubs.credentials.store import Credential, ProviderId
from litellm_mysubs.ui import mount
from litellm_mysubs.ui.service import MySubsService


class DemoStore:
    owns_refresh = True

    def __init__(self) -> None:
        self._creds: dict[str, Credential] = {
            "anthropic": Credential(
                provider="anthropic",
                access_token="AT",
                refresh_token="RT",
                expires_at=time.time() + 5400,
            ),
            "openai-codex": Credential(
                provider="openai-codex",
                access_token="AT",
                refresh_token="RT",
                expires_at=time.time() + 7200,
            ),
            "google-antigravity": Credential(
                provider="google-antigravity",
                access_token="AT",
                refresh_token="RT",
                expires_at=time.time() - 60,
                project_id="meu-projecto-123",
            ),
        }

    def get(self, provider: ProviderId) -> Credential:
        if provider not in self._creds:
            raise KeyError(provider)
        return self._creds[provider]

    def set(self, credential: Credential) -> None:
        self._creds[credential.provider] = credential

    def delete(self, provider: ProviderId) -> None:
        self._creds.pop(provider, None)

    def reload(self) -> None: ...

    def connected(self) -> list[ProviderId]:
        return list(self._creds)  # type: ignore[arg-type]


class DemoRouter:
    def __init__(self) -> None:
        self.model_list: list[dict[str, Any]] = []

    def set_model_list(self, model_list: list[dict[str, Any]]) -> None:
        self.model_list = model_list


router = DemoRouter()
service = MySubsService(store=DemoStore(), router_source=lambda: router)

# Um provedor já com modelos descobertos, para a página mostrar os três estados.
service.discovered["anthropic"] = [
    DiscoveredModel(wire_name="claude-opus-5", suggested_name="claude-opus-5", verified=True),
    DiscoveredModel(wire_name="claude-sonnet-5", suggested_name="claude-sonnet-5", verified=True),
    DiscoveredModel(
        wire_name="claude-haiku-4-5",
        suggested_name="claude-haiku-4-5",
        verified=False,
        note="a sonda não chegou ao upstream (ConnectError: rede em baixo)",
    ),
]

# uso real, medido contra o proxy
service.observe("anthropic", {
    "llm_provider-anthropic-ratelimit-unified-5h-utilization": "0.03",
    "llm_provider-anthropic-ratelimit-unified-5h-reset": str(time.time() + 4200),
    "llm_provider-anthropic-ratelimit-unified-7d-utilization": "0.24",
    "llm_provider-anthropic-ratelimit-unified-7d-reset": str(time.time() + 415000),
})
service.observe("openai-codex", {
    "x-codex-primary-used-percent": "78", "x-codex-primary-window-minutes": "300",
    "x-codex-primary-reset-at": str(time.time() + 9000),
    "x-codex-secondary-used-percent": "93", "x-codex-secondary-window-minutes": "10080",
    "x-codex-secondary-reset-at": str(time.time() + 220000),
    "x-codex-plan-type": "plus", "x-codex-credits-balance": "0",
})

app = FastAPI()
mount(app, service)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8137, log_level="warning")
