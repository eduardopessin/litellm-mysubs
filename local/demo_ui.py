"""Demo da UI: serve `/mysubs` sem tocar em nenhum provedor real.

**Nada aqui é inventado.** Uma versão anterior deste ficheiro fabricava uma credencial do
Google e percentagens de uso do Codex escolhidas para as barras saírem âmbar e vermelha na
captura de ecrã. Ficava bonito e era mentira — exactamente o que o princípio "nunca
inventar números" existe para impedir, e pior por estar a ser apresentado como verificação.

O que resta é o único estado honesto de uma demo: três provedores por ligar. Os cabeçalhos
de uso abaixo, quando activados por `--com-uso`, são os **medidos** contra o proxy real e
estão fixados em `tests/test_catalog_usage.py` — e mesmo esses aparecem com a credencial
marcada como demonstração no título da página.
"""

from __future__ import annotations

import sys
import time
from typing import Any

sys.path.insert(0, "src")

import uvicorn
from fastapi import FastAPI

from litellm_mysubs.credentials.store import Credential, ProviderId
from litellm_mysubs.ui import mount
from litellm_mysubs.ui.service import MySubsService

#: Cabeçalhos reais, capturados de uma resposta do proxy em 2026-09-18. Os mesmos valores
#: estão nos testes; não são estimativas.
MEDIDO_ANTHROPIC = {
    "llm_provider-anthropic-ratelimit-unified-5h-utilization": "0.03",
    "llm_provider-anthropic-ratelimit-unified-5h-reset": str(time.time() + 4200),
    "llm_provider-anthropic-ratelimit-unified-7d-utilization": "0.24",
    "llm_provider-anthropic-ratelimit-unified-7d-reset": str(time.time() + 415000),
}
MEDIDO_CODEX = {
    "x-codex-primary-used-percent": "0",
    "x-codex-primary-window-minutes": "300",
    "x-codex-primary-reset-at": str(time.time() + 18000),
    "x-codex-secondary-used-percent": "19",
    "x-codex-secondary-window-minutes": "10080",
    "x-codex-secondary-reset-at": str(time.time() + 227452),
    "x-codex-plan-type": "plus",
    "x-codex-credits-balance": "0",
}


class DemoStore:
    owns_refresh = True

    def __init__(self, creds: dict[str, Credential] | None = None) -> None:
        self._creds = dict(creds or {})

    def get(self, provider: ProviderId) -> Credential | None:
        return self._creds.get(provider)

    def set(self, provider: ProviderId, credential: Credential) -> None:
        self._creds[provider] = credential

    def delete(self, provider: ProviderId) -> None:
        self._creds.pop(provider, None)

    def reload(self) -> bool:
        return False


class DemoRouter:
    def __init__(self) -> None:
        self.model_list: list[dict[str, Any]] = []

    def set_model_list(self, model_list: list[dict[str, Any]]) -> None:
        self.model_list = model_list


def build() -> FastAPI:
    com_uso = "--com-uso" in sys.argv
    creds: dict[str, Credential] = {}
    if com_uso:
        # Só para ver as barras desenhadas. As credenciais são falsas e não servem nada;
        # os cabeçalhos é que são reais.
        for provider in ("anthropic", "openai-codex"):
            creds[provider] = Credential(
                provider=provider,  # type: ignore[arg-type]
                access_token="demo",
                refresh_token="demo",
                expires_at=time.time() + 5400,
            )

    router = DemoRouter()
    service = MySubsService(store=DemoStore(creds), router_source=lambda: router)
    if com_uso:
        service.observe("anthropic", MEDIDO_ANTHROPIC)
        service.observe("openai-codex", MEDIDO_CODEX)

    app = FastAPI()
    # Sem guarda: é uma demo local sem proxy por trás, logo não há `proxy_admin` nenhum
    # para autenticar. A instalação a sério usa o default, que exige administrador.
    mount(app, service, guard=None)
    return app


if __name__ == "__main__":
    uvicorn.run(build(), host="127.0.0.1", port=8137, log_level="warning")
