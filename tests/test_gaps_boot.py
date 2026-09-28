"""The startup boot: the models a user applied must come back on a Router that shows up late.

`install()` runs inside the proxy's startup event, with a loop already running and
``llm_router`` not built yet. What it schedules — wait for the Router, bind the
Responses/Messages routes on it, reapply the stored selection, start the refresher — is the
only thing that brings the models back after a restart. Here it runs exactly that way:
`install()` from inside a running loop, the Router assigned to ``proxy_server.llm_router``
only afterwards, and the result read the way a client reads it — ``/v1/models`` and a
``/v1/messages`` turn through the real proxy app. The selection lives in the real
`SelectionStore` on disk (under the test's throwaway ``HOME``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import types
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Final

import httpx
import litellm
import litellm.proxy.proxy_server as proxy_server
import pytest
from fastapi import FastAPI
from litellm._logging import verbose_proxy_logger

from litellm_mysubs import plugin
from litellm_mysubs.catalog.deployments import to_deployment
from litellm_mysubs.catalog.discovery import DiscoveredModel
from litellm_mysubs.catalog.selection import SelectionStore
from litellm_mysubs.credentials import refresher
from litellm_mysubs.credentials.store import Credential
from litellm_mysubs.ui import install as ui_install
from litellm_mysubs.ui.service import MySubsService
from tests.test_plugin import FakeStore, FakeTransport, codex_events

OPERATOR_MODEL: Final = "house-model"


@pytest.fixture(autouse=True)
def startup(monkeypatch: pytest.MonkeyPatch) -> Iterable[None]:
    """A proxy mid-startup: no Router yet, and the plugin not bound to any."""
    plugin.uninstall()
    # A Router registers its deployments' models into the process-wide cost map; a name
    # left there reprices other tests' calls (measured: `gemini/gemini-3-flash` made the
    # `-agent` cost identity resolve to it).
    monkeypatch.setattr(litellm, "model_cost", dict(litellm.model_cost))
    monkeypatch.setattr(proxy_server, "llm_router", None)
    monkeypatch.setattr(proxy_server, "master_key", None)
    monkeypatch.setattr(ui_install, "_SERVICE", None)
    # The real cadence polls every 250 ms for 30 s; the sequence is what is under test.
    monkeypatch.setattr(ui_install, "_ROUTER_WAIT_S", 0.01)
    yield
    plugin.unbind_responses_route()
    plugin.unbind_messages_route()
    plugin.uninstall()


def saved_codex_model() -> str:
    """Applied in a previous life of the proxy: written to disk, not in any Router."""
    deployment = to_deployment(
        DiscoveredModel(wire_name="gpt-5.5", suggested_name="gpt-5.5", verified=True),
        "openai-codex",
    )
    SelectionStore().save("openai-codex", [deployment])
    return str(deployment["model_name"])


def store() -> FakeStore:
    return FakeStore({"openai-codex": Credential(provider="openai-codex", access_token="tok")})


def operator_router() -> litellm.Router:
    return litellm.Router(
        model_list=[
            {
                "model_name": OPERATOR_MODEL,
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-operator"},
            }
        ]
    )


async def booted(service: MySubsService, *, within_s: float = 10.0) -> None:
    """Until the boot has run to its last step, which starts the refresher."""
    async with asyncio.timeout(within_s):
        while service.refresher is None:
            await asyncio.sleep(0.01)


async def proxy_get_models() -> list[str]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy_server.app), base_url="http://proxy"
    ) as client:
        response = await client.get("/v1/models")
    assert response.status_code == 200, response.text
    return [entry["id"] for entry in response.json()["data"]]


async def proxy_messages(model: str) -> dict[str, Any]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy_server.app), base_url="http://proxy"
    ) as client:
        response = await client.post(
            "/v1/messages",
            json={
                "model": model,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert response.status_code == 200, response.text
    return dict(response.json())


async def install_during_startup(credentials: FakeStore) -> MySubsService:
    """`install()` as the proxy's startup event runs it: inside a live loop."""
    ui_install.install(FastAPI(), store=credentials)
    service = ui_install.shared_service()
    assert service is not None
    return service


class OwningStore(FakeStore):
    """The proxy's own store: the one that owns the refresh token."""

    owns_refresh = True


class TestTheRouterArrivesLate:
    async def test_the_applied_models_are_listed_again(self) -> None:
        """Measured before the wait existed: the reapply ran against a Router that did
        not exist yet, returned 0, and ``/v1/models`` came back without what the user had
        applied. The operator's own models must still be there next to ours."""
        name = saved_codex_model()
        service = await install_during_startup(store())
        await asyncio.sleep(0.05)
        router = operator_router()
        proxy_server.llm_router = router

        await booted(service)
        await service.stop_refresher()

        assert sorted(await proxy_get_models()) == sorted([OPERATOR_MODEL, name])

    async def test_the_late_router_answers_messages_through_the_subscription(self) -> None:
        """The Messages route is bound per Router instance, so the bind has to find the
        Router that arrived late — not the `None` there was when `install()` ran."""
        name = saved_codex_model()
        credentials = store()
        transport = FakeTransport(codex_events(text="from codex"))
        plugin.configure(store=credentials, transport=transport)
        service = await install_during_startup(credentials)
        await asyncio.sleep(0.05)
        proxy_server.llm_router = operator_router()

        await booted(service)
        await service.stop_refresher()
        answer = await proxy_messages(name)

        assert [block["text"] for block in answer["content"]] == ["from codex"]
        assert transport.specs[0].headers["Authorization"] == "Bearer tok"

    async def test_a_router_that_never_arrives_still_keeps_the_tokens_renewed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The wait has a ceiling so a proxy that never builds a Router does not keep a task
        alive forever. Past it the boot still runs to the end: the credential about to
        expire is renewed at the provider's token endpoint and written back, so the proxy
        is not found with every subscription expired the morning after."""
        monkeypatch.setattr(ui_install, "_ROUTER_WAIT_TRIES", 3)
        exchanged: list[str] = []

        def token_endpoint(request: httpx.Request) -> httpx.Response:
            exchanged.append(str(request.url))
            return httpx.Response(
                200,
                json={"access_token": "renewed", "refresh_token": "RT-2", "expires_in": 3600},
            )

        monkeypatch.setattr(
            refresher,
            "httpx",
            types.SimpleNamespace(
                AsyncClient=lambda: httpx.AsyncClient(transport=httpx.MockTransport(token_endpoint))
            ),
        )
        # The sweep's lock file sits next to the credentials; this store has no file.
        monkeypatch.setattr(refresher, "DEFAULT_PATH", tmp_path / "credentials.json")
        expiring = Credential(
            provider="openai-codex",
            access_token="expiring",
            refresh_token="RT",
            expires_at=time.time() + 30,
        )
        credentials = OwningStore({"openai-codex": expiring})
        service = await install_during_startup(credentials)

        await booted(service)
        async with asyncio.timeout(10):
            while (credentials.get("openai-codex") or expiring).access_token == "expiring":
                await asyncio.sleep(0.01)
        await service.stop_refresher()

        assert exchanged == ["https://auth.openai.com/oauth/token"]
        renewed = credentials.get("openai-codex")
        assert renewed is not None
        assert (renewed.access_token, renewed.refresh_token) == ("renewed", "RT-2")


class TestBootFailuresAreReported:
    @pytest.fixture
    def records(self) -> Iterable[list[logging.LogRecord]]:
        """What the proxy's logger emits. It does not propagate to the root, so the capture
        is attached to it directly."""
        captured: list[logging.LogRecord] = []

        class Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured.append(record)

        handler = Capture(level=logging.DEBUG)
        previous = verbose_proxy_logger.level
        verbose_proxy_logger.addHandler(handler)
        verbose_proxy_logger.setLevel(logging.DEBUG)
        yield captured
        verbose_proxy_logger.removeHandler(handler)
        verbose_proxy_logger.setLevel(previous)

    async def test_a_selection_the_router_refuses_is_logged_and_the_routes_still_bind(
        self, records: list[logging.LogRecord]
    ) -> None:
        """A ``models.json`` the Router refuses (here an entry whose model is not a string)
        must not take the rest of the boot down with it: the operator reads why the models
        are missing in the log, and the routes are still bound so the subscription keeps
        answering what the Router does know."""
        path = SelectionStore().path
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "providers": {
                        "openai-codex": [
                            {
                                "model_name": "broken",
                                "litellm_params": {"model": 5},
                                "model_info": {"managed_by": "mysubs"},
                            }
                        ]
                    },
                }
            ),
            "utf-8",
        )
        credentials = store()
        transport = FakeTransport(codex_events(text="still served"))
        plugin.configure(store=credentials, transport=transport)
        service = await install_during_startup(credentials)
        # A subscription model from `config.yaml`, which the Router knows on its own.
        proxy_server.llm_router = litellm.Router(
            model_list=[
                {
                    "model_name": "mysubs/codex/gpt-5.5",
                    "litellm_params": {"model": "openai/gpt-5.5"},
                    "model_info": {"id": "codex-1", "mysubs_provider": "openai-codex"},
                }
            ]
        )

        await booted(service)
        await service.stop_refresher()

        failures = [r for r in records if r.levelno >= logging.ERROR]
        assert [r.getMessage() for r in failures] == [
            "mysubs: reapply failed; stored models may be missing"
        ]
        assert failures[0].exc_info is not None
        answer = await proxy_messages("mysubs/codex/gpt-5.5")
        assert [block["text"] for block in answer["content"]] == ["still served"]

    async def test_each_bind_reports_what_it_bound(self, records: list[logging.LogRecord]) -> None:
        """Without this line a route that silently fell back to LiteLLM's native path — no
        spend row, nothing else wrong — took several deploys to attribute."""
        service = await install_during_startup(store())
        proxy_server.llm_router = operator_router()

        await booted(service)
        await service.stop_refresher()

        lines = [r.getMessage() for r in records if r.getMessage().startswith("mysubs: a")]
        assert lines == [
            "mysubs: aresponses bound=True router=True",
            "mysubs: aanthropic_messages bound=True router=True",
        ]
