"""Codex against omp 18.4.4: turn state, quota headers, tool schemas, service tiers.

The request goes through the real proxy app with the `openai` SDK as client, a real
`litellm.Router` and the real `Transport`; only the Codex backend is fake — an
`httpx.MockTransport` that answers SSE with the response headers each test scripts.

The schema corpus is differential, like `test_schema_differential.py`: the expected
outputs are what omp's own TypeScript produced for the same inputs, run with Bun over the
18.8.6 `utils/schema/*` sources (``adaptSchemaForStrict(sanitizeSchemaForOpenAIResponses(
toolWireSchema({parameters: case})), false).schema``, the Codex path of
`convertOpenAICodexResponsesTools`) — not what the Python produces. Regenerate both files
when the pinned omp version goes up; `codex_schema_expected.json` holds one output per case
of ``[*cca_schema_cases.json, *codex_schema_cases.json]``, ``null`` where omp throws. The
18.4.4 → 18.8.6 move left the Codex output of every pre-existing case unchanged.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Final

import httpx
import litellm
import litellm.proxy.proxy_server as proxy_server
import openai
import pytest
from fastapi import FastAPI

from litellm_mysubs import plugin
from litellm_mysubs.catalog.discovery import discover
from litellm_mysubs.credentials.store import Credential
from litellm_mysubs.transport.client import Transport
from litellm_mysubs.ui import install as ui_install
from litellm_mysubs.ui.app import mount
from litellm_mysubs.ui.service import MySubsService
from litellm_mysubs.wire import codex
from litellm_mysubs.wire.openai_schema import codex_tool_parameters
from tests.test_plugin import FakeStore, codex_events
from tests.test_router_ownership_real import _LITELLM_ENTRY_POINTS

CODEX: Final = "mysubs/codex/gpt-5.5"
FIXTURES: Final = Path(__file__).parent / "fixtures"


def _jwt(payload: dict[str, Any]) -> str:
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"header.{body}.signature"


#: A ChatGPT token naming its account, as the real ones do.
TOKEN: Final = _jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-1"}})

WEATHER_TOOL: Final = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


def _call(call_id: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
            }
        ],
    }


def _result(call_id: str, text: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": text}


OPENING: Final = [{"role": "user", "content": "weather in Paris?"}]


class Backend:
    """The Codex backend: records each request and answers it with the next scripted
    response headers (none once the script runs out)."""

    def __init__(self, *headers: dict[str, str]) -> None:
        self.script = list(headers)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        extra = self.script.pop(0) if self.script else {}
        body = "".join(f"data: {json.dumps(event)}\n\n" for event in codex_events(text="ok"))
        return httpx.Response(
            200, text=body, headers={"content-type": "text/event-stream", **extra}
        )

    def sent(self, header: str) -> list[str | None]:
        return [request.headers.get(header) for request in self.requests]

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(request.content) for request in self.requests]


@pytest.fixture
def router(monkeypatch: pytest.MonkeyPatch) -> Iterable[Any]:
    """The proxy with a Router serving our Codex deployment; no remembered Codex state."""
    plugin.uninstall()
    monkeypatch.setattr(litellm, "model_cost", dict(litellm.model_cost))
    for (module, name), function in _LITELLM_ENTRY_POINTS.items():
        monkeypatch.setattr(module, name, function)
    router = litellm.Router(
        model_list=[
            {
                "model_name": CODEX,
                "litellm_params": {"model": "openai/gpt-5.5"},
                "model_info": {"id": CODEX, "mysubs_provider": "openai-codex"},
            }
        ]
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "master_key", None)
    monkeypatch.setattr(codex, "_metadata_sessions", type(codex._metadata_sessions)())
    monkeypatch.setattr(codex, "_advertised_service_tiers", {})
    monkeypatch.setattr(ui_install, "_SERVICE", None)
    plugin.install()
    yield router
    plugin.uninstall()


def serve(backend: Backend) -> FakeStore:
    """The real `Transport`, talking to ``backend`` instead of chatgpt.com."""
    store = FakeStore({"openai-codex": Credential(provider="openai-codex", access_token=TOKEN)})
    client = httpx.AsyncClient(transport=httpx.MockTransport(backend))
    plugin.configure(store=store, transport=Transport(client=client))
    return store


async def chat(
    messages: list[dict[str, Any]], *, stream: bool = False, **kwargs: Any
) -> str:
    """One chat completion through the proxy, as the `openai` SDK sends it."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy_server.app), base_url="http://proxy"
    ) as http:
        client = openai.AsyncOpenAI(
            api_key="sk-anything", base_url="http://proxy/v1", http_client=http, max_retries=0
        )
        if not stream:
            answer = await client.chat.completions.create(
                model=CODEX, messages=messages, **kwargs  # type: ignore[arg-type]
            )
            return answer.choices[0].message.content or ""
        chunks = await client.chat.completions.create(
            model=CODEX, messages=messages, stream=True, **kwargs  # type: ignore[arg-type]
        )
        return "".join([c.choices[0].delta.content or "" async for c in chunks if c.choices])


STREAMING: Final = [pytest.param(False, id="whole"), pytest.param(True, id="stream")]


@pytest.mark.usefixtures("router")
class TestTurnState:
    """`x-codex-turn-state` is the backend's sticky-routing token for one turn
    (openai-codex-responses.ts :: updateCodexSessionMetadataFromHeaders, createCodexHeaders,
    clearCodexTurnStatesForNewTurn)."""

    @pytest.mark.parametrize("stream", STREAMING)
    async def test_the_token_rides_the_turn_and_dies_with_it(self, stream: bool) -> None:
        """The first token a turn receives goes back on every tool-result follow-up of that
        turn — a later one does not replace it — and a new user turn starts without one.
        ``x-models-etag`` is not per turn: the latest one keeps going out."""
        backend = Backend(
            {"x-codex-turn-state": "ts-1", "x-models-etag": "etag-1"},
            {"x-codex-turn-state": "ts-2"},
            {},
            {"x-codex-turn-state": "ts-3"},
            {},
        )
        serve(backend)
        first_turn = [*OPENING, _call("call_1"), _result("call_1", "sun")]
        deeper = [*first_turn, _call("call_2"), _result("call_2", "warm")]
        next_turn = [
            *deeper,
            {"role": "assistant", "content": "Sunny and warm."},
            {"role": "user", "content": "and tomorrow?"},
        ]

        for messages in (
            OPENING,
            first_turn,
            deeper,
            next_turn,
            [*next_turn, _call("call_3"), _result("call_3", "rain")],
        ):
            assert await chat(messages, stream=stream, tools=[WEATHER_TOOL]) == "ok"

        assert backend.sent("x-codex-turn-state") == [None, "ts-1", "ts-1", None, "ts-3"]
        assert backend.sent("x-models-etag") == [None, "etag-1", "etag-1", "etag-1", "etag-1"]

    async def test_a_token_stays_in_its_own_conversation(self) -> None:
        """Two conversations the client names apart, each continuing its turn: the token
        one of them received never reaches the other."""
        backend = Backend({"x-codex-turn-state": "ts-a"}, {}, {}, {})
        serve(backend)
        continued = [*OPENING, _call("call_1"), _result("call_1", "sun")]

        await chat(OPENING, tools=[WEATHER_TOOL], extra_headers={"session_id": "conv-a"})
        await chat(continued, tools=[WEATHER_TOOL], extra_headers={"session_id": "conv-b"})
        await chat(continued, tools=[WEATHER_TOOL], extra_headers={"session_id": "conv-a"})

        assert backend.sent("x-codex-turn-state") == [None, None, "ts-a"]


#: Real `x-codex-*` response headers of `gpt-5.5`, measured on 2026-09-18
#: (`test_catalog_usage.py`).
QUOTA_HEADERS: Final = {
    "x-codex-active-limit": "premium",
    "x-codex-credits-balance": "0",
    "x-codex-plan-type": "plus",
    "x-codex-primary-reset-at": "1789790331",
    "x-codex-primary-used-percent": "0",
    "x-codex-primary-window-minutes": "300",
    "x-codex-secondary-reset-at": "1789999782",
    "x-codex-secondary-used-percent": "19",
    "x-codex-secondary-window-minutes": "10080",
}


def _no_network(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"the page must not probe {request.url} for a fresh reading")


@pytest.mark.usefixtures("router")
class TestQuotaHeaders:
    @pytest.mark.parametrize("stream", STREAMING)
    async def test_an_inference_response_moves_the_subscription_card(
        self, stream: bool, router: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """omp ingests the `x-codex-*` headers of every response
        (usage/openai-codex.ts :: parseCodexRateLimitHeaders). The page read after one
        request shows them, without polling the quota endpoint."""
        store = serve(Backend(QUOTA_HEADERS))
        service = MySubsService(
            store=store,
            router_source=lambda: router,
            client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(_no_network)),
        )
        monkeypatch.setattr(ui_install, "_SERVICE", service)

        assert await chat(OPENING, stream=stream) == "ok"

        ui = FastAPI()
        mount(ui, service, guard=None)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=ui), base_url="http://ui"
        ) as http:
            page = (await http.get("/mysubs/")).text
        (card,) = [c for c in service.cards() if c.provider == "openai-codex"]
        assert [(w.label, w.used_percent) for w in card.usage.windows] == [
            ("5h", 0.0),
            ("7d", 19.0),
        ]
        assert card.usage.plan == "plus"
        assert '<span class="pct">19%</span>' in page
        assert "plan plus" in page


def _cases() -> list[tuple[object, object]]:
    cases = [
        *json.loads((FIXTURES / "cca_schema_cases.json").read_text("utf-8")),
        *json.loads((FIXTURES / "codex_schema_cases.json").read_text("utf-8")),
    ]
    expected = json.loads((FIXTURES / "codex_schema_expected.json").read_text("utf-8"))
    assert len(cases) == len(expected), "an unpaired corpus silences the test"
    # omp throws on a boolean root (`toolWireSchema` stamps it in a WeakMap); this path
    # never sends one — a missing or falsy `parameters` becomes the object schema.
    return [(case, out) for case, out in zip(cases, expected, strict=True) if out is not None]


SCHEMA_CASES: Final = _cases()


class TestToolSchemas:
    @pytest.mark.parametrize(("case", "expected"), SCHEMA_CASES)
    def test_matches_the_typescript_output(self, case: object, expected: object) -> None:
        """Identical to omp's Codex tool parameters, key order included."""
        assert json.dumps(codex_tool_parameters(case)) == json.dumps(expected)

    @pytest.mark.usefixtures("router")
    async def test_the_backend_receives_the_normalized_schema(self) -> None:
        """`oneOf` becomes `anyOf`, an object without `properties` gets them, a lookahead
        `pattern` goes, a nullable scalar union becomes a type array — on the wire, from a
        client's tool, while a hosted tool still passes through untouched."""
        backend = Backend()
        serve(backend)
        tool = {
            "type": "function",
            "function": {
                "name": "search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"oneOf": [{"type": "string"}, {"type": "object"}]},
                        "glob": {"type": "string", "pattern": "^(?!tmp/).*"},
                        "limit": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                    },
                    "required": ["target"],
                },
            },
        }

        await chat(OPENING, tools=[tool, {"type": "web_search"}])

        (body,) = backend.bodies
        assert body["tools"] == [
            {
                "type": "function",
                "name": "search",
                "description": "",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "anyOf": [{"type": "string"}, {"type": "object", "properties": {}}]
                        },
                        "glob": {"type": "string"},
                        "limit": {"type": ["integer", "null"]},
                    },
                    "required": ["target"],
                },
            },
            {"type": "web_search"},
        ]


async def _discover_tiers(entry: dict[str, Any]) -> None:
    """Codex discovery over a `/models` answer listing ``entry``."""

    def models(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [entry]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(models)) as http:
        await discover(Credential(provider="openai-codex", access_token=TOKEN), client=http)


@pytest.mark.usefixtures("router")
class TestServiceTier:
    """types.ts :: shouldSendServiceTier for Codex models: the tiers a model's discovery
    advertises gate `priority`/`scale`; an empty or missing list is "not reported"."""

    @pytest.mark.parametrize(
        ("advertised", "requested", "sent"),
        [
            pytest.param([{"id": "flex", "name": "Flex"}], "priority", None, id="omitted"),
            pytest.param([{"id": "flex"}], "scale", None, id="scale-omitted"),
            pytest.param([{"id": "priority", "name": "Fast"}], "priority", "priority", id="listed"),
            pytest.param([], "priority", "priority", id="empty-list"),
            pytest.param(None, "priority", "priority", id="no-list"),
            pytest.param([{"id": "priority"}], "flex", "flex", id="flex-never-gated"),
        ],
    )
    async def test_the_advertised_tiers_gate_the_request(
        self, advertised: list[dict[str, str]] | None, requested: str, sent: str | None
    ) -> None:
        entry: dict[str, Any] = {"slug": "gpt-5.5"}
        if advertised is not None:
            entry["service_tiers"] = advertised
        await _discover_tiers(entry)
        backend = Backend()
        serve(backend)

        await chat(OPENING, service_tier=requested)

        (body,) = backend.bodies
        assert body.get("service_tier") == sent
        hint = "model=gpt-5.5" if sent is None else f"model=gpt-5.5;tier={sent}"
        assert backend.sent("x-codex-routing-hint") == [hint]
