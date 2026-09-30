"""The Cloud Code request omp 18.4.4 builds, read back through the real proxy.

Each class is one piece of ``buildRequest`` / ``convertMessages`` / ``convertTools``
(``pi-ai/src/providers/google-gemini-cli.ts``, ``google-shared.ts``) with the pi-catalog
compat behind it, driven the way a client drives it: the OpenAI or Anthropic SDK, the real
proxy app over ``httpx.ASGITransport``, a real ``litellm.Router``. Only the subscription's
HTTP is faked, and the body it would have received is what gets asserted.

The expected schemas are omp's own output for the same input — ``normalizeSchemaForCCA``
over ``normalizeSchemaForGoogle(toolWireSchema(tool))`` for Gemini, over
``toolWireSchema(tool)`` for Claude — run from the 18.4.4 sources.
"""

from __future__ import annotations

import contextlib
import re
from collections import OrderedDict
from collections.abc import AsyncIterator, Iterable
from typing import Any, Final

import httpx
import litellm
import litellm.main
import litellm.proxy.proxy_server as proxy_server
import openai
import pytest

from litellm_mysubs import plugin, specs
from litellm_mysubs.wire import antigravity as ag
from litellm_mysubs.wire import antigravity_models
from litellm_mysubs.wire.antigravity import FORCED_TOOL_DIRECTIVE, SIGNATURE_SENTINEL
from tests.test_messages_stream_real import sdk_client
from tests.test_plugin import FakeTransport, gemini_events, install_transport

GEMINI: Final = "mysubs/antigravity/gemini-3-pro"
GEMINI_25: Final = "mysubs/antigravity/gemini-2.5-flash"
CLAUDE: Final = "mysubs/antigravity/claude-sonnet-4-6"
GPT_OSS: Final = "mysubs/antigravity/gpt-oss-120b-medium"
GEMINI_31: Final = "mysubs/antigravity/gemini-3.1-pro"
FLASH_38: Final = "mysubs/antigravity/gemini-3.8-flash"

WEATHER: Final[dict[str, Any]] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}

#: A tool whose fields carry ``null`` three ways: a null-typed field, a type array and a
#: pydantic ``T | None`` union.
NULLABLE: Final[dict[str, Any]] = {
    "type": "function",
    "function": {
        "name": "tag",
        "description": "Tag a note.",
        "parameters": {
            "type": "object",
            "properties": {
                "note": {"type": "null"},
                "tag": {"type": ["string", "null"]},
                "n": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            },
            "required": ["tag"],
        },
    },
}

_LITELLM_ENTRY_POINTS: Final = {
    (module, name): getattr(module, name)
    for module in (litellm, litellm.main)
    for name in ("acompletion", "completion")
}


@pytest.fixture(autouse=True)
def proxy(monkeypatch: pytest.MonkeyPatch) -> Iterable[None]:
    plugin.uninstall()
    # A Router registers its deployments' models into the process-wide cost map; a name
    # left there reprices other tests' calls.
    monkeypatch.setattr(litellm, "model_cost", dict(litellm.model_cost))
    for (module, name), function in _LITELLM_ENTRY_POINTS.items():
        monkeypatch.setattr(module, name, function)
    router = litellm.Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {"model": f"gemini/{model.rsplit('/', 1)[-1]}"},
                "model_info": {"id": model, "mysubs_provider": "google-antigravity"},
            }
            for model in (GEMINI, GEMINI_25, CLAUDE, GPT_OSS, GEMINI_31, FLASH_38)
        ]
    )
    assert plugin.bind_messages_route(router)
    assert plugin.bind_responses_route(router)
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "master_key", None)
    # The account's catalog, already fetched: without it only the static Gemini map
    # resolves, and `claude-*` / `gpt-oss-*` are refused as not served.
    catalog = antigravity_models.ModelCatalog()
    catalog.update(
        {
            "models": {
                "gemini-3-pro-low": {},
                # The catalog entries as the account reports them (measured 2026-09-30).
                "gemini-2.5-flash": {
                    "thinkingBudget": -1,
                    "minThinkingBudget": 128,
                    "maxOutputTokens": 65535,
                },
                "gemini-3.8-flash-low": {
                    "thinkingBudget": 1000,
                    "minThinkingBudget": 32,
                    "maxOutputTokens": 65536,
                },
                "gpt-oss-120b-medium": {"thinkingBudget": 8192, "maxOutputTokens": 32768},
                # As the account's catalog declares it (measured 2026-09-30).
                "gemini-3.1-pro-low": {
                    "thinkingBudget": 1001,
                    "minThinkingBudget": 128,
                    "maxOutputTokens": 65535,
                },
                "claude-sonnet-4-6": {"thinkingBudget": 1024, "maxOutputTokens": 64000},
            }
        }
    )
    monkeypatch.setattr(specs._state, "catalog", catalog)
    # Conversation state is process-wide: each test starts with none.
    monkeypatch.setattr(ag, "_sessions", OrderedDict(), raising=False)
    plugin.install()
    yield
    plugin.uninstall()


@pytest.fixture
async def client() -> AsyncIterator[openai.AsyncOpenAI]:
    sdk = openai.AsyncOpenAI(
        api_key="unused",
        base_url="http://proxy/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy_server.app), base_url="http://proxy"
        ),
    )
    yield sdk
    await sdk.close()


async def chat_request(
    client: openai.AsyncOpenAI, model: str, messages: list[Any] | None = None, **extra: Any
) -> dict[str, Any]:
    """The Cloud Code ``request`` the subscription receives for this chat call."""
    transport = install_transport(FakeTransport(gemini_events()))
    await client.chat.completions.create(
        model=model, messages=messages or [{"role": "user", "content": "go"}], **extra
    )
    assert len(transport.specs) == 1
    return dict(transport.specs[0].body["request"])


def declared(request: dict[str, Any]) -> dict[str, Any]:
    """``parameters`` of each declared function, by name."""
    (tools,) = request["tools"]
    return {d["name"]: d["parameters"] for d in tools["functionDeclarations"]}


def function_parts(request: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    """Every ``functionCall`` / ``functionResponse`` part, in order."""
    return [part for c in request["contents"] for part in c["parts"] if kind in part]


DIRECTIVE_TURN: Final = {"role": "user", "parts": [{"text": FORCED_TOOL_DIRECTIVE}]}


class TestToolSchemas:
    async def test_a_null_field_no_longer_costs_gemini_the_whole_schema(
        self, client: openai.AsyncOpenAI
    ) -> None:
        """omp takes Gemini through the Google dialect first, where a null-typed field or
        branch is merely nullable. Straight into the CCA pass it is a residue the backend
        rejects, and the whole tool fell back to an open object: the model lost every
        argument's type because one field could be null. Claude takes that direct path in
        omp too, so it keeps the fallback."""
        gemini = declared(await chat_request(client, GEMINI, tools=[NULLABLE]))
        claude = declared(await chat_request(client, CLAUDE, tools=[NULLABLE]))

        assert gemini["tag"] == {
            "type": "object",
            "properties": {"note": {}, "tag": {"type": "string"}, "n": {"type": "integer"}},
            "required": ["tag"],
            "propertyOrdering": ["note", "tag", "n"],
        }
        assert claude["tag"] == {"type": "object", "properties": {}}

    async def test_gemini_gets_the_argument_order_claude_does_not(
        self, client: openai.AsyncOpenAI
    ) -> None:
        tool = {
            "type": "function",
            "function": {
                "name": "move",
                "description": "Move a file.",
                "parameters": {
                    "type": "object",
                    "properties": {"to": {"type": "string"}, "from": {"type": "string"}},
                },
            },
        }
        gemini = declared(await chat_request(client, GEMINI, tools=[tool]))
        claude = declared(await chat_request(client, CLAUDE, tools=[tool]))

        assert gemini["move"]["propertyOrdering"] == ["to", "from"]
        assert "propertyOrdering" not in claude["move"]

    async def test_a_tool_without_parameters_declares_an_empty_schema(
        self, client: openai.AsyncOpenAI
    ) -> None:
        """OpenAI lets a function omit ``parameters``; omp's ``buildTools`` reads that as
        ``{}`` and every pass keeps it ``{}``. Claude gets an object schema instead: the
        backend hands it to Anthropic as ``input_schema``, and measured on the live backend
        (2026-09-30) ``{}`` answered 400 "tools.0.custom.input_schema.type: Field required"
        while ``{"type": "object", "properties": {}}`` answered 200."""
        ping = {"type": "function", "function": {"name": "ping", "description": "Ping."}}
        gemini = declared(await chat_request(client, GEMINI, tools=[ping]))
        claude = declared(
            await chat_request(client, CLAUDE, tools=[ping], tool_choice="required")
        )

        assert gemini == {"ping": {}}
        assert claude == {"ping": {"type": "object", "properties": {}}}


class TestToolChoice:
    @pytest.mark.parametrize(
        ("choice", "config"),
        [
            pytest.param("required", {"mode": "ANY"}, id="required"),
            pytest.param(
                {"type": "function", "function": {"name": "get_weather"}},
                {"mode": "ANY", "allowedFunctionNames": ["get_weather"]},
                id="named",
            ),
        ],
    )
    async def test_a_forced_gemini_call_is_restated_in_the_transcript(
        self, client: openai.AsyncOpenAI, choice: Any, config: dict[str, Any]
    ) -> None:
        """omp: Cloud Code Assist drops ``toolConfig`` on Antigravity's Gemini routes and
        answers in text under ``ANY``, so the forced choice also goes in as a final user
        turn."""
        request = await chat_request(client, GEMINI, tools=[WEATHER], tool_choice=choice)

        assert request["toolConfig"] == {"functionCallingConfig": config}
        assert request["contents"][-1] == DIRECTIVE_TURN
        assert request["contents"][0] == {"role": "user", "parts": [{"text": "go"}]}

    @pytest.mark.parametrize(
        ("choice", "config"),
        [
            pytest.param("auto", {"mode": "VALIDATED"}, id="auto"),
            pytest.param("none", {"mode": "NONE"}, id="none"),
        ],
    )
    async def test_an_unforced_gemini_choice_adds_no_turn(
        self, client: openai.AsyncOpenAI, choice: str, config: dict[str, Any]
    ) -> None:
        request = await chat_request(client, GEMINI, tools=[WEATHER], tool_choice=choice)

        assert request["toolConfig"] == {"functionCallingConfig": config}
        assert request["contents"] == [{"role": "user", "parts": [{"text": "go"}]}]

    async def test_a_named_choice_goes_out_as_asked(self, client: openai.AsyncOpenAI) -> None:
        """omp sends the name the client chose without checking it against the declared
        tools; the backend is the judge. Falling back to ``VALIDATED`` served a turn the
        client had not asked for."""
        request = await chat_request(
            client,
            GEMINI,
            tools=[WEATHER],
            tool_choice={"type": "function", "function": {"name": "undeclared"}},
        )

        assert request["toolConfig"] == {
            "functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["undeclared"]}
        }

    @pytest.mark.parametrize(
        "choice",
        [
            "required",
            "none",
            {"type": "function", "function": {"name": "get_weather"}},
        ],
    )
    async def test_claude_always_goes_out_validated(
        self, client: openai.AsyncOpenAI, choice: Any
    ) -> None:
        """pi-catalog's ``antigravity-claude-tool-mode``: ``buildRequest`` overwrites any
        choice with ``VALIDATED`` for Claude on Antigravity, and adds no directive turn."""
        request = await chat_request(client, CLAUDE, tools=[WEATHER], tool_choice=choice)

        assert request["toolConfig"] == {"functionCallingConfig": {"mode": "VALIDATED"}}
        assert request["contents"] == [{"role": "user", "parts": [{"text": "go"}]}]

    async def test_claude_is_validated_even_without_tools(
        self, client: openai.AsyncOpenAI
    ) -> None:
        claude = await chat_request(client, CLAUDE)
        gemini = await chat_request(client, GEMINI)

        assert claude["toolConfig"] == {"functionCallingConfig": {"mode": "VALIDATED"}}
        assert "tools" not in claude
        assert "toolConfig" not in gemini

    async def test_the_messages_route_forces_gemini_the_same_way(self) -> None:
        """Anthropic's ``{"type": "any"}`` from the Anthropic SDK, through ``/v1/messages``."""
        _, sdk = sdk_client()
        transport = install_transport(FakeTransport(gemini_events()))
        await sdk.messages.create(
            model=GEMINI,
            max_tokens=256,
            messages=[{"role": "user", "content": "go"}],
            tools=[
                {
                    "name": "get_weather",
                    "description": "Weather for a city.",
                    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
                }
            ],
            tool_choice={"type": "any"},
        )
        await sdk.close()

        request = transport.specs[0].body["request"]
        assert request["toolConfig"] == {"functionCallingConfig": {"mode": "ANY"}}
        assert request["contents"][-1] == DIRECTIVE_TURN

    async def test_the_responses_route_forces_gemini_the_same_way(
        self, client: openai.AsyncOpenAI
    ) -> None:
        """The Responses API names a function flat: ``{"type": "function", "name"}``."""
        transport = install_transport(FakeTransport(gemini_events()))
        await client.responses.create(
            model=GEMINI,
            input="go",
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Weather for a city.",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                }
            ],
            tool_choice={"type": "function", "name": "get_weather"},
        )

        request = transport.specs[0].body["request"]
        assert request["toolConfig"] == {
            "functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["get_weather"]}
        }
        assert request["contents"][-1] == DIRECTIVE_TURN


def exchange(call_id: str, *, arguments: str = "{}", result_name: str | None = None) -> list[Any]:
    """A user turn, one assistant call and its result."""
    result: dict[str, Any] = {"role": "tool", "tool_call_id": call_id, "content": "sunny"}
    if result_name is not None:
        result["name"] = result_name
    return [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": arguments},
                }
            ],
        },
        result,
    ]


class TestFunctionParts:
    async def test_ids_travel_only_where_the_catalog_grants_them(
        self, client: openai.AsyncOpenAI
    ) -> None:
        """pi-catalog's ``supports-function-part-id`` on this host: the Anthropic class and
        ``gpt-oss``, not Gemini (its id is a public-API contract)."""
        for model, expected in ((CLAUDE, "c1"), (GPT_OSS, "c1"), (GEMINI, None)):
            request = await chat_request(client, model, exchange("c1"), tools=[WEATHER])
            (call,) = function_parts(request, "functionCall")
            (result,) = function_parts(request, "functionResponse")
            assert call["functionCall"].get("id") == expected, model
            assert result["functionResponse"].get("id") == expected, model

    async def test_an_id_the_backend_rejects_is_rewritten_on_both_sides(
        self, client: openai.AsyncOpenAI
    ) -> None:
        """Anthropic takes ``tool_use.id`` only as ``[a-zA-Z0-9_-]{1,64}``; ids minted by
        other providers carry dots and colons."""
        request = await chat_request(
            client, CLAUDE, exchange("functions.get_weather:0"), tools=[WEATHER]
        )
        (call,) = function_parts(request, "functionCall")
        (result,) = function_parts(request, "functionResponse")

        assert call["functionCall"]["id"] == "functions_get_weather_0"
        assert result["functionResponse"]["id"] == "functions_get_weather_0"

    async def test_the_sentinel_goes_only_to_gemini_3(self, client: openai.AsyncOpenAI) -> None:
        """``requires-skip-thought-signature-on-first-function-call`` is a Gemini 3+ contract
        of this host; Claude, ``gpt-oss`` and Gemini 2.5 validate no signature."""
        for model, expected in (
            (GEMINI, SIGNATURE_SENTINEL),
            (CLAUDE, None),
            (GPT_OSS, None),
            (GEMINI_25, None),
        ):
            request = await chat_request(client, model, exchange("c1"), tools=[WEATHER])
            (call,) = function_parts(request, "functionCall")
            assert call.get("thoughtSignature") == expected, model

    async def test_the_result_is_named_after_the_call_it_answers(
        self, client: openai.AsyncOpenAI
    ) -> None:
        """omp names ``functionResponse`` after the emitted call first; a result carrying
        another name must not break the pair."""
        request = await chat_request(
            client, GEMINI, exchange("c1", result_name="something_else"), tools=[WEATHER]
        )
        (result,) = function_parts(request, "functionResponse")

        assert result["functionResponse"]["name"] == "get_weather"

    async def test_arguments_that_are_not_an_object_are_kept_raw(
        self, client: openai.AsyncOpenAI
    ) -> None:
        """``args`` is a Struct: a JSON array there is refused. omp keeps the text under
        ``__raw``, as it does for text that is not JSON at all."""
        request = await chat_request(
            client, GEMINI, exchange("c1", arguments="[1, 2]"), tools=[WEATHER]
        )
        (call,) = function_parts(request, "functionCall")

        assert call["functionCall"]["args"] == {"__raw": "[1, 2]"}


class TestSystemAndSampling:
    async def test_system_messages_become_one_prompt(self, client: openai.AsyncOpenAI) -> None:
        """omp's chat server joins every system message into a single prompt; the text
        parts of one message join with nothing between them."""
        request = await chat_request(
            client,
            GEMINI,
            [
                {"role": "system", "content": "Be brief."},
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "Use "},
                        {"type": "text", "text": "metric."},
                    ],
                },
                {"role": "user", "content": "go"},
            ],
        )

        assert request["systemInstruction"] == {
            "role": "user",
            "parts": [{"text": "Be brief.\n\nUse metric."}],
        }

    async def test_the_callers_sampling_reaches_generation_config(
        self, client: openai.AsyncOpenAI
    ) -> None:
        """omp carries temperature, top-p and presence penalty into ``generationConfig``;
        dropping them served every request at the backend's defaults."""
        request = await chat_request(
            client, GPT_OSS, temperature=0.3, top_p=0.9, presence_penalty=0.5, max_tokens=900
        )
        generation = request["generationConfig"]

        assert list(generation) == [
            "temperature",
            "maxOutputTokens",
            "topP",
            "presencePenalty",
            "thinkingConfig",
        ]
        assert (generation["temperature"], generation["topP"]) == (0.3, 0.9)
        assert generation["presencePenalty"] == 0.5

    async def test_a_field_the_model_refuses_is_left_out(
        self, client: openai.AsyncOpenAI
    ) -> None:
        """Measured on the live backend (2026-09-30): every Gemini answers a penalty with
        400 "Penalty is not enabled for this model", and a thinking Claude answers ``top_p``
        under 0.95 with 400. Sent as-is — omp's way — the whole turn failed; the rest of
        the caller's sampling still goes."""
        sampling = {"temperature": 0.3, "top_p": 0.9, "presence_penalty": 0.5}
        gemini = (await chat_request(client, GEMINI, **sampling))["generationConfig"]
        claude = (await chat_request(client, CLAUDE, **sampling))["generationConfig"]
        claude_high = (await chat_request(client, CLAUDE, top_p=0.97))["generationConfig"]

        assert (gemini["temperature"], gemini["topP"]) == (0.3, 0.9)
        assert "presencePenalty" not in gemini
        assert (claude["temperature"], claude["presencePenalty"]) == (0.3, 0.5)
        assert "topP" not in claude
        assert claude_high["topP"] == 0.97

    async def test_top_k_arrives_from_the_messages_route(self) -> None:
        _, sdk = sdk_client()
        transport = install_transport(FakeTransport(gemini_events()))
        await sdk.messages.create(
            model=GEMINI,
            max_tokens=256,
            messages=[{"role": "user", "content": "go"}],
            # The installed SDK types neither field any more; the Messages API takes both.
            extra_body={"top_k": 40, "temperature": 0.2},
        )
        await sdk.close()

        generation = transport.specs[0].body["request"]["generationConfig"]
        assert generation["topK"] == 40
        assert generation["temperature"] == 0.2


class TestReplayedReasoning:
    @pytest.mark.parametrize(
        ("model", "demoted"),
        [
            pytest.param(GEMINI, "```thinking\nCheck the sky.\n```", id="gemini"),
            pytest.param(CLAUDE, "Check the sky.", id="claude"),
            pytest.param(GPT_OSS, "<think>\nCheck the sky.\n</think>", id="gpt-oss"),
        ],
    )
    async def test_reasoning_sent_back_leads_the_model_turn_as_text(
        self, client: openai.AsyncOpenAI, model: str, demoted: str
    ) -> None:
        """omp's chat server keeps an assistant's ``reasoning_content`` and, toward another
        API, demotes it to text in the target's own thinking form — a replayed unsigned
        ``thought`` part is discarded by Gemini. Dropping it lost the reasoning the client
        chose to send back."""
        request = await chat_request(
            client,
            model,
            [
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "content": "Sunny.", "reasoning_content": "Check the sky."},
                {"role": "user", "content": "sure?"},
            ],
        )

        assert request["contents"][1] == {
            "role": "model",
            "parts": [{"text": demoted}, {"text": "Sunny."}],
        }

    async def test_reasoning_alone_loses_its_trailing_whitespace(
        self, client: openai.AsyncOpenAI
    ) -> None:
        """Anthropic refuses a final assistant text that ends in whitespace."""
        request = await chat_request(
            client,
            CLAUDE,
            [
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "content": None, "reasoning_content": "Look outside. \n"},
                {"role": "user", "content": "and?"},
            ],
        )

        assert request["contents"][1] == {"role": "model", "parts": [{"text": "Look outside."}]}


CALL: Final[dict[str, Any]] = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
    ],
}


def turns(request: dict[str, Any]) -> list[tuple[str, Any]]:
    """Each content as ``(role, what it carries)`` — text, call name or response value."""
    out: list[tuple[str, Any]] = []
    for content in request["contents"]:
        for part in content["parts"]:
            if "functionCall" in part:
                out.append((content["role"], ("call", part["functionCall"]["name"])))
            elif "functionResponse" in part:
                response = part["functionResponse"]
                out.append((content["role"], (response["name"], response["response"])))
            else:
                out.append((content["role"], part["text"]))
    return out


class TestToolPairing:
    """omp's ``transformMessages`` leaves every call followed by exactly one result before
    the request is built; Cloud Code refuses a turn whose responses do not match its calls."""

    async def test_an_unanswered_call_gets_an_error_result(
        self, client: openai.AsyncOpenAI
    ) -> None:
        request = await chat_request(
            client,
            GEMINI,
            [{"role": "user", "content": "weather?"}, CALL, {"role": "user", "content": "skip"}],
            tools=[WEATHER],
        )

        assert turns(request) == [
            ("user", "weather?"),
            ("model", ("call", "get_weather")),
            ("user", ("get_weather", {"error": "No result provided"})),
            ("user", "skip"),
        ]

    async def test_a_late_result_is_pulled_behind_its_call(
        self, client: openai.AsyncOpenAI
    ) -> None:
        request = await chat_request(
            client,
            GEMINI,
            [
                {"role": "user", "content": "weather?"},
                CALL,
                {"role": "user", "content": "wait"},
                {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
            ],
            tools=[WEATHER],
        )

        assert turns(request) == [
            ("user", "weather?"),
            ("model", ("call", "get_weather")),
            ("user", ("get_weather", {"output": "sunny"})),
            ("user", "wait"),
        ]

    async def test_a_second_result_for_the_same_call_is_dropped(
        self, client: openai.AsyncOpenAI
    ) -> None:
        request = await chat_request(
            client,
            GEMINI,
            [
                {"role": "user", "content": "weather?"},
                CALL,
                {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
                {"role": "tool", "tool_call_id": "c1", "content": "rainy"},
            ],
            tools=[WEATHER],
        )

        assert turns(request)[2:] == [("user", ("get_weather", {"output": "sunny"}))]

    async def test_a_result_whose_call_is_gone_becomes_a_note(
        self, client: openai.AsyncOpenAI
    ) -> None:
        """After compaction folds the calling turn away, the result survives as context
        in a ``<stale-tool-result>`` user note instead of a response to nothing."""
        request = await chat_request(
            client,
            GEMINI,
            [
                {"role": "user", "content": "weather?"},
                {"role": "tool", "tool_call_id": "gone", "content": "sunny"},
                {"role": "user", "content": "so?"},
            ],
        )

        assert turns(request) == [
            ("user", "weather?"),
            ("user", '<stale-tool-result tool="" id="gone">\nsunny\n</stale-tool-result>'),
            ("user", "so?"),
        ]


class TestAssistantText:
    async def test_an_assistant_turn_replays_as_one_text(self, client: openai.AsyncOpenAI) -> None:
        """omp's ``buildAssistantMessage`` reads the content with ``stringifyContent``: the
        text parts joined as they were streamed, anything else dropped."""
        request = await chat_request(
            client,
            GEMINI,
            [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Sun"},
                        {"type": "text", "text": "ny."},
                        {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
                    ],
                },
                {"role": "user", "content": "thanks"},
            ],
        )

        assert request["contents"][1] == {"role": "model", "parts": [{"text": "Sunny."}]}


def answered(response_id: str, *, finish: str = "STOP") -> list[dict[str, Any]]:
    """A Cloud Code answer whose events carry ``responseId``, as the backend's do."""
    events = gemini_events(finish=finish)
    for event in events:
        event["response"]["responseId"] = response_id
    return events


REQUEST_ID = re.compile(
    r"^agent/(?P<agent>[0-9a-f-]{36})/\d+/(?P<trajectory>[0-9a-f-]{36})/(?P<step>\d+)$"
)


class TestRequestEnvelope:
    """omp's ``buildAntigravityRequestEnvelope``: ``sessionId`` and ``labels`` inside
    ``request``, and a ``requestId`` that walks one conversation's steps. Measured on the
    live backend (2026-09-30): accepted with HTTP 200 on gemini-3-flash and
    claude-sonnet-4-6, a second turn carrying ``labels.last_execution_id`` included."""

    async def turn(
        self, client: openai.AsyncOpenAI, model: str, messages: list[Any], events: Any
    ) -> dict[str, Any]:
        transport = install_transport(FakeTransport(events))
        with contextlib.suppress(openai.APIError):
            await client.chat.completions.create(model=model, messages=messages)
        (spec,) = transport.specs
        return dict(spec.body)

    async def test_one_conversation_walks_its_steps(self, client: openai.AsyncOpenAI) -> None:
        first = [{"role": "user", "content": "weather?"}]
        second = [
            *first,
            {"role": "assistant", "content": "Sunny."},
            {"role": "user", "content": "sure?"},
        ]
        one = await self.turn(client, GEMINI, first, answered("resp-1"))
        two = await self.turn(client, GEMINI, second, answered("resp-2"))

        ids = [REQUEST_ID.match(body["requestId"]) for body in (one, two)]
        assert all(ids), (one["requestId"], two["requestId"])
        assert ids[0]["agent"] == ids[1]["agent"]
        assert ids[0]["trajectory"] == ids[1]["trajectory"]
        assert (ids[0]["step"], ids[1]["step"]) == ("2", "3")
        assert one["request"]["sessionId"] == two["request"]["sessionId"]
        assert re.fullmatch(r"-\d+", one["request"]["sessionId"])
        assert one["request"]["labels"] == {
            "last_step_index": "1",
            "trajectory_id": ids[0]["trajectory"],
            "used_claude": "false",
            "used_claude_conservative": "false",
        }
        assert two["request"]["labels"] == {
            "last_execution_id": "resp-1",
            "last_step_index": "2",
            "trajectory_id": ids[0]["trajectory"],
            "used_claude": "false",
            "used_claude_conservative": "false",
        }

    async def test_two_conversations_do_not_share_a_session(
        self, client: openai.AsyncOpenAI
    ) -> None:
        one = await self.turn(client, GEMINI, [{"role": "user", "content": "a"}], answered("r"))
        two = await self.turn(client, GEMINI, [{"role": "user", "content": "b"}], answered("r"))

        assert one["request"]["sessionId"] != two["request"]["sessionId"]
        assert "last_execution_id" not in two["request"]["labels"]
        assert two["requestId"].endswith("/2")

    async def test_a_failed_turn_does_not_become_the_last_execution(
        self, client: openai.AsyncOpenAI
    ) -> None:
        """omp commits ``lastExecutionId`` only after a fully successful attempt."""
        first = [{"role": "user", "content": "weather?"}]
        await self.turn(client, GEMINI, first, answered("resp-ok"))
        await self.turn(client, GEMINI, first, answered("resp-blocked", finish="SAFETY"))
        third = await self.turn(client, GEMINI, first, answered("resp-3"))

        assert third["request"]["labels"]["last_execution_id"] == "resp-ok"
        assert third["requestId"].endswith("/4")

    async def test_claude_and_profiled_ids_are_labelled_as_omp_labels_them(
        self, client: openai.AsyncOpenAI
    ) -> None:
        """``used_claude`` from the model class; ``model_enum`` from omp's per-wire-id
        profiles, which name ``gemini-3.1-pro-low``."""
        messages = [{"role": "user", "content": "hi"}]
        claude = await self.turn(client, CLAUDE, messages, answered("r"))
        gemini = await self.turn(client, GEMINI_31, messages, answered("r"))

        assert claude["request"]["labels"]["used_claude"] == "true"
        assert claude["request"]["labels"]["used_claude_conservative"] == "true"
        assert "model_enum" not in claude["request"]["labels"]
        assert gemini["model"] == "gemini-3.1-pro-low"
        assert gemini["request"]["labels"]["model_enum"] == "MODEL_PLACEHOLDER_M36"


class TestTheCallersCeilingBoundsTheAnswer:
    """``max_tokens`` is the total ``maxOutputTokens``, and the thinking budget fits under
    it. Live (2026-09-30): with the budget on top, claude-sonnet-4-6 asked for 64 tokens
    answered 623 words; with 64 as the total every family stopped ``MAX_TOKENS`` within
    it, and the thinking-only ones kept a valid budget."""

    @pytest.mark.parametrize("effort", [None, "high"])
    async def test_claude_thinks_only_when_the_ceiling_has_room(
        self, client: openai.AsyncOpenAI, effort: str | None
    ) -> None:
        extra: dict[str, Any] = {"max_tokens": 64}
        if effort:
            extra["reasoning_effort"] = effort
        request = await chat_request(client, CLAUDE, **extra)

        assert request["generationConfig"] == {
            "maxOutputTokens": 64,
            "thinkingConfig": {"includeThoughts": False, "thinkingBudget": 0},
        }

    async def test_a_thinking_only_model_keeps_a_budget(self, client: openai.AsyncOpenAI) -> None:
        """gemini-3.1-pro-low refuses a budget of 0 — "Budget 0 is invalid. This model only
        works in thinking mode." — and takes its ``minThinkingBudget`` above the ceiling."""
        request = await chat_request(client, GEMINI_31, max_tokens=64)

        assert request["generationConfig"] == {
            "maxOutputTokens": 64,
            "thinkingConfig": {"includeThoughts": True, "thinkingBudget": 128},
        }


class TestNoReasoningOnAModelThatCannotStopThinking:
    """``reasoning_effort: "none"`` sent ``thinkingBudget: 0`` everywhere. Live
    (2026-09-30) six ids refuse it with 400 — gemini-3.1-pro-low and gemini-pro-agent
    ("Budget 0 is invalid. This model only works in thinking mode."), gemini-2.5-flash,
    gemini-2.5-flash-lite, gemini-3.5-flash-lite and gpt-oss-120b-medium — and each
    answered 200 at its ``minThinkingBudget`` (128), or gpt-oss at its own budget.
    Claude and the flash ids with a 32 minimum took the budget of 0."""

    @pytest.mark.parametrize(
        ("model", "thinking"),
        [
            pytest.param(
                GEMINI_31, {"includeThoughts": True, "thinkingBudget": 128}, id="gemini-3.1-pro"
            ),
            pytest.param(
                GEMINI_25, {"includeThoughts": True, "thinkingBudget": 128}, id="gemini-2.5-flash"
            ),
            pytest.param(
                GPT_OSS, {"includeThoughts": True, "thinkingBudget": 8192}, id="gpt-oss"
            ),
            pytest.param(
                CLAUDE, {"includeThoughts": False, "thinkingBudget": 0}, id="claude"
            ),
            pytest.param(
                FLASH_38, {"includeThoughts": False, "thinkingBudget": 0}, id="gemini-3.8-flash"
            ),
        ],
    )
    async def test_effort_none_sends_what_the_model_accepts(
        self, client: openai.AsyncOpenAI, model: str, thinking: dict[str, Any]
    ) -> None:
        request = await chat_request(client, model, reasoning_effort="none")

        assert request["generationConfig"]["thinkingConfig"] == thinking
