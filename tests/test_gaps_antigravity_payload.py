"""Tools reaching Antigravity: what the Cloud Code request carries for a client's tools.

`test_wire_antigravity.py` checks `tools_to_declarations` and `tool_config` on their own and
that a request *without* tools carries neither field. Here the tools come from a client —
the OpenAI SDK, through the real proxy app and a real `litellm.Router` — with the schemas a
real tool library emits (pydantic 2: ``$defs``/``$ref``, ``anyOf`` with ``null``,
``const`` unions, numeric bounds), and the request body the subscription receives is read
back. Only the subscription's HTTP is faked.

The Claude-on-Antigravity expectations are omp 18.4.1's own output for the same schemas —
``normalizeSchemaForCCA(toolWireSchema(tool))``, the path `convertTools` takes when
``ccaLegacyParametersSchema`` is set (pi-catalog ``rules/classes/anthropic.kdl``), run from
the tarball. For Gemini omp inserts ``normalizeSchemaForGoogle`` first, which adds
``propertyOrdering``; that step is not ported, so the Gemini test asserts what both paths
agree on rather than a byte-for-byte copy.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Final

import httpx
import litellm
import litellm.main
import litellm.proxy.proxy_server as proxy_server
import openai
import pytest

from litellm_mysubs import plugin, specs
from litellm_mysubs.wire import antigravity_models
from tests.test_plugin import FakeTransport, gemini_events, install_transport

GEMINI: Final = "mysubs/antigravity/gemini-3-pro"
CLAUDE: Final = "mysubs/antigravity/claude-sonnet-4-6"

#: ``GetWeather.model_json_schema()`` as pydantic 2 emits it.
GET_WEATHER: Final[dict[str, Any]] = {
    "$defs": {
        "Location": {
            "properties": {
                "city": {"description": "City name", "title": "City", "type": "string"},
                "country": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                    "description": "ISO country code",
                    "title": "Country",
                },
            },
            "required": ["city"],
            "title": "Location",
            "type": "object",
        },
        "Unit": {"enum": ["metric", "imperial"], "title": "Unit", "type": "string"},
    },
    "description": "Current weather and forecast for a place.",
    "properties": {
        "location": {"$ref": "#/$defs/Location"},
        "units": {"$ref": "#/$defs/Unit", "default": "metric"},
        "days": {
            "default": 1,
            "description": "Days of forecast",
            "maximum": 14,
            "minimum": 1,
            "title": "Days",
            "type": "integer",
        },
        "include": {
            "items": {"enum": ["wind", "rain"], "type": "string"},
            "title": "Include",
            "type": "array",
        },
    },
    "required": ["location"],
    "title": "GetWeather",
    "type": "object",
}

#: ``ApplyEdits.model_json_schema()``: a list of nested objects, an optional flag and a
#: ``Literal[...] | Literal[...]`` union, which pydantic emits as ``anyOf`` of ``const``.
APPLY_EDITS: Final[dict[str, Any]] = {
    "$defs": {
        "Edit": {
            "properties": {
                "path": {"title": "Path", "type": "string"},
                "old": {"title": "Old", "type": "string"},
                "new": {"title": "New", "type": "string"},
            },
            "required": ["path", "old", "new"],
            "title": "Edit",
            "type": "object",
        }
    },
    "properties": {
        "edits": {
            "items": {"$ref": "#/$defs/Edit"},
            "minItems": 1,
            "title": "Edits",
            "type": "array",
        },
        "dry_run": {
            "anyOf": [{"type": "boolean"}, {"type": "null"}],
            "default": None,
            "title": "Dry Run",
        },
        "mode": {
            "anyOf": [{"const": "strict", "type": "string"}, {"const": "fuzzy", "type": "string"}],
            "default": "strict",
            "title": "Mode",
        },
    },
    "required": ["edits"],
    "title": "ApplyEdits",
    "type": "object",
}

#: omp 18.4.1, ``normalizeSchemaForCCA(toolWireSchema(tool))`` on `GET_WEATHER`.
GET_WEATHER_CCA: Final[dict[str, Any]] = {
    "description": "Current weather and forecast for a place.",
    "properties": {
        "location": {
            "properties": {
                "city": {"description": "City name", "title": "City", "type": "string"},
                "country": {
                    "default": None,
                    "description": "ISO country code",
                    "title": "Country",
                    "type": "string",
                },
            },
            "required": ["city"],
            "title": "Location",
            "type": "object",
        },
        "units": {
            "enum": ["metric", "imperial"],
            "title": "Unit",
            "type": "string",
            "default": "metric",
        },
        "days": {
            "default": 1,
            "description": "Days of forecast\n\n{maximum: 14, minimum: 1}",
            "title": "Days",
            "type": "integer",
        },
        "include": {
            "items": {"enum": ["wind", "rain"], "type": "string"},
            "title": "Include",
            "type": "array",
        },
    },
    "required": ["location"],
    "title": "GetWeather",
    "type": "object",
}

#: omp 18.4.1, ``normalizeSchemaForCCA(toolWireSchema(tool))`` on `APPLY_EDITS`.
APPLY_EDITS_CCA: Final[dict[str, Any]] = {
    "properties": {
        "edits": {
            "items": {
                "properties": {
                    "path": {"title": "Path", "type": "string"},
                    "old": {"title": "Old", "type": "string"},
                    "new": {"title": "New", "type": "string"},
                },
                "required": ["path", "old", "new"],
                "title": "Edit",
                "type": "object",
            },
            "title": "Edits",
            "type": "array",
            "description": "{minItems: 1}",
        },
        "dry_run": {"default": None, "title": "Dry Run", "type": "boolean"},
        "mode": {
            "enum": ["strict", "fuzzy"],
            "type": "string",
            "default": "strict",
            "title": "Mode",
        },
    },
    "required": ["edits"],
    "title": "ApplyEdits",
    "type": "object",
}

TOOLS: Final = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather for a city.",
            "parameters": GET_WEATHER,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_edits",
            "description": "Apply text edits.",
            "parameters": APPLY_EDITS,
        },
    },
]

_LITELLM_ENTRY_POINTS: Final = {
    (module, name): getattr(module, name)
    for module in (litellm, litellm.main)
    for name in ("acompletion", "completion")
}


@pytest.fixture(autouse=True)
def proxy(monkeypatch: pytest.MonkeyPatch) -> Iterable[None]:
    plugin.uninstall()
    # A Router registers its deployments' models into the process-wide cost map; a name
    # left there reprices other tests' calls (measured: `gemini/gemini-3-flash` made the
    # `-agent` cost identity resolve to it).
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
            for model in (GEMINI, CLAUDE)
        ]
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "master_key", None)
    # The account's catalog, already fetched: without it only the static Gemini map
    # resolves, and `claude-sonnet-4-6` is refused as not served.
    catalog = antigravity_models.ModelCatalog()
    catalog.update({"models": {"gemini-3-pro-high": {}, "claude-sonnet-4-6": {}}})
    monkeypatch.setattr(specs._state, "catalog", catalog)
    plugin.install()
    yield
    plugin.uninstall()


async def upstream_request(model: str, **extra: Any) -> dict[str, Any]:
    """The Cloud Code ``request`` object the subscription receives for this chat call."""
    transport = install_transport(FakeTransport(gemini_events()))
    client = openai.AsyncOpenAI(
        api_key="unused",
        base_url="http://proxy/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy_server.app), base_url="http://proxy"
        ),
    )
    await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": "go"}], **extra
    )
    await client.close()
    assert len(transport.specs) == 1
    return dict(transport.specs[0].body["request"])


def walk(value: Any) -> Iterable[str]:
    """Every key anywhere in a JSON value."""
    if isinstance(value, dict):
        for key, inner in value.items():
            yield key
            yield from walk(inner)
    elif isinstance(value, list):
        for inner in value:
            yield from walk(inner)


class TestToolsLandInTheRequest:
    async def test_claude_gets_omps_declarations(self) -> None:
        """Claude on Antigravity takes the legacy ``parameters`` field, which the backend
        translates into Anthropic's ``input_schema``. What goes there must be what omp
        sends: a schema CCA accepts, with the ``$ref``s inlined, the nullable unions folded
        and the bounds it cannot express moved into the description."""
        request = await upstream_request(CLAUDE, tools=TOOLS)

        assert request["tools"] == [
            {
                "functionDeclarations": [
                    {
                        "name": "get_weather",
                        "description": "Weather for a city.",
                        "parameters": GET_WEATHER_CCA,
                    },
                    {
                        "name": "apply_edits",
                        "description": "Apply text edits.",
                        "parameters": APPLY_EDITS_CCA,
                    },
                ]
            }
        ]
        assert request["toolConfig"] == {"functionCallingConfig": {"mode": "VALIDATED"}}

    async def test_gemini_gets_a_schema_the_backend_accepts(self) -> None:
        """The constructs CCA answers 400 to must not survive anywhere in the tree, and
        what they meant must: the referenced object inlined, the optional field typed, the
        literal union an enum, the bounds in the description."""
        request = await upstream_request(GEMINI, tools=TOOLS)

        (declarations,) = request["tools"]
        weather, edits = declarations["functionDeclarations"]
        assert [weather["name"], edits["name"]] == ["get_weather", "apply_edits"]
        rejected = {"$ref", "$defs", "anyOf", "oneOf", "allOf", "const", "not", "minItems"}
        assert rejected.isdisjoint(walk(declarations))

        location = weather["parameters"]["properties"]["location"]
        assert location["required"] == ["city"]
        assert location["properties"]["country"]["type"] == "string"
        assert weather["parameters"]["properties"]["units"]["enum"] == ["metric", "imperial"]
        assert (
            "{maximum: 14, minimum: 1}"
            in weather["parameters"]["properties"]["days"]["description"]
        )
        assert edits["parameters"]["properties"]["mode"]["enum"] == ["strict", "fuzzy"]
        assert edits["parameters"]["properties"]["edits"]["items"]["required"] == [
            "path",
            "old",
            "new",
        ]
        assert request["toolConfig"] == {"functionCallingConfig": {"mode": "VALIDATED"}}

    async def test_a_schema_cca_cannot_express_degrades_to_an_open_object(self) -> None:
        """One tool with a ``not`` must not fail the whole request with a 400: that tool
        goes out with an open object schema, as omp sends it, and the others intact."""
        odd = {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "exclude": {"not": {"type": "string", "pattern": "^x"}},
                    },
                    "required": ["query"],
                },
            },
        }
        request = await upstream_request(GEMINI, tools=[odd, TOOLS[0]])

        search, weather = request["tools"][0]["functionDeclarations"]
        assert search == {
            "name": "search",
            "description": "Search.",
            "parameters": {"type": "object", "properties": {}},
        }
        assert weather["parameters"]["properties"]["location"]["type"] == "object"


class TestToolChoiceLandsInTheRequest:
    @pytest.mark.parametrize(
        ("choice", "config"),
        [
            pytest.param("auto", {"mode": "VALIDATED"}, id="auto"),
            pytest.param("none", {"mode": "NONE"}, id="none"),
            pytest.param("required", {"mode": "ANY"}, id="required"),
            pytest.param(
                {"type": "function", "function": {"name": "apply_edits"}},
                {"mode": "ANY", "allowedFunctionNames": ["apply_edits"]},
                id="named",
            ),
        ],
    )
    async def test_gemini(self, choice: Any, config: dict[str, Any]) -> None:
        request = await upstream_request(GEMINI, tools=TOOLS, tool_choice=choice)

        assert request["toolConfig"] == {"functionCallingConfig": config}
        declared = request["tools"][0]["functionDeclarations"]
        assert [d["name"] for d in declared] == ["get_weather", "apply_edits"]
