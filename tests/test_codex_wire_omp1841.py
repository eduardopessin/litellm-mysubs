"""The exact Codex request, checked against omp 18.4.1.

Every expectation below is read off omp's source (``@oh-my-pi/pi-ai`` / ``pi-catalog`` /
``pi-utils`` 18.4.1); the docstrings cite file and line. The request is taken where it
leaves this package — the `RequestSpec` handed to the transport — so the URL, the headers
and the JSON body are the bytes the backend would receive. The last two classes drive it
through a real `litellm.Router` and through the real proxy app with the `openai` SDK as
client; only the subscription's HTTP is faked.

omp's analogue of this proxy is its auth gateway: `openai-chat-server.ts` parses a chat
completions request, `auth-gateway/server.ts` names its session, and the Codex provider
(`openai-codex-responses.ts`, `openai-codex/request-transformer.ts`) builds the wire.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterable
from typing import Any

import httpx
import litellm
import litellm.proxy.proxy_server as proxy_server
import openai
import pytest

from litellm_mysubs import plugin, specs
from litellm_mysubs.credentials.store import Credential
from litellm_mysubs.transport.client import RequestSpec
from litellm_mysubs.wire import codex
from tests.test_plugin import FakeStore, FakeTransport, codex_events, install_transport
from tests.test_router_ownership_real import _LITELLM_ENTRY_POINTS

CODEX = "mysubs/codex/gpt-5.5"

#: The keys of `x-codex-turn-metadata`, in omp's order
#: (openai-codex-responses.ts:630-650, non-compaction turn).
TURN_METADATA_KEYS = [
    "installation_id",
    "session_id",
    "thread_id",
    "turn_id",
    "window_id",
    "request_kind",
    "turn_started_at_unix_ms",
]

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


def jwt(payload: dict[str, Any]) -> str:
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"header.{body}.signature"


@pytest.fixture(autouse=True)
def clean(monkeypatch: pytest.MonkeyPatch) -> Iterable[None]:
    """Unpatched LiteLLM, neutral dependencies and no remembered Codex sessions."""
    plugin.uninstall()
    for (module, name), function in _LITELLM_ENTRY_POINTS.items():
        monkeypatch.setattr(module, name, function)
    plugin.configure(store=FakeStore(), transport=FakeTransport())
    codex._metadata_sessions.clear()
    yield
    plugin.uninstall()
    codex._metadata_sessions.clear()


async def spec_for(
    messages: list[dict[str, Any]], *, model: str = "gpt-5.5", **extra: Any
) -> RequestSpec:
    install_transport(FakeTransport())
    return await specs._codex_spec(model, messages, extra)


def turn_metadata(spec: RequestSpec) -> dict[str, Any]:
    parsed: dict[str, Any] = json.loads(spec.headers["x-codex-turn-metadata"])
    return parsed


USER = [{"role": "user", "content": "hi"}]

#: An assistant turn that calls the weather tool.
CALL: dict[str, Any] = {
    "role": "assistant",
    "content": "Checking.",
    "tool_calls": [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
        }
    ],
}


class TestPlainTurn:
    async def test_url_headers_and_body(self) -> None:
        """A system prompt and one user message, the client naming its session.

        - URL: `CODEX_BASE_URL` + `/codex/responses` (pi-catalog wire/codex.ts:5,
          openai-codex-responses.ts:4855-4861).
        - Headers: createCodexHeaders (openai-codex-responses.ts:4752-4824) —
          `conversation_id`/`session_id`/`x-client-request-id` carry the session (4784-4787),
          `session-id`/`thread-id`/`x-codex-window-id`/`x-codex-turn-metadata` the request
          identity (683-692), the installation id header is deleted (4793).
        - Body: buildTransformedCodexRequestBody (1530-1577) + transformRequestBody
          (request-transformer.ts:416-549): user items are bare ``{role, content}``
          (openai-codex-responses.ts:4919), `store: false`, `include` the encrypted
          reasoning, and `client_metadata` the same identity (1488, 663-673).
        """
        spec = await spec_for(
            [{"role": "system", "content": "Be terse."}, *USER],
            reasoning_effort="high",
            prompt_cache_key="conv-1",
        )
        meta = turn_metadata(spec)

        assert spec.url == "https://chatgpt.com/backend-api/codex/responses"
        assert spec.headers == {
            "Authorization": "Bearer tok-codex",
            "x-codex-routing-hint": "model=gpt-5.5",
            "OpenAI-Beta": "responses=experimental",
            "originator": "omp",
            "version": "0.159.0",
            "User-Agent": "omp/18.4.1",
            "conversation_id": "conv-1",
            "session_id": "conv-1",
            "x-client-request-id": "conv-1",
            "session-id": "conv-1",
            "thread-id": meta["thread_id"],
            "x-codex-window-id": meta["window_id"],
            "x-codex-turn-metadata": spec.headers["x-codex-turn-metadata"],
            "accept": "text/event-stream",
            "Content-Type": "application/json",
        }
        assert spec.body == {
            "model": "gpt-5.5",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            "stream": True,
            "prompt_cache_key": "conv-1",
            "instructions": "Be terse.",
            "store": False,
            "reasoning": {"effort": "high", "summary": "auto"},
            "include": ["reasoning.encrypted_content"],
            "client_metadata": {
                "x-codex-installation-id": codex.INSTALLATION_ID,
                "session_id": "conv-1",
                "thread_id": meta["thread_id"],
                "x-codex-window-id": meta["window_id"],
                "turn_id": meta["turn_id"],
                "x-codex-turn-metadata": spec.headers["x-codex-turn-metadata"],
            },
        }

    async def test_turn_metadata_is_omps_projection(self) -> None:
        """createCodexRequestMetadata (openai-codex-responses.ts:630-657): the installation
        id rides inside the JSON, the thread id is the session's own random UUID — not the
        session id (520, 633) — and the header is compact ASCII JSON (toAsciiJsonString,
        596-601)."""
        spec = await spec_for(USER, prompt_cache_key="conv-1")
        raw = spec.headers["x-codex-turn-metadata"]
        meta = json.loads(raw)

        assert list(meta) == TURN_METADATA_KEYS
        assert meta["installation_id"] == codex.INSTALLATION_ID
        assert meta["request_kind"] == "turn"
        assert meta["thread_id"] != meta["session_id"]
        assert raw == json.dumps(meta, separators=(",", ":"), ensure_ascii=True)
        assert "x-codex-installation-id" not in spec.headers

    async def test_non_ascii_session_is_escaped_in_the_header(self) -> None:
        """toAsciiJsonString escapes everything from \\x7f up (596-601): the value travels
        in an HTTP header."""
        spec = await spec_for(USER, prompt_cache_key="sessão")
        assert "\\u00e3" in spec.headers["x-codex-turn-metadata"]
        assert spec.headers["x-codex-turn-metadata"].isascii()


class TestSessionScope:
    """auth-gateway/server.ts:178-182: the client's key, else `deriveSessionId`."""

    async def test_body_key_wins(self) -> None:
        """readBodyCacheKey (auth-gateway/http.ts:182-198): `prompt_cache_key` first."""
        spec = await spec_for(
            USER, prompt_cache_key="body", metadata={"session_id": "meta"}, user="alice"
        )
        assert spec.body["prompt_cache_key"] == "body"
        assert spec.headers["conversation_id"] == "body"

    async def test_metadata_bag(self) -> None:
        """http.ts:190-197: then `metadata.{prompt_cache_key, session_id, conversation_id}`."""
        spec = await spec_for(USER, metadata={"conversation_id": "meta-conv"})
        assert spec.body["prompt_cache_key"] == "meta-conv"

    async def test_inbound_headers(self) -> None:
        """http.ts:174-180, 211-220: then the inbound `x-prompt-cache-key`, `session_id`,
        `conversation_id`, `x-session-id`, `x-conversation-id` headers — read from the copy
        LiteLLM's proxy keeps in `proxy_server_request`."""
        spec = await spec_for(
            USER,
            proxy_server_request={"headers": {"X-Conversation-Id": "hdr", "session_id": "sid"}},
            litellm_session_id="litellm",
        )
        assert spec.body["prompt_cache_key"] == "sid"

    async def test_litellm_session_after_omps_sources(self) -> None:
        spec = await spec_for(USER, litellm_session_id="litellm-7")
        assert spec.headers["session_id"] == "litellm-7"

    async def test_user_is_not_a_session(self) -> None:
        """`user` names a person (openai-chat-server.ts:199 keeps it apart from the cache
        key); two chats of one user must not share a session."""
        first = await spec_for([{"role": "user", "content": "chat A"}], user="alice")
        second = await spec_for([{"role": "user", "content": "chat B"}], user="alice")
        assert first.body["prompt_cache_key"] != second.body["prompt_cache_key"]
        assert "alice" not in (first.body["prompt_cache_key"], second.body["prompt_cache_key"])

    async def test_blank_key_counts_as_none(self) -> None:
        """normalizeClientSessionKey (auth-gateway/dispatch.ts:44-46)."""
        spec = await spec_for(USER, prompt_cache_key="   ")
        assert spec.body["prompt_cache_key"].strip()
        assert spec.body["prompt_cache_key"] != "   "

    async def test_derived_key_is_stable_across_turns(self) -> None:
        """deriveSessionId (server.ts:114-133): model, system prompt, tools and the first
        message — the parts a client re-sends unchanged — laid out by deterministicUuid
        (utils/deterministic-id.ts:17-20)."""
        system = {"role": "system", "content": "rules"}
        first = await spec_for([system, {"role": "user", "content": "q1"}])
        later = await spec_for(
            [
                system,
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
            ]
        )
        key = first.body["prompt_cache_key"]
        assert later.body["prompt_cache_key"] == key
        assert first.headers["conversation_id"] == key
        assert len(key) == 36 and key.count("-") == 4

    async def test_long_key_is_hashed_the_same_everywhere(self) -> None:
        """normalizeOpenAIPromptCacheKey (openai-shared.ts:474-476, 1378-1383): headers and
        cache key share the normalized id (1428, 3289-3290)."""
        spec = await spec_for(USER, prompt_cache_key="k" * 100)
        key = spec.body["prompt_cache_key"]
        assert key.startswith("pc_") and len(key) <= 64
        assert spec.headers["session_id"] == key == turn_metadata(spec)["session_id"]

    async def test_cache_retention_none_keeps_the_session(self) -> None:
        """getOpenAIPromptCacheKey drops the key (openai-shared.ts:483-486); the transport
        session does not depend on it (1428)."""
        spec = await spec_for(USER, prompt_cache_key="conv", cache_retention="none")
        assert "prompt_cache_key" not in spec.body
        assert spec.headers["session_id"] == "conv"


class TestTurnIdentity:
    """Thread and window ids live on the session (openai-codex-responses.ts:517-524), the
    turn id changes only when a new turn starts (615-618, 1470, 1957-1964)."""

    async def test_turn_id_survives_a_tool_result_continuation(self) -> None:
        opening = [{"role": "user", "content": "weather?"}]
        continued = [
            *opening,
            CALL,
            {"role": "tool", "tool_call_id": "call_1", "content": "sun"},
        ]
        next_turn = [*continued, {"role": "assistant", "content": "Sunny."}, USER[0]]

        first = turn_metadata(await spec_for(opening, tools=[WEATHER_TOOL]))
        second = turn_metadata(await spec_for(continued, tools=[WEATHER_TOOL]))
        third = turn_metadata(await spec_for(next_turn, tools=[WEATHER_TOOL]))

        assert first["session_id"] == second["session_id"] == third["session_id"]
        assert second["turn_id"] == first["turn_id"]
        assert second["turn_started_at_unix_ms"] == first["turn_started_at_unix_ms"]
        assert third["turn_id"] != first["turn_id"]
        assert (first["thread_id"], first["window_id"]) == (third["thread_id"], third["window_id"])

    async def test_two_conversations_share_no_identity(self) -> None:
        """Per session, not per process: the port pinned one window id for the whole proxy
        and used the session id as thread id."""
        a = await spec_for([{"role": "user", "content": "conversation A"}])
        b = await spec_for([{"role": "user", "content": "conversation B"}])
        for header in ("session_id", "thread-id", "x-codex-window-id"):
            assert a.headers[header] != b.headers[header], header


class TestConversation:
    async def test_tool_exchange_items(self) -> None:
        """Assistant text is a completed `output_text` message with `annotations`
        (openai-shared.ts:2255-2264); arguments are re-serialized from the parsed object
        (openai-chat-server.ts:280-291, openai-shared.ts:2335); the result is a
        `function_call_output` (2412-2416); the choice is mapped only for an offered tool
        (openai-codex-responses.ts:1214-1253)."""
        spec = await spec_for(
            [
                {"role": "user", "content": "weather?"},
                CALL,
                {"role": "tool", "tool_call_id": "call_1", "content": "sun"},
            ],
            tools=[WEATHER_TOOL],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )
        assert spec.body["input"] == [
            {"role": "user", "content": [{"type": "input_text", "text": "weather?"}]},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Checking.", "annotations": []}],
                "status": "completed",
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"city":"Paris"}',
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "sun"},
        ]
        assert spec.body["tools"] == [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Current weather.",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ]
        assert spec.body["tool_choice"] == {"type": "function", "name": "get_weather"}

    async def test_unparseable_arguments_travel_raw(self) -> None:
        """openai-chat-server.ts:283-290: `{__raw: …}` instead of a broken string."""
        call = {
            **CALL,
            "tool_calls": [
                {"id": "c", "type": "function", "function": {"name": "f", "arguments": "{oops"}}
            ],
        }
        spec = await spec_for(
            [USER[0], call, {"role": "tool", "tool_call_id": "c", "content": "r"}]
        )
        assert spec.body["input"][2]["arguments"] == '{"__raw":"{oops"}'

    async def test_malformed_call_is_dropped_with_its_result(self) -> None:
        """sanitizeMalformedToolCalls (transform-messages.ts:283-340)."""
        call = {
            "role": "assistant",
            "tool_calls": [
                {"id": " ", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                {"id": "ok", "type": "function", "function": {"name": "g", "arguments": "{}"}},
            ],
        }
        spec = await spec_for(
            [
                USER[0],
                call,
                {"role": "tool", "tool_call_id": " ", "content": "rejected"},
                {"role": "tool", "tool_call_id": "ok", "content": "fine"},
            ]
        )
        assert [(i.get("type"), i.get("call_id")) for i in spec.body["input"][1:]] == [
            ("function_call", "ok"),
            ("function_call_output", "ok"),
        ]

    async def test_all_system_prompts_join_into_instructions(self) -> None:
        """openai-chat-server.ts:115-118, 164-167: every system message, joined by a blank
        line, is the single system prompt; none becomes a developer item."""
        spec = await spec_for(
            [{"role": "system", "content": "base"}, USER[0], {"role": "system", "content": "late"}]
        )
        assert spec.body["instructions"] == "base\n\nlate"
        assert [i["role"] for i in spec.body["input"]] == ["user"]

    async def test_whitespace_only_user_message_is_dropped(self) -> None:
        """normalizeInputMessageContent (openai-codex-responses.ts:5017, 4918)."""
        spec = await spec_for([{"role": "user", "content": "   "}, USER[0]])
        assert spec.body["input"] == [
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}
        ]

    async def test_forced_unknown_tool_is_omitted(self) -> None:
        """normalizeCodexToolChoice returns undefined for a tool that is not offered
        (1227-1228), and for hosted types it does not map (1252)."""
        for choice in ({"type": "function", "function": {"name": "nope"}}, {"type": "web_search"}):
            spec = await spec_for(USER, tools=[WEATHER_TOOL], tool_choice=choice)
            assert "tool_choice" not in spec.body, choice

    async def test_no_tool_choice_without_tools(self) -> None:
        """buildTransformedCodexRequestBody sets it only inside `if (context.tools…)`
        (1547-1555)."""
        spec = await spec_for(USER, tool_choice="required")
        assert "tool_choice" not in spec.body

    async def test_hosted_tools_pass_through(self) -> None:
        """Kept on purpose (see `HOSTED_TOOL_TYPES`): omp has no hosted passthrough on this
        path (openai-chat-server.ts:381), but the Codex backend serves them."""
        spec = await spec_for(USER, tools=[{"type": "web_search"}, WEATHER_TOOL])
        assert spec.body["tools"][0] == {"type": "web_search"}


class TestImages:
    async def test_text_first_then_images(self) -> None:
        """convertResponsesInputContent partitions text ahead of images
        (openai-shared.ts:1754-1769, vision-guard.ts:5-19)."""
        spec = await spec_for(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,AA", "detail": "high"},
                        },
                        {"type": "text", "text": "what is it?"},
                    ],
                }
            ]
        )
        assert spec.body["input"][0]["content"] == [
            {"type": "input_text", "text": "what is it?"},
            {"type": "input_image", "detail": "high", "image_url": "data:image/png;base64,AA"},
        ]

    async def test_tool_result_images_are_hoisted(self) -> None:
        """pushToolResultMessages (openai-chat-server.ts:326-375): text stays in the result,
        joined by newlines (openai-shared.ts:2386-2389); images follow as a user message."""
        spec = await spec_for(
            [
                USER[0],
                CALL,
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {"type": "text", "text": "line 1"},
                        {"type": "text", "text": "line 2"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,BB"}},
                    ],
                },
            ]
        )
        assert spec.body["input"][-2:] == [
            {"type": "function_call_output", "call_id": "call_1", "output": "line 1\nline 2"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "detail": "auto",
                        "image_url": "data:image/png;base64,BB",
                    }
                ],
            },
        ]


EFFORTS = ["minimal", "low", "medium", "high", "xhigh", "max"]
MODELS = ["gpt-5.5", "gpt-6-astra", "gpt-6-sol", "gpt-6-luna"]


class TestReasoning:
    @pytest.mark.parametrize("model", ["gpt-5.5", "gpt-5.4-mini"])
    async def test_none_turns_reasoning_off_below_5_6(self, model: str) -> None:
        """openai-chat-server.ts:203-204 maps `none` to `forceReasoningOff`, which the Codex
        transformer sends as `{effort: "none"}` (request-transformer.ts:489-491)."""
        spec = await spec_for(USER, model=model, reasoning_effort="none")
        assert spec.body["reasoning"] == {"effort": "none"}
        assert not any("Juice" in json.dumps(item) for item in spec.body["input"])

    @pytest.mark.parametrize(
        "model",
        ["gpt-5.6-sol", "gpt-6-astra", "gpt-6-sol", "gpt-6-luna", "gpt-daybreak-blue-latest"],
    )
    async def test_none_turns_reasoning_off_with_juice_from_5_6(self, model: str) -> None:
        """The live backend answers 400 "'none' is not supported with the 'gpt-6-astra'
        model", and omp's Codex path has no effort fallback. From the 5.6 generation
        (pi-catalog requires-reasoning-off-juice-instruction) `none` sends no `reasoning`
        and ends the input with Juice 0, which measured 0 reasoning tokens live."""
        spec = await spec_for(USER, model=model, reasoning_effort="none")
        assert "reasoning" not in spec.body
        assert spec.body["input"][-1] == {
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_text", "text": "# Juice: 0 !important"}],
        }

    @pytest.mark.parametrize("model", MODELS)
    @pytest.mark.parametrize("effort", EFFORTS)
    async def test_effort_carries_the_summary(self, model: str, effort: str) -> None:
        """getReasoningConfig (request-transformer.ts:153-171): the effort and
        `summary: "auto"`."""
        spec = await spec_for(USER, model=model, reasoning_effort=effort)
        assert spec.body["reasoning"] == {"effort": effort, "summary": "auto"}

    @pytest.mark.parametrize("value", [None, "turbo"])
    async def test_no_known_effort_sends_no_reasoning(self, value: str | None) -> None:
        """isReasoningEffort filters (openai-chat-server.ts:39-48, 205), and with no effort
        the transformer deletes `reasoning` (request-transformer.ts:510-512)."""
        extra = {} if value is None else {"reasoning_effort": value}
        spec = await spec_for(USER, **extra)
        assert "reasoning" not in spec.body

    async def test_responses_reasoning_object(self) -> None:
        spec = await spec_for(USER, reasoning_effort={"effort": "low", "summary": "detailed"})
        assert spec.body["reasoning"] == {"effort": "low", "summary": "detailed"}


class TestServiceTierAndResidency:
    async def test_auto_tier_is_never_sent(self) -> None:
        """shouldSendServiceTier (types.ts:232-236): the Codex endpoint rejects `auto`; the
        routing hint reads the body's tier (openai-codex-responses.ts:4758-4759)."""
        spec = await spec_for(USER, service_tier="auto")
        assert "service_tier" not in spec.body
        assert spec.headers["x-codex-routing-hint"] == "model=gpt-5.5"

    async def test_priority_tier_reaches_body_and_hint(self) -> None:
        spec = await spec_for(USER, service_tier="priority")
        assert spec.body["service_tier"] == "priority"
        assert spec.headers["x-codex-routing-hint"] == "model=gpt-5.5;tier=priority"

    @pytest.mark.parametrize(
        ("claims", "expected"),
        [
            ({"chatgpt_data_residency": "no_constraint"}, "no_constraint"),
            ({"chatgpt_data_residency": " ", "chatgpt_compute_residency": "eu"}, "eu"),
        ],
    )
    async def test_residency_is_omps(self, claims: dict[str, str], expected: str) -> None:
        """getCodexResidency (pi-catalog wire/codex.ts:102-120): the first non-blank of the
        data then compute claim, trimmed, with no value excluded."""
        token = jwt({"https://api.openai.com/auth": claims})
        plugin.configure(
            store=FakeStore(
                {"openai-codex": Credential(provider="openai-codex", access_token=token)}
            ),
            transport=FakeTransport(),
        )
        spec = await specs._codex_spec("gpt-5.5", USER, {})
        assert spec.headers["x-openai-internal-codex-residency"] == expected


class TestThroughTheRouter:
    """Real `litellm.Router`, a deployment marked as ours, `router.acompletion`."""

    @staticmethod
    def router() -> Any:
        return litellm.Router(
            model_list=[
                {
                    "model_name": CODEX,
                    "litellm_params": {"model": "openai/gpt-5.5"},
                    "model_info": {"id": CODEX, "mysubs_provider": "openai-codex"},
                }
            ]
        )

    async def test_the_spec_on_the_wire(self) -> None:
        router = self.router()
        transport = install_transport(FakeTransport(codex_events(text="ok")))
        plugin.install()
        opening = [{"role": "user", "content": "weather?"}]

        await router.acompletion(
            model=CODEX, messages=opening, tools=[WEATHER_TOOL], reasoning_effort="none"
        )
        await router.acompletion(
            model=CODEX,
            messages=[
                *opening,
                CALL,
                {"role": "tool", "tool_call_id": "call_1", "content": "sun"},
            ],
            tools=[WEATHER_TOOL],
            reasoning_effort="none",
        )

        first, second = transport.specs
        assert first.body["reasoning"] == {"effort": "none"}
        assert first.body["input"] == [
            {"role": "user", "content": [{"type": "input_text", "text": "weather?"}]}
        ]
        assert first.body["client_metadata"]["turn_id"] == second.body["client_metadata"]["turn_id"]
        assert turn_metadata(first)["turn_id"] == turn_metadata(second)["turn_id"]
        assert first.headers["conversation_id"] == first.body["prompt_cache_key"]
        assert second.headers["thread-id"] == first.headers["thread-id"]
        assert list(turn_metadata(second)) == TURN_METADATA_KEYS
        assert second.body["input"][-1] == {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "sun",
        }


class TestThroughTheProxy:
    """The proxy app with the `openai` SDK as client: the session header the client sends
    is what names the Codex session (auth-gateway/http.ts:174-180)."""

    @pytest.fixture
    def proxy(self, monkeypatch: pytest.MonkeyPatch) -> FakeTransport:
        router = TestThroughTheRouter.router()
        transport = install_transport(FakeTransport(codex_events(text="ok")))
        plugin.install()
        monkeypatch.setattr(proxy_server, "llm_router", router)
        monkeypatch.setattr(proxy_server, "master_key", None)
        return transport

    async def test_session_header_names_the_session(self, proxy: FakeTransport) -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy_server.app), base_url="http://proxy"
        ) as http:
            client = openai.AsyncOpenAI(
                api_key="sk-anything", base_url="http://proxy/v1", http_client=http
            )
            answer = await client.chat.completions.create(
                model=CODEX,
                messages=[{"role": "user", "content": "hi"}],
                reasoning_effort="low",
                extra_headers={"session_id": "conv-from-client"},
            )

        assert answer.choices[0].message.content == "ok"
        (spec,) = proxy.specs
        assert spec.headers["conversation_id"] == "conv-from-client"
        assert spec.body["prompt_cache_key"] == "conv-from-client"
        assert spec.body["client_metadata"]["session_id"] == "conv-from-client"
        assert spec.body["reasoning"] == {"effort": "low", "summary": "auto"}
