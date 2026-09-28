"""Who owns a call on the Router, measured against a real `litellm.Router`.

The regression these guard (S1, reproduced on 1.101.0 with a real Router): the chat
wrapper handed `dispatch` a `None` provider for any deployment without our mark, and the
name heuristic took over. An operator's own `gpt-4o` with its own key was answered by the
Codex subscription, and an operator's `claude-*` with its own `sk-ant-api03-...` key went
upstream with the Claude Max token — which the Router then read as a client-side
credential and cloned the deployment for. The same leak ran again one hop down: the
Router calls `litellm.acompletion`, which calls `litellm.completion`, both patched, with
the deployment's wire name — and `/v1/responses` reaches that hop through LiteLLM's
bridge for providers without a native Responses API.

Fakes cannot show any of it: the clone is made inside the Router, and the inner hops only
exist because real LiteLLM calls the patched module functions. So every test here builds
a real Router, installs the plugin, and reads what reached LiteLLM's real `completion`
(answered by `mock_response`) or the operator's host (a local HTTP server), and what
reached the subscription transport. No network.
"""

from __future__ import annotations

import copy
import json
import threading
from collections.abc import Iterable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Final

import litellm
import litellm.main
import pytest

from litellm_mysubs import plugin
from litellm_mysubs.catalog.deployments import to_deployment
from litellm_mysubs.catalog.discovery import DiscoveredModel
from litellm_mysubs.credentials.store import Credential
from litellm_mysubs.registry import ModelRegistry
from tests.test_plugin import FakeStore, FakeTransport, codex_events

MESSAGES = [{"role": "user", "content": "hi"}]
OPERATOR_OPENAI_KEY = "sk-operator-openai"
OPERATOR_ANTHROPIC_KEY = "sk-ant-api03-operator"


#: LiteLLM's own entry points, taken at import — i.e. at collection, before any test runs.
#: Tests elsewhere swap `plugin._state.original_acompletion` for a stub, and `uninstall`
#: writes that stub back onto `litellm.acompletion`; measured, running `test_plugin.py`
#: first left these tests talking to a function that returns `"from-original"`.
_LITELLM_ENTRY_POINTS: Final = {
    (module, name): getattr(module, name)
    for module in (litellm, litellm.main)
    for name in ("acompletion", "completion")
}


@pytest.fixture(autouse=True)
def clean_plugin(monkeypatch: pytest.MonkeyPatch) -> Iterable[None]:
    plugin.uninstall()
    for (module, name), function in _LITELLM_ENTRY_POINTS.items():
        monkeypatch.setattr(module, name, function)
    yield
    plugin.uninstall()


class Upstream:
    """The two places a chat call can end: our subscription transport or LiteLLM itself.

    `litellm_calls` records the kwargs LiteLLM's real `completion` received — LiteLLM's
    `acompletion` runs it in an executor, and it is the last point before a provider
    client is built — and forwards them, so `mock_response` answers without network. Both
    patched hops sit in front of it: the Router's call into `litellm.acompletion`, and
    that function's call into `litellm.completion`.
    """

    def __init__(self, *events: dict[str, Any]) -> None:
        self.store = FakeStore(
            {
                "openai-codex": Credential(provider="openai-codex", access_token="tok-codex"),
                "anthropic": Credential(provider="anthropic", access_token="tok-max-1"),
            }
        )
        self.transport = FakeTransport(events)
        self.litellm_calls: list[dict[str, Any]] = []
        plugin.configure(store=self.store, transport=self.transport)
        plugin.install()
        real = plugin._state.original_completion
        assert real is not None

        def spy(*args: Any, **kwargs: Any) -> Any:
            self.litellm_calls.append(dict(kwargs))
            return real(*args, **kwargs)

        plugin._state.original_completion = spy

    def rotate(self, token: str) -> None:
        self.store.set("anthropic", Credential(provider="anthropic", access_token=token))


def claude_prompt_in(call: dict[str, Any]) -> bool:
    return "Claude Code" in json.dumps(call.get("messages"))


class TestAWildcardWithItsOwnKeyIsNotOurs:
    async def test_a_wildcard_with_its_own_key_is_not_ours_either(self) -> None:
        """A name absent from `model_list` resolves through the Router's own wildcard; the
        deployment it lands on is the operator's, so the token has no business there."""
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "anthropic/*",
                    "litellm_params": {"model": "anthropic/*", "api_key": OPERATOR_ANTHROPIC_KEY},
                }
            ]
        )
        upstream = Upstream()

        await router.acompletion(
            model="anthropic/claude-opus-4-5", messages=MESSAGES, mock_response="ok"
        )

        assert [c["api_key"] for c in upstream.litellm_calls] == [OPERATOR_ANTHROPIC_KEY]
        assert len(router.model_list) == 1


class OperatorUpstream:
    """A local HTTP server standing in for the operator's own OpenAI and Anthropic hosts.

    `mock_response` stops `/v1/responses` before LiteLLM's bridge to `litellm.acompletion`,
    which is exactly where that route leaked the token, so these tests let LiteLLM build
    the real provider request and read the credential off the wire.
    """

    def __init__(self) -> None:
        #: ``(path, credential header, whether the Claude Code identity rode along)``.
        self.requests: list[tuple[str, str, bool]] = []
        seen = self.requests

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers["content-length"]))
                key = self.headers.get("x-api-key") or self.headers.get("authorization") or ""
                seen.append((self.path, key, b"Claude Code" in body))
                self._answer(
                    {
                        "id": "c1",
                        "object": "chat.completion",
                        "created": 1,
                        "model": "gpt-4o",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "operator"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    }
                    if self.path.endswith("/chat/completions")
                    else {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-opus-4-5",
                        "content": [{"type": "text", "text": "operator"}],
                        "stop_reason": "end_turn",
                        "stop_sequence": None,
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    }
                )

            def _answer(self, payload: dict[str, Any]) -> None:
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.fixture
def operator_upstream() -> Iterable[OperatorUpstream]:
    upstream = OperatorUpstream()
    yield upstream
    upstream.server.shutdown()
    upstream.server.server_close()


class TestTheOperatorsKeyIsWhatGoesOnTheWire:
    @staticmethod
    def router(base: str) -> Any:
        return litellm.Router(
            model_list=[
                {
                    "model_name": "gpt-4o",
                    "litellm_params": {
                        "model": "openai/gpt-4o",
                        "api_key": OPERATOR_OPENAI_KEY,
                        "api_base": f"{base}/v1",
                    },
                },
                {
                    "model_name": "claude-opus-4-5",
                    "litellm_params": {
                        "model": "anthropic/claude-opus-4-5",
                        "api_key": OPERATOR_ANTHROPIC_KEY,
                        "api_base": base,
                    },
                },
            ]
        )

    async def test_chat(self, operator_upstream: OperatorUpstream) -> None:
        """S1: the operator's `gpt-4o` was answered by Codex, and its `claude-*` went out
        with the Claude Max token and identity and was cloned once per rotation."""
        router = self.router(operator_upstream.base)
        upstream = Upstream(*codex_events(text="from-codex"))
        before = copy.deepcopy(router.model_list)

        for token in ("tok-max-1", "tok-max-2", "tok-max-3"):
            upstream.rotate(token)
            await router.acompletion(model="claude-opus-4-5", messages=MESSAGES)
        answer = await router.acompletion(model="gpt-4o", messages=MESSAGES)

        assert answer.choices[0].message.content == "operator"
        assert upstream.transport.specs == []
        assert operator_upstream.requests == [
            ("/v1/messages", OPERATOR_ANTHROPIC_KEY, False)
        ] * 3 + [("/v1/chat/completions", f"Bearer {OPERATOR_OPENAI_KEY}", False)]
        assert router.model_list == before

    async def test_responses(self, operator_upstream: OperatorUpstream) -> None:
        """LiteLLM answers an Anthropic deployment on `/v1/responses` by bridging it
        through `litellm.acompletion`. Before the fix, that hop injected the Claude Max
        token into the operator's own `claude-*` deployment — on a route whose wrapper
        had already left it alone."""
        router = self.router(operator_upstream.base)
        Upstream()
        plugin.bind_responses_route(router)

        await router.aresponses(model="claude-opus-4-5", input="hi")

        assert operator_upstream.requests == [("/v1/messages", OPERATOR_ANTHROPIC_KEY, False)]

    async def test_messages(self, operator_upstream: OperatorUpstream) -> None:
        """LiteLLM's `/v1/messages` adapter answers a chat-only provider through
        `litellm.acompletion`. Before the fix, an operator's self-hosted `gpt-oss` there
        matched `gpt-` one hop down and was answered by the Codex subscription."""
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "gpt-oss",
                    "litellm_params": {
                        "model": "hosted_vllm/gpt-oss-120b",
                        "api_key": OPERATOR_OPENAI_KEY,
                        "api_base": f"{operator_upstream.base}/v1",
                    },
                }
            ]
        )
        upstream = Upstream(*codex_events(text="from-codex"))
        plugin.bind_messages_route(router)

        await router.aanthropic_messages(model="gpt-oss", messages=MESSAGES, max_tokens=5)

        assert upstream.transport.specs == []
        assert operator_upstream.requests == [
            ("/v1/chat/completions", f"Bearer {OPERATOR_OPENAI_KEY}", False)
        ]


class TestAConfigClaudeWithoutCredentialsIsStillServedByMax:
    """DECISIONS.md D2: `config.yaml` may declare `claude-*` entries — aliases included —
    with no key, and those are served by the Claude Max subscription by name. The token
    now rides on the deployment, so a rotation no longer clones it."""

    async def test_three_rotations_carry_the_current_token_and_leave_one_deployment(
        self,
    ) -> None:
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "claude-opus",
                    "litellm_params": {"model": "anthropic/claude-opus-4-8"},
                }
            ]
        )
        upstream = Upstream()

        for token in ("tok-max-1", "tok-max-2", "tok-max-3"):
            upstream.rotate(token)
            await router.acompletion(model="claude-opus", messages=MESSAGES, mock_response="ok")

        assert [c["api_key"] for c in upstream.litellm_calls] == [
            "tok-max-1",
            "tok-max-2",
            "tok-max-3",
        ]
        assert all(claude_prompt_in(c) for c in upstream.litellm_calls)
        assert len(router.model_list) == 1, "every rotation minted a client-side clone"

    async def test_a_wildcard_without_credentials_still_gets_the_token(self) -> None:
        """The D2 safety net: a family name nobody declared yet, resolved by `claude-*`."""
        router = litellm.Router(
            model_list=[
                {"model_name": "claude-*", "litellm_params": {"model": "anthropic/claude-*"}}
            ]
        )
        upstream = Upstream()

        await router.acompletion(model="claude-opus-9", messages=MESSAGES, mock_response="ok")

        assert [c["api_key"] for c in upstream.litellm_calls] == ["tok-max-1"]
        assert claude_prompt_in(upstream.litellm_calls[0])


def applied(router: Any, wire: str, provider: Any, **extra: Any) -> str:
    """Injects a deployment the way the UI does: real builder, real registry."""
    deployment = to_deployment(
        DiscoveredModel(wire_name=wire, suggested_name=wire, verified=True, **extra), provider
    )
    ModelRegistry(router).apply([deployment])
    return str(deployment["model_name"])


#: What the proxy adds to every call it hands the Router.
def proxy_kwargs() -> dict[str, Any]:
    return {
        "metadata": {"user_api_key": "hashed-key", "user_api_key_alias": "ci"},
        "litellm_call_id": "call-123",
        "proxy_server_request": {"url": "http://proxy/v1/chat/completions"},
    }


class TestOurCodexDeploymentIsServedThroughTheRouter:
    @pytest.mark.parametrize("stream", [False, True])
    async def test_served_by_the_subscription(self, stream: bool) -> None:
        router = litellm.Router(model_list=[])
        name = applied(router, "gpt-5.5", "openai-codex")
        upstream = Upstream(*codex_events(chunks=["ser", "ved"]))
        before = copy.deepcopy(router.model_list)

        response = await router.acompletion(
            model=name, messages=MESSAGES, stream=stream, **proxy_kwargs()
        )
        if stream:
            text = "".join(
                [chunk.choices[0].delta.content or "" async for chunk in response]
            )
        else:
            text = response.choices[0].message.content

        assert text == "served"
        assert upstream.litellm_calls == [], "our model reached LiteLLM's native client"
        [spec] = upstream.transport.specs
        assert spec.body["model"] == "gpt-5.5"
        assert "mysubs_" not in json.dumps(spec.body), "a private kwarg reached the wire"
        assert router.model_list == before


class TestOurClaudeDeploymentCarriesTheToken:
    async def test_the_token_rides_on_the_deployment_without_clones(self) -> None:
        router = litellm.Router(model_list=[])
        name = applied(router, "claude-opus-5", "anthropic")
        upstream = Upstream()

        for token in ("tok-max-1", "tok-max-2", "tok-max-3"):
            upstream.rotate(token)
            await router.acompletion(
                model=name, messages=MESSAGES, mock_response="ok", **proxy_kwargs()
            )

        assert [c["api_key"] for c in upstream.litellm_calls] == [
            "tok-max-1",
            "tok-max-2",
            "tok-max-3",
        ]
        assert all(claude_prompt_in(c) for c in upstream.litellm_calls)
        assert len(router.model_list) == 1
        assert router.model_list[0]["litellm_params"]["api_key"] == "tok-max-3"
