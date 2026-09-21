"""Entry point: patching, normalisation, dispatch and the shape of the responses.

No network: the transport is a double that returns already-decoded events, which is exactly
what the real `Transport` delivers (`AsyncIterator[dict]`).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Iterable
from types import SimpleNamespace
from typing import Any

import litellm
import litellm.main
import pytest

from litellm_mysubs import observability, plugin, routes, specs
from litellm_mysubs.credentials.store import Credential, CredentialStore, ProviderId
from litellm_mysubs.transport.client import (
    RedeemRequired,
    RemapRequired,
    RequestSpec,
    Response,
    UpstreamError,
)
from litellm_mysubs.wire.antigravity_models import ModelNotServedError


class FakeStore(CredentialStore):
    """In-memory store; never touches the disk nor the environment."""

    owns_refresh = False

    def __init__(self, credentials: dict[ProviderId, Credential] | None = None) -> None:
        self._credentials = dict(credentials or {})
        self.reloads = 0

    def get(self, provider: ProviderId) -> Credential | None:
        return self._credentials.get(provider)

    def set(self, provider: ProviderId, credential: Credential) -> None:
        self._credentials[provider] = credential

    def delete(self, provider: ProviderId) -> None:
        self._credentials.pop(provider, None)

    def reload(self) -> bool:
        self.reloads += 1
        return False


class FakeTransport:
    """Double of `Transport`: returns stored events and records the specs it received."""

    def __init__(
        self,
        events: Iterable[dict[str, Any]] = (),
        *,
        error: Exception | None = None,
        delay: float = 0.0,
    ) -> None:
        self.events = list(events)
        self.error = error
        #: Seconds before the first event. Without it a test cannot tell the moment the
        #: stream opened from the moment the model answered, which is the whole of TTFT.
        self.delay = delay
        self.specs: list[RequestSpec] = []

    async def request(self, spec: RequestSpec) -> Response:  # pragma: no cover - unused
        self.specs.append(spec)
        raise AssertionError("dispatch always uses stream()")

    async def stream(self, spec: RequestSpec) -> AsyncIterator[dict[str, Any]]:
        self.specs.append(spec)
        if self.error is not None:
            raise self.error
        if self.delay:
            await asyncio.sleep(self.delay)
        for event in self.events:
            yield event


def codex_events(
    *, text: str = "hello", status: str = "completed", usage: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    return [
        {"type": "response.output_text.delta", "delta": text},
        {
            "type": "response.completed" if status == "completed" else "response.incomplete",
            "response": {"status": status, "usage": usage or {}},
        },
    ]


def gemini_events(
    *,
    text: str = "hello",
    finish: str = "STOP",
    usage: dict[str, Any] | None = None,
    chunks: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Cloud Code events. `chunks` splits the answer across several upstream events.

    A single-event fixture cannot tell an incremental stream from one that blocks and
    replays the finished turn, which is how a two-event non-stream reached production on
    the Responses route.
    """
    parts = chunks if chunks is not None else [text]
    return [
        {
            "response": {
                "candidates": [
                    {
                        "content": {"parts": [{"text": part}]},
                        **({"finishReason": finish} if last else {}),
                    }
                ],
                "usageMetadata": (usage or {}) if last else {},
            }
        }
        for part, last in ((p, i == len(parts) - 1) for i, p in enumerate(parts))
    ]


@pytest.fixture(autouse=True)
def clean_plugin() -> Iterable[None]:
    """Every test starts unpatched and with neutral dependencies."""
    plugin.uninstall()
    plugin.configure(store=FakeStore(), transport=FakeTransport())
    plugin._state.signatures.clear()
    yield
    plugin.uninstall()


def install_transport(transport: FakeTransport) -> FakeTransport:
    plugin.configure(
        store=FakeStore(
            {
                "openai-codex": Credential(provider="openai-codex", access_token="tok-codex"),
                "google-antigravity": Credential(
                    provider="google-antigravity",
                    access_token="tok-google",
                    project_id="proj-1",
                ),
                "anthropic": Credential(provider="anthropic", access_token="tok-claude"),
            }
        ),
        transport=transport,
    )
    return transport


class TestInstall:
    def test_install_twice_does_not_chain_wrappers(self) -> None:
        """Two chained wrappers would make every request go through dispatch twice, and
        `uninstall` would leave the patch half applied."""
        original = litellm.main.acompletion
        plugin.install()
        first = litellm.main.acompletion
        plugin.install()

        assert litellm.main.acompletion is first
        plugin.uninstall()
        assert litellm.main.acompletion is original

    def test_uninstall_restores_both_entrypoints(self) -> None:
        original_async = litellm.main.acompletion
        original_sync = litellm.main.completion
        plugin.install()
        assert litellm.main.acompletion is not original_async
        assert litellm.main.completion is not original_sync

        plugin.uninstall()
        assert litellm.main.acompletion is original_async
        assert litellm.main.completion is original_sync

    def test_the_package_level_name_is_patched_too(self) -> None:
        """`litellm.acompletion` is the documented way of calling the library.

        `litellm/__init__.py` does `from .main import acompletion`, which copies the
        reference. Patching only `litellm.main` left `litellm.acompletion` pointing at the
        original, and a request down that path reached the native route with a name no
        provider knows — `BadRequestError: LLM Provider NOT provided`.

        The tests only looked at `litellm.main`, so they passed with the patch half applied.
        """
        original = litellm.acompletion
        plugin.install()
        try:
            assert litellm.acompletion is not original
            assert litellm.acompletion is litellm.main.acompletion
            assert litellm.completion is litellm.main.completion
        finally:
            plugin.uninstall()
        assert litellm.acompletion is original

    def test_uninstall_without_install_is_a_noop(self) -> None:
        original = litellm.main.acompletion
        plugin.uninstall()
        assert litellm.main.acompletion is original


class TestDelegation:
    """A model that is not ours falls through to the original — with the Claude prompt
    applied."""

    async def test_foreign_model_reaches_original_with_claude_prompt(self) -> None:
        seen: dict[str, Any] = {}

        async def fake_original(**kwargs: Any) -> str:
            seen.update(kwargs)
            return "from-original"

        install_transport(FakeTransport())
        plugin.install()
        plugin._state.original_acompletion = fake_original

        result = await litellm.main.acompletion(
            model="claude-sonnet-4-6", messages=[{"role": "user", "content": "hi"}]
        )

        assert result == "from-original"
        # `build_request` puts the Claude Code identity as the first system message plus the
        # subscription credential; without that Anthropic refuses the OAuth request.
        assert seen["api_key"] == "tok-claude"
        assert seen["messages"][0]["role"] == "system"
        assert "Claude Code" in json.dumps(seen["messages"][0]["content"])
        assert "anthropic-beta" in seen["extra_headers"]

    async def test_non_claude_foreign_model_is_untouched(self) -> None:
        """A third-party model carries neither Anthropic headers nor our credential."""
        seen: dict[str, Any] = {}

        async def fake_original(**kwargs: Any) -> str:
            seen.update(kwargs)
            return "from-original"

        install_transport(FakeTransport())
        plugin.install()
        plugin._state.original_acompletion = fake_original

        await litellm.main.acompletion(
            model="mistral-large", messages=[{"role": "user", "content": "hi"}]
        )

        assert "api_key" not in seen
        assert "extra_headers" not in seen

    async def test_our_model_never_reaches_the_original(self) -> None:
        async def fake_original(**kwargs: Any) -> str:  # pragma: no cover - must not run
            raise AssertionError("a model of ours must not fall through to the original")

        install_transport(FakeTransport(codex_events(text="served")))
        plugin.install()
        plugin._state.original_acompletion = fake_original

        response = await litellm.main.acompletion(
            model="gpt-5.5", messages=[{"role": "user", "content": "hi"}]
        )
        assert response.choices[0].message.content == "served"


class TestPositionalArguments:
    async def test_positional_model_and_messages_are_dispatched(self) -> None:
        """Without normalisation, `acompletion("gpt-5.5", msgs)` fell entirely through to the
        original."""
        transport = install_transport(FakeTransport(codex_events(text="positional")))
        plugin.install()

        response = await litellm.main.acompletion("gpt-5.5", [{"role": "user", "content": "hi"}])

        assert response.choices[0].message.content == "positional"
        assert transport.specs[0].model == "gpt-5.5"
        assert transport.specs[0].body["input"]

    async def test_keyword_wins_over_positional(self) -> None:
        transport = install_transport(FakeTransport(codex_events()))
        plugin.install()

        await litellm.main.acompletion(
            "gpt-5.5", [{"role": "user", "content": "x"}], model="gpt-5.5-codex"
        )

        assert transport.specs[0].model == "gpt-5.5-codex"


class TestDispatchRouting:
    @pytest.mark.parametrize("model", ["gemini-3-pro", "GEMINI-3-FLASH", "gemini-2.5-pro"])
    async def test_gemini_names_go_to_antigravity(self, model: str) -> None:
        transport = install_transport(FakeTransport(gemini_events()))
        await plugin.dispatch(model=model, messages=[{"role": "user", "content": "hi"}])
        assert transport.specs[0].provider == "antigravity"

    @pytest.mark.parametrize("model", ["antigravity-fast", "gemini-3-gpt-preview"])
    async def test_unserved_gemini_name_fails_instead_of_being_substituted(
        self, model: str
    ) -> None:
        """It was routed to Antigravity — and refused there by name.

        Only that branch raises `ModelNotServedError`; the error arriving proves both the
        routing and the fail-loud principle at once: ``gemini-3-gpt-preview`` also matches
        `is_codex_model`, and serving it through another provider (or under another name)
        returned 200 with the ``model`` field echoing the request.
        """
        transport = install_transport(FakeTransport(gemini_events()))
        with pytest.raises(ModelNotServedError):
            await plugin.dispatch(model=model, messages=[{"role": "user", "content": "hi"}])
        assert transport.specs == []

    @pytest.mark.parametrize("model", ["gpt-5.5", "gpt-5.5-codex", "codex-mini"])
    async def test_codex_names_go_to_codex(self, model: str) -> None:
        transport = install_transport(FakeTransport(codex_events()))
        await plugin.dispatch(model=model, messages=[{"role": "user", "content": "hi"}])
        assert transport.specs[0].provider == "codex"

    @pytest.mark.parametrize("model", ["claude-sonnet-4-6", "mistral-large", ""])
    async def test_foreign_models_return_none(self, model: str) -> None:
        transport = install_transport(FakeTransport())
        assert await plugin.dispatch(model=model, messages=[]) is None
        assert transport.specs == []


class TestUpstreamErrors:
    """Fail loud: never substitute a model nor fabricate a response."""

    @pytest.mark.parametrize(
        "error",
        [
            UpstreamError(500, "boom"),
            RemapRequired(400, "is not supported when using Codex"),
            RedeemRequired(429, "quota"),
        ],
    )
    async def test_transport_errors_propagate(self, error: UpstreamError) -> None:
        install_transport(FakeTransport(error=error))
        with pytest.raises(UpstreamError) as caught:
            await plugin.dispatch(model="gpt-5.5", messages=[{"role": "user", "content": "x"}])
        assert caught.value is error

    async def test_a_429_reaches_the_client_as_a_rate_limit(self) -> None:
        """A quota refusal has to keep its status across the LiteLLM boundary.

        `UpstreamError` carries the real 429, but it means nothing to the proxy: an
        exception it does not recognise is reported as `internal_server_error` with HTTP
        500, and the 429 survives only as text inside the message. Measured on the live
        gateway, that is exactly what the client saw — and a client cannot back off on a
        500, which is the one thing it should do here.
        """
        install_transport(FakeTransport(error=UpstreamError(429, "RESOURCE_EXHAUSTED")))
        with pytest.raises(litellm.exceptions.RateLimitError) as caught:
            await plugin.dispatch(model="gpt-5.5", messages=[{"role": "user", "content": "x"}])
        assert caught.value.status_code == 429
        assert "RESOURCE_EXHAUSTED" in str(caught.value)

    async def test_in_band_codex_failure_is_not_a_response(self) -> None:
        """`response.failed` arrives with HTTP 200; swallowing it delivered an empty turn."""
        install_transport(
            FakeTransport([{"type": "response.failed", "response": {"error": "refused"}}])
        )
        with pytest.raises(plugin.StreamError, match="refused"):
            await plugin.dispatch(model="gpt-5.5", messages=[{"role": "user", "content": "x"}])

    async def test_truncated_codex_stream_is_a_failure(self) -> None:
        """With no terminal event the response is cut short: returning it lied to the
        client."""
        install_transport(FakeTransport([{"type": "response.output_text.delta", "delta": "half "}]))
        with pytest.raises(plugin.StreamError, match=r"response\.completed"):
            await plugin.dispatch(model="gpt-5.5", messages=[{"role": "user", "content": "x"}])

    async def test_antigravity_in_band_error_propagates(self) -> None:
        install_transport(FakeTransport([{"error": {"code": 429, "message": "out of quota"}}]))
        with pytest.raises(plugin.StreamError, match="out of quota"):
            await plugin.dispatch(model="gemini-3-pro", messages=[{"role": "user", "content": "x"}])

    async def test_antigravity_blocked_content_propagates(self) -> None:
        install_transport(
            FakeTransport([{"response": {"promptFeedback": {"blockReason": "SAFETY"}}}])
        )
        with pytest.raises(plugin.StreamError, match="SAFETY"):
            await plugin.dispatch(model="gemini-3-pro", messages=[{"role": "user", "content": "x"}])


class TestNonStreamingShape:
    async def test_codex_response_carries_content_and_usage(self) -> None:
        install_transport(
            FakeTransport(
                codex_events(
                    text="answer",
                    usage={
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "input_tokens_details": {"cached_tokens": 40},
                    },
                )
            )
        )
        response = await plugin.dispatch(
            model="gpt-5.5", messages=[{"role": "user", "content": "x"}]
        )

        assert isinstance(response, litellm.ModelResponse)
        assert response.model == "gpt-5.5"
        assert response.choices[0].message.content == "answer"
        assert response.choices[0].finish_reason == "stop"
        assert response.usage.prompt_tokens == 100
        # Without this attribute the cached tokens are invisible in /spend/logs.
        assert response.usage.cache_read_input_tokens == 40

    async def test_incomplete_status_is_length_not_stop(self) -> None:
        """A turn cut short by the output limit arrived as a clean stop."""
        install_transport(FakeTransport(codex_events(status="incomplete")))
        response = await plugin.dispatch(
            model="gpt-5.5", messages=[{"role": "user", "content": "x"}]
        )
        assert response.choices[0].finish_reason == "length"

    async def test_incomplete_without_status_field_is_still_length(self) -> None:
        """The event does not always carry ``status``; the event type is the only clue left.

        Without the fallback on the type, a truncated turn arrived as a clean stop again.
        """
        install_transport(
            FakeTransport(
                [
                    {"type": "response.output_text.delta", "delta": "trunca"},
                    {"type": "response.incomplete", "response": {"usage": {}}},
                ]
            )
        )
        response = await plugin.dispatch(
            model="gpt-5.5", messages=[{"role": "user", "content": "x"}]
        )
        assert response.choices[0].finish_reason == "length"

    async def test_reasoning_lands_in_its_own_field(self) -> None:
        install_transport(
            FakeTransport(
                [
                    {"type": "response.reasoning_text.delta", "delta": "thinking"},
                    {"type": "response.output_text.delta", "delta": "visible"},
                    {"type": "response.completed", "response": {"status": "completed"}},
                ]
            )
        )
        response = await plugin.dispatch(
            model="gpt-5.5", messages=[{"role": "user", "content": "x"}]
        )
        assert response.choices[0].message.content == "visible"
        assert response.choices[0].message.reasoning_content == "thinking"

    async def test_codex_tool_call_is_assembled_from_deltas(self) -> None:
        install_transport(
            FakeTransport(
                [
                    {
                        "type": "response.output_item.added",
                        "item": {
                            "type": "function_call",
                            "id": "item-1",
                            "call_id": "call-1",
                            "name": "read",
                        },
                    },
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": "item-1",
                        "delta": '{"path":',
                    },
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": "item-1",
                        "delta": '"a.txt"}',
                    },
                    {
                        "type": "response.output_item.done",
                        "item": {"type": "function_call", "id": "item-1"},
                    },
                    {"type": "response.completed", "response": {"status": "completed"}},
                ]
            )
        )
        response = await plugin.dispatch(
            model="gpt-5.5", messages=[{"role": "user", "content": "x"}]
        )
        call = response.choices[0].message.tool_calls[0]
        assert call.function.name == "read"
        assert json.loads(call.function.arguments) == {"path": "a.txt"}
        assert response.choices[0].finish_reason == "tool_calls"

    async def test_gemini_finish_reason_maps_from_google_names(self) -> None:
        install_transport(FakeTransport(gemini_events(finish="MAX_TOKENS")))
        response = await plugin.dispatch(
            model="gemini-3-pro", messages=[{"role": "user", "content": "x"}]
        )
        assert response.choices[0].finish_reason == "length"

    async def test_gemini_thought_is_reasoning_not_content(self) -> None:
        install_transport(
            FakeTransport(
                [
                    {
                        "response": {
                            "candidates": [
                                {
                                    "content": {
                                        "parts": [
                                            {"text": "internal", "thought": True},
                                            {"text": "visible"},
                                        ]
                                    },
                                    "finishReason": "STOP",
                                }
                            ]
                        }
                    }
                ]
            )
        )
        response = await plugin.dispatch(
            model="gemini-3-pro", messages=[{"role": "user", "content": "x"}]
        )
        assert response.choices[0].message.content == "visible"
        assert response.choices[0].message.reasoning_content == "internal"


class TestStreamingShape:
    async def test_streaming_returns_the_litellm_wrapper(self) -> None:
        install_transport(FakeTransport(codex_events(text="stream")))
        result = await plugin.dispatch(
            model="gpt-5.5", messages=[{"role": "user", "content": "x"}], stream=True
        )
        assert isinstance(result, litellm.CustomStreamWrapper)

    async def test_stream_ends_with_finish_then_usage(self) -> None:
        """The usage chunk has to arrive and has to carry a non-empty `choices`: the iterator
        of the /v1/responses route does `chunk.choices[0]` with no guard."""
        install_transport(FakeTransport(codex_events(text="stream", usage={"input_tokens": 7})))
        chunks = await collect_stream(model="gpt-5.5", messages=[{"role": "user", "content": "x"}])

        assert [c.choices[0].delta.content for c in chunks if c.choices[0].delta.content] == [
            "stream"
        ]
        assert chunks[-2].choices[0].finish_reason == "stop"
        assert chunks[-1].choices
        assert chunks[-1].usage.prompt_tokens == 7

    async def test_stream_emits_tool_call_open_then_arguments(self) -> None:
        install_transport(
            FakeTransport(
                [
                    {
                        "response": {
                            "candidates": [
                                {
                                    "content": {
                                        "parts": [
                                            {
                                                "functionCall": {
                                                    "id": "c1",
                                                    "name": "read",
                                                    "args": {"path": "a"},
                                                }
                                            }
                                        ]
                                    },
                                    "finishReason": "STOP",
                                }
                            ]
                        }
                    }
                ]
            )
        )
        chunks = await collect_stream(
            model="gemini-3-pro", messages=[{"role": "user", "content": "x"}]
        )
        tool_chunks = [c for c in chunks if c.choices[0].delta.tool_calls]

        assert tool_chunks[0].choices[0].delta.tool_calls[0].function.name == "read"
        assert json.loads(tool_chunks[1].choices[0].delta.tool_calls[0].function.arguments) == {
            "path": "a"
        }
        assert chunks[-2].choices[0].finish_reason == "tool_calls"

    async def test_stream_error_is_not_swallowed(self) -> None:
        install_transport(FakeTransport([{"type": "response.output_text.delta", "delta": "half "}]))
        with pytest.raises(plugin.StreamError):
            await collect_stream(model="gpt-5.5", messages=[{"role": "user", "content": "x"}])


async def collect_stream(**kwargs: Any) -> list[Any]:
    """Drain the LiteLLM wrapper to the end."""
    wrapper = await plugin.dispatch(stream=True, **kwargs)
    return [chunk async for chunk in wrapper.completion_stream]


class TestRequestConstruction:
    async def test_codex_request_carries_token_and_endpoint(self) -> None:
        transport = install_transport(FakeTransport(codex_events()))
        await plugin.dispatch(model="gpt-5.5", messages=[{"role": "user", "content": "x"}])
        spec = transport.specs[0]

        assert spec.url == plugin.CODEX_URL
        assert spec.headers["Authorization"] == "Bearer tok-codex"

    async def test_wire_name_is_resolved_but_response_echoes_the_request(self) -> None:
        """``gpt-6`` is a family alias and gets resolved; the response names what was asked
        for.

        Returning the wire name in the ``model`` field made the spend log bill a model the
        client never named.
        """
        transport = install_transport(FakeTransport(codex_events()))
        response = await plugin.dispatch(model="gpt-6", messages=[{"role": "user", "content": "x"}])

        assert transport.specs[0].body["model"] == "gpt-6-astra"
        assert response.model == "gpt-6"

    async def test_antigravity_request_carries_project_and_token(self) -> None:
        transport = install_transport(FakeTransport(gemini_events()))
        await plugin.dispatch(model="gemini-3-pro", messages=[{"role": "user", "content": "x"}])
        spec = transport.specs[0]

        assert spec.body["project"] == "proj-1"
        assert spec.headers["Authorization"] == "Bearer tok-google"
        assert spec.url.endswith("streamGenerateContent?alt=sse")

    async def test_thought_signature_is_kept_for_the_next_turn(self) -> None:
        """Without the signature sent back, the CCA rejects the next turn carrying the tool
        call."""
        install_transport(
            FakeTransport(
                [
                    {
                        "response": {
                            "candidates": [
                                {
                                    "content": {
                                        "parts": [
                                            {
                                                "functionCall": {"id": "c1", "name": "read"},
                                                "thoughtSignature": "sig-abc",
                                            }
                                        ]
                                    },
                                    "finishReason": "STOP",
                                }
                            ]
                        }
                    }
                ]
            )
        )
        await plugin.dispatch(model="gemini-3-pro", messages=[{"role": "user", "content": "x"}])
        assert plugin._state.signatures["c1"] == "sig-abc"


class TestRefresh:
    async def test_refresh_rereads_the_store_instead_of_rotating(self) -> None:
        """The refresh token belongs to the store; here the source is only re-read."""
        store = FakeStore({"openai-codex": Credential(provider="openai-codex", access_token="new")})
        plugin.configure(store=store)

        token = await specs._refresh("codex")

        assert token == "new"
        assert store.reloads == 1


class TestTokenRenewal:
    """Automatic renewal. Before this, an expired token required pressing a button."""

    def _store(self, *, expired: bool, owns: bool = True, refresh_token: str = "RT") -> Any:
        class Store:
            owns_refresh = owns

            def __init__(self) -> None:
                self.credential = Credential(
                    provider="openai-codex",
                    access_token="AT-old",
                    refresh_token=refresh_token,
                    expires_at=time.time() + (-10 if expired else 3600),
                )
                self.written: list[str] = []

            def get(self, provider: str) -> Credential:
                return self.credential

            def set(self, provider: str, credential: Credential) -> None:
                self.credential = credential
                self.written.append(credential.access_token)

            def delete(self, provider: str) -> None: ...

            def reload(self) -> bool:
                self.reloads += 1
                return False

        store = Store()
        store.reloads = 0
        return store

    @pytest.fixture(autouse=True)
    def _fake_oauth(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def refresh(credential: Credential, *, client: Any, store: Any = None) -> Credential:
            return credential.with_access_token("AT-new", expires_at=time.time() + 3600)

        monkeypatch.setattr("litellm_mysubs.credentials.oauth.refresh", refresh)

    @pytest.mark.asyncio
    async def test_an_expired_token_is_renewed_before_the_request(self) -> None:
        """Renewing here, instead of waiting for the 401, avoids one round trip to the
        upstream per expiring token — and avoids a streaming request failing halfway, where
        it is no longer recoverable: the transport only replays what it has not delivered
        yet."""
        store = self._store(expired=True)
        plugin.configure(store=store)
        assert await plugin._access_token("codex") == "AT-new"

    @pytest.mark.asyncio
    async def test_the_renewed_credential_is_persisted(self) -> None:
        """Without writing it down, every request would spend a single-use refresh token —
        and the second would fail with `invalid_grant`."""
        store = self._store(expired=True)
        plugin.configure(store=store)
        await plugin._access_token("codex")
        assert store.written == ["AT-new"]

    @pytest.mark.asyncio
    async def test_a_valid_token_is_not_renewed(self) -> None:
        """Spending a rotation for no reason is the easiest way to break a session that was
        fine."""
        store = self._store(expired=False)
        plugin.configure(store=store)
        assert await plugin._access_token("codex") == "AT-old"
        assert store.written == []

    @pytest.mark.asyncio
    async def test_a_store_that_is_not_the_owner_never_rotates(self) -> None:
        """Two refreshers on rotating single-use tokens invalidate each other's copy and
        produce a loop of `invalid_grant`, forcing a manual re-login."""
        store = self._store(expired=True, owns=False)
        plugin.configure(store=store)
        assert await plugin._access_token("codex") == "AT-old"
        assert store.written == []

    @pytest.mark.asyncio
    async def test_without_a_refresh_token_there_is_nothing_to_rotate(self) -> None:
        store = self._store(expired=True, refresh_token="")
        plugin.configure(store=store)
        assert await plugin._access_token("codex") == "AT-old"
        assert store.written == []

    @pytest.mark.asyncio
    async def test_a_valid_token_does_not_even_consult_the_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Returning the right token is not enough: the provider must not be contacted at
        all.

        An unnecessary rotation spends a single-use refresh token and turns a good session
        into one that needs a re-login. The previous test passed even with `if True:` because
        `_refresh` re-reads the source before rotating — only counting the calls makes the
        difference visible.
        """
        calls: list[str] = []

        async def refresh(credential: Credential, *, client: Any, store: Any = None) -> Credential:
            calls.append(credential.access_token)
            return credential.with_access_token("AT-new", expires_at=time.time() + 3600)

        monkeypatch.setattr("litellm_mysubs.credentials.oauth.refresh", refresh)
        plugin.configure(store=self._store(expired=False))
        assert await plugin._access_token("codex") == "AT-old"
        assert calls == []

    @pytest.mark.asyncio
    async def test_a_valid_token_does_not_touch_the_store_source(self) -> None:
        """The source is not even re-read.

        `_refresh` starts with a `store.reload()` — a disk read or a call to the vault.
        Calling it with a valid token is one I/O per served request, on the hot path. The
        `is_expired` guard is what avoids it, and only counting the `reload` calls makes it
        visible: the returned result is the same with or without it.
        """
        store = self._store(expired=False)
        plugin.configure(store=store)
        await plugin._access_token("codex")
        assert store.reloads == 0


class TestProviderComesFromTheDeployment:
    """The provider comes from the deployment, not from the model name.

    Measured on the real catalogue of the Antigravity account: of the 32 models served,
    **seven** do not have "gemini" in the name. `claude-sonnet-4-6`,
    `claude-opus-4-6-thinking`, `chat_23310`, `chat_20706`, `tab_flash_lite_preview` and
    `tab_jump_flash_lite_preview` fell into LiteLLM's native path — which has no credential
    and blows up with `Illegal header value`. Worse: `gpt-oss-120b-medium` was dispatched to
    **Codex**, a different subscription and a different account.
    """

    def router_with(self, model: str, provider: str) -> Any:
        return SimpleNamespace(
            model_list=[
                {
                    "model_name": model,
                    "litellm_params": {"model": f"openai/{model}"},
                    "model_info": {"mysubs_provider": provider, "id": "x"},
                }
            ]
        )

    def test_a_claude_model_served_by_antigravity_is_not_treated_as_anthropic(self) -> None:
        from litellm_mysubs.plugin import provider_of_deployment

        router = self.router_with("claude-sonnet-4-6", "google-antigravity")
        assert provider_of_deployment(router, "claude-sonnet-4-6") == "google-antigravity"

    def test_a_gpt_model_served_by_antigravity_does_not_go_to_codex(self) -> None:
        """`gpt-oss-120b-medium` matches `is_codex_model` through the `gpt-` prefix. Without
        the deployment's marker, the request went out through the wrong subscription."""
        from litellm_mysubs.plugin import provider_of_deployment
        from litellm_mysubs.wire import codex

        assert codex.is_codex_model("gpt-oss-120b-medium"), "premise of the test"
        router = self.router_with("gpt-oss-120b-medium", "google-antigravity")
        assert provider_of_deployment(router, "gpt-oss-120b-medium") == "google-antigravity"

    def test_an_unnamed_model_is_resolved_too(self) -> None:
        """`chat_23310` and `tab_flash_lite_preview` have nothing in the name to identify
        them."""
        from litellm_mysubs.plugin import provider_of_deployment

        for model in ("chat_23310", "tab_flash_lite_preview"):
            router = self.router_with(model, "google-antigravity")
            assert provider_of_deployment(router, model) == "google-antigravity", model

    def test_a_model_that_is_not_ours_returns_none(self) -> None:
        """A model from `config.yaml` carries no marker: dispatch has to let it through."""
        from litellm_mysubs.plugin import provider_of_deployment

        router = SimpleNamespace(
            model_list=[{"model_name": "eco", "litellm_params": {"model": "openai/eco"},
                         "model_info": {"id": "eco"}}]
        )
        assert provider_of_deployment(router, "eco") is None

    def test_an_absent_router_does_not_break_dispatch(self) -> None:
        from litellm_mysubs.plugin import provider_of_deployment

        assert provider_of_deployment(None, "whatever-it-is") is None
        assert provider_of_deployment(SimpleNamespace(), "x") is None


class TestDispatchHonoursTheDeclaredProvider:
    """`dispatch` has to obey the deployment's provider, not merely know it.

    Without these tests, deleting the `declared = provider_of_deployment(...)` in the Router
    wrapper went unnoticed: the resolution stayed correct and nobody used it.
    """

    async def test_a_claude_name_declared_as_antigravity_goes_to_antigravity(self) -> None:
        import litellm_mysubs.plugin as plugin

        called: list[str] = []

        async def fake_antigravity(model, messages, extra):
            called.append("antigravity")
            return "answer"

        original = routes._antigravity_turn
        routes._antigravity_turn = fake_antigravity  # type: ignore[assignment]
        try:
            out = await plugin.dispatch(
                provider="google-antigravity", model="claude-sonnet-4-6", messages=[]
            )
        finally:
            routes._antigravity_turn = original  # type: ignore[assignment]
        assert out == "answer"
        assert called == ["antigravity"], "the name beat the declared provider"

    async def test_a_gpt_name_declared_as_antigravity_does_not_reach_codex(self) -> None:
        """This was the worst case: a request billed to the wrong subscription, with no
        visible error."""
        import litellm_mysubs.plugin as plugin

        called: list[str] = []

        async def fake_antigravity(model, messages, extra):
            called.append("antigravity")
            return "ok"

        async def fake_codex(model, messages, extra):
            called.append("codex")
            return "ok"

        a, c = routes._antigravity_turn, routes._codex_turn
        routes._antigravity_turn = fake_antigravity  # type: ignore[assignment]
        routes._codex_turn = fake_codex  # type: ignore[assignment]
        try:
            await plugin.dispatch(
                provider="google-antigravity", model="gpt-oss-120b-medium", messages=[]
            )
        finally:
            routes._antigravity_turn, routes._codex_turn = a, c  # type: ignore[assignment]
        assert called == ["antigravity"], f"it went to the wrong place: {called}"

    async def test_without_a_declared_provider_the_name_still_decides(self) -> None:
        """Whoever calls `litellm.acompletion` directly has neither Router nor deployment: the
        name heuristic is still all there is."""
        import litellm_mysubs.plugin as plugin

        called: list[str] = []

        async def fake_antigravity(model, messages, extra):
            called.append("antigravity")
            return "ok"

        original = routes._antigravity_turn
        routes._antigravity_turn = fake_antigravity  # type: ignore[assignment]
        try:
            await plugin.dispatch(model="gemini-3-flash", messages=[])
        finally:
            routes._antigravity_turn = original  # type: ignore[assignment]
        assert called == ["antigravity"]

    async def test_a_foreign_model_is_still_not_ours(self) -> None:
        import litellm_mysubs.plugin as plugin

        assert await plugin.dispatch(model="eco", messages=[]) is None

    async def test_the_anthropic_token_is_not_injected_into_another_provider(self) -> None:
        """`claude-sonnet-4-6` exists in both catalogues. Without a guard, the Anthropic
        subscription's token went out to a Google endpoint — one account's credential sent to
        another.

        With a real token in the store, otherwise the test passes through absence of a
        credential rather than because of the guard.
        """
        import litellm_mysubs.plugin as plugin

        async def token(_provider: str) -> str:
            return "sk-ant-secret"

        original = plugin._access_token
        plugin._access_token = token  # type: ignore[assignment]
        try:
            foreign = await plugin._delegate_kwargs(
                {"model": "claude-sonnet-4-6", "messages": []}, provider="google-antigravity"
            )
            ours = await plugin._delegate_kwargs(
                {"model": "claude-sonnet-4-6", "messages": []}, provider="anthropic"
            )
        finally:
            plugin._access_token = original  # type: ignore[assignment]

        assert foreign.get("api_key") != "sk-ant-secret", "the Anthropic token went elsewhere"
        assert ours.get("api_key") == "sk-ant-secret", "the legitimate injection stopped"


class TestStreamedCallsCarryACostableIdentity:
    """A streamed call has to be priceable, or the spend log records it as free.

    The regression these guard: `_wrap_stream` handed the wrapper the public model name
    and `custom_openai`, neither of which has a rate, so `response_cost_calculator`
    returned `0.0` for every streamed request. Measured on a live proxy: 74 of 89
    billable calls logged at zero.
    """

    @pytest.mark.parametrize(
        ("public", "wire", "expected"),
        [
            (
                "mysubs/claudecode/claude-opus-5",
                "anthropic/claude-opus-5",
                ("anthropic/claude-opus-5", "anthropic"),
            ),
            ("mysubs/codex/gpt-5.5", "openai/gpt-5.5", ("openai/gpt-5.5", "openai")),
            (
                "mysubs/antigravity/gemini-2.5-pro",
                "gemini/gemini-2.5-pro",
                ("gemini/gemini-2.5-pro", "gemini"),
            ),
        ],
    )
    def test_the_wire_pair_is_what_reaches_the_cost_calculation(
        self, public: str, wire: str, expected: tuple[str, str]
    ) -> None:
        identity = observability._cost_identity(public, {observability._WIRE_MODEL_KEY: wire})
        assert identity == expected

    def test_without_a_deployment_the_name_is_not_guessed(self) -> None:
        """A direct `litellm.acompletion` call has no Router to ask.

        Guessing a family from the name would price the call against another model's
        rate, which is worse than not pricing it.
        """
        assert observability._cost_identity("gpt-5.5", {}) == ("gpt-5.5", "custom_openai")

    @pytest.mark.parametrize(
        ("wire", "expected"),
        [
            # Effort is part of the Antigravity model id and absent from the price map;
            # the base name carries the rate. Measured on the live gateway: every Gemini
            # row on all six routes read spend=0.0 while usage was recorded correctly.
            ("gemini/gemini-3.6-flash-low", "gemini/gemini-3.6-flash"),
            ("gemini/gemini-3.8-flash-high", "gemini/gemini-3.8-flash"),
            ("gemini/gemini-3.6-flash-tiered", "gemini/gemini-3.6-flash"),
            # Longest-first matching: `-low` must not eat the tail of `-extra-low`.
            ("gemini/gemini-3.5-flash-extra-low", "gemini/gemini-3.5-flash"),
            # A name the map already knows is never trimmed.
            ("gemini/gemini-2.5-flash", "gemini/gemini-2.5-flash"),
            # Trimming only applies when the base actually has a rate. `-agent` is a
            # distinct deployment, not an effort of `gemini-3-flash`, and neither name is
            # priced — repricing it against a sibling would invent a number.
            ("gemini/gemini-3-flash-agent", "gemini/gemini-3-flash-agent"),
            ("gemini/gemini-pro-agent", "gemini/gemini-pro-agent"),
        ],
    )
    def test_an_effort_suffix_is_priced_against_the_base_model(
        self, wire: str, expected: str
    ) -> None:
        """Effort changes the thinking budget, not the per-token rate."""
        identity = observability._cost_identity(
            "mysubs/antigravity/x", {observability._WIRE_MODEL_KEY: wire}
        )
        assert identity == (expected, "gemini")

    def test_the_wrapper_is_built_with_the_wire_identity(self) -> None:
        """What the wrapper is given is what the cost calculation sees."""

        async def chunks() -> AsyncIterator[Any]:
            if False:  # pragma: no cover - an empty stream is enough here
                yield None

        wrapper = observability._wrap_stream(
            chunks(),
            "mysubs/codex/gpt-5.5",
            {observability._WIRE_MODEL_KEY: "openai/gpt-5.5", "messages": []},
        )
        # The wrapper keeps the prefix; what matters is that the pair is the priceable
        # one, not the public name with `custom_openai`.
        assert wrapper.model == "openai/gpt-5.5"
        assert wrapper.custom_llm_provider == "openai"


class TestTheWireModelIsReadOffOurOwnDeployments:
    def test_our_deployment_yields_its_wire_name(self) -> None:
        router = SimpleNamespace(
            model_list=[
                {
                    "model_name": "mysubs/codex/gpt-5.5",
                    "litellm_params": {"model": "openai/gpt-5.5"},
                    "model_info": {"mysubs_provider": "openai-codex"},
                }
            ]
        )
        assert plugin.wire_model_of_deployment(router, "mysubs/codex/gpt-5.5") == "openai/gpt-5.5"

    def test_a_deployment_without_our_mark_is_not_read(self) -> None:
        """Pricing a call this plugin never served would attribute someone else's cost."""
        router = SimpleNamespace(
            model_list=[
                {
                    "model_name": "gpt-4",
                    "litellm_params": {"model": "openai/gpt-4"},
                    "model_info": {},
                }
            ]
        )
        assert plugin.wire_model_of_deployment(router, "gpt-4") is None

    def test_an_unknown_name_yields_nothing(self) -> None:
        assert plugin.wire_model_of_deployment(SimpleNamespace(model_list=[]), "x") is None


class TestThePrivateKwargNeverReachesTheProvider:
    """`mysubs_wire_model` is private to the hop between the Router and the wrapper.

    An unknown kwarg reaching the provider client raises `unexpected keyword argument`
    and the whole request is lost — a worse failure than the missing cost it fixes.
    """

    @pytest.mark.asyncio
    async def test_the_delegated_call_does_not_receive_it(self, monkeypatch: Any) -> None:
        seen: dict[str, Any] = {}

        async def fake_original(**kwargs: Any) -> str:
            seen.update(kwargs)
            return "delegated"

        monkeypatch.setattr(plugin._state, "original_acompletion", fake_original)
        monkeypatch.setattr(plugin._state, "store", FakeStore())

        out = await plugin._wrapped_acompletion(
            model="some-other-model",
            messages=[{"role": "user", "content": "hi"}],
            **{observability._WIRE_MODEL_KEY: "openai/gpt-5.5"},
        )

        assert out == "delegated"
        assert observability._WIRE_MODEL_KEY not in seen


def codex_responses_events(
    *,
    text: str = "hello",
    status: str = "completed",
    usage: dict[str, Any] | None = None,
    carry_output: bool = False,
    chunks: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Events as the real endpoint sends them.

    `response.completed` closes the turn **without** repeating `output`: the items arrived
    in the `response.output_item.done` events during the stream. Measured against the live
    endpoint after a passthrough shipped `output: []` with non-zero `output_tokens`.

    `carry_output=True` covers the opposite case — an upstream that does include it, which
    must then be preserved rather than rebuilt.
    """
    response: dict[str, Any] = {
        "id": "resp_upstream_1",
        "created_at": 1700000000,
        "object": "response",
        "status": status,
        "model": "gpt-5.5",
        "usage": usage or {},
    }
    if carry_output:
        response["output"] = [
            {
                "type": "message",
                "id": "msg_from_upstream",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ]
    return [
        *(
            {"type": "response.output_text.delta", "delta": part}
            for part in (chunks if chunks is not None else [text])
        ),
        {
            "type": "response.completed" if status == "completed" else "response.incomplete",
            "response": response,
        },
    ]


class TestTheResponsesRouteIsServed:
    """`/v1/responses` reaches `Router.aresponses`, which `install()` cannot patch.

    The regression these guard: the plugin patched `Router.acompletion` only, so a Codex
    request on the Responses route bypassed dispatch entirely, reached LiteLLM's native
    OpenAI path with no `api_key` — the credential is OAuth and lives in the store — and
    came back `Incorrect API key provided: None`.
    """

    @pytest.mark.asyncio
    async def test_the_text_survives_when_the_terminal_event_omits_output(self) -> None:
        """The regression: `output: []` with non-zero `output_tokens`.

        `response.completed` does not repeat the items, so a straight passthrough returned
        a completed turn whose text had vanished. Measured on the live gateway: the chat
        route answered "4" while this one answered nothing.
        """
        install_transport(
            FakeTransport(
                codex_responses_events(
                    text="served", usage={"input_tokens": 19, "output_tokens": 17}
                )
            )
        )

        out = await plugin.dispatch_responses(
            provider="openai-codex", model="mysubs/codex/gpt-5.5", input="hi"
        )

        assert out is not None
        assert out.output, "a turn that billed output tokens must carry output items"
        assert out.output[0].content[0].text == "served"
        assert out.usage.output_tokens == 17
        # The public name wins over the wire name the upstream echoes: it is what the
        # caller asked for and what the spend log records.
        assert out.model == "mysubs/codex/gpt-5.5"
        assert out.id == "resp_upstream_1"

    @pytest.mark.asyncio
    async def test_an_upstream_that_sends_output_keeps_it(self) -> None:
        """Rebuilding is the fallback, not the rule: the upstream's own items win."""
        install_transport(FakeTransport(codex_responses_events(text="served", carry_output=True)))

        out = await plugin.dispatch_responses(
            provider="openai-codex", model="mysubs/codex/gpt-5.5", input="hi"
        )

        assert out is not None
        assert out.output[0].id == "msg_from_upstream"

    @pytest.mark.asyncio
    async def test_tool_calls_become_function_call_items(self) -> None:
        """A tool turn has to replay as `function_call` items, with the call id preserved."""
        install_transport(
            FakeTransport(
                [
                    {
                        "type": "response.output_item.added",
                        "item": {
                            "type": "function_call",
                            "id": "item_1",
                            "call_id": "call_abc",
                            "name": "get_weather",
                        },
                    },
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": "item_1",
                        "delta": '{"city":"Lisbon"}',
                    },
                    {
                        "type": "response.output_item.done",
                        "item": {
                            "type": "function_call",
                            "id": "item_1",
                            "call_id": "call_abc",
                            "name": "get_weather",
                            "arguments": '{"city":"Lisbon"}',
                        },
                    },
                    {
                        "type": "response.completed",
                        "response": {
                            "id": "resp_tools",
                            "created_at": 1700000000,
                            "status": "completed",
                            "usage": {},
                        },
                    },
                ]
            )
        )

        out = await plugin.dispatch_responses(
            provider="openai-codex", model="mysubs/codex/gpt-5.5", input="weather?"
        )

        assert out is not None
        call = out.output[0]
        assert call.type == "function_call"
        assert call.name == "get_weather"
        assert call.arguments == '{"city":"Lisbon"}'
        assert "call_abc" in call.call_id

    @pytest.mark.asyncio
    async def test_a_string_input_becomes_a_user_turn(self) -> None:
        """`/v1/responses` carries `input`, not `messages`."""
        transport = install_transport(FakeTransport(codex_responses_events()))

        await plugin.dispatch_responses(
            provider="openai-codex", model="mysubs/codex/gpt-5.5", input="what is 2+2"
        )

        body = transport.specs[0].body
        assert json.dumps(body["input"]).find("what is 2+2") != -1

    @pytest.mark.asyncio
    async def test_streaming_emits_responses_api_events(self) -> None:
        """Streaming returns Responses events, not chat chunks.

        Event order captured from this proxy's own native `/v1/responses` path
        (`qwen-agent-coder`, `stream: true`) rather than assumed.
        """
        install_transport(FakeTransport(codex_responses_events(text="served")))

        stream = await plugin.dispatch_responses(
            provider="openai-codex", model="mysubs/codex/gpt-5.5", input="hi", stream=True
        )

        assert stream is not None
        events = [e async for e in stream]
        kinds = [e.type for e in events]

        assert kinds[0] == "response.created"
        assert kinds[1] == "response.in_progress"
        assert "response.output_item.added" in kinds
        assert "response.content_part.added" in kinds
        assert "response.output_text.delta" in kinds
        assert kinds[-1] == "response.completed"

    @pytest.mark.asyncio
    async def test_every_streamed_event_serialises_to_json(self) -> None:
        """The proxy serialises a chunk with `.model_dump_json()`.

        A plain dict falls through to `str()` and reaches the client as a Python repr with
        single quotes, which no JSON parser accepts. Measured against a real proxy.
        """
        install_transport(FakeTransport(codex_responses_events(text="served")))

        stream = await plugin.dispatch_responses(
            provider="openai-codex", model="mysubs/codex/gpt-5.5", input="hi", stream=True
        )

        async for event in stream:
            assert hasattr(event, "model_dump_json"), f"{event!r} is not a typed event"
            json.loads(event.model_dump_json(exclude_none=True, exclude_unset=True))

    @pytest.mark.asyncio
    async def test_the_streamed_text_matches_the_terminal_payload(self) -> None:
        """The deltas and the final `output` have to agree."""
        install_transport(FakeTransport(codex_responses_events(text="served")))

        stream = await plugin.dispatch_responses(
            provider="openai-codex", model="mysubs/codex/gpt-5.5", input="hi", stream=True
        )
        events = [e async for e in stream]

        streamed = "".join(e.delta for e in events if e.type == "response.output_text.delta")
        final = next(e for e in events if e.type == "response.completed")
        text = next(
            part.text
            for item in final.response.output
            if getattr(item, "type", None) == "message"
            for part in item.content
        )

        assert streamed == "served"
        assert text == streamed

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("provider", "model", "events"),
        [
            (
                "openai-codex",
                "mysubs/codex/gpt-5.5",
                codex_responses_events(chunks=["a", "b", "c", "d"]),
            ),
            (
                "google-antigravity",
                "mysubs/antigravity/gemini-3.6-flash-low",
                gemini_events(chunks=["a", "b", "c", "d"]),
            ),
        ],
    )
    async def test_a_streamed_cell_emits_a_delta_per_upstream_chunk(
        self, provider: Any, model: str, events: list[dict[str, Any]]
    ) -> None:
        """Both subscriptions must stream at the same granularity on this route.

        Membership assertions — "a delta appeared somewhere" — pass with one delta and
        with a thousand, which is how a version that awaited the whole turn and emitted
        two events shipped as a stream. Measured on the live gateway before this was
        fixed, same route and prompt: Codex `events=7117 deltas=7107 ttft=30ms`,
        Antigravity `events=2 deltas=0 ttft=20911ms`. A client reading `text_deltas` got
        nothing from one of them until the turn had finished.

        The count is asserted against the upstream chunk count rather than a literal, so
        it pins the incremental property without pinning an event total that legitimately
        varies.
        """
        install_transport(FakeTransport(events))

        stream = await plugin.dispatch_responses(
            provider=provider, model=model, input="hi", stream=True
        )
        kinds = [e.type async for e in stream]

        deltas = [k for k in kinds if k == "response.output_text.delta"]
        assert len(deltas) == 4, f"one delta per upstream chunk, got {len(deltas)}"
        assert kinds[0] == "response.created"
        assert kinds[-1] == "response.completed"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("provider", "model"),
        [
            ("anthropic", "mysubs/claudecode/claude-opus-5"),
            (None, "some-other-model"),
        ],
    )
    async def test_everything_else_is_not_ours(self, provider: Any, model: str) -> None:
        """Anthropic is served by LiteLLM's native path with the token we inject.

        Antigravity is **not** in this list any more: delegating it left the turn priced
        by the native path, which costs from the response object — public name, no rate.
        It is served here now, translated with LiteLLM's own bridge.
        """
        install_transport(FakeTransport(codex_responses_events()))

        assert await plugin.dispatch_responses(provider=provider, model=model, input="hi") is None


class TestTheSpendLogRecordsAPriceableIdentity:
    """The Logs tab reads `custom_llm_provider` off the logging object, per request.

    A request this plugin serves never reaches the provider client that would fill those
    in, so without the stamp the row lands with the public name and no provider at all.
    Measured on a live gateway: rows served natively read `anthropic/claude-opus-5` +
    `anthropic`, rows served here read `mysubs/antigravity/...` with an empty provider —
    and an empty provider is what leaves the row without an icon.

    Declaring `custom_llm_provider` on the deployment fixes the Models tab only: that one
    is built from the Router, this one is built per request.
    """

    @staticmethod
    def _logging_obj(model: str) -> Any:
        return SimpleNamespace(
            model_call_details={"model": model, "custom_llm_provider": None}
        )

    def test_the_wire_pair_reaches_the_logging_object(self) -> None:
        public = "mysubs/antigravity/gemini-3-flash"
        log = self._logging_obj(public)

        observability._stamp_logging_identity(
            public,
            {
                "litellm_logging_obj": log,
                observability._WIRE_MODEL_KEY: "gemini/gemini-3-flash",
            },
        )

        assert log.model_call_details["model"] == "gemini/gemini-3-flash"
        assert log.model_call_details["custom_llm_provider"] == "gemini"

    def test_the_public_name_survives_as_the_model_group(self) -> None:
        """The client asked for the public name; the row still has to show it."""
        public = "mysubs/codex/gpt-5.5"
        log = self._logging_obj(public)

        observability._stamp_logging_identity(
            public,
            {"litellm_logging_obj": log, observability._WIRE_MODEL_KEY: "openai/gpt-5.5"},
        )

        assert log.model_call_details["model_group"] == public

    def test_without_a_deployment_nothing_is_invented(self) -> None:
        """A direct library call has no Router, so there is no wire name to read.

        Guessing a family would bill the call against another model's rate.
        """
        public = "mysubs/codex/gpt-5.5"
        log = self._logging_obj(public)

        observability._stamp_logging_identity(public, {"litellm_logging_obj": log})

        assert log.model_call_details["model"] == public
        assert log.model_call_details["custom_llm_provider"] is None

    def test_a_caller_without_a_logging_object_is_not_a_failure(self) -> None:
        observability._stamp_logging_identity("m", {observability._WIRE_MODEL_KEY: "openai/m"})


class TestTheNonStreamingPathIsLogged:
    """A non-streaming turn returns straight out of `dispatch`.

    So it never reaches the `@client` wrapper in `litellm.utils` that calls
    `async_success_handler`, and a request that never logs produces **no** spend row at
    all. Measured on a live gateway, same model two seconds apart: the streamed call was
    priced at `0.00121`, the non-streamed one left no row of any kind.

    The consequence is an accounting hole rather than a pricing bug — `x-litellm-key-spend`
    undercounts, and per-key budgets never see these calls.
    """

    class Recorder:
        def __init__(self, model: str) -> None:
            self.model_call_details: dict[str, Any] = {
                "model": model,
                "custom_llm_provider": None,
            }
            self.calls: list[dict[str, Any]] = []

        async def async_success_handler(
            self, result: Any = None, start_time: Any = None, end_time: Any = None
        ) -> None:
            payload = getattr(result, "response", None) or result
            self.calls.append(
                {
                    "model": self.model_call_details.get("model"),
                    "provider": self.model_call_details.get("custom_llm_provider"),
                    "usage": getattr(payload, "usage", None),
                    "cost": self.model_call_details.get("response_cost"),
                    "start": start_time,
                    "end": end_time,
                    "first_token": self.model_call_details.get("completion_start_time"),
                    "result": result,
                }
            )

    @pytest.mark.asyncio
    async def test_a_non_streamed_turn_dispatches_success_logging(self) -> None:
        install_transport(FakeTransport(codex_events(text="served")))
        log = self.Recorder("mysubs/codex/gpt-5.5")

        await plugin.dispatch(
            provider="openai-codex",
            model="mysubs/codex/gpt-5.5",
            messages=[{"role": "user", "content": "hi"}],
            litellm_logging_obj=log,
            **{observability._WIRE_MODEL_KEY: "openai/gpt-5.5"},
        )

        assert len(log.calls) == 1, "exactly one success event per turn"
        assert log.calls[0]["model"] == "openai/gpt-5.5"
        assert log.calls[0]["provider"] == "openai"
        assert log.calls[0]["usage"] is not None, "a row without usage cannot be priced"

    @pytest.mark.asyncio
    async def test_the_row_measures_how_long_the_turn_took(self) -> None:
        """A row whose two timestamps are the same instant reads as 0 ms.

        The first version marked `now` once, after the await, and passed it as both
        `start_time` and `end_time`. Measured on the live gateway's Logs tab, every row
        this plugin served read 0 ms while the natively-served `anthropic/claude-opus-5`
        ones read 3.9 s to 14 s — the duration was never measured, not merely small.
        """
        install_transport(FakeTransport(codex_events(text="served")))
        log = self.Recorder("mysubs/codex/gpt-5.5")

        await plugin.dispatch(
            provider="openai-codex",
            model="mysubs/codex/gpt-5.5",
            messages=[{"role": "user", "content": "hi"}],
            litellm_logging_obj=log,
            **{observability._WIRE_MODEL_KEY: "openai/gpt-5.5"},
        )

        call = log.calls[0]
        assert call["end"] > call["start"], "the turn took time; the row has to show it"

    @pytest.mark.asyncio
    async def test_the_row_carries_the_cost_of_the_turn(self) -> None:
        """A logged row with no cost is a row the UI shows at zero.

        `get_standard_logging_object_payload` reads the number from
        `kwargs["response_cost"]`, which the `@client` wrapper in `litellm.utils` fills in
        — and that wrapper is exactly what a served request never reaches. Measured on the
        live gateway's own Logs tab: 49 of 50 rows at zero, including
        `anthropic/claude-opus-5` turns of 123k tokens whose rate is in the map.

        The identity fix was necessary but not sufficient: the row names the right model
        and still bills nothing.
        """
        install_transport(
            FakeTransport(
                codex_events(
                    text="served",
                    usage={"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500},
                )
            )
        )
        log = self.Recorder("mysubs/codex/gpt-5.5")

        await plugin.dispatch(
            provider="openai-codex",
            model="mysubs/codex/gpt-5.5",
            messages=[{"role": "user", "content": "hi"}],
            litellm_logging_obj=log,
            **{observability._WIRE_MODEL_KEY: "openai/gpt-5.5"},
        )

        assert log.calls[0]["cost"], "the row has to carry what the turn cost"

    @pytest.mark.asyncio
    async def test_a_logging_failure_does_not_lose_the_response(self) -> None:
        """The subscription has already been charged; losing the answer is worse."""
        install_transport(FakeTransport(codex_events(text="served")))

        class Broken(self.Recorder):
            async def async_success_handler(self, **_: Any) -> None:
                raise RuntimeError("logging backend down")

        out = await plugin.dispatch(
            provider="openai-codex",
            model="mysubs/codex/gpt-5.5",
            messages=[{"role": "user", "content": "hi"}],
            litellm_logging_obj=Broken("mysubs/codex/gpt-5.5"),
            **{observability._WIRE_MODEL_KEY: "openai/gpt-5.5"},
        )

        assert out.choices[0].message.content == "served"

    @pytest.mark.asyncio
    async def test_a_caller_without_a_logging_object_still_gets_its_answer(self) -> None:
        install_transport(FakeTransport(codex_events(text="served")))

        out = await plugin.dispatch(
            provider="openai-codex",
            model="mysubs/codex/gpt-5.5",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert out.choices[0].message.content == "served"


class TestTheResponsesRouteIsLogged:
    """`/v1/responses` is a third path, and it had neither of the other two's logging.

    `dispatch` covers chat both ways — `_wrap_stream` hands the streamed turn to
    `CustomStreamWrapper`, which fires the handler at end of stream, and `_logged` covers
    the non-streamed one. `dispatch_responses` returned its turn bare.

    It is the path that matters most in practice: a client that discovers models through
    LiteLLM routes every OpenAI-backed model here. Measured on the live gateway, an omp run
    that exercised all five Codex models end to end left **no** Codex row at all, while
    Anthropic and Gemini — which go through chat — logged 33 and 50.
    """

    Recorder = TestTheNonStreamingPathIsLogged.Recorder

    @pytest.mark.asyncio
    async def test_a_gemini_turn_is_served_and_priced_on_this_route(self) -> None:
        """Delegating Gemini here left the turn unpriced, so it is served instead.

        The native path costs from the response object, which carries the public name and
        `cost: None`. Stamping the identity first was not enough: measured on the live
        gateway the provider stuck and the model name did not, and the row stayed at zero
        while the same model on chat and messages logged 0.0005265 for identical usage.

        Serving it through `_logged` is what prices it, and the Responses shape comes
        from LiteLLM's own `LiteLLMCompletionResponsesConfig` rather than hand-built
        items.
        """
        install_transport(
            FakeTransport(
                gemini_events(
                    text="served",
                    usage={"promptTokenCount": 1000, "candidatesTokenCount": 500},
                )
            )
        )
        log = self.Recorder("mysubs/antigravity/gemini-3.6-flash-low")

        answer = await plugin.dispatch_responses(
            provider="google-antigravity",
            model="mysubs/antigravity/gemini-3.6-flash-low",
            input="hi",
            litellm_logging_obj=log,
            **{observability._WIRE_MODEL_KEY: "gemini/gemini-3.6-flash-low"},
        )

        assert answer is not None, "the turn is served here now, not delegated"
        assert len(log.calls) == 1, "a turn the subscription paid for has to leave a row"
        assert log.model_call_details["model"] == "gemini/gemini-3.6-flash"
        assert log.model_call_details["custom_llm_provider"] == "gemini"
        assert log.calls[0]["cost"], "the name in the row is only useful if it prices"

    @pytest.mark.asyncio
    async def test_a_streamed_gemini_turn_leaves_exactly_one_row(self) -> None:
        """One turn, one row — the cell that had no test at all.

        The first version of this path served the turn through `_logged` and then replayed
        it through a second logging wrapper. Swapping the non-logging wrapper for the
        logging one billed the subscription twice and the whole suite stayed green, which
        is the most expensive defect reachable here.
        """
        install_transport(
            FakeTransport(
                gemini_events(
                    chunks=["a", "b", "c"],
                    usage={"promptTokenCount": 1000, "candidatesTokenCount": 500},
                )
            )
        )
        log = self.Recorder("mysubs/antigravity/gemini-3.6-flash-low")

        stream = await plugin.dispatch_responses(
            provider="google-antigravity",
            model="mysubs/antigravity/gemini-3.6-flash-low",
            input="hi",
            stream=True,
            litellm_logging_obj=log,
            **{observability._WIRE_MODEL_KEY: "gemini/gemini-3.6-flash-low"},
        )
        async for _ in stream:
            pass

        assert len(log.calls) == 1, "billed once per turn, never twice"
        assert log.calls[0]["cost"], "a streamed row still has to carry its cost"

    @pytest.mark.asyncio
    async def test_a_non_streamed_responses_turn_is_logged(self) -> None:
        install_transport(FakeTransport(codex_responses_events(text="served")))
        log = self.Recorder("mysubs/codex/gpt-5.5")

        await plugin.dispatch_responses(
            provider="openai-codex",
            model="mysubs/codex/gpt-5.5",
            input="hi",
            litellm_logging_obj=log,
            **{observability._WIRE_MODEL_KEY: "openai/gpt-5.5"},
        )

        assert len(log.calls) == 1, "exactly one success event per turn"
        assert log.calls[0]["model"] == "openai/gpt-5.5"
        assert log.calls[0]["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_a_streamed_responses_turn_is_logged_once_at_the_end(self) -> None:
        """The handler fires after the last event, not per event.

        This is the shape omp actually sends: `stream: true` on the Responses route.
        """
        install_transport(FakeTransport(codex_responses_events(text="served")))
        log = self.Recorder("mysubs/codex/gpt-5.5")

        stream = await plugin.dispatch_responses(
            provider="openai-codex",
            model="mysubs/codex/gpt-5.5",
            input="hi",
            stream=True,
            litellm_logging_obj=log,
            **{observability._WIRE_MODEL_KEY: "openai/gpt-5.5"},
        )
        events = [e async for e in stream]

        assert events, "the stream still delivers its events"
        assert len(log.calls) == 1, "one row per stream, fired at the end"
        assert log.calls[0]["model"] == "openai/gpt-5.5"

    @pytest.mark.asyncio
    async def test_a_logging_failure_does_not_lose_the_stream(self) -> None:
        """The subscription has already been charged; losing the answer is worse."""
        install_transport(FakeTransport(codex_responses_events(text="served")))

        class Broken(self.Recorder):
            async def async_success_handler(self, **_: Any) -> None:
                raise RuntimeError("logging backend down")

        stream = await plugin.dispatch_responses(
            provider="openai-codex",
            model="mysubs/codex/gpt-5.5",
            input="hi",
            stream=True,
            litellm_logging_obj=Broken("mysubs/codex/gpt-5.5"),
        )
        events = [e async for e in stream]

        assert events[-1].type == "response.completed"

    @pytest.mark.asyncio
    async def test_a_streamed_row_records_when_the_first_token_left(self) -> None:
        """Time-to-first-token is only measurable while the stream runs.

        By the time it ends the moment has passed, and the upstream does not report it —
        so a row without it shows no TTFT at all, which is what the Logs tab displayed.

        It must also measure the **model**, not the envelope. `response.created` is
        emitted the instant the stream opens, before the upstream has said anything;
        timing it produced `ttft=1ms` on a 114-second Codex turn on the live gateway,
        next to a Gemini turn reporting an honest 736ms on the same route.
        """
        install_transport(FakeTransport(codex_responses_events(text="served"), delay=0.05))
        log = self.Recorder("mysubs/codex/gpt-5.5")

        stream = await plugin.dispatch_responses(
            provider="openai-codex",
            model="mysubs/codex/gpt-5.5",
            input="hi",
            stream=True,
            litellm_logging_obj=log,
            **{observability._WIRE_MODEL_KEY: "openai/gpt-5.5"},
        )
        async for _ in stream:
            pass

        call = log.calls[0]
        assert call["first_token"] is not None, "the row has to know when output began"
        assert call["start"] <= call["first_token"] <= call["end"]
        assert call["end"] > call["start"], "a streamed turn takes time"
        assert (call["first_token"] - call["start"]).total_seconds() >= 0.05, (
            "TTFT has to time the model, not the envelope event we emit ourselves"
        )

    @pytest.mark.asyncio
    async def test_a_consumer_that_stops_at_the_terminal_event_still_gets_a_row(
        self,
    ) -> None:
        """The proxy stops reading at `response.completed`; it does not drain the stream.

        Logging used to live after the `async for`, so closing the generator raised
        `GeneratorExit` at the `yield` and that code never ran. Every other test in this
        class drains to exhaustion, which is why the suite stayed green while three
        streamed `/v1/responses` calls on the live gateway left no spend row at all and the
        non-streamed one on the same model logged normally.
        """
        install_transport(FakeTransport(codex_responses_events(text="served")))
        log = self.Recorder("mysubs/codex/gpt-5.5")

        stream = await plugin.dispatch_responses(
            provider="openai-codex",
            model="mysubs/codex/gpt-5.5",
            input="hi",
            stream=True,
            litellm_logging_obj=log,
            **{observability._WIRE_MODEL_KEY: "openai/gpt-5.5"},
        )
        async for event in stream:
            if getattr(event, "type", None) == "response.completed":
                break
        await stream.aclose()

        assert len(log.calls) == 1, "a turn the subscription paid for has to leave a row"
        assert log.calls[0]["usage"] is not None, "a row without usage cannot be priced"

    @pytest.mark.asyncio
    async def test_the_router_recognises_the_stream_as_a_responses_iterator(self) -> None:
        """`Router._aresponses_with_streaming_fallbacks` gates on the type::

            if kwargs.get("stream") and isinstance(response, BaseResponsesAPIStreamingIterator):
                return await self._aresponses_streaming_iterator(...)
            return response

        Anything else is handed back raw and never reaches the path that logs the turn.
        Measured on the live gateway while this object was a plain async iterator: the
        streamed turn answered correctly, logged nothing, and emitted no diagnostic of its
        own because the code that would have logged was never reached.
        """
        from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator

        install_transport(FakeTransport(codex_responses_events(text="served")))

        stream = await plugin.dispatch_responses(
            provider="openai-codex",
            model="mysubs/codex/gpt-5.5",
            input="hi",
            stream=True,
            litellm_logging_obj=self.Recorder("mysubs/codex/gpt-5.5"),
            **{observability._WIRE_MODEL_KEY: "openai/gpt-5.5"},
        )

        assert isinstance(stream, BaseResponsesAPIStreamingIterator), (
            "the Router hands anything else back raw, unlogged"
        )

    @pytest.mark.asyncio
    async def test_the_stream_mirrors_every_attribute_the_base_sets(self) -> None:
        """Subclassing for the `isinstance` gate means inheriting the base's methods.

        `super().__init__` cannot run — it wants an `httpx.Response` this object does not
        have — so every attribute it would set is mirrored by hand. Miss one and the
        inherited reader raises mid-stream: `_check_max_streaming_duration` reads
        `_stream_created_time` on every `__anext__`, and the proxy's
        `get_hidden_params_dict` reads `_hidden_params` to build the response headers.

        The expected names come from the base constructor's **AST**, not from a regex over
        its source. `self.x = ...` and `self.x: T = ...` are different nodes and the
        obvious pattern only matches the first, so a regex silently under-reported by
        seven names — including both readers above — and the assertion passed while the
        object had holes.
        """
        import ast
        import inspect
        import textwrap

        from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator

        tree = ast.parse(
            textwrap.dedent(inspect.getsource(BaseResponsesAPIStreamingIterator.__init__))
        )
        expected = {
            target.attr
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            if isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        }

        install_transport(FakeTransport(codex_responses_events(text="served")))
        stream = await plugin.dispatch_responses(
            provider="openai-codex",
            model="mysubs/codex/gpt-5.5",
            input="hi",
            stream=True,
            litellm_logging_obj=self.Recorder("mysubs/codex/gpt-5.5"),
            **{observability._WIRE_MODEL_KEY: "openai/gpt-5.5"},
        )

        missing = sorted(name for name in expected if not hasattr(stream, name))
        assert not missing, f"inherited methods read these: {missing}"
        # Not just present — usable. This is the reader that runs on every `__anext__`.
        stream._check_max_streaming_duration()


    @pytest.mark.asyncio
    async def test_the_handler_is_given_the_terminal_event_not_the_response(self) -> None:
        """`Logging._get_assembled_streaming_response` keys off the event type::

            elif isinstance(result, (ResponseCompletedEvent, ResponseIncompleteEvent,
                                     ResponseFailedEvent)):
                return result.response
            else:
                return None

        Handing it `result.response` takes the `else`, returns `None`, and writes no row.
        Measured on the live gateway: the plugin logged `responses stream finished,
        terminal=True` with no exception, and the row still did not exist.

        The proxy wants the opposite shape on `completed_response`, so both are kept.
        """
        from litellm.types.llms.openai import ResponseCompletedEvent

        install_transport(FakeTransport(codex_responses_events(text="served")))
        log = self.Recorder("mysubs/codex/gpt-5.5")

        stream = await plugin.dispatch_responses(
            provider="openai-codex",
            model="mysubs/codex/gpt-5.5",
            input="hi",
            stream=True,
            litellm_logging_obj=log,
            **{observability._WIRE_MODEL_KEY: "openai/gpt-5.5"},
        )
        async for _ in stream:
            pass

        assert isinstance(log.calls[0]["result"], ResponseCompletedEvent), (
            "the handler drops a result that is not the terminal event"
        )
        assert not isinstance(stream.completed_response, ResponseCompletedEvent), (
            "the proxy wants the response, not the event"
        )

    @pytest.mark.asyncio
    async def test_the_stream_carries_the_finished_turn_on_itself(self) -> None:
        """The proxy reads the turn off the object, not off anything the stream yields.

        `_extract_completed_responses_response` does `attribute_of(stream_response,
        "completed_response")`, so a bare `async_generator` — which streams perfectly well
        — hands it nothing. Measured on the live gateway with the generator this replaces:

            Container ownership recording skipped on streaming /v1/responses:
            no completed_response on stream iterator async_generator

        Same minute, same model, same key: the streamed chat turn logged dur=3958
        ttft=3794 and the streamed Responses turn left no row at all.
        """
        install_transport(FakeTransport(codex_responses_events(text="served")))
        log = self.Recorder("mysubs/codex/gpt-5.5")

        stream = await plugin.dispatch_responses(
            provider="openai-codex",
            model="mysubs/codex/gpt-5.5",
            input="hi",
            stream=True,
            litellm_logging_obj=log,
            **{observability._WIRE_MODEL_KEY: "openai/gpt-5.5"},
        )
        assert stream.completed_response is None, "nothing has finished before iteration"

        async for event in stream:
            if getattr(event, "type", None) == "response.completed":
                break

        assert stream.completed_response is not None, (
            "the proxy reads the finished turn off this attribute"
        )
        assert getattr(stream.completed_response, "usage", None) is not None


class TestTheResponsesRouteIsBoundPerRouter:
    """`Router.aresponses` is built per instance, so there is no class attribute to patch.

    `Router.__init__` does `self.aresponses = self.factory_function(litellm.aresponses, ...)`,
    capturing the module function by value. Measured: patching the class is a no-op, and
    patching `litellm.aresponses` after the Router exists is too late — which is always,
    because the proxy builds the Router before loading this package.
    """

    @staticmethod
    def _router() -> Any:
        calls: list[dict[str, Any]] = []

        async def original(**kwargs: Any) -> str:
            calls.append(kwargs)
            return "original"

        router = SimpleNamespace(
            aresponses=original,
            model_list=[
                {
                    "model_name": "mysubs/codex/gpt-5.5",
                    "litellm_params": {"model": "openai/gpt-5.5"},
                    "model_info": {"mysubs_provider": "openai-codex"},
                }
            ],
        )
        return router, original, calls

    @pytest.mark.asyncio
    async def test_ours_is_served_and_the_original_is_not_called(self) -> None:
        install_transport(FakeTransport(codex_responses_events(text="bound")))
        router, _original, calls = self._router()

        assert plugin.bind_responses_route(router) is True
        out = await router.aresponses(model="mysubs/codex/gpt-5.5", input="hi")

        assert out.output[0].content[0].text == "bound"
        assert calls == [], "the original must not be reached for one of our models"

        plugin.unbind_responses_route()

    @pytest.mark.asyncio
    async def test_someone_elses_model_reaches_the_original(self) -> None:
        install_transport(FakeTransport(codex_responses_events()))
        router, _original, calls = self._router()
        plugin.bind_responses_route(router)

        out = await router.aresponses(model="gpt-4o", input="hi")

        assert out == "original"
        assert calls and calls[0]["model"] == "gpt-4o"

        plugin.unbind_responses_route()

    def test_binding_twice_does_not_chain_wrappers(self) -> None:
        """The second bind would save our own wrapper as the original to restore."""
        router, original, _calls = self._router()

        assert plugin.bind_responses_route(router) is True
        assert plugin.bind_responses_route(router) is False

        plugin.unbind_responses_route()
        assert router.aresponses is original

    def test_unbind_restores_the_attribute(self) -> None:
        router, original, _calls = self._router()
        plugin.bind_responses_route(router)
        assert router.aresponses is not original

        plugin.unbind_responses_route()

        assert router.aresponses is original

    def test_uninstall_unbinds(self) -> None:
        """`uninstall` has to leave no loose ends, the Responses route included."""
        router, original, _calls = self._router()
        plugin.install()
        plugin.bind_responses_route(router)

        plugin.uninstall()

        assert router.aresponses is original

    def test_a_router_without_the_attribute_is_skipped(self) -> None:
        assert plugin.bind_responses_route(SimpleNamespace()) is False
        assert plugin.bind_responses_route(None) is False


class TestTheMessagesRouteIsBoundPerRouter:
    """`/v1/messages` is the third dialect, and a client should not have to adapt.

    Anthropic-native clients speak Messages, and a proxy that only answers
    chat-completions and Responses forces each of them to change. Measured on the live
    gateway before this existed::

        /v1/messages  mysubs/claudecode/*  401 Missing Anthropic API Key
        /v1/messages  mysubs/codex/*       401 AuthenticationError

    The 401 is the same failure mode the Responses route had before 0.1.3: without
    interception the request reaches LiteLLM's native client with no key, because the
    credential is OAuth and lives in the store, not in `config.yaml`.

    `Router.aanthropic_messages` is built per instance exactly like `aresponses`
    (`self.aanthropic_messages = self.factory_function(litellm.anthropic_messages, ...)`),
    so the same binding strategy applies.
    """

    @staticmethod
    def _router() -> Any:
        calls: list[dict[str, Any]] = []

        async def original(**kwargs: Any) -> str:
            calls.append(kwargs)
            return "original"

        router = SimpleNamespace(
            aanthropic_messages=original,
            anthropic_messages=original,
            model_list=[
                {
                    "model_name": "mysubs/claudecode/claude-opus-5",
                    "litellm_params": {"model": "anthropic/claude-opus-5"},
                    "model_info": {"mysubs_provider": "anthropic"},
                },
                {
                    "model_name": "mysubs/codex/gpt-5.5",
                    "litellm_params": {"model": "openai/gpt-5.5"},
                    "model_info": {"mysubs_provider": "openai-codex"},
                },
            ],
        )
        return router, original, calls

    def test_binding_replaces_both_spellings(self) -> None:
        """The Router exposes the call under two names, and the proxy may use either."""
        router, original, _calls = self._router()

        assert plugin.bind_messages_route(router) is True
        assert router.aanthropic_messages is not original
        assert router.anthropic_messages is not original

        plugin.unbind_messages_route()
        assert router.aanthropic_messages is original
        assert router.anthropic_messages is original

    @pytest.mark.asyncio
    async def test_someone_elses_model_reaches_the_original(self) -> None:
        router, _original, calls = self._router()
        plugin.bind_messages_route(router)

        out = await router.aanthropic_messages(
            model="claude-3-5-sonnet", messages=[{"role": "user", "content": "hi"}]
        )

        assert out == "original"
        assert calls and calls[0]["model"] == "claude-3-5-sonnet"

        plugin.unbind_messages_route()

    def test_binding_twice_does_not_chain_wrappers(self) -> None:
        router, original, _calls = self._router()

        assert plugin.bind_messages_route(router) is True
        assert plugin.bind_messages_route(router) is False

        plugin.unbind_messages_route()
        assert router.aanthropic_messages is original

    def test_uninstall_unbinds(self) -> None:
        router, original, _calls = self._router()
        plugin.install()
        plugin.bind_messages_route(router)

        plugin.uninstall()

        assert router.aanthropic_messages is original

    def test_a_router_without_the_attribute_is_skipped(self) -> None:
        assert plugin.bind_messages_route(SimpleNamespace()) is False
        assert plugin.bind_messages_route(None) is False


class TestEveryDialectReachesEverySubscription:
    """The matrix a client should never have to think about.

    Three wire dialects, three subscriptions. A client that speaks Messages and one that
    speaks chat-completions must both reach the same model without adapting, which is the
    whole reason a proxy sits here. Measured on the live gateway before `/v1/messages` was
    served, two of the three columns were 401s.

    Claude Max is the deliberate gap in `dispatch_messages`: Messages *is* its wire, so it
    returns `None` and LiteLLM's native path answers it with the token `_delegate_kwargs`
    injects — the same routing `dispatch` uses for chat.
    """

    @pytest.mark.asyncio
    async def test_messages_reaches_codex(self) -> None:
        transport = install_transport(FakeTransport(codex_events(text="served")))

        out = await plugin.dispatch_messages(
            provider="openai-codex",
            model="mysubs/codex/gpt-5.5",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert transport.specs, "the request has to reach the Codex wire"
        assert out["type"] == "message"
        assert out["content"][0]["text"] == "served"
        assert out["stop_reason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_messages_reaches_antigravity(self) -> None:
        install_transport(FakeTransport(gemini_events(text="served")))

        out = await plugin.dispatch_messages(
            provider="google-antigravity",
            model="mysubs/antigravity/gemini-3-pro",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert out["content"][0]["text"] == "served"

    @pytest.mark.asyncio
    async def test_claude_max_is_left_to_the_native_path(self) -> None:
        """Messages is Anthropic's own wire; translating it would only lose fidelity."""
        assert (
            await plugin.dispatch_messages(
                provider="anthropic",
                model="mysubs/claudecode/claude-opus-5",
                messages=[{"role": "user", "content": "hi"}],
            )
            is None
        )

    @pytest.mark.asyncio
    async def test_claude_max_still_gets_its_token(self) -> None:
        """Declining to translate is not the same as needing no credential.

        `dispatch_messages` returns `None` for Claude Max because Messages is already its
        wire, and the first version stopped there — the native client then answered
        `Missing Anthropic API Key`, measured on the live gateway. The token has to be
        injected before delegating, exactly as the chat route does.
        """
        seen: list[dict[str, Any]] = []

        async def original(**kwargs: Any) -> str:
            seen.append(kwargs)
            return "native"

        install_transport(FakeTransport())
        router = SimpleNamespace(
            aanthropic_messages=original,
            anthropic_messages=original,
            model_list=[
                {
                    "model_name": "mysubs/claudecode/claude-opus-5",
                    "litellm_params": {"model": "anthropic/claude-opus-5"},
                    "model_info": {"mysubs_provider": "anthropic"},
                }
            ],
        )
        plugin.bind_messages_route(router)
        try:
            out = await router.aanthropic_messages(
                model="mysubs/claudecode/claude-opus-5",
                messages=[{"role": "user", "content": "hi"}],
            )
        finally:
            plugin.unbind_messages_route()

        assert out == "native", "the native path still answers it"
        assert seen and seen[0].get("api_key"), "without a key the upstream returns 401"
        # `/v1/messages` has its own top-level `system`, and the upstream rejects the
        # identity as `messages[0]` outright:
        #   400 messages.0: use the top-level 'system' parameter for the initial system
        #       prompt
        assert seen[0].get("system"), "the identity has to ride in the native field"
        assert all(
            message.get("role") != "system" for message in seen[0].get("messages") or []
        ), "no system message may survive in the turn list"

    @pytest.mark.asyncio
    async def test_the_system_prompt_is_not_lost(self) -> None:
        """Messages carries `system` at the top level, the canonical form as a message.

        Dropping it silently would change the answer rather than fail the request, which is
        the worse failure.
        """
        transport = install_transport(FakeTransport(codex_events()))

        await plugin.dispatch_messages(
            provider="openai-codex",
            model="mysubs/codex/gpt-5.5",
            system="be terse",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert "be terse" in json.dumps(transport.specs[0].body)

    @pytest.mark.asyncio
    async def test_a_tool_roundtrip_keeps_its_ids(self) -> None:
        """`tool_use.id` has to survive: Vertex rejects a tool result that cannot name its
        call, which is the 0.1.5 fix this route must not undo."""
        transport = install_transport(FakeTransport(codex_events()))

        await plugin.dispatch_messages(
            provider="openai-codex",
            model="mysubs/codex/gpt-5.5",
            messages=[
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "toolu_1", "name": "get", "input": {"a": 1}}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "21C"}
                    ],
                },
            ],
            tools=[{"name": "get", "description": "d", "input_schema": {"type": "object"}}],
        )

        body = json.dumps(transport.specs[0].body)
        assert "toolu_1" in body, "the call id has to reach the wire"
        assert "21C" in body, "the result has to reach the wire"

    @pytest.mark.asyncio
    async def test_streaming_replays_the_anthropic_event_sequence(self) -> None:
        """A Messages client tracks block indices, so they have to be contiguous."""
        install_transport(FakeTransport(codex_events(text="served")))

        stream = await plugin.dispatch_messages(
            provider="openai-codex",
            model="mysubs/codex/gpt-5.5",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        kinds = [event["type"] async for event in stream]

        assert kinds[0] == "message_start"
        assert kinds[-1] == "message_stop"
        assert "content_block_start" in kinds
        assert "message_delta" in kinds


class TestEveryRouteRecordsAPriceableIdentity:
    """The wire pair has to reach the spend log on all three routes, not just chat.

    `_stamp_logging_identity` rewrites the identity from `_cost_identity`, which reads
    `_WIRE_MODEL_KEY` — and only the chat wrapper was injecting it. Measured on the live
    gateway, the same five Codex models appear twice under different names:

        openai/gpt-5.5          prov=openai    <- chat
        mysubs/codex/gpt-5.5    prov=          <- responses, messages

    An empty provider is what leaves the row without an icon, and the public name has no
    rate in the price map, so those rows also priced at zero. Same defect the chat route
    already fixed, on the two routes that did not inherit it.
    """

    @staticmethod
    def _router(attr: str) -> Any:
        async def original(**kwargs: Any) -> str:
            return "original"

        return SimpleNamespace(
            **{attr: original},
            model_list=[
                {
                    "model_name": "mysubs/codex/gpt-5.5",
                    "litellm_params": {"model": "openai/gpt-5.5"},
                    "model_info": {"mysubs_provider": "openai-codex"},
                }
            ],
        )

    class Recorder:
        def __init__(self) -> None:
            self.model_call_details: dict[str, Any] = {
                "model": "mysubs/codex/gpt-5.5",
                "custom_llm_provider": None,
            }

        async def async_success_handler(self, **_: Any) -> None:
            return None

    @pytest.mark.asyncio
    async def test_the_responses_route_stamps_the_wire_pair(self) -> None:
        install_transport(FakeTransport(codex_responses_events()))
        router = self._router("aresponses")
        log = self.Recorder()
        plugin.bind_responses_route(router)
        try:
            await router.aresponses(
                model="mysubs/codex/gpt-5.5", input="hi", litellm_logging_obj=log
            )
        finally:
            plugin.unbind_responses_route()

        assert log.model_call_details["model"] == "openai/gpt-5.5"
        assert log.model_call_details["custom_llm_provider"] == "openai"

    @pytest.mark.asyncio
    async def test_the_messages_route_stamps_the_wire_pair(self) -> None:
        install_transport(FakeTransport(codex_events()))
        router = self._router("aanthropic_messages")
        log = self.Recorder()
        plugin.bind_messages_route(router)
        try:
            await router.aanthropic_messages(
                model="mysubs/codex/gpt-5.5",
                messages=[{"role": "user", "content": "hi"}],
                litellm_logging_obj=log,
            )
        finally:
            plugin.unbind_messages_route()

        assert log.model_call_details["model"] == "openai/gpt-5.5"
        assert log.model_call_details["custom_llm_provider"] == "openai"
