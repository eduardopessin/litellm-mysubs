"""Entry point: patching, normalisation, dispatch and the shape of the responses.

No network: the transport is a double that returns already-decoded events, which is exactly
what the real `Transport` delivers (`AsyncIterator[dict]`).
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Iterable
from types import SimpleNamespace
from typing import Any

import litellm
import litellm.main
import pytest

from litellm_mysubs import plugin
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
    ) -> None:
        self.events = list(events)
        self.error = error
        self.specs: list[RequestSpec] = []

    async def request(self, spec: RequestSpec) -> Response:  # pragma: no cover - unused
        self.specs.append(spec)
        raise AssertionError("dispatch always uses stream()")

    async def stream(self, spec: RequestSpec) -> AsyncIterator[dict[str, Any]]:
        self.specs.append(spec)
        if self.error is not None:
            raise self.error
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
    *, text: str = "hello", finish: str = "STOP", usage: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    return [
        {
            "response": {
                "candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": finish}],
                "usageMetadata": usage or {},
            }
        }
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

        token = await plugin._refresh("codex")

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

        original = plugin._antigravity_turn
        plugin._antigravity_turn = fake_antigravity  # type: ignore[assignment]
        try:
            out = await plugin.dispatch(
                provider="google-antigravity", model="claude-sonnet-4-6", messages=[]
            )
        finally:
            plugin._antigravity_turn = original  # type: ignore[assignment]
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

        a, c = plugin._antigravity_turn, plugin._codex_turn
        plugin._antigravity_turn = fake_antigravity  # type: ignore[assignment]
        plugin._codex_turn = fake_codex  # type: ignore[assignment]
        try:
            await plugin.dispatch(
                provider="google-antigravity", model="gpt-oss-120b-medium", messages=[]
            )
        finally:
            plugin._antigravity_turn, plugin._codex_turn = a, c  # type: ignore[assignment]
        assert called == ["antigravity"], f"it went to the wrong place: {called}"

    async def test_without_a_declared_provider_the_name_still_decides(self) -> None:
        """Whoever calls `litellm.acompletion` directly has neither Router nor deployment: the
        name heuristic is still all there is."""
        import litellm_mysubs.plugin as plugin

        called: list[str] = []

        async def fake_antigravity(model, messages, extra):
            called.append("antigravity")
            return "ok"

        original = plugin._antigravity_turn
        plugin._antigravity_turn = fake_antigravity  # type: ignore[assignment]
        try:
            await plugin.dispatch(model="gemini-3-flash", messages=[])
        finally:
            plugin._antigravity_turn = original  # type: ignore[assignment]
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
