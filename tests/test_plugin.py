"""Ponto de entrada: patch, normalização, despacho e forma das respostas.

Sem rede: o transporte é um duplo que devolve eventos já decodificados, que é exactamente
o que o `Transport` real entrega (`AsyncIterator[dict]`, contrato em `local/CONTRACT.md`).
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Iterable
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
    """Store em memória; nunca toca no disco nem no ambiente."""

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
    """Duplo do `Transport`: devolve eventos guardados e regista os specs recebidos."""

    def __init__(
        self,
        events: Iterable[dict[str, Any]] = (),
        *,
        error: Exception | None = None,
    ) -> None:
        self.events = list(events)
        self.error = error
        self.specs: list[RequestSpec] = []

    async def request(self, spec: RequestSpec) -> Response:  # pragma: no cover - não usado
        self.specs.append(spec)
        raise AssertionError("o despacho usa sempre stream()")

    async def stream(self, spec: RequestSpec) -> AsyncIterator[dict[str, Any]]:
        self.specs.append(spec)
        if self.error is not None:
            raise self.error
        for event in self.events:
            yield event


def codex_events(
    *, text: str = "olá", status: str = "completed", usage: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    return [
        {"type": "response.output_text.delta", "delta": text},
        {
            "type": "response.completed" if status == "completed" else "response.incomplete",
            "response": {"status": status, "usage": usage or {}},
        },
    ]


def gemini_events(
    *, text: str = "olá", finish: str = "STOP", usage: dict[str, Any] | None = None
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
    """Cada teste começa sem patch e com dependências neutras."""
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
        """Dois wrappers encadeados fariam cada pedido passar duas vezes pelo despacho e
        `uninstall` deixaria o patch meio posto."""
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
        """`litellm.acompletion` é a forma documentada de chamar a biblioteca.

        `litellm/__init__.py` faz `from .main import acompletion`, o que copia a
        referência. Patchar só `litellm.main` deixava `litellm.acompletion` a apontar para
        o original, e um pedido por essa via chegava ao caminho nativo com um nome que
        nenhum provedor conhece — `BadRequestError: LLM Provider NOT provided`.

        Os testes só olhavam para `litellm.main`, por isso passavam com o patch meio posto.
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
    """Um modelo que não é nosso cai no original — com o prompt Claude aplicado."""

    async def test_foreign_model_reaches_original_with_claude_prompt(self) -> None:
        seen: dict[str, Any] = {}

        async def fake_original(**kwargs: Any) -> str:
            seen.update(kwargs)
            return "do-original"

        install_transport(FakeTransport())
        plugin.install()
        plugin._state.original_acompletion = fake_original

        result = await litellm.main.acompletion(
            model="claude-sonnet-4-6", messages=[{"role": "user", "content": "oi"}]
        )

        assert result == "do-original"
        # `build_request` põe a identidade do Claude Code como primeira system message e a
        # credencial da subscrição; sem isso a Anthropic recusa o pedido OAuth.
        assert seen["api_key"] == "tok-claude"
        assert seen["messages"][0]["role"] == "system"
        assert "Claude Code" in json.dumps(seen["messages"][0]["content"])
        assert "anthropic-beta" in seen["extra_headers"]

    async def test_non_claude_foreign_model_is_untouched(self) -> None:
        """Um modelo de terceiros não leva headers da Anthropic nem a nossa credencial."""
        seen: dict[str, Any] = {}

        async def fake_original(**kwargs: Any) -> str:
            seen.update(kwargs)
            return "do-original"

        install_transport(FakeTransport())
        plugin.install()
        plugin._state.original_acompletion = fake_original

        await litellm.main.acompletion(
            model="mistral-large", messages=[{"role": "user", "content": "oi"}]
        )

        assert "api_key" not in seen
        assert "extra_headers" not in seen

    async def test_our_model_never_reaches_the_original(self) -> None:
        async def fake_original(**kwargs: Any) -> str:  # pragma: no cover - não deve correr
            raise AssertionError("um modelo nosso não pode cair no original")

        install_transport(FakeTransport(codex_events(text="servido")))
        plugin.install()
        plugin._state.original_acompletion = fake_original

        response = await litellm.main.acompletion(
            model="gpt-5.5", messages=[{"role": "user", "content": "oi"}]
        )
        assert response.choices[0].message.content == "servido"


class TestPositionalArguments:
    async def test_positional_model_and_messages_are_dispatched(self) -> None:
        """Sem normalização, `acompletion("gpt-5.5", msgs)` caía todo no original."""
        transport = install_transport(FakeTransport(codex_events(text="posicional")))
        plugin.install()

        response = await litellm.main.acompletion("gpt-5.5", [{"role": "user", "content": "oi"}])

        assert response.choices[0].message.content == "posicional"
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
        await plugin.dispatch(model=model, messages=[{"role": "user", "content": "oi"}])
        assert transport.specs[0].provider == "antigravity"

    @pytest.mark.parametrize("model", ["antigravity-fast", "gemini-3-gpt-preview"])
    async def test_unserved_gemini_name_fails_instead_of_being_substituted(
        self, model: str
    ) -> None:
        """Foi encaminhado para o Antigravity — e lá recusado por nome.

        Só esse ramo levanta `ModelNotServedError`; que o erro chegue prova ao mesmo tempo
        o encaminhamento e o princípio de falhar alto: ``gemini-3-gpt-preview`` casa
        também com `is_codex_model`, e servi-lo por outro provedor (ou por outro nome)
        devolvia 200 com o campo ``model`` a ecoar o pedido.
        """
        transport = install_transport(FakeTransport(gemini_events()))
        with pytest.raises(ModelNotServedError):
            await plugin.dispatch(model=model, messages=[{"role": "user", "content": "oi"}])
        assert transport.specs == []

    @pytest.mark.parametrize("model", ["gpt-5.5", "gpt-5.5-codex", "codex-mini"])
    async def test_codex_names_go_to_codex(self, model: str) -> None:
        transport = install_transport(FakeTransport(codex_events()))
        await plugin.dispatch(model=model, messages=[{"role": "user", "content": "oi"}])
        assert transport.specs[0].provider == "codex"

    @pytest.mark.parametrize("model", ["claude-sonnet-4-6", "mistral-large", ""])
    async def test_foreign_models_return_none(self, model: str) -> None:
        transport = install_transport(FakeTransport())
        assert await plugin.dispatch(model=model, messages=[]) is None
        assert transport.specs == []


class TestUpstreamErrors:
    """Falhar alto: nunca substituir modelo nem fabricar resposta."""

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
        """`response.failed` chega com HTTP 200; engoli-lo entregava turno vazio."""
        install_transport(
            FakeTransport([{"type": "response.failed", "response": {"error": "recusado"}}])
        )
        with pytest.raises(plugin.StreamError, match="recusado"):
            await plugin.dispatch(model="gpt-5.5", messages=[{"role": "user", "content": "x"}])

    async def test_truncated_codex_stream_is_a_failure(self) -> None:
        """Sem evento terminal a resposta está cortada: devolvê-la mentia ao cliente."""
        install_transport(FakeTransport([{"type": "response.output_text.delta", "delta": "meia "}]))
        with pytest.raises(plugin.StreamError, match=r"response\.completed"):
            await plugin.dispatch(model="gpt-5.5", messages=[{"role": "user", "content": "x"}])

    async def test_antigravity_in_band_error_propagates(self) -> None:
        install_transport(FakeTransport([{"error": {"code": 429, "message": "sem quota"}}]))
        with pytest.raises(plugin.StreamError, match="sem quota"):
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
                    text="resposta",
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
        assert response.choices[0].message.content == "resposta"
        assert response.choices[0].finish_reason == "stop"
        assert response.usage.prompt_tokens == 100
        # Sem este atributo os tokens em cache ficam invisíveis no /spend/logs.
        assert response.usage.cache_read_input_tokens == 40

    async def test_incomplete_status_is_length_not_stop(self) -> None:
        """Um turno cortado por limite de output chegava como stop limpo."""
        install_transport(FakeTransport(codex_events(status="incomplete")))
        response = await plugin.dispatch(
            model="gpt-5.5", messages=[{"role": "user", "content": "x"}]
        )
        assert response.choices[0].finish_reason == "length"

    async def test_incomplete_without_status_field_is_still_length(self) -> None:
        """O evento nem sempre traz ``status``; o tipo do evento é a única pista restante.

        Sem o fallback pelo tipo, um turno truncado voltava a chegar como stop limpo.
        """
        install_transport(
            FakeTransport(
                [
                    {"type": "response.output_text.delta", "delta": "cortad"},
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
                    {"type": "response.reasoning_text.delta", "delta": "a pensar"},
                    {"type": "response.output_text.delta", "delta": "visível"},
                    {"type": "response.completed", "response": {"status": "completed"}},
                ]
            )
        )
        response = await plugin.dispatch(
            model="gpt-5.5", messages=[{"role": "user", "content": "x"}]
        )
        assert response.choices[0].message.content == "visível"
        assert response.choices[0].message.reasoning_content == "a pensar"

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
                            "name": "ler",
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
        assert call.function.name == "ler"
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
                                            {"text": "interno", "thought": True},
                                            {"text": "visível"},
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
        assert response.choices[0].message.content == "visível"
        assert response.choices[0].message.reasoning_content == "interno"


class TestStreamingShape:
    async def test_streaming_returns_the_litellm_wrapper(self) -> None:
        install_transport(FakeTransport(codex_events(text="fluxo")))
        result = await plugin.dispatch(
            model="gpt-5.5", messages=[{"role": "user", "content": "x"}], stream=True
        )
        assert isinstance(result, litellm.CustomStreamWrapper)

    async def test_stream_ends_with_finish_then_usage(self) -> None:
        """O chunk de usage tem de vir e tem de trazer `choices` não vazio: o iterador da
        rota /v1/responses faz `chunk.choices[0]` sem guarda."""
        install_transport(FakeTransport(codex_events(text="fluxo", usage={"input_tokens": 7})))
        chunks = await collect_stream(model="gpt-5.5", messages=[{"role": "user", "content": "x"}])

        assert [c.choices[0].delta.content for c in chunks if c.choices[0].delta.content] == [
            "fluxo"
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
                                                    "name": "ler",
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

        assert tool_chunks[0].choices[0].delta.tool_calls[0].function.name == "ler"
        assert json.loads(tool_chunks[1].choices[0].delta.tool_calls[0].function.arguments) == {
            "path": "a"
        }
        assert chunks[-2].choices[0].finish_reason == "tool_calls"

    async def test_stream_error_is_not_swallowed(self) -> None:
        install_transport(FakeTransport([{"type": "response.output_text.delta", "delta": "meia "}]))
        with pytest.raises(plugin.StreamError):
            await collect_stream(model="gpt-5.5", messages=[{"role": "user", "content": "x"}])


async def collect_stream(**kwargs: Any) -> list[Any]:
    """Consome o wrapper do LiteLLM até ao fim."""
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
        """``gpt-6`` é um alias de família e resolve-se; a resposta nomeia o que se pediu.

        Devolver o nome de fio no campo ``model`` fazia o spend log cobrar um modelo que o
        cliente nunca nomeou.
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
        """Sem a assinatura de volta, o CCA rejeita o turno seguinte com a tool call."""
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
                                                "functionCall": {"id": "c1", "name": "ler"},
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
        """Quem detém o refresh token é o store; aqui só se relê a fonte."""
        store = FakeStore(
            {"openai-codex": Credential(provider="openai-codex", access_token="novo")}
        )
        plugin.configure(store=store)

        token = await plugin._refresh("codex")

        assert token == "novo"
        assert store.reloads == 1


class TestTokenRenewal:
    """Renovação automática. Antes disto, um token expirado exigia carregar num botão."""

    def _store(self, *, expired: bool, owns: bool = True, refresh_token: str = "RT") -> Any:
        class Store:
            owns_refresh = owns

            def __init__(self) -> None:
                self.credential = Credential(
                    provider="openai-codex",
                    access_token="AT-velho",
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
            return credential.with_access_token("AT-novo", expires_at=time.time() + 3600)

        monkeypatch.setattr("litellm_mysubs.credentials.oauth.refresh", refresh)

    @pytest.mark.asyncio
    async def test_an_expired_token_is_renewed_before_the_request(self) -> None:
        """Renovar aqui, e não à espera do 401, evita uma ida ao upstream por cada token
        que expira — e evita que um pedido em streaming falhe a meio, onde já não é
        recuperável: o transporte só repete o que ainda não entregou."""
        store = self._store(expired=True)
        plugin.configure(store=store)
        assert await plugin._access_token("codex") == "AT-novo"

    @pytest.mark.asyncio
    async def test_the_renewed_credential_is_persisted(self) -> None:
        """Sem gravar, cada pedido gastaria um refresh token de uso único — e o segundo
        falharia com `invalid_grant`."""
        store = self._store(expired=True)
        plugin.configure(store=store)
        await plugin._access_token("codex")
        assert store.written == ["AT-novo"]

    @pytest.mark.asyncio
    async def test_a_valid_token_is_not_renewed(self) -> None:
        """Gastar uma rotação sem necessidade é a forma mais fácil de partir uma sessão
        que estava boa."""
        store = self._store(expired=False)
        plugin.configure(store=store)
        assert await plugin._access_token("codex") == "AT-velho"
        assert store.written == []

    @pytest.mark.asyncio
    async def test_a_store_that_is_not_the_owner_never_rotates(self) -> None:
        """Dois renovadores sobre tokens rotativos de uso único invalidam a cópia um do
        outro e produzem `invalid_grant` em ciclo, forçando re-login manual."""
        store = self._store(expired=True, owns=False)
        plugin.configure(store=store)
        assert await plugin._access_token("codex") == "AT-velho"
        assert store.written == []

    @pytest.mark.asyncio
    async def test_without_a_refresh_token_there_is_nothing_to_rotate(self) -> None:
        store = self._store(expired=True, refresh_token="")
        plugin.configure(store=store)
        assert await plugin._access_token("codex") == "AT-velho"
        assert store.written == []

    @pytest.mark.asyncio
    async def test_a_valid_token_does_not_even_consult_the_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Não basta devolver o token certo: não se pode falar com o provedor de todo.

        Uma rotação desnecessária gasta um refresh token de uso único e transforma uma
        sessão boa numa que precisa de re-login. O teste anterior passava mesmo com
        `if True:` porque o `_refresh` relê a fonte antes de rodar — só contando as
        chamadas é que a diferença aparece.
        """
        calls: list[str] = []

        async def refresh(credential: Credential, *, client: Any, store: Any = None) -> Credential:
            calls.append(credential.access_token)
            return credential.with_access_token("AT-novo", expires_at=time.time() + 3600)

        monkeypatch.setattr("litellm_mysubs.credentials.oauth.refresh", refresh)
        plugin.configure(store=self._store(expired=False))
        assert await plugin._access_token("codex") == "AT-velho"
        assert calls == []

    @pytest.mark.asyncio
    async def test_a_valid_token_does_not_touch_the_store_source(self) -> None:
        """Nem sequer se relê a fonte.

        O `_refresh` começa por um `store.reload()` — leitura de disco ou chamada ao cofre.
        Chamá-lo com um token válido é um I/O por pedido servido, no caminho quente. A
        guarda `is_expired` é o que o evita, e só contar os `reload` a torna visível: o
        resultado devolvido é o mesmo com ou sem ela.
        """
        store = self._store(expired=False)
        plugin.configure(store=store)
        await plugin._access_token("codex")
        assert store.reloads == 0
