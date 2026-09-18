"""Descoberta de modelos: o que a lista diz sobre a conta, e o que não pode dizer.

O que se testa aqui não é que um POST devolva 200. É a fronteira entre facto e hipótese —
`verified` — e as duas maneiras de a atravessar por engano:

* marcar como servido um nome que ninguém confirmou;
* marcar como não servido um nome que a rede desta máquina não deixou perguntar.

As duas produzem uma lista plausível e errada, que é exactamente o modo de falha que o
resto do pacote existe para evitar.

Tudo com `httpx.MockTransport`: sem rede, sem relógio.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

import httpx
import pytest

from litellm_mysubs.catalog.discovery import (
    CURATED_ANTHROPIC,
    CURATED_CODEX,
    PROBE_CONCURRENCY,
    DiscoveredModel,
    DiscoveryError,
    discover,
    suggested_name,
)
from litellm_mysubs.credentials.store import Credential
from litellm_mysubs.transport.hosts import HOSTS, MODELS_PATH
from litellm_mysubs.wire.antigravity_models import BROKEN_WIRE, ModelCatalog
from litellm_mysubs.wire.codex import resolve_model

Handler = Callable[[httpx.Request], httpx.Response]

#: JWT sem assinatura com a claim de conta que o `codex.build_headers` lê. O corpo é
#: `{"https://api.openai.com/auth": {"chatgpt_account_id": "acc-1"}}` em base64url.
CODEX_TOKEN = (
    "eyJhbGciOiJub25lIn0."
    "eyJodHRwczovL2FwaS5vcGVuYWkuY29tL2F1dGgiOnsiY2hhdGdwdF9hY2NvdW50X2lkIjoiYWNjLTEifX0."
)

ANTHROPIC = Credential(provider="anthropic", access_token="tok-a")
CODEX = Credential(provider="openai-codex", access_token=CODEX_TOKEN)
GOOGLE = Credential(provider="google-antigravity", access_token="tok-g")


def client(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def by_name(models: list[DiscoveredModel]) -> dict[str, DiscoveredModel]:
    return {m.wire_name: m for m in models}


def catalog_payload(*ids: str, deprecated: tuple[str, ...] = ()) -> dict[str, object]:
    return {
        "models": {model: {"displayName": model} for model in ids},
        "deprecatedModelIds": list(deprecated),
    }


class TestGoogleCatalog:
    """O único provedor com catálogo real. A verdade é a resposta, não uma lista nossa."""

    async def test_deprecated_model_never_reaches_the_user(self) -> None:
        """Um id em `deprecatedModelIds` está no payload e não pode sair na lista.

        É a razão de ser do desconto: o catálogo anuncia variantes que o
        `streamGenerateContent` recusa com 400, e oferecê-las produz um deployment que só
        sabe falhar.
        """

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json=catalog_payload(
                    "gemini-3.1-pro", "gemini-3.1-pro-high", deprecated=("gemini-3.1-pro-high",)
                ),
            )

        async with client(handler) as http:
            models = await discover(GOOGLE, client=http)

        assert [m.wire_name for m in models] == ["gemini-3.1-pro"]

    async def test_catalog_ids_are_verified_without_probing(self) -> None:
        """O catálogo é a resposta do próprio upstream: não precisa de confirmação."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=catalog_payload("gemini-3.1-pro"))

        async with client(handler) as http:
            models = await discover(GOOGLE, client=http)

        assert models == [
            DiscoveredModel(
                wire_name="gemini-3.1-pro", suggested_name="gemini-3.1-pro", verified=True
            )
        ]

    async def test_broken_wire_variant_is_listed_unverified(self) -> None:
        """`gemini-3.1-pro-high` está no catálogo e dá 400 no fio.

        Escondê-lo perdia informação que a conta deu; anunciá-lo como servido repetia o
        defeito. Fica listado, com `verified=False` e a razão.
        """
        broken = BROKEN_WIRE[0]

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=catalog_payload("gemini-3.1-pro", broken))

        async with client(handler) as http:
            found = by_name(await discover(GOOGLE, client=http))

        assert found[broken].verified is False
        assert "400" in found[broken].note
        assert found["gemini-3.1-pro"].verified is True

    async def test_empty_catalog_keeps_previous_snapshot_labelled_with_age(self) -> None:
        """Uma resposta sem modelos não apaga o que já se sabia, e não passa por actual.

        `ModelCatalog.update` ignora um payload vazio de propósito. Devolver o
        instantâneo é honesto; devolvê-lo sem idade seria apresentar dados velhos como
        frescos — o "número plausível fabricado" que o projecto proíbe.
        """
        catalog = ModelCatalog()
        catalog.update(catalog_payload("gemini-3.1-pro"), now=1000.0)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"models": {}})

        async with client(handler) as http:
            models = await discover(GOOGLE, client=http, catalog=catalog, now=1300.0)

        assert [m.wire_name for m in models] == ["gemini-3.1-pro"]
        assert "300 s" in models[0].note

    async def test_all_deprecated_payload_is_treated_as_stale_not_fresh(self) -> None:
        """200 cujos modelos estão todos deprecados não renova nada.

        Bastava olhar para "houve resposta" para etiquetar isto como fresco e servir um
        catálogo velho como actual.
        """
        catalog = ModelCatalog()
        catalog.update(catalog_payload("gemini-3.1-pro"), now=1000.0)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json=catalog_payload("gemini-9-novo", deprecated=("gemini-9-novo",))
            )

        async with client(handler) as http:
            models = await discover(GOOGLE, client=http, catalog=catalog, now=1042.0)

        assert [m.wire_name for m in models] == ["gemini-3.1-pro"]
        assert "42 s" in models[0].note

    async def test_fresh_catalog_carries_no_age_note(self) -> None:
        """A etiqueta de idade só aparece quando há idade: caso contrário é ruído."""
        catalog = ModelCatalog()
        catalog.update(catalog_payload("gemini-antigo"), now=1000.0)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=catalog_payload("gemini-3.1-pro"))

        async with client(handler) as http:
            models = await discover(GOOGLE, client=http, catalog=catalog, now=9000.0)

        assert [m.wire_name for m in models] == ["gemini-3.1-pro"]
        assert models[0].note == ""

    async def test_unreachable_catalog_without_snapshot_raises(self) -> None:
        """Sem catálogo e sem instantâneo não há resposta honesta em forma de lista."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("sem rota", request=request)

        async with client(handler) as http:
            with pytest.raises(DiscoveryError):
                await discover(GOOGLE, client=http)

    async def test_second_host_is_tried_when_the_first_fails(self) -> None:
        """Um host em baixo não é uma conta sem modelos."""
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            if str(request.url).startswith(HOSTS[0]):
                return httpx.Response(503, text="indisponível")
            return httpx.Response(200, json=catalog_payload("gemini-3.1-pro"))

        async with client(handler) as http:
            models = await discover(GOOGLE, client=http)

        assert seen == [HOSTS[0] + MODELS_PATH, HOSTS[1] + MODELS_PATH]
        assert [m.wire_name for m in models] == ["gemini-3.1-pro"]


class TestProbedProviders:
    """Anthropic e Codex: lista curada e sonda. Só o upstream decide."""

    async def test_upstream_not_found_removes_the_model(self) -> None:
        """404 com `not_found_error` é "esta conta não serve": sai da lista."""
        recusado = CURATED_ANTHROPIC[0]

        def handler(request: httpx.Request) -> httpx.Response:
            body = request.read().decode()
            if f'"{recusado}"' in body:
                return httpx.Response(404, json={"error": {"type": "not_found_error"}})
            return httpx.Response(200, json={})

        async with client(handler) as http:
            models = await discover(ANTHROPIC, client=http)

        assert recusado not in by_name(models)
        assert len(models) == len(CURATED_ANTHROPIC) - 1

    async def test_network_failure_keeps_the_model_unverified_with_reason(self) -> None:
        """Uma sonda que não chegou ao upstream não é um facto sobre a conta.

        Esta é a asserção central do módulo: tratar `ConnectError` como recusa apagava
        modelos servidos sempre que a máquina do utilizador tivesse a rede instável.
        """
        inalcancavel = CURATED_ANTHROPIC[1]

        def handler(request: httpx.Request) -> httpx.Response:
            if f'"{inalcancavel}"' in request.read().decode():
                raise httpx.ConnectError("sem rota", request=request)
            return httpx.Response(200, json={})

        async with client(handler) as http:
            found = by_name(await discover(ANTHROPIC, client=http))

        assert inalcancavel in found
        assert found[inalcancavel].verified is False
        assert "não chegou ao upstream" in found[inalcancavel].note

    async def test_verified_is_never_true_without_a_real_answer(self) -> None:
        """Nenhum caminho que não seja um 200 pode produzir `verified=True`."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("demorou", request=request)

        async with client(handler) as http:
            models = await discover(ANTHROPIC, client=http)

        assert [m.wire_name for m in models] == list(CURATED_ANTHROPIC)
        assert not any(m.verified for m in models)
        assert all(m.note for m in models)

    async def test_ambiguous_status_neither_confirms_nor_denies(self) -> None:
        """429 é quota, não inexistência.

        Tratá-lo como recusa desligava um modelo bom por causa de um pico de uso; tratá-lo
        como confirmação prometia um modelo que nunca respondeu.
        """

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, json={"error": {"type": "rate_limit_error"}})

        async with client(handler) as http:
            models = await discover(ANTHROPIC, client=http)

        assert [m.wire_name for m in models] == list(CURATED_ANTHROPIC)
        assert not any(m.verified for m in models)
        assert all("429" in m.note for m in models)

    async def test_404_without_the_marker_is_not_a_denial(self) -> None:
        """Um 404 de rota errada não é o upstream a recusar o modelo.

        O corpo é que nomeia o motivo; só o estado faria uma mudança de caminho na API
        apagar a lista curada inteira.
        """

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="<html>Not Found</html>")

        async with client(handler) as http:
            models = await discover(ANTHROPIC, client=http)

        assert len(models) == len(CURATED_ANTHROPIC)
        assert not any(m.verified for m in models)

    async def test_served_model_is_verified(self) -> None:
        """200 é a única fonte de `verified=True`, e não deixa nota."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={})

        async with client(handler) as http:
            models = await discover(ANTHROPIC, client=http)

        assert all(m.verified for m in models)
        assert all(m.note == "" for m in models)

    async def test_codex_unsupported_marker_removes_the_model(self) -> None:
        """A recusa do Codex é um 400 com marca própria, não um 404."""
        recusado = CURATED_CODEX[0]

        def handler(request: httpx.Request) -> httpx.Response:
            if f'"{recusado}"' in request.read().decode():
                return httpx.Response(
                    400,
                    json={
                        "error": {
                            "message": f"The '{recusado}' model is not supported when "
                            f"using Codex with a ChatGPT account"
                        }
                    },
                )
            return httpx.Response(200, json={})

        async with client(handler) as http:
            models = await discover(CODEX, client=http)

        assert recusado not in by_name(models)
        assert len(models) == len(CURATED_CODEX) - 1

    async def test_codex_generic_400_is_not_a_denial(self) -> None:
        """Um 400 sem a marca é um pedido malformado nosso, não um modelo inexistente."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": {"message": "invalid_request"}})

        async with client(handler) as http:
            models = await discover(CODEX, client=http)

        assert [m.wire_name for m in models] == list(CURATED_CODEX)
        assert not any(m.verified for m in models)

    async def test_probe_asks_for_the_curated_name_itself(self) -> None:
        """A sonda tem de perguntar pelo nome curado, não por um alias resolvido.

        Duas metades da mesma invariante, e nenhuma chega sozinha:

        1. O nome curado é o que viaja no corpo. Um alias substituído pelo caminho faria a
           lista afirmar servido um nome que nunca foi perguntado.
        2. Nenhuma entrada curada *é* um alias. `codex.resolve_model` mapeia
           `gpt-5` -> `gpt-5.5`; pôr `gpt-5` na lista fazia a sonda de `gpt-5` confirmar
           `gpt-5.5`, e o utilizador ficava com um deployment que nomeia um modelo e
           corre outro. A primeira asserção passa na mesma nesse caso — é esta que apanha.
        """
        pedidos: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            pedidos.append(str(json.loads(request.read())["model"]))
            return httpx.Response(200, json={})

        async with client(handler) as http:
            await discover(CODEX, client=http)

        assert sorted(pedidos) == sorted(CURATED_CODEX)
        assert [resolve_model(w) for w in CURATED_CODEX] == list(CURATED_CODEX)

    async def test_probes_respect_the_concurrency_ceiling(self) -> None:
        """Mais sondas que o tecto nunca estão em voo ao mesmo tempo.

        Sem tecto, ligar uma subscrição abria uma ligação por nome curado de uma vez só
        contra o mesmo backend.
        """
        assert len(CURATED_ANTHROPIC) > PROBE_CONCURRENCY, "a lista tem de exceder o tecto"
        em_voo = 0
        pico = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal em_voo, pico
            em_voo += 1
            pico = max(pico, em_voo)
            try:
                # Uma volta pelo escalonador: sem ela cada sonda corre até ao fim antes de
                # a seguinte começar e o pico seria 1 mesmo sem semáforo nenhum.
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                return httpx.Response(200, json={})
            finally:
                em_voo -= 1

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            models = await discover(ANTHROPIC, client=http)

        assert len(models) == len(CURATED_ANTHROPIC)
        assert pico > 1, "as sondas têm de correr em paralelo"
        assert pico <= PROBE_CONCURRENCY

    async def test_anthropic_probe_leads_with_the_identity_block(self) -> None:
        """Medido: `system` só com o prompt do cliente devolve 429 no caminho OAuth.

        Uma sonda que caísse nisso reportava toda a lista curada como "por verificar" por
        uma razão nossa, não da conta.
        """
        capturado: list[object] = []

        def handler(request: httpx.Request) -> httpx.Response:
            capturado.append(json.loads(request.read())["system"])
            return httpx.Response(200, json={})

        async with client(handler) as http:
            await discover(ANTHROPIC, client=http)

        primeiro = capturado[0]
        assert isinstance(primeiro, list)
        assert "Claude Code" in primeiro[0]["text"]

    async def test_probe_carries_the_subscription_credential(self) -> None:
        """Sem o token a sonda mede a rejeição do anónimo, não o que a conta serve."""
        tokens: set[str] = set()

        def handler(request: httpx.Request) -> httpx.Response:
            tokens.add(request.headers.get("authorization", ""))
            return httpx.Response(200, json={})

        async with client(handler) as http:
            await discover(ANTHROPIC, client=http)

        assert tokens == {"Bearer tok-a"}


class TestSuggestedName:
    """O nome público que vai para o `model_name` do deployment."""

    def test_provider_prefix_is_stripped(self) -> None:
        """O prefixo pertence a `litellm_params["model"]`, nunca ao `model_name`.

        É o `model_name` que ecoa no spend log. Um `anthropic/claude-opus-5` aí nomeia
        algo que nenhum cliente pediu, e o `registry.is_declared` compara-o com o
        `model_info["id"]` — um prefixo a mais fazia a entrada gerida parecer declarada.
        """
        assert suggested_name("anthropic/claude-opus-5") == "claude-opus-5"

    def test_bare_name_survives_intact(self) -> None:
        """O caso normal: o wire name do catálogo já vem nu e não pode ser mexido."""
        assert suggested_name("gemini-3.1-pro") == "gemini-3.1-pro"
