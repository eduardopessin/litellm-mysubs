"""Model discovery: what the list says about the account, and what it must not say.

What is tested here is not that a POST returns 200. It is the boundary between fact and
hypothesis — `verified` — and the two ways of crossing it by mistake:

* marking as served a name nobody confirmed;
* marking as not served a name this machine's network did not allow asking about.

Both produce a plausible, wrong list, which is exactly the failure mode the rest of the
package exists to avoid.

All of it with `httpx.MockTransport`: no network, no clock.
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

#: Unsigned JWT carrying the account claim that `codex.build_headers` reads. The body is
#: `{"https://api.openai.com/auth": {"chatgpt_account_id": "acc-1"}}` in base64url.
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
    """The only provider with a real catalogue. The truth is the response, not a list of
    ours."""

    async def test_deprecated_model_never_reaches_the_user(self) -> None:
        """An id in `deprecatedModelIds` is in the payload and must not come out in the list.

        That is the reason the subtraction exists: the catalogue advertises variants that
        `streamGenerateContent` refuses with 400, and offering them produces a deployment
        that only knows how to fail.
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
        """The catalogue is the upstream's own answer: it needs no confirmation."""

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
        """`gemini-3.1-pro-high` is in the catalogue and gives 400 on the wire.

        Hiding it lost information the account gave; advertising it as served repeated the
        defect. It stays listed, with `verified=False` and the reason.
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
        """A response with no models does not erase what was already known, and does not pass
        as current.

        `ModelCatalog.update` ignores an empty payload on purpose. Returning the snapshot is
        honest; returning it without its age would present old data as fresh — the
        "fabricated plausible number" the project forbids.
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
        """A 200 whose models are all deprecated refreshes nothing.

        Looking only at "there was a response" was enough to label this fresh and serve an
        old catalogue as current.
        """
        catalog = ModelCatalog()
        catalog.update(catalog_payload("gemini-3.1-pro"), now=1000.0)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json=catalog_payload("gemini-9-new", deprecated=("gemini-9-new",))
            )

        async with client(handler) as http:
            models = await discover(GOOGLE, client=http, catalog=catalog, now=1042.0)

        assert [m.wire_name for m in models] == ["gemini-3.1-pro"]
        assert "42 s" in models[0].note

    async def test_fresh_catalog_carries_no_age_note(self) -> None:
        """The age label only shows up when there is an age: otherwise it is noise."""
        catalog = ModelCatalog()
        catalog.update(catalog_payload("gemini-old"), now=1000.0)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=catalog_payload("gemini-3.1-pro"))

        async with client(handler) as http:
            models = await discover(GOOGLE, client=http, catalog=catalog, now=9000.0)

        assert [m.wire_name for m in models] == ["gemini-3.1-pro"]
        assert models[0].note == ""

    async def test_unreachable_catalog_without_snapshot_raises(self) -> None:
        """With no catalogue and no snapshot there is no honest answer shaped like a list."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route", request=request)

        async with client(handler) as http:
            with pytest.raises(DiscoveryError):
                await discover(GOOGLE, client=http)

    async def test_second_host_is_tried_when_the_first_fails(self) -> None:
        """A host that is down is not an account with no models."""
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            if str(request.url).startswith(HOSTS[0]):
                return httpx.Response(503, text="unavailable")
            return httpx.Response(200, json=catalog_payload("gemini-3.1-pro"))

        async with client(handler) as http:
            models = await discover(GOOGLE, client=http)

        assert seen == [HOSTS[0] + MODELS_PATH, HOSTS[1] + MODELS_PATH]
        assert [m.wire_name for m in models] == ["gemini-3.1-pro"]


def sse_stream(*, text: str = "", usage: dict[str, object] | None = None) -> str:
    """SSE body of one CCA turn, in the shape `:streamGenerateContent` returns."""
    event: dict[str, object] = {
        "response": {
            "candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}],
            "usageMetadata": usage if usage is not None else {"totalTokenCount": 12},
        }
    }
    return f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n"


#: The text measured on the real account for `gemini-3.5-flash-low` and `-extra-low`:
#: HTTP 200, `finishReason: STOP`, and zero tokens billed.
RETIREMENT_NOTICE = (
    "Gemini 3.5 Flash is no longer available. Please switch to Gemini 3.7 Flash in the "
    "latest version of Antigravity."
)


class TestGoogleProbe:
    """Being in the catalogue is not being served, and the name does not predict which is
    which.

    Measured on the same account: `gemini-3.5-flash-lite` answers and `-low` is dead;
    `tab_flash_lite_preview` answers and `tab_jump_flash_lite_preview` gives 400. Each test
    here pins one of the verdicts that tell those pairs apart, and the last one pins the
    rule that makes them safe — none of them leaves the list.
    """

    async def test_answering_model_is_verified(self) -> None:
        """200 with content and tokens billed: `gemini-3.5-flash-lite`, measured."""

        def handler(request: httpx.Request) -> httpx.Response:
            if MODELS_PATH in str(request.url):
                return httpx.Response(200, json=catalog_payload("gemini-3.5-flash-lite"))
            return httpx.Response(200, text=sse_stream(text="2 + 2 = 4"))

        async with client(handler) as http:
            models = await discover(GOOGLE, client=http, probe=True)

        assert [(m.wire_name, m.verified, m.note) for m in models] == [
            ("gemini-3.5-flash-lite", True, "")
        ]

    async def test_refused_model_is_unverified_citing_the_400(self) -> None:
        """`chat_23310` gives 400 "Request contains an invalid argument"; its `tab_*` sibling
        does not.

        The note cites the status because the status is what the user can check; a generic
        note left the 400 indistinguishable from a network failure.
        """

        def handler(request: httpx.Request) -> httpx.Response:
            if MODELS_PATH in str(request.url):
                return httpx.Response(
                    200, json=catalog_payload("chat_23310", "tab_flash_lite_preview")
                )
            if '"chat_23310"' in request.read().decode():
                return httpx.Response(400, json={"error": {"message": "invalid argument"}})
            return httpx.Response(200, text=sse_stream(text="Hello!"))

        async with client(handler) as http:
            found = by_name(await discover(GOOGLE, client=http, probe=True))

        assert found["chat_23310"].verified is False
        assert "400" in found["chat_23310"].note
        assert found["tab_flash_lite_preview"].verified is True

    async def test_retirement_notice_is_not_an_answer(self) -> None:
        """200 + notice + usage 0 is a dead model passing for a live one.

        This is the defect that hurts most: without it the notice enters the history as if
        the model had spoken. The note is its own — merging it with the 400 one erased the
        difference between "the upstream refused the request" and "the model no longer
        exists".
        """

        def handler(request: httpx.Request) -> httpx.Response:
            if MODELS_PATH in str(request.url):
                return httpx.Response(200, json=catalog_payload("gemini-3.5-flash-low"))
            return httpx.Response(
                200, text=sse_stream(text=RETIREMENT_NOTICE, usage={"totalTokenCount": 0})
            )

        async with client(handler) as http:
            found = by_name(await discover(GOOGLE, client=http, probe=True))

        dead = found["gemini-3.5-flash-low"]
        assert dead.verified is False
        assert dead.note == "model retired by upstream"
        assert "400" not in dead.note

    async def test_unreachable_probe_keeps_the_model_listed(self) -> None:
        """A timeout is a fact about this network, not about the account.

        Discarding the model here was lying about the subscription, so the proof is the
        length of the list: both catalogue names are still there.
        """

        def handler(request: httpx.Request) -> httpx.Response:
            if MODELS_PATH in str(request.url):
                return httpx.Response(200, json=catalog_payload("gemini-3.1-pro", "gemini-2.5-pro"))
            raise httpx.ReadTimeout("took too long", request=request)

        async with client(handler) as http:
            models = await discover(GOOGLE, client=http, probe=True)

        assert [m.wire_name for m in models] == ["gemini-3.1-pro", "gemini-2.5-pro"]
        assert not any(m.verified for m in models)
        assert all("could not probe" in m.note for m in models)

    async def test_transient_capacity_is_not_a_denial(self) -> None:
        """503 "No capacity available" is Google capacity, measured on `gemini-2.5-pro`.

        Treating it as a refusal disabled a good model until the next discovery, and the
        note has to say so without promising anything — `verified` stays false because
        nobody measured.
        """

        def handler(request: httpx.Request) -> httpx.Response:
            if MODELS_PATH in str(request.url):
                return httpx.Response(200, json=catalog_payload("gemini-2.5-pro"))
            return httpx.Response(503, text="No capacity available")

        async with client(handler) as http:
            found = by_name(await discover(GOOGLE, client=http, probe=True))

        note = found["gemini-2.5-pro"].note
        assert found["gemini-2.5-pro"].verified is False
        assert "could not probe" in note
        assert "503" in note

    async def test_in_band_error_inside_a_200_is_a_denial(self) -> None:
        """The CCA returns errors inside the stream with HTTP 200.

        The status alone said "served": it is exactly the failure mode that
        `plugin.py :: _raise_in_band` exists to catch on the real path.
        """
        event = {"error": {"code": 400, "message": "Request contains an invalid argument."}}

        def handler(request: httpx.Request) -> httpx.Response:
            if MODELS_PATH in str(request.url):
                return httpx.Response(200, json=catalog_payload("chat_20706"))
            return httpx.Response(200, text=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n")

        async with client(handler) as http:
            found = by_name(await discover(GOOGLE, client=http, probe=True))

        assert found["chat_20706"].verified is False
        assert "400" in found["chat_20706"].note

    async def test_probe_is_off_by_default(self) -> None:
        """The probe spends quota: one billed turn per catalogue name.

        Without `probe`, the transport can only see the catalogue request. One extra probe
        request here meant that connecting the subscription billed 32 turns without the user
        having asked for anything.
        """
        paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            paths.append(request.url.path)
            return httpx.Response(200, json=catalog_payload("gemini-3.1-pro", "chat_23310"))

        async with client(handler) as http:
            models = await discover(GOOGLE, client=http)

        assert paths == [MODELS_PATH]
        assert [m.wire_name for m in models] == ["gemini-3.1-pro", "chat_23310"]

    async def test_probes_respect_the_concurrency_ceiling(self) -> None:
        """The real catalogue has 32 names; without a ceiling that would be 32 connections at
        once.

        The backend itself answers that with 503, which would turn the spike into a whole
        list of "could not probe".
        """
        names = tuple(f"gemini-probe-{n}" for n in range(PROBE_CONCURRENCY * 3))
        in_flight = 0
        peak = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal in_flight, peak
            if MODELS_PATH in str(request.url):
                return httpx.Response(200, json=catalog_payload(*names))
            in_flight += 1
            peak = max(peak, in_flight)
            try:
                # Two turns through the scheduler: without them each probe runs to completion
                # before the next starts, and the peak was 1 even with no semaphore at all.
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                return httpx.Response(200, text=sse_stream(text="4"))
            finally:
                in_flight -= 1

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            models = await discover(GOOGLE, client=http, probe=True)

        assert len(models) == len(names)
        assert peak > 1, "the probes have to run in parallel"
        assert peak <= PROBE_CONCURRENCY

    async def test_measurement_overrides_the_static_broken_list(self) -> None:
        """`BROKEN_WIRE` is a stored guess; the probe is the measurement of now.

        If the upstream serves a name from that list again, keeping it unverified would be
        preferring the table to the response — the opposite of what this module defends.
        """
        broken = BROKEN_WIRE[0]

        def handler(request: httpx.Request) -> httpx.Response:
            if MODELS_PATH in str(request.url):
                return httpx.Response(200, json=catalog_payload(broken))
            return httpx.Response(200, text=sse_stream(text="2 + 2 = 4"))

        async with client(handler) as http:
            found = by_name(await discover(GOOGLE, client=http, probe=True))

        assert found[broken].verified is True
        assert found[broken].note == ""


class TestProbedProviders:
    """Anthropic and Codex: curated list plus probe. Only the upstream decides."""

    async def test_upstream_not_found_removes_the_model(self) -> None:
        """404 with `not_found_error` means "this account does not serve it": out of the
        list."""
        refused = CURATED_ANTHROPIC[0]

        def handler(request: httpx.Request) -> httpx.Response:
            body = request.read().decode()
            if f'"{refused}"' in body:
                return httpx.Response(404, json={"error": {"type": "not_found_error"}})
            return httpx.Response(200, json={})

        async with client(handler) as http:
            models = await discover(ANTHROPIC, client=http)

        assert refused not in by_name(models)
        assert len(models) == len(CURATED_ANTHROPIC) - 1

    async def test_network_failure_keeps_the_model_unverified_with_reason(self) -> None:
        """A probe that never reached the upstream is not a fact about the account.

        This is the module's central assertion: treating `ConnectError` as a refusal deleted
        served models whenever the user's machine had a flaky network.
        """
        unreachable = CURATED_ANTHROPIC[1]

        def handler(request: httpx.Request) -> httpx.Response:
            if f'"{unreachable}"' in request.read().decode():
                raise httpx.ConnectError("no route", request=request)
            return httpx.Response(200, json={})

        async with client(handler) as http:
            found = by_name(await discover(ANTHROPIC, client=http))

        assert unreachable in found
        assert found[unreachable].verified is False
        assert "did not reach the upstream" in found[unreachable].note

    async def test_verified_is_never_true_without_a_real_answer(self) -> None:
        """No path other than a 200 may produce `verified=True`."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("took too long", request=request)

        async with client(handler) as http:
            models = await discover(ANTHROPIC, client=http)

        assert [m.wire_name for m in models] == list(CURATED_ANTHROPIC)
        assert not any(m.verified for m in models)
        assert all(m.note for m in models)

    async def test_ambiguous_status_neither_confirms_nor_denies(self) -> None:
        """429 is quota, not non-existence.

        Treating it as a refusal disabled a good model over a usage spike; treating it as a
        confirmation promised a model that never answered.
        """

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, json={"error": {"type": "rate_limit_error"}})

        async with client(handler) as http:
            models = await discover(ANTHROPIC, client=http)

        assert [m.wire_name for m in models] == list(CURATED_ANTHROPIC)
        assert not any(m.verified for m in models)
        assert all("429" in m.note for m in models)

    async def test_404_without_the_marker_is_not_a_denial(self) -> None:
        """A 404 from a wrong route is not the upstream refusing the model.

        It is the body that names the reason; the status alone would let an API path change
        delete the entire curated list.
        """

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="<html>Not Found</html>")

        async with client(handler) as http:
            models = await discover(ANTHROPIC, client=http)

        assert len(models) == len(CURATED_ANTHROPIC)
        assert not any(m.verified for m in models)

    async def test_served_model_is_verified(self) -> None:
        """200 is the only source of `verified=True`, and it leaves no note."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={})

        async with client(handler) as http:
            models = await discover(ANTHROPIC, client=http)

        assert all(m.verified for m in models)
        assert all(m.note == "" for m in models)

    async def test_codex_unsupported_marker_removes_the_model(self) -> None:
        """The Codex refusal is a 400 with its own marker, not a 404."""
        refused = CURATED_CODEX[0]

        def handler(request: httpx.Request) -> httpx.Response:
            if f'"{refused}"' in request.read().decode():
                return httpx.Response(
                    400,
                    json={
                        "error": {
                            "message": f"The '{refused}' model is not supported when "
                            f"using Codex with a ChatGPT account"
                        }
                    },
                )
            return httpx.Response(200, json={})

        async with client(handler) as http:
            models = await discover(CODEX, client=http)

        assert refused not in by_name(models)
        assert len(models) == len(CURATED_CODEX) - 1

    async def test_codex_generic_400_is_not_a_denial(self) -> None:
        """A 400 without the marker is a malformed request of ours, not a missing model."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": {"message": "invalid_request"}})

        async with client(handler) as http:
            models = await discover(CODEX, client=http)

        assert [m.wire_name for m in models] == list(CURATED_CODEX)
        assert not any(m.verified for m in models)

    async def test_probe_asks_for_the_curated_name_itself(self) -> None:
        """The probe has to ask for the curated name itself, not for a resolved alias.

        Two halves of the same invariant, and neither is enough alone:

        1. The curated name is what travels in the body. An alias substituted along the way
           would make the list claim as served a name that was never asked about.
        2. No curated entry *is* an alias. `codex.resolve_model` maps `gpt-5` -> `gpt-5.5`;
           putting `gpt-5` in the list made the `gpt-5` probe confirm `gpt-5.5`, and the user
           ended up with a deployment that names one model and runs another. The first
           assertion still passes in that case — this is the one that catches it.
        """
        requested: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requested.append(str(json.loads(request.read())["model"]))
            return httpx.Response(200, json={})

        async with client(handler) as http:
            await discover(CODEX, client=http)

        assert sorted(requested) == sorted(CURATED_CODEX)
        assert [resolve_model(w) for w in CURATED_CODEX] == list(CURATED_CODEX)

    async def test_probes_respect_the_concurrency_ceiling(self) -> None:
        """More probes than the ceiling are never in flight at the same time.

        Without a ceiling, connecting a subscription opened one connection per curated name
        at once against the same backend.
        """
        assert len(CURATED_ANTHROPIC) > PROBE_CONCURRENCY, "the list has to exceed the ceiling"
        in_flight = 0
        peak = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            try:
                # One turn through the scheduler: without it each probe runs to completion
                # before the next starts and the peak would be 1 even with no semaphore.
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                return httpx.Response(200, json={})
            finally:
                in_flight -= 1

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            models = await discover(ANTHROPIC, client=http)

        assert len(models) == len(CURATED_ANTHROPIC)
        assert peak > 1, "the probes have to run in parallel"
        assert peak <= PROBE_CONCURRENCY

    async def test_anthropic_probe_leads_with_the_identity_block(self) -> None:
        """Measured: a `system` carrying only the client prompt returns 429 on the OAuth
        path.

        A probe that fell into that reported the whole curated list as "unverified" for a
        reason of ours, not of the account.
        """
        captured: list[object] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.read())["system"])
            return httpx.Response(200, json={})

        async with client(handler) as http:
            await discover(ANTHROPIC, client=http)

        first = captured[0]
        assert isinstance(first, list)
        assert "Claude Code" in first[0]["text"]

    async def test_probe_carries_the_subscription_credential(self) -> None:
        """Without the token the probe measures the anonymous rejection, not what the account
        serves."""
        tokens: set[str] = set()

        def handler(request: httpx.Request) -> httpx.Response:
            tokens.add(request.headers.get("authorization", ""))
            return httpx.Response(200, json={})

        async with client(handler) as http:
            await discover(ANTHROPIC, client=http)

        assert tokens == {"Bearer tok-a"}


class TestSuggestedName:
    """The public name that goes into the deployment's `model_name`."""

    def test_provider_prefix_is_stripped(self) -> None:
        """The prefix belongs in `litellm_params["model"]`, never in `model_name`.

        It is `model_name` that echoes in the spend log. An `anthropic/claude-opus-5` there
        names something no client asked for, and `registry.is_declared` compares it against
        `model_info["id"]` — one extra prefix made the managed entry look declared.
        """
        assert suggested_name("anthropic/claude-opus-5") == "claude-opus-5"

    def test_bare_name_survives_intact(self) -> None:
        """The normal case: the catalogue's wire name already comes bare and must not be
        touched."""
        assert suggested_name("gemini-3.1-pro") == "gemini-3.1-pro"
