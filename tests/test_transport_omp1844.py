"""omp 18.4.4 transport, seen from the client: retries, replays and failover end to end.

`test_transport_client` pins the policy on the transport alone. Here the request goes
through the real proxy app, a real `litellm.Router`, the real `Transport` and the OpenAI
SDK; only the subscription hosts behind it are fake. What is judged is what the client
reads and what the upstream received — a transient refusal the client never sees, a
response replayed without a single duplicated token, a refusal the other host is not
asked to repeat.
"""

from __future__ import annotations

import json
import time
from typing import Any, Final

import httpx
import openai
import pytest

from litellm_mysubs.credentials.store import Credential
from litellm_mysubs.transport.hosts import HOSTS
from tests.test_gaps_refresh import (
    ANTIGRAVITY,
    CODEX,
    STREAMING,
    Hosts,
    OwningStore,
    ask,
    proxy,  # noqa: F401 — the autouse fixture that installs the proxy and its Router
    serve,
    sse,
    valid,
)
from tests.test_plugin import codex_events, gemini_events

EMPTY_STOP: Final = {"response": {"candidates": [{"finishReason": "STOP"}]}}


def antigravity() -> Credential:
    return Credential(
        provider="google-antigravity",
        access_token="AT-g",
        refresh_token="RT",
        expires_at=time.time() + 3600,
        project_id="p",
    )


class Script:
    """Inference replies in order, and every URL the transport asked."""

    def __init__(self, *replies: httpx.Response) -> None:
        self.replies = list(replies)
        self.urls: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.urls.append(str(request.url))
        return self.replies.pop(0)


def cut(*events: dict[str, Any]) -> httpx.Response:
    """A 200 whose body ends after ``events``, before the response is complete."""
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


@pytest.mark.parametrize("stream", STREAMING)
class TestCodex:
    async def test_a_transient_refusal_never_reaches_the_client(
        self, stream: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Before, a 503 was the client's answer. omp re-sends it, here at once because the
        server said so (``retry-after-ms: 0``)."""
        script = Script(
            httpx.Response(503, text="busy", headers={"retry-after-ms": "0"}),
            sse(codex_events(text="answer")),
        )
        serve(Hosts(script), OwningStore({"openai-codex": valid()}), monkeypatch)

        assert await ask(CODEX, stream=stream) == "answer"
        assert len(script.urls) == 2

    async def test_a_response_cut_before_its_content_is_replayed_once(
        self, stream: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Before, the cut response was the answer: a stream error. omp sends the request
        again because nothing had reached the client — and the client reads the second
        answer once, with nothing of the first."""
        script = Script(
            cut({"type": "response.created", "response": {"id": "first"}}),
            sse(codex_events(text="answer")),
        )
        serve(Hosts(script), OwningStore({"openai-codex": valid()}), monkeypatch)

        assert await ask(CODEX, stream=stream) == "answer"
        assert len(script.urls) == 2


@pytest.mark.parametrize("stream", STREAMING)
class TestAntigravity:
    async def test_an_empty_answer_is_asked_again(
        self, stream: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Before, a ``STOP`` with nothing in it reached the client as an empty answer.
        omp's Cloud Code retry sends it again on the same host."""
        script = Script(cut(EMPTY_STOP), sse(gemini_events(text="answer")))
        serve(Hosts(script), OwningStore({"google-antigravity": antigravity()}), monkeypatch)

        assert await ask(ANTIGRAVITY, stream=stream) == "answer"
        assert [url.split("/v1internal")[0] for url in script.urls] == [HOSTS[0], HOSTS[0]]

    async def test_a_refused_model_is_not_asked_of_the_other_host(
        self, stream: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Before, a 404 went on to the sandbox host, which refuses the same request the
        same way. omp moves hosts only for a transient failure."""
        script = Script(httpx.Response(404, text="Requested entity was not found."))
        serve(Hosts(script), OwningStore({"google-antigravity": antigravity()}), monkeypatch)

        with pytest.raises(openai.APIError, match="Requested entity was not found"):
            await ask(ANTIGRAVITY, stream=stream)

        assert len(script.urls) == 1
        assert script.urls[0].startswith(HOSTS[0])
