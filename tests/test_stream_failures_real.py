"""How a failed turn reaches the client and the spend log, through the real proxy.

Three failures omp settles and this plugin got wrong, each judged on all three routes and
both delivery modes, the way a client and a spend log see them:

- **A stream that ends without a finish.** omp's providers throw when the upstream stops
  sending before its terminal event (Cloud Code: no ``finishReason``; Codex: no
  ``response.completed``), and its gateway answers 502 ``upstream_error``. Here the
  Antigravity turn was delivered as a finished answer, and the Codex one as a 500.
- **The usage of a failed turn.** omp's gateway records a turn's usage before looking at
  its stop reason, so a Gemini turn stopped for SAFETY still bills the 165 tokens the
  upstream reported. Here it was billed at zero, or at LiteLLM's token-count guess.
- **An upstream 401** the transport's refresh-once could not cure. omp classifies it as
  ``authentication_error``; here it reached the client as a 500.

Everything between the client and the subscription is real: the proxy app, a
`litellm.Router` with the plugin installed and both routes bound, LiteLLM's own failure
logging. Only the upstream HTTP is faked.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from typing import Any

import httpx
import litellm
import litellm.proxy.proxy_server as proxy_server
import openai
import pytest
from litellm.integrations.custom_logger import CustomLogger

from litellm_mysubs import plugin
from litellm_mysubs.transport.client import UpstreamError
from tests.test_plugin import FakeTransport, codex_events, install_transport

CODEX = "mysubs/codex/gpt-5.5"
ANTIGRAVITY = "mysubs/antigravity/gemini-2.5-pro"
ANTIGRAVITY_WIRE = "gemini/gemini-2.5-pro"
MESSAGES = [{"role": "user", "content": "hi"}]
USAGE = {"promptTokenCount": 120, "candidatesTokenCount": 45, "totalTokenCount": 165}

ROUTES = ["chat", "messages", "responses"]


def antigravity_events(finish: str | None) -> list[dict[str, Any]]:
    """Text over two events; the second carries the usage and, when given, the finish."""
    closing: dict[str, Any] = {"content": {"role": "model", "parts": [{"text": "answer"}]}}
    if finish:
        closing["finishReason"] = finish
    return [
        {"response": {"candidates": [{"content": {"parts": [{"text": "Part "}]}}]}},
        {"response": {"candidates": [closing], "usageMetadata": USAGE}},
    ]


class Failures(CustomLogger):
    """What LiteLLM's failure and success paths record for each call."""

    def __init__(self) -> None:
        super().__init__()
        #: ``request_data`` of the proxy's post-call failure hook: what the key is billed.
        self.billed: list[dict[str, Any]] = []
        #: ``standard_logging_object`` of the failure event: what a log row reads.
        self.failure_rows: list[dict[str, Any]] = []
        self.success_rows: list[dict[str, Any]] = []
        self.exceptions: list[BaseException] = []

    def reset(self) -> None:
        for records in (self.billed, self.failure_rows, self.success_rows, self.exceptions):
            records.clear()

    async def async_post_call_failure_hook(
        self,
        request_data: dict[str, Any],
        original_exception: Exception,
        user_api_key_dict: Any,
        traceback_str: str | None = None,
    ) -> None:
        self.billed.append(dict(request_data))
        self.exceptions.append(original_exception)

    async def async_log_failure_event(
        self, kwargs: Any, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        self.failure_rows.append(kwargs.get("standard_logging_object") or {})

    async def async_log_success_event(
        self, kwargs: Any, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        self.success_rows.append(kwargs.get("standard_logging_object") or {})

    async def settle(self) -> None:
        """Logging runs off the request path."""
        for _ in range(50):
            if self.billed:
                break
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.1)


#: One recorder for the whole module: LiteLLM registers a callback once per instance.
RECORDER = Failures()


@pytest.fixture(autouse=True)
def proxy(monkeypatch: pytest.MonkeyPatch) -> Iterable[Failures]:
    plugin.uninstall()
    router = litellm.Router(
        model_list=[
            {
                "model_name": CODEX,
                "litellm_params": {"model": "openai/gpt-5.5"},
                "model_info": {"id": CODEX, "mysubs_provider": "openai-codex"},
            },
            {
                "model_name": ANTIGRAVITY,
                "litellm_params": {"model": ANTIGRAVITY_WIRE},
                "model_info": {"id": ANTIGRAVITY, "mysubs_provider": "google-antigravity"},
            },
        ]
    )
    plugin.install()
    assert plugin.bind_messages_route(router)
    assert plugin.bind_responses_route(router)
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "master_key", None)
    recorder = RECORDER
    recorder.reset()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    yield recorder
    plugin.uninstall()


def http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy_server.app), base_url="http://proxy"
    )


async def call(route: str, model: str, *, stream: bool) -> httpx.Response:
    """One request on ``route``, the body read whole."""
    if route == "chat":
        path, body = "/v1/chat/completions", {"model": model, "messages": MESSAGES}
    elif route == "messages":
        path, body = "/v1/messages", {"model": model, "max_tokens": 64, "messages": MESSAGES}
    else:
        path, body = "/v1/responses", {"model": model, "input": "hi"}
    async with http_client() as client:
        response = await client.post(path, json={**body, "stream": stream})
    return response


def frames(body: str) -> list[dict[str, Any]]:
    """The JSON payloads of an SSE body."""
    out = []
    for frame in body.replace("\r\n", "\n").split("\n\n"):
        data = [line[5:].strip() for line in frame.split("\n") if line.startswith("data:")]
        if data and data[0] != "[DONE]":
            out.append(json.loads("\n".join(data)))
    return out


def error_of(response: httpx.Response, *, stream: bool) -> tuple[int | None, str]:
    """``(status, message)`` of the failure the client reads, streamed or not.

    A stream that already answered 200 carries the status in LiteLLM's own error frame.
    On ``/v1/responses`` LiteLLM 1.103 no longer appends one after omp's
    ``response.failed``, which carries the message alone: the status is ``None`` there.
    """
    if not stream:
        payload = response.json()
        error = payload.get("error") or {}
        return response.status_code, str(error.get("message"))
    payloads = frames(response.text)
    for payload in reversed(payloads):
        error = payload.get("error")
        if isinstance(error, dict) and error.get("code") is not None:
            return int(error["code"]), str(error.get("message"))
    for payload in payloads:
        if payload.get("type") == "response.failed":
            return None, str(payload["response"]["error"]["message"])
    raise AssertionError(f"no error frame in {response.text!r}")


def assert_status(status: int | None, expected: int, *, route: str, stream: bool) -> None:
    if status is None:
        assert (route, stream) == ("responses", True), "only omp's response.failed has none"
        return
    assert status == expected


def wire_cost(prompt: int, completion: int) -> float:
    info = litellm.get_model_info(ANTIGRAVITY_WIRE)
    return prompt * info["input_cost_per_token"] + completion * info["output_cost_per_token"]


@pytest.mark.parametrize("stream", [False, True], ids=["whole", "stream"])
@pytest.mark.parametrize("route", ROUTES)
class TestAStreamWithoutAFinishFails:
    async def test_antigravity(self, route: str, stream: bool) -> None:
        """omp: "Cloud Code Assist stream ended without a finish reason". Before: the
        text that did arrive was delivered as a finished answer, HTTP 200 and a clean
        stop, on every route."""
        install_transport(FakeTransport(antigravity_events(None)))

        response = await call(route, ANTIGRAVITY, stream=stream)

        status, message = error_of(response, stream=stream)
        assert_status(status, 502, route=route, stream=stream)
        assert "ended without a finish reason" in message

    async def test_codex(self, route: str, stream: bool) -> None:
        """omp: "Codex stream ended before terminal completion event". Before: 500
        ``internal_server_error`` — the proxy blamed itself for the upstream cutting off."""
        install_transport(FakeTransport(codex_events(text="half ")[:-1]))

        response = await call(route, CODEX, stream=stream)

        status, message = error_of(response, stream=stream)
        assert_status(status, 502, route=route, stream=stream)
        assert "terminal completion event" in message


@pytest.mark.parametrize("stream", [False, True], ids=["whole", "stream"])
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("finish", ["SAFETY", None], ids=["blocked", "cut"])
async def test_a_failed_turn_bills_what_the_upstream_reported(
    proxy: Failures, route: str, stream: bool, finish: str | None
) -> None:
    """The key is billed the 120 + 45 tokens the upstream reported, at the wire rate.

    Before: zero on every route, and on the chat stream LiteLLM's guess from the text
    that went out (8 + 2 tokens)."""
    install_transport(FakeTransport(antigravity_events(finish)))

    await call(route, ANTIGRAVITY, stream=stream)
    await proxy.settle()

    assert len(proxy.billed) == 1
    usage = proxy.billed[0].get("combined_usage_object")
    assert isinstance(usage, litellm.Usage)
    assert (usage.prompt_tokens, usage.completion_tokens) == (120, 45)
    assert proxy.billed[0]["response_cost"] == pytest.approx(wire_cost(120, 45))
    assert proxy.success_rows == [], "a failed turn is not also a success"
    for row in proxy.failure_rows:
        assert (row["prompt_tokens"], row["completion_tokens"]) == (120, 45)
        assert row["response_cost"] == pytest.approx(wire_cost(120, 45))


@pytest.mark.parametrize("route", ROUTES)
async def test_a_failure_before_any_usage_bills_nothing_invented(
    proxy: Failures, route: str
) -> None:
    """An in-band error on the first event carries no usage: none is made up."""
    install_transport(FakeTransport([{"error": {"code": 500, "message": "backend down"}}]))

    response = await call(route, ANTIGRAVITY, stream=False)
    await proxy.settle()

    assert response.status_code == 500
    assert proxy.billed and proxy.billed[0].get("combined_usage_object") is None


class TestAnUpstream401IsAnAuthenticationError:
    """The transport refreshes once on a 401; what survives that is the client's answer."""

    @pytest.mark.parametrize("model", [CODEX, ANTIGRAVITY])
    @pytest.mark.parametrize("route", ROUTES)
    async def test_whole(self, proxy: Failures, route: str, model: str) -> None:
        """Before: 500 ``internal_server_error``, logged as `UpstreamError`."""
        install_transport(FakeTransport(error=UpstreamError(401, '{"error": "invalid token"}')))

        response = await call(route, model, stream=False)
        await proxy.settle()

        assert response.status_code == 401
        assert "invalid token" in response.text
        assert isinstance(proxy.exceptions[0], litellm.exceptions.AuthenticationError)

    async def test_the_chat_stream_answers_401_before_it_opens(self, proxy: Failures) -> None:
        """Before: 500, with a Python traceback as the message."""
        install_transport(FakeTransport(error=UpstreamError(401, '{"error": "invalid token"}')))
        client = openai.AsyncOpenAI(
            api_key="unused", base_url="http://proxy/v1", http_client=http_client(), max_retries=0
        )

        with pytest.raises(openai.AuthenticationError) as caught:
            stream = await client.chat.completions.create(
                model=CODEX, messages=MESSAGES, stream=True
            )
            async for _ in stream:
                pass
        await client.close()

        assert caught.value.status_code == 401

    @pytest.mark.parametrize("route", ["messages", "responses"])
    async def test_a_stream_already_open_says_401(self, route: str) -> None:
        """The Messages and Responses streams answer 200 before the upstream is asked, as
        omp's do; the failure frame is what carries the status."""
        install_transport(FakeTransport(error=UpstreamError(401, '{"error": "invalid token"}')))

        response = await call(route, CODEX, stream=True)

        status, message = error_of(response, stream=True)
        assert_status(status, 401, route=route, stream=True)
        assert "invalid token" in message


async def test_an_in_band_401_is_an_authentication_error_too() -> None:
    """omp raises the in-band code as the status (`GeminiCliApiError`)."""
    install_transport(
        FakeTransport([{"error": {"code": 401, "message": "Request had invalid credentials"}}])
    )

    response = await call("chat", ANTIGRAVITY, stream=False)

    assert response.status_code == 401


class TestAnUpstreamStatusReachesTheClient:
    """omp's gateway answers an upstream HTTP error with the upstream's own status."""

    @pytest.mark.parametrize(
        ("status", "body"),
        [
            (503, '{"error": {"code": 503, "message": "No capacity available for model"}}'),
            (400, '{"error": {"message": "Invalid value for tool_choice"}}'),
            (403, '{"error": {"message": "The caller does not have permission"}}'),
            (504, "upstream timeout"),
        ],
    )
    @pytest.mark.parametrize("route", ROUTES)
    async def test_whole(self, route: str, status: int, body: str) -> None:
        """Measured live, 0.1.16 and the candidate: Cloud Code's 503 went out as 500
        ``internal_server_error``."""
        install_transport(FakeTransport(error=UpstreamError(status, body)))

        response = await call(route, ANTIGRAVITY, stream=False)

        assert response.status_code == status
        assert "HTTP " + str(status) in response.text

    async def test_the_chat_stream(self) -> None:
        install_transport(FakeTransport(error=UpstreamError(503, "No capacity available")))

        response = await call("chat", ANTIGRAVITY, stream=True)

        # Refused before the first event, so the proxy still answers with a status.
        assert response.status_code == 503
        assert "No capacity available" in response.json()["error"]["message"]
