"""Retired model: upstream answers 200 with a notice instead of running the model.

Measured on the real account, against the proxy:

    gemini-3.5-flash-low        -> "Gemini 3.5 Flash is no longer available. Please switch
                                    to Gemini 3.7 Flash in the latest version of
                                    Antigravity."   finish_reason=stop, total_tokens=0
    gemini-3.5-flash-extra-low  -> the same notice, usage 0
    gemini-3.5-flash-lite       -> "2 + 2 = 4", total_tokens=12          <- alive

There is no HTTP error: accepted, the notice enters the conversation history as if the model
had answered. Hence the guard. And hence, too, the negative tests: `-lite` really answers
while `-low` and `-extra-low` are dead (just as `tab_flash_lite_preview` answers and
`tab_jump_flash_lite_preview` gives 400), so no rule can look at the name — what is tested
here is always the response.

No network: the transport is a double that returns the recorded events.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from typing import Any, Final

import pytest

from litellm_mysubs import plugin
from litellm_mysubs.credentials.store import Credential, CredentialStore, ProviderId
from litellm_mysubs.transport.client import RequestSpec, Response
from litellm_mysubs.wire import antigravity, antigravity_models
from litellm_mysubs.wire.antigravity import ModelRetiredError

#: The notice exactly as it came from upstream, copied from the measurement.
RETIREMENT_NOTICE = (
    "Gemini 3.5 Flash is no longer available. Please switch to Gemini 3.7 Flash in the "
    "latest version of Antigravity."
)

#: Usage of a response that really ran (`"2 + 2 = 4"` cost 12 tokens).
LIVE_USAGE: dict[str, Any] = {
    "promptTokenCount": 5,
    "candidatesTokenCount": 7,
    "totalTokenCount": 12,
}

#: Usage of a retired model: upstream ran no model at all.
DEAD_USAGE: dict[str, Any] = {"totalTokenCount": 0}


class FakeStore(CredentialStore):
    """In-memory store; never touches the disk or the environment."""

    owns_refresh = False

    def __init__(self) -> None:
        self._credentials: dict[ProviderId, Credential] = {
            "google-antigravity": Credential(
                provider="google-antigravity", access_token="tok-google", project_id="proj-1"
            )
        }

    def get(self, provider: ProviderId) -> Credential | None:
        return self._credentials.get(provider)

    def set(self, provider: ProviderId, credential: Credential) -> None:
        self._credentials[provider] = credential

    def delete(self, provider: ProviderId) -> None:
        self._credentials.pop(provider, None)

    def reload(self) -> bool:
        return False


class FakeTransport:
    """Double of `Transport`: returns the recorded events, no network."""

    def __init__(self, events: Iterable[dict[str, Any]]) -> None:
        self.events = list(events)

    async def request(self, spec: RequestSpec) -> Response:  # pragma: no cover - unused
        raise AssertionError("dispatch always uses stream()")

    async def stream(self, spec: RequestSpec) -> AsyncIterator[dict[str, Any]]:
        for event in self.events:
            yield event


def gemini_events(
    *, text: str, usage: dict[str, Any], finish: str = "STOP"
) -> list[dict[str, Any]]:
    """One `:streamGenerateContent` event in the measured shape: text, finishReason, usage."""
    parts = [{"text": text}] if text else []
    return [
        {
            "response": {
                "candidates": [{"content": {"parts": parts}, "finishReason": finish}],
                "usageMetadata": usage,
            }
        }
    ]


#: Catalogue of the real account, reduced to the names that matter here. Without it
#: `map_model` falls back to the static map, which does not know `gemini-3.5-flash-lite` —
#: and the live-model test died on name resolution, before reaching the guard under test.
ACCOUNT_MODELS: Final[tuple[str, ...]] = (
    "gemini-3.5-flash-low",
    "gemini-3.5-flash-extra-low",
    "gemini-3.5-flash-lite",
)


@pytest.fixture(autouse=True)
def _clean_plugin() -> Iterable[None]:
    plugin.uninstall()
    plugin._state.catalog = antigravity_models.ModelCatalog()
    plugin._state.catalog.update({"models": {n: {} for n in ACCOUNT_MODELS}})
    yield
    plugin.uninstall()
    plugin._state.catalog = antigravity_models.ModelCatalog()


def install(events: list[dict[str, Any]]) -> None:
    plugin.configure(store=FakeStore(), transport=FakeTransport(events))


async def complete(model: str = "gemini-3.5-flash-low") -> Any:
    """Non-streaming path."""
    return await plugin.dispatch(model=model, messages=[{"role": "user", "content": "2+2?"}])


async def stream(model: str = "gemini-3.5-flash-low") -> list[Any]:
    """Streaming path, consumed to the end."""
    wrapper = await plugin.dispatch(
        model=model, messages=[{"role": "user", "content": "2+2?"}], stream=True
    )
    return [chunk async for chunk in wrapper.completion_stream]


def content_of(response: Any) -> str:
    return str(response.choices[0].message.content or "")


class TestDetection:
    """The pure rule, without the reader around it."""

    def test_notice_with_zero_usage_is_a_retired_model(self) -> None:
        assert antigravity.is_retired_response(RETIREMENT_NOTICE, DEAD_USAGE)

    def test_the_same_notice_with_tokens_spent_is_a_real_answer(self) -> None:
        """An answer that *talks about* retired models cost tokens; the dead model's cost
        none. It is the usage that separates the two."""
        assert not antigravity.is_retired_response(RETIREMENT_NOTICE, LIVE_USAGE)

    def test_an_empty_answer_with_zero_usage_is_not_retirement(self) -> None:
        """`gemini-pro-agent` answered empty content to "oi" and is still alive."""
        assert not antigravity.is_retired_response("", DEAD_USAGE)
        assert not antigravity.is_retired_response("", {})
        assert not antigravity.is_retired_response("", None)

    def test_a_normal_answer_is_not_retirement(self) -> None:
        assert not antigravity.is_retired_response("2 + 2 = 4", LIVE_USAGE)

    @pytest.mark.parametrize(
        "text",
        [
            RETIREMENT_NOTICE,
            RETIREMENT_NOTICE.lower(),
            RETIREMENT_NOTICE.upper(),
            "gemini 3.5 flash IS NO LONGER AVAILABLE. please SWITCH TO gemini 3.7 flash.",
        ],
    )
    def test_the_case_of_the_notice_does_not_matter(self, text: str) -> None:
        """Upstream may change the case of the sentence between client versions; a
        case-sensitive `in` stopped catching the notice without anything visibly failing."""
        assert antigravity.is_retired_response(text, DEAD_USAGE)

    def test_half_the_marks_is_not_enough(self) -> None:
        """Both marks are required: "please switch to" shows up in legitimate answers
        about migrations, and on its own it proves no retirement at all."""
        assert not antigravity.is_retirement_notice("Please switch to the streaming API.")

    def test_any_counted_token_means_the_model_ran(self) -> None:
        """`totalTokenCount` absent but `candidatesTokenCount` present still counts —
        the CCA does not always send the total in the same event."""
        assert antigravity.usage_is_zero({"totalTokenCount": 0, "promptTokenCount": 0})
        assert not antigravity.usage_is_zero({"candidatesTokenCount": 7})
        assert not antigravity.usage_is_zero({"promptTokenCount": 5})


class TestNonStreaming:
    async def test_a_retired_model_raises_instead_of_answering(self) -> None:
        install(gemini_events(text=RETIREMENT_NOTICE, usage=DEAD_USAGE))
        with pytest.raises(ModelRetiredError) as excinfo:
            await complete()
        # The message has to quote upstream: it is upstream that says where to migrate.
        assert "Gemini 3.7" in str(excinfo.value)
        assert excinfo.value.notice == RETIREMENT_NOTICE

    async def test_the_error_is_distinguishable_from_a_transport_failure(self) -> None:
        """A network failure is worth retrying; a retired model never answers again.
        Whoever catches has to be able to separate the two cases."""
        install(gemini_events(text=RETIREMENT_NOTICE, usage=DEAD_USAGE))
        with pytest.raises(ModelRetiredError) as excinfo:
            await complete()
        assert not isinstance(excinfo.value, plugin.StreamError)
        assert excinfo.value.wire_model

    async def test_the_same_text_with_tokens_spent_comes_through(self) -> None:
        install(gemini_events(text=RETIREMENT_NOTICE, usage=LIVE_USAGE))
        response = await complete()
        assert content_of(response) == RETIREMENT_NOTICE

    async def test_an_empty_answer_with_zero_usage_comes_through(self) -> None:
        install(gemini_events(text="", usage=DEAD_USAGE))
        response = await complete()
        assert content_of(response) == ""

    async def test_a_live_model_answers(self) -> None:
        """`gemini-3.5-flash-lite` lives in the same family as the dead ones: if the guard
        looked at the name, this request would fail."""
        install(gemini_events(text="2 + 2 = 4", usage=LIVE_USAGE))
        response = await complete(model="gemini-3.5-flash-lite")
        assert content_of(response) == "2 + 2 = 4"
        assert response.usage.total_tokens == 12


class TestStreaming:
    async def test_a_retired_model_raises_before_emitting_the_notice(self) -> None:
        """In streaming the text goes out chunk by chunk: if the guard ran after emitting,
        the client already had the notice on screen and in history, and the exception
        arrived too late."""
        install(gemini_events(text=RETIREMENT_NOTICE, usage=DEAD_USAGE))
        wrapper = await plugin.dispatch(
            model="gemini-3.5-flash-low",
            messages=[{"role": "user", "content": "2+2?"}],
            stream=True,
        )
        seen: list[str] = []
        with pytest.raises(ModelRetiredError, match=r"Gemini 3\.7"):
            async for chunk in wrapper.completion_stream:
                seen.append(str(chunk.choices[0].delta.content or ""))
        assert "".join(seen) == "", f"the notice went out before the guard: {seen}"

    async def test_the_same_text_with_tokens_spent_streams_normally(self) -> None:
        install(gemini_events(text=RETIREMENT_NOTICE, usage=LIVE_USAGE))
        chunks = await stream()
        assert RETIREMENT_NOTICE in "".join(
            str(chunk.choices[0].delta.content or "") for chunk in chunks
        )

    async def test_a_live_model_streams(self) -> None:
        install(gemini_events(text="2 + 2 = 4", usage=LIVE_USAGE))
        chunks = await stream(model="gemini-3.5-flash-lite")
        assert "2 + 2 = 4" in "".join(
            str(chunk.choices[0].delta.content or "") for chunk in chunks
        )

    async def test_an_empty_answer_with_zero_usage_streams(self) -> None:
        install(gemini_events(text="", usage=DEAD_USAGE))
        chunks = await stream()
        assert "".join(str(chunk.choices[0].delta.content or "") for chunk in chunks) == ""


class TestSplitNotice:
    """The notice may arrive split over several events; neither half alone matches."""

    @staticmethod
    def split_events() -> list[dict[str, Any]]:
        head, tail = RETIREMENT_NOTICE.split(". ", 1)
        return [
            {"response": {"candidates": [{"content": {"parts": [{"text": head + ". "}]}}]}},
            {
                "response": {
                    "candidates": [
                        {"content": {"parts": [{"text": tail}]}, "finishReason": "STOP"}
                    ],
                    "usageMetadata": DEAD_USAGE,
                }
            },
        ]

    async def test_the_notice_is_caught_even_when_split(self) -> None:
        install(self.split_events())
        with pytest.raises(ModelRetiredError, match=r"Gemini 3\.7"):
            await stream()

    async def test_it_raises_once_per_response(self) -> None:
        """One exception per request, not one per chunk.

        The complete turn is seen in the event carrying the `finishReason` and again on
        `close`; if both raised, the same request produced repeated exceptions.
        """
        install(self.split_events())
        calls: list[str] = []
        original = antigravity.raise_if_retired

        def counting(text: object, usage: Any, wire_model: str) -> None:
            try:
                original(text, usage, wire_model)
            except ModelRetiredError:
                calls.append(wire_model)
                raise

        plugin.antigravity.raise_if_retired = counting  # type: ignore[assignment]
        try:
            with pytest.raises(ModelRetiredError, match=r"Gemini 3\.7"):
                await stream()
        finally:
            plugin.antigravity.raise_if_retired = original  # type: ignore[assignment]
        assert calls == ["gemini-3.5-flash-low"]
