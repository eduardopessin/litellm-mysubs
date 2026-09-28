"""System cache breakpoint as omp 18.4.1 places it, measured on LiteLLM's real wire body.

omp 18.4.1 (`providers/anthropic.ts`, #13104/#13556) anchors the system breakpoint on the
block right before the first `<project-context>` or `<memories>` segment and *moves* an
earlier system marker there instead of adding a second one. A client running 18.4.1
arrives with its own markers already placed that way; one running 18.3.2 arrives with its
system marker on `<project-context>`. Either way the request must leave here with at most
4 markers — Anthropic answers 400 "A maximum of 4 blocks with cache_control may be
provided. Found 5." to a fifth.

The markers are counted on the body LiteLLM actually sends, captured at its HTTP client:
the chat path through `litellm.completion` (the Anthropic chat transformation that hoists
system messages) and `/v1/messages` through `litellm.anthropic_messages`. A count taken on
our kwargs would miss what the transformation drops, merges or keeps.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from typing import Any

import httpx
import litellm
import pytest
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

from litellm_mysubs.catalog.discovery import discover
from litellm_mysubs.credentials.store import Credential
from litellm_mysubs.wire import anthropic as ant

MODEL = "claude-opus-5"
STATIC = " ".join(["You are omp, a coding agent."] * 40)
PROJECT = "<project-context>\n/home/me/worktree-a\n</project-context>"
MEMORIES = "<memories>\nyesterday you fixed the cache\n</memories>"
ROLE = "You are the reviewer subagent."

FIVE_MIN: dict[str, str] = {"type": "ephemeral"}
ONE_HOUR: dict[str, str] = {"type": "ephemeral", "ttl": "1h"}


class _SentError(Exception):
    """Raised by the patched HTTP client once the body is captured: nothing leaves."""


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    """The JSON body LiteLLM hands to its HTTP client, for either route."""
    body: dict[str, Any] = {}

    def capture(kwargs: dict[str, Any]) -> None:
        payload = kwargs.get("data") if kwargs.get("data") is not None else kwargs.get("json")
        if isinstance(payload, (bytes, str)):
            payload = json.loads(payload)
        assert isinstance(payload, dict)
        body.clear()
        body.update(payload)
        raise _SentError

    def sync_post(self: object, url: str, *args: object, **kwargs: Any) -> None:
        capture(kwargs)

    async def async_post(self: object, url: str, *args: object, **kwargs: Any) -> None:
        capture(kwargs)

    monkeypatch.setattr(HTTPHandler, "post", sync_post)
    monkeypatch.setattr(AsyncHTTPHandler, "post", async_post)
    yield body


def send_chat(wire: dict[str, Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """`build_request` on the chat route, then LiteLLM's own Anthropic chat transformation."""
    built = ant.build_request(copy.deepcopy(kwargs), MODEL, "tok")
    built.pop(ant.TOOL_ALIAS_KEY, None)
    with pytest.raises(Exception):  # noqa: B017 - LiteLLM re-wraps `_SentError`
        litellm.completion(model=f"anthropic/{MODEL}", **built)
    assert wire, "LiteLLM never reached its HTTP client"
    return wire


async def send_native(wire: dict[str, Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """`build_request` on `/v1/messages`, then `litellm.anthropic_messages`."""
    built = ant.build_request(copy.deepcopy(kwargs), MODEL, "tok", native_system=True)
    built.pop(ant.TOOL_ALIAS_KEY, None)
    with pytest.raises(Exception):  # noqa: B017 - LiteLLM re-wraps `_SentError`
        await litellm.anthropic_messages(model=f"anthropic/{MODEL}", **built)
    assert wire, "LiteLLM never reached its HTTP client"
    return wire


def markers(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Every `cache_control` on the wire, in Anthropic's order: tools, system, messages."""
    found: list[dict[str, Any]] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("cache_control"), dict):
                found.append(node["cache_control"])
            for key, value in node.items():
                if key != "cache_control":
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    for section in ("tools", "system", "messages"):
        walk(body.get(section))
    return found


def system_marks(body: dict[str, Any]) -> list[tuple[str, bool]]:
    return [(b["text"][:20], "cache_control" in b) for b in body["system"]]


def marked_system_texts(body: dict[str, Any]) -> list[str]:
    return [b["text"] for b in body["system"] if "cache_control" in b]


def assert_ttl_order(body: dict[str, Any]) -> None:
    """400 "a ttl='1h' cache_control block must not come after a ttl='5m' one"."""
    seen_short = False
    for control in markers(body):
        if control.get("ttl") == "1h":
            assert not seen_short, markers(body)
        else:
            seen_short = True


# -- /v1/messages -----------------------------------------------------------------


def native_tools(*, marked: bool) -> list[dict[str, Any]]:
    last: dict[str, Any] = {"name": "edit", "input_schema": {"type": "object"}}
    if marked:
        last["cache_control"] = dict(FIVE_MIN)
    return [{"name": "read", "input_schema": {"type": "object"}}, last]


def native_turns(marked: int, control: dict[str, str] = FIVE_MIN) -> list[dict[str, Any]]:
    """Alternating turns ending on a user one; the last ``marked`` user turns carry a marker."""
    turns: list[dict[str, Any]] = []
    for index in range(4):
        turns.append({"role": "user", "content": [{"type": "text", "text": f"q{index}"}]})
        turns.append({"role": "assistant", "content": [{"type": "text", "text": f"a{index}"}]})
    turns.append({"role": "user", "content": [{"type": "text", "text": "now"}]})
    users = [t for t in turns if t["role"] == "user"]
    for turn in users[len(users) - marked :] if marked else ():
        turn["content"][-1]["cache_control"] = dict(control)
    return turns


def native_system(*texts: str, marked: str | None = None) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for text in texts:
        block: dict[str, Any] = {"type": "text", "text": text}
        if text == marked:
            block["cache_control"] = dict(FIVE_MIN)
        blocks.append(block)
    return blocks


class TestNativeRoute:
    async def test_anchor_is_the_block_before_the_first_volatile_one(
        self, wire: dict[str, Any]
    ) -> None:
        """[identity, static, <project-context>, <memories>]: only static is marked, so the
        head is shared across working directories and a recall refresh."""
        body = await send_native(
            wire,
            {
                "system": native_system(STATIC, PROJECT, MEMORIES),
                "messages": native_turns(0),
            },
        )
        assert marked_system_texts(body) == [STATIC]
        assert len(markers(body)) <= ant.CACHE_BREAKPOINT_CEILING

    async def test_stable_block_after_a_volatile_one_stays_out_of_the_head(
        self, wire: dict[str, Any]
    ) -> None:
        """Prefix caching is positional: a role block behind `<project-context>` can never
        extend the cached head, so the anchor stays on static."""
        body = await send_native(
            wire,
            {"system": native_system(STATIC, PROJECT, ROLE), "messages": native_turns(0)},
        )
        assert marked_system_texts(body) == [STATIC]

    async def test_omp_1841_client_stays_at_four(self, wire: dict[str, Any]) -> None:
        """omp 18.4.1 already marks static, its last tool and two turns. v0.1.14 still saw
        `<project-context>` as stable and added a 5th on it."""
        body = await send_native(
            wire,
            {
                "tools": native_tools(marked=True),
                "system": native_system(STATIC, PROJECT, MEMORIES, marked=STATIC),
                "messages": native_turns(2),
            },
        )
        assert len(markers(body)) == ant.CACHE_BREAKPOINT_CEILING
        assert marked_system_texts(body) == [STATIC]

    async def test_omp_1832_marker_on_project_context_moves_to_static(
        self, wire: dict[str, Any]
    ) -> None:
        """omp 18.3.2 anchors the block before `<memories>`, i.e. `<project-context>`.
        The marker is relocated, not duplicated."""
        body = await send_native(
            wire,
            {
                "tools": native_tools(marked=True),
                "system": native_system(STATIC, PROJECT, MEMORIES, marked=PROJECT),
                "messages": native_turns(2),
            },
        )
        assert marked_system_texts(body) == [STATIC]
        assert len(markers(body)) == ant.CACHE_BREAKPOINT_CEILING

    async def test_four_markers_outside_system_get_no_fifth(self, wire: dict[str, Any]) -> None:
        """1 tool + 3 turn markers and none on system: v0.1.14 added the system one."""
        body = await send_native(
            wire,
            {
                "tools": native_tools(marked=True),
                "system": native_system(STATIC, PROJECT),
                "messages": native_turns(3),
            },
        )
        assert len(markers(body)) == ant.CACHE_BREAKPOINT_CEILING
        assert marked_system_texts(body) == []

    async def test_four_turn_markers_get_no_tool_marker(self, wire: dict[str, Any]) -> None:
        body = await send_native(
            wire,
            {
                "tools": native_tools(marked=False),
                "system": native_system(STATIC),
                "messages": native_turns(4),
            },
        )
        assert len(markers(body)) == ant.CACHE_BREAKPOINT_CEILING
        assert all("cache_control" not in tool for tool in body["tools"])
        assert marked_system_texts(body) == []

    async def test_last_slot_goes_to_system_not_tools(self, wire: dict[str, Any]) -> None:
        """The system anchor's prefix contains every tool; the tool anchor's does not
        contain the system."""
        body = await send_native(
            wire,
            {
                "tools": native_tools(marked=False),
                "system": native_system(STATIC, PROJECT),
                "messages": native_turns(3),
            },
        )
        assert len(markers(body)) == ant.CACHE_BREAKPOINT_CEILING
        assert marked_system_texts(body) == [STATIC]
        assert all("cache_control" not in tool for tool in body["tools"])

    async def test_short_ttl_client_gets_short_relocated_marker(
        self, wire: dict[str, Any]
    ) -> None:
        """A 5m client marker ahead of a 1h one of ours is a 400."""
        body = await send_native(
            wire,
            {
                "tools": native_tools(marked=False),
                "system": native_system(STATIC, PROJECT, MEMORIES, marked=PROJECT),
                "messages": native_turns(1),
            },
        )
        assert marked_system_texts(body) == [STATIC]
        assert all(control == FIVE_MIN for control in markers(body))
        assert_ttl_order(body)

    async def test_long_ttl_client_keeps_long_markers(self, wire: dict[str, Any]) -> None:
        system = native_system(STATIC, PROJECT, MEMORIES)
        system[1]["cache_control"] = dict(ONE_HOUR)
        body = await send_native(
            wire,
            {
                "tools": native_tools(marked=False),
                "system": system,
                "messages": native_turns(1, ONE_HOUR),
            },
        )
        assert marked_system_texts(body) == [STATIC]
        assert all(control == ONE_HOUR for control in markers(body))
        assert_ttl_order(body)

    async def test_client_system_opening_volatile_keeps_its_tail_marker(
        self, wire: dict[str, Any]
    ) -> None:
        """omp with a gateway key sends no identity, sees [<project-context>, static] as
        all-volatile and marks static. Our identity must not pull that marker onto itself."""
        body = await send_native(
            wire,
            {
                "system": native_system(PROJECT, STATIC, marked=STATIC),
                "messages": native_turns(0),
            },
        )
        assert marked_system_texts(body) == [STATIC]


# -- chat completions -------------------------------------------------------------


def chat_turns(marked: int) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for index in range(4):
        turns.append({"role": "user", "content": [{"type": "text", "text": f"q{index}"}]})
        turns.append({"role": "assistant", "content": f"a{index}"})
    turns.append({"role": "user", "content": [{"type": "text", "text": "now"}]})
    users = [t for t in turns if t["role"] == "user"]
    for turn in users[len(users) - marked :] if marked else ():
        turn["content"][-1]["cache_control"] = dict(FIVE_MIN)
    return turns


def chat_tools(*, marked: bool) -> list[dict[str, Any]]:
    def tool(name: str) -> dict[str, Any]:
        return {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}

    last = tool("edit")
    if marked:
        last["cache_control"] = dict(FIVE_MIN)
    return [tool("read"), last]


def chat_system(*texts: str, role: str = "system") -> list[dict[str, Any]]:
    """One message per system prompt: what omp's openai-completions provider sends."""
    return [{"role": role, "content": text} for text in texts]


class TestChatRoute:
    def test_each_system_message_is_its_own_block(self, wire: dict[str, Any]) -> None:
        """Joined into one block, `[static, <project-context>]` did not start with a tag
        and the marker cached the per-directory text."""
        body = send_chat(
            wire, {"messages": [*chat_system(STATIC, PROJECT, MEMORIES), *chat_turns(0)]}
        )
        assert [text for text, _ in system_marks(body)] == [
            ant.CLAUDE_CODE_PROMPT[:20],
            STATIC[:20],
            PROJECT[:20],
            MEMORIES[:20],
        ]
        assert marked_system_texts(body) == [STATIC]

    def test_role_block_after_project_context_is_not_the_anchor(
        self, wire: dict[str, Any]
    ) -> None:
        body = send_chat(wire, {"messages": [*chat_system(STATIC, PROJECT, ROLE), *chat_turns(0)]})
        assert marked_system_texts(body) == [STATIC]

    @pytest.mark.parametrize("opening", [PROJECT, MEMORIES])
    def test_volatile_first_message_does_not_anchor_the_identity_alone(
        self, wire: dict[str, Any], opening: str
    ) -> None:
        """The joined block opened with a tag, so the only stable block left was the
        identity line: tools plus one sentence cached."""
        body = send_chat(wire, {"messages": [*chat_system(opening, STATIC), *chat_turns(0)]})
        assert marked_system_texts(body) == [STATIC]

    def test_developer_messages_join_the_head(self, wire: dict[str, Any]) -> None:
        """LiteLLM rewrites `developer` to `system` after us; left in the messages, the head
        anchor only saw the identity."""
        body = send_chat(
            wire,
            {"messages": [*chat_system(STATIC, PROJECT, role="developer"), *chat_turns(0)]},
        )
        assert marked_system_texts(body) == [STATIC]
        assert len(body["system"]) == 3

    def test_omp_1841_client_stays_at_four(self, wire: dict[str, Any]) -> None:
        system = chat_system(STATIC, PROJECT, MEMORIES)
        system[0]["cache_control"] = dict(FIVE_MIN)
        body = send_chat(
            wire, {"tools": chat_tools(marked=True), "messages": [*system, *chat_turns(2)]}
        )
        assert len(markers(body)) == ant.CACHE_BREAKPOINT_CEILING
        assert marked_system_texts(body) == [STATIC]
        assert_ttl_order(body)

    def test_four_markers_outside_system_get_no_fifth(self, wire: dict[str, Any]) -> None:
        body = send_chat(
            wire,
            {
                "tools": chat_tools(marked=True),
                "messages": [*chat_system(STATIC, PROJECT), *chat_turns(3)],
            },
        )
        assert len(markers(body)) == ant.CACHE_BREAKPOINT_CEILING
        assert marked_system_texts(body) == []

    def test_identity_duplicated_by_the_client_is_dropped(self, wire: dict[str, Any]) -> None:
        body = send_chat(
            wire,
            {"messages": [*chat_system(ant.CLAUDE_CODE_PROMPT, STATIC), *chat_turns(0)]},
        )
        texts = [b["text"] for b in body["system"]]
        assert texts == [ant.CLAUDE_CODE_PROMPT, STATIC]


class TestHeadCacheUnit:
    def test_all_volatile_falls_back_to_the_tail(self) -> None:
        blocks: list[Any] = [{"type": "text", "text": PROJECT}, {"type": "text", "text": MEMORIES}]
        ant.apply_head_cache(blocks, None)
        assert [("cache_control" in b) for b in blocks] == [False, True]

    def test_marked_anchor_keeps_other_client_markers(self) -> None:
        """omp only relocates when the anchor is bare; an anchor already marked is left as
        the client placed everything."""
        blocks: list[Any] = [
            {"type": "text", "text": ant.CLAUDE_CODE_PROMPT, "cache_control": dict(FIVE_MIN)},
            {"type": "text", "text": STATIC, "cache_control": dict(FIVE_MIN)},
            {"type": "text", "text": PROJECT},
        ]
        assert ant.apply_head_cache(blocks, None, None) == 2
        assert [("cache_control" in b) for b in blocks] == [True, True, False]


class TestProbeBody:
    async def test_probe_system_is_the_identity_block_alone(self) -> None:
        """`build_system_blocks()` with no client blocks must keep the discovery probe's
        wire `system` byte-identical: the identity, nothing else."""
        captured: list[object] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.read())["system"])
            return httpx.Response(200, json={})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http:
            await discover(Credential(provider="anthropic", access_token="tok-a"), client=http)

        assert captured
        assert all(
            system == [{"type": "text", "text": ant.CLAUDE_CODE_PROMPT}] for system in captured
        )
