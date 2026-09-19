"""Pairing codes between the `/mysubs` UI and `mysubs-login`."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from litellm_mysubs.ui import pairing as pairing_mod
from litellm_mysubs.ui.pairing import PairingError, PairingRegistry


class FakeClock:
    """Controlled clock. The TTL is measured in minutes; a test that crossed it by
    sleeping would be minutes of suite time per assertion."""

    def __init__(self, start: float = 1_800_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeClock]:
    """Pin the clock the registry uses to timestamp issued codes."""
    fake = FakeClock()
    monkeypatch.setattr(pairing_mod.time, "time", fake)
    yield fake


class TestRedeem:
    def test_the_issued_code_redeems_and_carries_the_provider_the_button_chose(self) -> None:
        """The provider travels with the code. That is what stops the client from
        choosing where the credential lands."""
        registry = PairingRegistry()
        issued = registry.issue("openai-codex")

        redeemed = registry.redeem(issued.code)

        assert redeemed.provider == "openai-codex"
        assert redeemed.code == issued.code

    def test_the_second_redeem_of_the_same_code_fails(self) -> None:
        """Single use: the code stays in the terminal scrollback, and what prevents a
        replay is no longer being in the registry."""
        registry = PairingRegistry()
        issued = registry.issue("anthropic")
        registry.redeem(issued.code)

        with pytest.raises(PairingError):
            registry.redeem(issued.code)

    def test_two_consecutive_codes_differ(self) -> None:
        registry = PairingRegistry()

        first = registry.issue("anthropic")
        second = registry.issue("anthropic")

        assert first.code != second.code

    def test_an_unknown_code_fails(self) -> None:
        registry = PairingRegistry()
        registry.issue("anthropic")

        with pytest.raises(PairingError):
            registry.redeem("ZZZZ-ZZZZ-ZZZZ")


class TestExpiry:
    def test_the_expired_code_fails_and_the_message_says_to_press_again(self) -> None:
        registry = PairingRegistry(ttl_s=600.0)
        issued = registry.issue("google-antigravity")

        with pytest.raises(PairingError) as error:
            registry.redeem(issued.code, now=issued.expires_at + 1.0)

        assert "expired" in str(error.value)

    def test_expired_and_unknown_say_different_things(self) -> None:
        """Whoever reads the message decides between pressing the button again and
        re-reading what they pasted. The two failures demand different actions, so they
        cannot read the same."""
        registry = PairingRegistry(ttl_s=600.0)
        issued = registry.issue("anthropic")

        with pytest.raises(PairingError) as expired:
            registry.redeem(issued.code, now=issued.expires_at + 1.0)
        with pytest.raises(PairingError) as unknown:
            registry.redeem("ZZZZ-ZZZZ-ZZZZ")

        assert str(expired.value) != str(unknown.value)

    def test_within_the_deadline_it_still_redeems(self) -> None:
        registry = PairingRegistry(ttl_s=600.0)
        issued = registry.issue("anthropic")

        assert registry.redeem(issued.code, now=issued.expires_at - 1.0) is not None


class TestPurge:
    def test_purge_removes_only_the_expired_and_returns_the_count(self, clock: FakeClock) -> None:
        registry = PairingRegistry(ttl_s=600.0)
        old = registry.issue("anthropic")
        clock.advance(500.0)
        new = registry.issue("openai-codex")

        # An instant at which the first has already lapsed and the second has not.
        removed = registry.purge(now=old.expires_at + 1.0)

        assert removed == 1
        assert registry.redeem(new.code, now=old.expires_at + 1.0).provider == "openai-codex"

    def test_purge_with_nothing_to_remove_returns_zero(self, clock: FakeClock) -> None:
        registry = PairingRegistry(ttl_s=600.0)
        registry.issue("anthropic")

        assert registry.purge(now=clock.now) == 0

    def test_issue_sweeps_the_expired_so_the_registry_does_not_grow(self, clock: FakeClock) -> None:
        """A long-lived proxy accumulates one entry per button pressed and never
        redeemed; without the sweep inside `issue` nobody deletes them."""
        registry = PairingRegistry(ttl_s=600.0)
        abandoned = registry.issue("anthropic")
        clock.advance(601.0)

        registry.issue("anthropic")

        # It is no longer there to be swept a second time.
        assert registry.purge(now=clock.now) == 0
        with pytest.raises(PairingError):
            registry.redeem(abandoned.code, now=clock.now)


class TestPasting:
    def test_surrounding_whitespace_and_a_different_case_still_redeem(self) -> None:
        """Copy-and-paste from a terminal drags whitespace along; whoever types it by
        hand types in whatever case the keyboard is in."""
        registry = PairingRegistry()
        issued = registry.issue("anthropic")

        redeemed = registry.redeem(f"  {issued.code.lower()}\n")

        assert redeemed.provider == "anthropic"

    def test_a_paste_with_odd_characters_is_an_invalid_code_not_an_error(self) -> None:
        """A mail client swaps the hyphen for an em dash. That is a code that does not
        pair, not an exception the UI does not know how to catch."""
        registry = PairingRegistry()
        registry.issue("anthropic")

        with pytest.raises(PairingError):
            registry.redeem("ABCD—EFGH—JKLM")


class TestCode:
    def test_the_code_avoids_the_characters_that_look_alike_on_screen(self) -> None:
        """A transcription error costs a trip back to the UI. `0`/`O` and `1`/`l`/`I` are
        where it happens."""
        registry = PairingRegistry()

        for _ in range(50):
            code = registry.issue("anthropic").code
            assert set(code).isdisjoint({"0", "O", "1", "l", "I"})

    def test_the_code_has_the_shape_the_ui_shows(self) -> None:
        """Twelve symbols from a 32-character alphabet are the 60 bits of entropy the
        format promises; the groups are for whoever copies it by hand."""
        code = PairingRegistry().issue("anthropic").code

        groups = code.split("-")
        assert len(groups) == 3
        assert all(len(g) == 4 for g in groups)
