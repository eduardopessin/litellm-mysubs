"""Single-use pairing codes between the `/mysubs` UI and `mysubs-login`.

The problem is concrete: the providers' OAuth flow requires a loopback port, and the
loopback that counts is the user's machine, not the container the proxy runs in.
`mysubs-login` runs on the workstation, captures the callback, and then has to deposit the
credential into a remote proxy. For that it needs authorisation.

The alternative was the user copying the proxy's administrator key onto the command line.
That trades a short-lived secret (one provider's credential) for a long-lived one with
total reach (the key that opens the whole proxy), and leaves it in the shell history. A
pairing code inverts the relation: it is worth a few minutes, it is worth one use, and it
is worth only for the provider the UI button picked.
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass
from typing import Final

from ..credentials.store import ProviderId

#: The code's lifetime window. Ten minutes is the time it takes to go from the UI to the
#: terminal and paste; longer than that is a code sleeping in the scrollback waiting for
#: whoever reads the screen.
DEFAULT_TTL_S: float = 600.0

#: Alphabet without the pairs that get confused on a terminal: neither `0`/`O` nor
#: `1`/`l`/`I`. Whoever reads the code off a screen and types it by hand gets these wrong
#: and no others, and a transcription error costs a trip back to the UI to press the
#: button again.
#:
#: That leaves 32 symbols, and 32 is a power of two on purpose: each symbol is worth
#: exactly 5 bits, with no waste from rounding.
_ALPHABET: Final = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

#: Three groups of four, `XXXX-XXXX-XXXX`. The groups exist for whoever copies by hand: a
#: block of twelve characters gets lost halfway, three blocks of four do not.
#:
#: The entropy arithmetic: 32 symbols = 5 bits each, 12 symbols = **60 bits**. Against a
#: remote guesser that is ample — even at a million attempts per second, exhausting the
#: ten-minute window covers less than 2^-30 of the space.
_GROUPS: Final = 3
_GROUP_LEN: Final = 4


class PairingError(RuntimeError):
    """Invalid code: unknown, already used, or out of date."""


@dataclass(frozen=True, slots=True)
class Pairing:
    """An issued code and what it authorises.

    The `provider` is decided here, at the button that issued the code, and travels with
    it. The redeeming client neither picks it nor can contradict it — that is the
    difference between pairing one concrete login session and accepting any write from
    whoever holds a code.
    """

    code: str
    provider: ProviderId
    expires_at: float


def _normalize(code: str) -> bytes:
    """What the user pasted, reduced to the comparable canonical form.

    Two tolerances, both because of how the code travels: copy-and-paste from a terminal
    drags surrounding spaces and line breaks, and whoever types it by hand types it in the
    case their keyboard is in. Neither of those is a different code, and refusing them
    sends the user back to the UI for a reason they cannot see.

    It comes out as ASCII bytes because that is what `compare_digest` accepts without
    complaining: given a `str` with a non-ASCII character — an accent caught from a
    keyboard, an em dash a mail client substituted for the hyphen — it raises `TypeError`,
    and a clumsy paste has to be an invalid code, not a 500 in the UI.
    """
    return code.strip().upper().encode("ascii", "replace")


def _generate() -> str:
    """A new code. `secrets`, not `random`: `random` is a seeded, observable Mersenne
    Twister, and what is generated here is an authorisation."""
    groups = (
        "".join(secrets.choice(_ALPHABET) for _ in range(_GROUP_LEN)) for _ in range(_GROUPS)
    )
    return "-".join(groups)


class PairingRegistry:
    """The live codes of this proxy.

    In memory on purpose: a code that survives a proxy restart is a code that survives the
    terminal it was shown in. Losing the pending ones on a restart costs a click;
    persisting them costs a surface.
    """

    def __init__(self, *, ttl_s: float = DEFAULT_TTL_S) -> None:
        self._ttl_s = ttl_s
        # A list, not `dict[code]`: see `redeem`. Order does not matter, the number of live
        # entries is in the single digits, and the linear scan is what makes the comparison
        # independent of the code presented.
        self._pairings: list[Pairing] = []
        # The UI is served by an async proxy with a thread pool; issuing and redeeming can
        # land on different threads and both mutate the list.
        self._lock = threading.Lock()

    def issue(self, provider: ProviderId) -> Pairing:
        """Issues a code for this provider."""
        now = time.time()
        pairing = Pairing(code=_generate(), provider=provider, expires_at=now + self._ttl_s)
        with self._lock:
            # Sweeping here is what keeps the registry bounded. Without it, a long-lived
            # proxy accumulates one entry per button pressed and never redeemed — nobody
            # clears them, because the only other way out is the `redeem` that never
            # arrives.
            self._purge_locked(now)
            self._pairings.append(pairing)
        return pairing

    def redeem(self, code: str, *, now: float | None = None) -> Pairing:
        """Consumes the code and returns what it authorises.

        `now` is injectable so the test can cross the TTL without sleeping.
        """
        moment = time.time() if now is None else now
        candidate = _normalize(code)

        with self._lock:
            # Full scan, no `break`, and `compare_digest` instead of `==`. In a
            # `dict[code]` — or in an `==` that exits at the first differing character —
            # the duration of the failure grows with the prefix matched, and that is an
            # oracle that can be queried from the network.
            #
            # The surface here is small: TTL of minutes, single use, few entries. That is
            # not why it is left open — closing it costs one line and costs nothing else.
            found: Pairing | None = None
            for pairing in self._pairings:
                if secrets.compare_digest(_normalize(pairing.code), candidate):
                    found = pairing
            if found is None:
                raise PairingError(
                    "Unknown pairing code. Check that you copied it whole from the "
                    "subscriptions page."
                )
            # It leaves the registry before it is known whether it was still good. This is
            # what makes it single use: the code was visible on the terminal and in the
            # scrollback, and what stops it being replayed is it no longer being here.
            self._pairings.remove(found)

        if moment >= found.expires_at:
            raise PairingError(
                "The pairing code has expired. Press 'Issue code' again on the "
                "subscriptions page to get a new one."
            )
        return found

    def purge(self, *, now: float | None = None) -> int:
        """Discards the expired ones. Returns how many left."""
        moment = time.time() if now is None else now
        with self._lock:
            return self._purge_locked(moment)

    def _purge_locked(self, now: float) -> int:
        alive = [p for p in self._pairings if now < p.expires_at]
        removed = len(self._pairings) - len(alive)
        self._pairings = alive
        return removed
