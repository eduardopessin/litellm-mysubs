"""Antigravity endpoint failover.

Two hosts accept the same envelope. The last one that answered **in full** is remembered,
so the first host's failure is not paid for on every request of a session.

Two invariants that come from the source and are easy to lose:

1. The last-good host is only committed after a **complete** stream — content and finish
   reason. Marking it on the first OK response pins the rotation to a host that accepted
   the connection and then cut the stream in half.
2. Failover is only legal while **nothing** has been emitted. After the first event the
   client has already seen part of the response, and restarting on another host would
   duplicate it. That is tracked per request, by the transport.
3. Only a transient failure moves to the other host — 408, 5xx, a network failure, a
   first-event timeout, an empty response. A 4xx is the request's own fault and the other
   host would refuse it the same way (`retry.decide_antigravity`).

Note established in production: when both fail, the error message names the **last** host
tried — which is what makes `sandbox` show up in errors without it being "stuck".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# omp: providers/google-gemini-cli.ts :: ANTIGRAVITY_ENDPOINT_FALLBACKS
HOSTS: Final[tuple[str, ...]] = (
    "https://daily-cloudcode-pa.googleapis.com",
    "https://daily-cloudcode-pa.sandbox.googleapis.com",
)

STREAM_PATH: Final = "/v1internal:streamGenerateContent?alt=sse"
MODELS_PATH: Final = "/v1internal:fetchAvailableModels"

# omp: providers/google-shared.ts :: MAX_EMPTY_STREAM_RETRIES
#: **Additional** attempts when the stream closes with no content at all.
MAX_EMPTY_RETRIES: Final = 2
# omp: providers/google-shared.ts :: EMPTY_STREAM_BASE_DELAY_MS
#: Exponential backoff base: 500 ms, 1 s.
EMPTY_RETRY_BASE_S: Final = 0.5


def empty_retry_delay(attempt: int) -> float:
    """Delay before attempt ``attempt`` (1-indexed): ``base * 2^(n-1)``."""
    return float(EMPTY_RETRY_BASE_S * (2 ** max(0, attempt - 1)))


# omp: providers/google-gemini-cli.ts :: DEFAULT_FIRST_EVENT_TIMEOUT_MS
#: Wait for the first event — response headers included. Five minutes is the allowance
#: omp keeps for a cold Pro reasoning start.
FIRST_EVENT_TIMEOUT_S: Final = 300.0
# omp: compat/rules/classes/gemini.kdl :: flash
#: ``stream-first-event-timeout-ms 60000`` on the Gemini flash family, on the Antigravity
#: and Gemini CLI hosts: flash does not inherit the Pro allowance.
FLASH_FIRST_EVENT_TIMEOUT_S: Final = 60.0
# omp: providers/google-gemini-cli.ts :: FIRST_EVENT_TIMEOUT_ERROR
FIRST_EVENT_TIMEOUT_ERROR: Final = (
    "Cloud Code Assist stream timed out while waiting for the first event"
)


def first_event_timeout(model: str) -> float:
    """The first-event watchdog for a wire model name.

    The catalog's flash family is ``flash`` without ``lite``: the lite line is a family of
    its own and keeps the default, as does every Claude and GPT model the host serves.
    """
    name = model.rsplit("/", 1)[-1].lower()
    if "flash" in name and "lite" not in name:
        return FLASH_FIRST_EVENT_TIMEOUT_S
    return FIRST_EVENT_TIMEOUT_S


# omp: providers/google-gemini-cli.ts :: lastGoodEndpoint
@dataclass(slots=True)
class HostRotation:
    """Endpoint try order, with memory of the last good one.

    An instance rather than a global: two clients in the same process must not share the
    memory of which host answered. What a single request has already emitted is not kept
    here: the rotation is shared by every request in flight, so a flag on it let one
    stream that had started forbid failover to another that had not.
    """

    hosts: tuple[str, ...] = HOSTS
    index: int = 0

    def urls(self, path: str = STREAM_PATH) -> list[str]:
        """Every endpoint, starting at the last good one."""
        count = len(self.hosts)
        return [self.hosts[(self.index + offset) % count] + path for offset in range(count)]

    def commit(self, url: str) -> None:
        """Remember the host **after** a complete stream.

        Must only be called once content and a finish reason have been received — not on
        the first response with status 200.
        """
        for position, host in enumerate(self.hosts):
            if url.startswith(host):
                self.index = position
                return

    @property
    def current(self) -> str:
        return self.hosts[self.index]
