"""Connection reopening policy, separated from the transport.

In the original, the decision of what to do with a 401, a 400 or a 429 was embedded inside
the ``httpx`` loops, duplicated between the synchronous and the asynchronous version — and
the two had drifted into different shapes of the same rule. Here the decision is a pure
function over ``(status, body)``, and the transport merely executes it.

That makes the part that matters — *when* a retry happens and why — testable without
opening connections.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final


class Action(Enum):
    """What to do next."""

    RETURN = "return"
    """Good response: deliver it."""

    REFRESH_TOKEN = "refresh_token"
    """Credential rejected: re-read it and try again with the new one."""

    REMAP_MODEL = "remap_model"
    """The account does not serve this name; if it is a known alias, reroute."""

    REDEEM_CREDIT = "redeem_credit"
    """Quota exhausted and there is unused reset credit."""

    FAIL = "fail"
    """Nothing to do: propagate the upstream error."""

    ABORT = "abort"
    """Propagate at once: trying another endpoint cannot change the answer."""


@dataclass(frozen=True, slots=True)
class Decision:
    action: Action
    reason: str = ""

    @property
    def should_retry(self) -> bool:
        return self.action in (Action.REFRESH_TOKEN, Action.REMAP_MODEL, Action.REDEEM_CREDIT)


#: Open attempts. Three is enough to refresh the token and remap the model once each.
MAX_ATTEMPTS: Final = 3

#: Marker of the ChatGPT account refusing a model.
UNSUPPORTED_MARKER: Final = "is not supported when using Codex"


def is_unsupported_model(body: str) -> bool:
    return UNSUPPORTED_MARKER in str(body)


def decide_codex(
    status: int,
    body: str = "",
    *,
    can_remap: bool = False,
    can_redeem: bool = False,
) -> Decision:
    """What to do with the Codex open response.

    ``can_remap`` and ``can_redeem`` are capabilities of the caller, not of global state:
    with no known alias and no credit, the decision has to be to fail — retrying the same
    request would give the same error three times and triple the latency before saying so.
    """
    if status == 200:
        return Decision(Action.RETURN)
    if status == 401:
        return Decision(Action.REFRESH_TOKEN, "credential rejected")
    if status == 400 and is_unsupported_model(body):
        if can_remap:
            return Decision(Action.REMAP_MODEL, "known alias of a served model")
        # An arbitrary refused name is the correct answer: substituting another model for
        # it returned 200 with the `model` field echoing the request, and billing started
        # to lie.
        return Decision(Action.FAIL, "the account does not serve this model")
    if status == 429 and can_redeem:
        return Decision(Action.REDEEM_CREDIT, "quota exhausted, reset credit available")
    return Decision(Action.FAIL, f"HTTP {status}")


def decide_antigravity(status: int) -> Decision:
    """What to do with the Antigravity open response.

    Failover is on the *endpoint* only: a 404 or a 503 does not authorise answering with
    another model. A 404 is "this account does not serve this model" and a 503 is
    capacity; in either case the next host is tried and, once exhausted, the error
    propagates.

    A 429 is the exception, and it is not an endpoint problem: both hosts front the same
    account and the same quota, so asking the second one repeats a refusal that is already
    known. Measured against the real backend, the rotation turned an ~11 s failure into
    ~22 s and changed nothing else. `ABORT` propagates it at the first host.
    """
    if status == 200:
        return Decision(Action.RETURN)
    if status == 401:
        return Decision(Action.REFRESH_TOKEN, "credential rejected")
    if status == 429:
        return Decision(Action.ABORT, "quota exhausted: the other host serves the same account")
    return Decision(Action.FAIL, f"HTTP {status}")
