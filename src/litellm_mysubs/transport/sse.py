"""Server-Sent Events reading.

Kept apart from the transport because this is the deceptive part: an event split in half, a
`[DONE]` with stray spaces, a keep-alive comment. Each of these has already cost someone a
stream, and none of them needs a socket to be tested.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any, Final

DATA_PREFIX: Final = "data: "
DONE: Final = "[DONE]"


def parse_line(line: str) -> tuple[bool, Any]:
    """Parse one SSE line.

    Returns ``(done, payload)``. ``payload`` is ``None`` when the line carries no usable
    data — a comment, a blank line, or JSON that does not parse.

    A malformed event is ignored, not raised: the upstream interleaves keep-alives and
    fragments, and killing the stream over one of them would lose the whole response.
    """
    stripped = line.strip()
    if not stripped.startswith(DATA_PREFIX):
        return False, None

    payload = stripped[len(DATA_PREFIX) :].strip()
    if payload == DONE:
        return True, None
    if not payload:
        return False, None

    try:
        return False, json.loads(payload)
    except (ValueError, TypeError):
        return False, None


def iter_events(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Decoded events, stopping at ``[DONE]``."""
    for line in lines:
        done, event = parse_line(line)
        if done:
            return
        if isinstance(event, dict):
            yield event
