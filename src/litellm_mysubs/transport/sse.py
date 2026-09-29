"""Server-Sent Events reading, as omp's `pi-utils` reads them.

Kept apart from the transport because this is the deceptive part: an event split across
several ``data:`` lines, a ``data:`` with no space after the colon, a keep-alive comment,
a body cut in the middle of the last event. Each of these has already cost someone a
stream, and none of them needs a socket to be tested.

Two layers, as in omp. `readSseEvents` frames the lines into events — an event ends on a
blank line, its ``data:`` lines are joined with ``\\n``. `readSseFrames` turns each event's
data into a JSON value: empty data is skipped, ``[DONE]`` ends the stream, and a trailing
event that is a cut-off object (the connection dropped mid-event) ends it cleanly.

Reading one line at a time as if every line were an event — what this module did before —
dropped both a ``data:{...}`` without the space and every event whose JSON spanned more
than one ``data:`` line, silently: each fragment failed to parse on its own and was
skipped.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Final

DONE: Final = "[DONE]"

#: Stripped from the first line only; omp's `TextDecoder` drops it the same way.
_BOM: Final = "\ufeff"


@dataclass(frozen=True, slots=True)
class Malformed:
    """A ``data:`` frame JSON rejected: omp's text lane (`readSseJsonOrText`).

    Returned rather than raised so each reader chooses, as omp's two readers do: the
    transport raises it (`readSseJson`), the discovery probe skips it.
    """

    raw: str
    error: json.JSONDecodeError


class SseDecoder:
    """Incremental decoder: lines in (without their terminator), JSON frames out.

    ``feed`` takes the lines `httpx` splits on ``\\n``, ``\\r\\n`` and ``\\r`` — the same
    three omp splits on — and ``close`` flushes what the body left pending at EOF. After
    ``[DONE]`` the decoder is ``done`` and ignores the rest.

    Only ``data`` is kept of each event. omp also records ``event``, ``id`` and ``retry``,
    but its JSON readers never look at them, and an event that carries only those has
    empty data — skipped either way.
    """

    __slots__ = ("_data", "_first", "done")

    def __init__(self) -> None:
        #: ``None`` until a ``data:`` field arrives — distinct from a ``data:`` left empty.
        self._data: str | None = None
        self._first = True
        self.done = False

    def feed(self, line: str) -> list[Any]:
        """The frames the line completes: at most one, since only a blank line dispatches."""
        if self._first:
            self._first = False
            line = line.removeprefix(_BOM)
        if self.done or not self._push(line):
            return []
        return self._frames(trailing=False)

    def close(self) -> list[Any]:
        """The event a body ended without a blank line — real services do that."""
        if self.done:
            return []
        return self._frames(trailing=True)

    # omp: stream.ts :: pushSseLine
    def _push(self, line: str) -> bool:
        """Apply one line to the pending event; ``True`` when the line dispatches it."""
        if not line:
            return True
        field, colon, value = line.partition(":")
        # A comment is the empty field name: `: keep-alive`.
        if field == "data":
            if colon and value.startswith(" "):
                value = value[1:]
            self._data = value if self._data is None else f"{self._data}\n{value}"
        return False

    # omp: stream.ts :: flushSseEvent
    # omp: stream.ts :: readSseFrames
    # omp: stream.ts :: isRecoverableTrailingJson
    def _frames(self, *, trailing: bool) -> list[Any]:
        data, self._data = self._data, None
        if not data:
            return []
        if data == DONE:
            self.done = True
            return []
        try:
            return [json.loads(data)]
        except json.JSONDecodeError as error:
            # A cut-off object at the very end is a dropped connection, not a malformed
            # event: omp ends iteration cleanly. Anything else goes to the reader.
            if trailing and data.lstrip()[:1] in ("{", "["):
                return []
            return [Malformed(data, error)]


# omp: stream.ts :: readSseEvents
def iter_events(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Decoded object events, stopping at ``[DONE]``; for readers that already hold a body.

    Frames JSON rejects are skipped rather than raised — the lenient lane, fit for a
    probe that only counts what it can read.
    """
    decoder = SseDecoder()
    for line in lines:
        yield from _objects(decoder.feed(line))
        if decoder.done:
            return
    yield from _objects(decoder.close())


def _objects(frames: list[Any]) -> Iterator[dict[str, Any]]:
    for frame in frames:
        if isinstance(frame, dict):
            yield frame
