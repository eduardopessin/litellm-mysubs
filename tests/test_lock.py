"""Cross-process exclusion lock for the refresh token exchange.

These tests use real processes, not threads, on purpose: what the module promises is
exclusion *between processes* (each LiteLLM worker is one), and a `threading.Lock` would
pass any test written with threads alone.

The children load `lock.py` directly from the file instead of importing
`litellm_mysubs.credentials.lock`. The package `__init__` instantiates the LiteLLM
callback and costs a measured ~15s of startup — times one child per assertion, that would
be minutes of suite to exercise a module that depends on nothing in the package.
"""

from __future__ import annotations

import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from litellm_mysubs.credentials import lock as lock_mod
from litellm_mysubs.credentials.lock import LockBusyError, file_lock

_LOCK_PY = Path(lock_mod.__file__)

#: Prelude that gives the child `file_lock` and `LockBusyError` without touching the
#: package.
_PRELUDE = """
import importlib.util, sys
_spec = importlib.util.spec_from_file_location("_lock", sys.argv[1])
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
file_lock, LockBusyError = _mod.file_lock, _mod.LockBusyError
"""

#: Child that holds the lock: it announces it has it (by creating `ready`) and releases it
#: when `go` shows up, or after `max_hold_s`. The safety deadline exists so that a failure
#: in the parent does not leave a process clinging to the lock and the suite hanging in
#: `wait()`.
_HOLDER = _PRELUDE + """
import time
from pathlib import Path

target, ready, go = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
max_hold_s = float(sys.argv[5])
with file_lock(target):
    ready.write_text("ok")
    deadline = time.monotonic() + max_hold_s
    while not go.exists() and time.monotonic() < deadline:
        time.sleep(0.005)
"""

#: Child that tries to enter once: 0 if it got in, 3 if it found the lock busy.
_PROBE = _PRELUDE + """
from pathlib import Path

try:
    with file_lock(Path(sys.argv[2])):
        sys.exit(0)
except LockBusyError:
    sys.exit(3)
"""

_TIMEOUT_S = 15.0


class Holder:
    """A separate process that holds the lock over `target`."""

    def __init__(self, target: Path, workdir: Path, max_hold_s: float) -> None:
        self._ready = workdir / "ready"
        self._go = workdir / "go"
        self.proc = subprocess.Popen(
            [
                sys.executable, "-c", _HOLDER, str(_LOCK_PY), str(target),
                str(self._ready), str(self._go), str(max_hold_s),
            ],
        )

    def wait_until_holding(self) -> None:
        """Only after this is the lock really taken — without this barrier the test would
        measure the interpreter startup speed, not the exclusion."""
        deadline = time.monotonic() + _TIMEOUT_S
        while not self._ready.exists():
            if time.monotonic() > deadline:
                raise AssertionError("the process holding the lock never started")
            if self.proc.poll() is not None:
                raise AssertionError(f"the child died with code {self.proc.returncode}")
            time.sleep(0.005)

    def release(self) -> None:
        self._go.write_text("ok")

    def close(self) -> None:
        self.release()
        self.proc.wait(timeout=_TIMEOUT_S)


def _probe(target: Path) -> int:
    """Exit code of a process that tries to enter the lock exactly once."""
    return subprocess.run(
        [sys.executable, "-c", _PROBE, str(_LOCK_PY), str(target)], timeout=_TIMEOUT_S
    ).returncode


@pytest.fixture
def target(tmp_path: Path) -> Path:
    """Target inside a directory that does not exist yet: the lock has to create it."""
    return tmp_path / "creds" / "credentials.json"


@pytest.fixture
def hold(tmp_path: Path) -> Iterator[Callable[..., Holder]]:
    """Process factory, with guaranteed teardown even when the assertion fails."""
    holders: list[Holder] = []

    def make(target: Path, max_hold_s: float = 10.0) -> Holder:
        workdir = tmp_path / f"holder{len(holders)}"
        workdir.mkdir()
        holder = Holder(target, workdir, max_hold_s)
        holders.append(holder)
        holder.wait_until_holding()
        return holder

    yield make
    for holder in holders:
        holder.close()


class TestExclusionBetweenProcesses:
    def test_the_second_process_does_not_enter_while_the_first_holds(
        self, target: Path, hold: Callable[..., Holder]
    ) -> None:
        """The case that motivated the module: two refreshers, two processes, a single
        rotating refresh token. If both entered, the second would spend a token that had
        already been exchanged."""
        hold(target)

        with pytest.raises(LockBusyError), file_lock(target):
            pytest.fail("entered a lock that another process holds")

    def test_the_lock_held_here_excludes_another_process(self, target: Path) -> None:
        """The mirror of the test above. Without it, "acquired" in the following tests
        could mean only that this process never actually locks anything."""
        with file_lock(target):
            busy = _probe(target)
        free = _probe(target)

        assert (busy, free) == (3, 0)

    def test_when_the_other_process_releases_the_lock_becomes_free(
        self, target: Path, hold: Callable[..., Holder]
    ) -> None:
        """The exclusion is temporary. If the lock never became free again, refreshing
        would stop for good after the first sweep."""
        holder = hold(target)
        holder.close()

        with file_lock(target):
            pass


class TestRelease:
    def test_the_lock_is_released_at_the_end_of_the_with(self, target: Path) -> None:
        with file_lock(target):
            pass

        assert _probe(target) == 0

    def test_an_exception_inside_the_with_releases_the_lock(self, target: Path) -> None:
        """A stuck lock produces no error at all: the symptom would be "it stopped
        refreshing", silently. That is why the release lives in a `finally`, and this is
        what proves it."""
        with pytest.raises(ZeroDivisionError), file_lock(target):
            _ = 1 / 0

        assert _probe(target) == 0


class TestDeadline:
    def test_without_a_deadline_it_gives_up_immediately(
        self, target: Path, hold: Callable[..., Holder]
    ) -> None:
        """The default does not wait. A sweep blocking here would delay the other
        providers of the same cycle for a refresh another worker is already doing."""
        hold(target)

        started = time.monotonic()
        with pytest.raises(LockBusyError), file_lock(target):
            pytest.fail("entered a lock that another process holds")

        assert time.monotonic() - started < 0.5

    def test_with_a_deadline_it_waits_for_the_other_and_ends_up_entering(
        self, target: Path, hold: Callable[..., Holder]
    ) -> None:
        """`timeout_s > 0` is for those who cannot give up — a login depositing the
        credential has no next cycle in which to try again."""
        holder = hold(target)
        # Release the lock only after the parent is already waiting: releasing earlier
        # would make the first attempt enough and the waiting path would never be
        # exercised.
        threading.Timer(0.2, holder.release).start()

        started = time.monotonic()
        with file_lock(target, timeout_s=_TIMEOUT_S):
            elapsed = time.monotonic() - started

        assert elapsed >= 0.2

    def test_with_a_short_deadline_it_gives_up_if_the_other_does_not_release(
        self, target: Path, hold: Callable[..., Holder]
    ) -> None:
        """Waiting has to end: without a deadline, the N workers would all pile up on the
        lock."""
        hold(target)

        started = time.monotonic()
        with pytest.raises(LockBusyError), file_lock(target, timeout_s=0.3):
            pytest.fail("entered a lock that another process holds")
        elapsed = time.monotonic() - started

        # It really waited — an ignored `timeout_s` would return this immediately.
        assert elapsed >= 0.3


class TestLockFile:
    def test_the_lock_file_is_not_readable_by_anyone_else(self, target: Path) -> None:
        """The `.lock` sits next to `credentials.json`, in a directory where the file name
        itself already says there are credentials there."""
        with file_lock(target):
            pass

        lock_file = target.with_name(target.name + ".lock")
        assert stat.S_IMODE(lock_file.stat().st_mode) == 0o600
        assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700

    def test_the_lock_file_survives_the_end_of_the_with(self, target: Path) -> None:
        """Deleting the `.lock` on exit would open the race the lock exists to close: two
        processes locking different inodes under the same name, both convinced they have
        exclusivity."""
        with file_lock(target):
            pass
        lock_file = target.with_name(target.name + ".lock")
        inode = lock_file.stat().st_ino

        with file_lock(target):
            pass

        assert lock_file.stat().st_ino == inode

    def test_the_target_is_not_created_by_the_lock(self, target: Path) -> None:
        """The store writes the target with `os.replace`, which swaps the inode. If the
        lock lived in the file itself, every write would leave the owner locking an orphan
        inode."""
        with file_lock(target):
            pass

        assert not target.exists()
