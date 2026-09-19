"""**Cross-process** exclusion for the exchange of a refresh token.

Why a `threading.Lock` is not enough: LiteLLM runs with `--num_workers N`, and each worker
is a separate *process*, with its own background refresher looking at the same credentials
file. A thread lock lives inside one interpreter and sees nothing of what the other N-1
processes do.

Why that matters here and nowhere else: Anthropic's and OpenAI's refresh tokens are
single-use and rotating — exchanging one invalidates it and returns another. Two processes
exchanging the same credential at the same time each end up with half the result: the one
that writes first loses its token to the second one's write, and from then on both present
an already-spent refresh token. The measured symptom is `invalid_grant` in a loop, across
every worker, until somebody logs in by hand.

The lock is a file lock (`fcntl.flock`) precisely because the arbitration has to happen in
the kernel, which is the only place the N processes share.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

try:
    import fcntl
except ImportError:  # pragma: no cover - only happens on Windows
    # `fcntl` does not exist on Windows. Blowing up at import would make the whole package
    # unimportable on that platform because of an optional feature — the background
    # refresher is a convenience, not the request path.
    #
    # LIMITATION, and a material one: without `fcntl` the lock below degrades to a
    # `threading.Lock` and **only protects within the process**. On Windows, with
    # `--num_workers > 1`, the refresh token race described at the top of the module remains
    # possible. A single worker (the default) is still correct.
    _FCNTL: ModuleType | None = None
else:
    _FCNTL = fcntl

#: Wait between attempts when there is a deadline. It starts short so as not to waste the
#: deadline and grows to this ceiling: exchanging a token takes a network round trip
#: (hundreds of ms), and probing faster than this only burns CPU hearing the same "busy".
_MAX_POLL_S: float = 0.05
_MIN_POLL_S: float = 0.001

#: Per-path locks for the degraded mode. Kept in a dictionary because two `file_lock` calls
#: over the same target have to find the same object — a fresh lock per call would exclude
#: nobody.
_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


class LockBusyError(RuntimeError):
    """Another process holds the lock and the deadline ran out (or there was none)."""


def _lock_path(path: Path) -> Path:
    """Lock file next to the target, so it lands on the same file system."""
    return path.with_name(path.name + ".lock")


def _open_lock_file(path: Path) -> int:
    """Descriptor for the lock file, with directory and file permissions already tight."""
    lock_file = _lock_path(path)
    # The credentials directory is sensitive: whoever can list it learns which providers
    # are connected, and the `.lock` beside `credentials.json` gives that away.
    lock_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(lock_file, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        # The mode given to `os.open` goes through the umask and only applies on creation.
        # A `.lock` left by an earlier version with loose bits would stay that way forever.
        os.fchmod(fd, 0o600)
    except OSError:
        os.close(fd)
        raise
    return fd


def _try_acquire(fd: int, fcntl_mod: ModuleType) -> bool:
    """A single non-blocking attempt. `False` means busy, which is not an error."""
    try:
        fcntl_mod.flock(fd, fcntl_mod.LOCK_EX | fcntl_mod.LOCK_NB)
    except OSError:
        return False
    return True


@contextmanager
def _flock_guard(path: Path, timeout_s: float, fcntl_mod: ModuleType) -> Iterator[None]:
    fd = _open_lock_file(path)
    try:
        if not _try_acquire(fd, fcntl_mod):
            if timeout_s <= 0:
                # No wait by default: this is called from a periodic sweep. If another
                # worker is already renewing, this one has no work — the renewal that
                # matters will happen anyway, done by that worker. Waiting would pile all N
                # workers onto one lock for a task that is not urgent, and on the next cycle
                # they find the token already fresh.
                raise LockBusyError(f"lock held by another process: {_lock_path(path)}")
            deadline = time.monotonic() + timeout_s
            delay = _MIN_POLL_S
            while not _try_acquire(fd, fcntl_mod):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LockBusyError(f"lock still held after {timeout_s:g}s: {_lock_path(path)}")
                time.sleep(min(delay, remaining))
                delay = min(delay * 2, _MAX_POLL_S)
        try:
            yield
        finally:
            # Releasing has to happen even on an exception. A stuck lock raises nothing:
            # the symptom would be "the token stopped renewing", silent, on every subsequent
            # cycle, and the cause would sit in a process that no longer exists.
            fcntl_mod.flock(fd, fcntl_mod.LOCK_UN)
    finally:
        # Closing the descriptor releases the lock by itself; the `LOCK_UN` above is
        # explicit so the order reads clearly. What is NEVER done here is deleting the file:
        # between one process's `unlink` and its `close`, another can create and lock a new
        # file with the same name — two locks over different inodes, and both owners
        # convinced they have exclusivity. The file stays, empty, costing one inode.
        os.close(fd)


@contextmanager
def _thread_guard(path: Path, timeout_s: float) -> Iterator[None]:
    """Degraded mode without `fcntl`: exclusion within this process only."""
    key = os.path.abspath(_lock_path(path))
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.setdefault(key, threading.Lock())
    if not lock.acquire(blocking=timeout_s > 0, timeout=timeout_s if timeout_s > 0 else -1):
        raise LockBusyError(f"lock held by another thread: {_lock_path(path)}")
    try:
        yield
    finally:
        lock.release()


@contextmanager
def file_lock(path: Path, *, timeout_s: float = 0.0) -> Iterator[None]:
    """Holds exclusivity over `path` for the duration of the block.

    `path` is the target to protect (typically `credentials.json`); the lock itself lives in
    a `<path>.lock` next to it. The target is neither opened nor touched — that way the
    store's atomic write, which replaces the inode with `os.replace`, does not drag the lock
    along with it.

    `timeout_s=0.0` does not wait: it raises `LockBusyError` immediately if another process
    holds the lock. Larger values probe until the deadline and only then give up.
    """
    fcntl_mod = _FCNTL
    if fcntl_mod is None:  # pragma: no cover - Windows only
        with _thread_guard(path, timeout_s):
            yield
    else:
        with _flock_guard(path, timeout_s, fcntl_mod):
            yield
