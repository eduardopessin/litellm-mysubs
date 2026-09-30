"""Isolation of process-wide state during the tests.

This exists because of a measured defect: the suite wrote to `~/.litellm/mysubs/models.json`
— the user's real file — because `MySubsService` builds the `SelectionStore` with the
default path. A test that erases the model selection of whoever runs it is worse than a
test that fails: it gives no warning, and the effect only shows up on the proxy's next
restart.

Redirecting `HOME` covers the three things that land there — credentials, selection and
the refresher's locks — without every test having to remember to pass a `tmp_path`.

LiteLLM's callback lists are the other process-wide state: see `_isolate_callbacks`.
"""

from __future__ import annotations

from pathlib import Path

import litellm
import pytest

#: Every `litellm.Router` and every logged call append to these module-level lists, and
#: LiteLLM's `LoggingCallbackManager` refuses new entries past `MAX_CALLBACKS` without an
#: error.
_CALLBACK_LISTS = (
    "callbacks",
    "input_callback",
    "success_callback",
    "failure_callback",
    "service_callback",
    "_async_success_callback",
    "_async_failure_callback",
    "_async_input_callback",
)


@pytest.fixture(autouse=True)
def _isolate_callbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test its own copy of LiteLLM's callback lists.

    Measured: with one Router per test the lists filled up, and a later test's spend
    recorder was silently never registered — `test_the_streamed_usage_reaches_the_client_and_
    the_spend_row` failed in the full suite, passed alone, and passed in reverse order.
    """
    for name in _CALLBACK_LISTS:
        if isinstance(getattr(litellm, name, None), list):
            monkeypatch.setattr(litellm, name, list(getattr(litellm, name)))


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point on-disk state at a throwaway directory, in every test.

    `autouse` on purpose: the default path is resolved in a dataclass `default_factory`,
    far from any test, and requiring each one to protect itself would guarantee that the
    next one written would not.

    Swapping `Path.home()` is not enough: the `DEFAULT_PATH` values are module constants,
    evaluated **at import time** — long before any fixture runs. Measured: with only
    `home` swapped, the suite kept writing to `~/.litellm/mysubs/models.json`. So the
    constants are replaced too, along with the signature defaults that captured them.
    """
    from litellm_mysubs.catalog import selection
    from litellm_mysubs.credentials import file_store, refresher

    home = tmp_path / "home"
    (home / ".litellm" / "mysubs").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    for module, name in ((selection, "models.json"), (file_store, "credentials.json")):
        target = home / ".litellm" / "mysubs" / name
        monkeypatch.setattr(module, "DEFAULT_PATH", target)
        # The signature default was captured at import time: restoring `__defaults__` is
        # what makes a bare `SelectionStore()` point at the throwaway location.
        cls = selection.SelectionStore if module is selection else file_store.FileCredentialStore
        cls.__init__.__defaults__ = (target,)
    # The refresher imported the constant by value (`from .file_store import DEFAULT_PATH`)
    # and derives its lock files from it: without this the suite left `credentials.json.*.lock`
    # files in the real home of whoever ran it.
    monkeypatch.setattr(
        refresher, "DEFAULT_PATH", home / ".litellm" / "mysubs" / "credentials.json"
    )
