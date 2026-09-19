"""Connects subscriptions (Claude Max, ChatGPT Plus, Google Antigravity) to LiteLLM.

Installation is one line in `config.yaml`::

    litellm_settings:
      callbacks: ["litellm_mysubs.proxy_handler_instance"]

Or, interactively, with `mysubs-setup`.

`proxy_handler_instance` is an **instance**, not the class: the proxy refuses a class with
`ValueError` at startup — verified — because a callback that is not dispatchable would load
without complaint and then be ignored on every request.
"""

from __future__ import annotations

from .callback import MySubs

#: The name `config.yaml` refers to. The convention is LiteLLM's, which suggests it in the
#: error message itself when a class is pointed at it.
proxy_handler_instance = MySubs()

__all__ = ["MySubs", "proxy_handler_instance"]
