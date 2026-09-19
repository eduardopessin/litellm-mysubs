"""The `/mysubs` sub-app mounted on the LiteLLM proxy."""

from __future__ import annotations

from .app import build_app, mount

__all__ = ["build_app", "mount"]
