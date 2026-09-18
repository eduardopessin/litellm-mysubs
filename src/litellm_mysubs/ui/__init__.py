"""Sub-app `/mysubs` montada no proxy do LiteLLM."""

from __future__ import annotations

from .app import build_app, mount

__all__ = ["build_app", "mount"]
