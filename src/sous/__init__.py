"""Sous — a Meal & Nutrition Concierge multi-agent system built on Google's ADK."""

from .agent import root_agent
from .runtime import app

# ``app`` is the discovery entrypoint for ``adk web``/``adk api_server`` (they look
# for ``sous.app`` before ``root_agent``), so the plugins and observability config
# are active in the primary run path — not only via ``build_runner``.
__all__ = ["app", "root_agent"]
