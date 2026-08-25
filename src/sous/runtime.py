"""Runtime wiring: session persistence + a Runner for the coordinator.

The pantry and dietary profile are stored under ``user:``-scoped session state, so
with a persistent ``SessionService`` they survive across separate conversations —
the "memory" the concierge relies on to avoid re-asking what it already knows.
"""

from __future__ import annotations

import os

from google.adk.runners import Runner
from google.adk.sessions import (
    BaseSessionService,
    DatabaseSessionService,
    InMemorySessionService,
)

from .agent import root_agent

APP_NAME = "sous"


def _async_db_url(url: str) -> str:
    """Normalise a plain ``sqlite://`` URL to the async ``aiosqlite`` driver.

    ADK's DatabaseSessionService uses SQLAlchemy's async engine, which needs an
    async driver. This keeps ``.env`` simple — users can write ``sqlite:///...``.
    """
    if url.startswith("sqlite://") and "+aiosqlite" not in url:
        return url.replace("sqlite://", "sqlite+aiosqlite://", 1)
    return url


def get_session_service(db_url: str | None = None) -> BaseSessionService:
    """Return a session service.

    Uses a persistent SQLite-backed ``DatabaseSessionService`` when a database URL
    is provided (via arg or the ``SOUS_SESSION_DB`` env var), otherwise falls back
    to an in-memory service (handy for tests and quick local runs).
    """
    url = db_url or os.environ.get("SOUS_SESSION_DB")
    if url:
        return DatabaseSessionService(db_url=_async_db_url(url))
    return InMemorySessionService()


def build_runner(session_service: BaseSessionService | None = None) -> Runner:
    """Build a Runner for the Sous coordinator agent."""
    return Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service or get_session_service(),
    )
