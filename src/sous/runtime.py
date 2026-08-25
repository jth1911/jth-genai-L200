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


# Maps a bare URL scheme to the SQLAlchemy scheme with an async driver. ADK's
# DatabaseSessionService uses SQLAlchemy's async engine, which requires an async
# driver — so a plain ``sqlite://`` / ``postgresql://`` URL would otherwise fail.
_ASYNC_SCHEMES = {
    "sqlite": "sqlite+aiosqlite",
    "postgresql": "postgresql+asyncpg",
    "postgres": "postgresql+asyncpg",  # common alias
}


def _async_db_url(url: str) -> str:
    """Normalise a database URL to use an async driver.

    Keeps ``.env`` simple — users can write ``sqlite:///...`` or
    ``postgresql://user:pass@host/db`` and get the async driver automatically. A
    URL that already specifies a driver (``scheme+driver://``) is left untouched.
    """
    scheme, sep, rest = url.partition("://")
    if not sep or "+" in scheme:
        return url
    async_scheme = _ASYNC_SCHEMES.get(scheme)
    return f"{async_scheme}{sep}{rest}" if async_scheme else url


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
