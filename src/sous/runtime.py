"""Runtime wiring: session persistence + a Runner for the coordinator.

The pantry and dietary profile are stored under ``user:``-scoped session state, so
with a persistent ``SessionService`` they survive across separate conversations —
the "memory" the concierge relies on to avoid re-asking what it already knows.
"""

from __future__ import annotations

import os

from google.adk.apps import App
from google.adk.memory import (
    BaseMemoryService,
    InMemoryMemoryService,
    VertexAiMemoryBankService,
)
from google.adk.runners import Runner
from google.adk.sessions import (
    BaseSessionService,
    DatabaseSessionService,
    InMemorySessionService,
)

from .agent import root_agent
from .observability import ObservabilityPlugin, configure_logging, configure_telemetry
from .plugins import PolicyPlugin

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


def get_memory_service(backend: str | None = None) -> BaseMemoryService:
    """Return a long-term memory service.

    Mirrors :func:`get_session_service`'s env-driven pattern. Defaults to the
    in-memory (keyword-search) service — ideal for tests and local runs. Set
    ``SOUS_MEMORY_BACKEND=vertex`` (plus ``SOUS_VERTEX_PROJECT`` /
    ``SOUS_VERTEX_LOCATION`` / ``SOUS_VERTEX_AGENT_ENGINE_ID``) to use the managed
    Vertex AI Memory Bank, which extracts and semantically searches memories.
    """
    backend = (backend or os.environ.get("SOUS_MEMORY_BACKEND", "")).lower()
    if backend == "vertex":
        return VertexAiMemoryBankService(
            project=os.environ.get("SOUS_VERTEX_PROJECT"),
            location=os.environ.get("SOUS_VERTEX_LOCATION"),
            agent_engine_id=os.environ.get("SOUS_VERTEX_AGENT_ENGINE_ID"),
        )
    return InMemoryMemoryService()


def build_runner(
    session_service: BaseSessionService | None = None,
    memory_service: BaseMemoryService | None = None,
) -> Runner:
    """Build a Runner for the Sous coordinator agent.

    Reuses the module-level :data:`app`, so a programmatic runner and the one
    ``adk web``/``adk api_server`` build from discovering ``sous.app`` are wired
    identically — same plugins, same observability config.
    """
    return Runner(
        app=app,
        session_service=session_service or get_session_service(),
        memory_service=memory_service or get_memory_service(),
    )


def build_app() -> App:
    """Construct the Sous ``App`` and configure observability as a side effect.

    Plugins are registered on the ``App`` (not per-agent) so they apply globally to
    every agent and tool: ``PolicyPlugin`` enforces runtime guardrails (issue #7)
    and ``ObservabilityPlugin`` captures intent/outcome events (issue #9). Building
    the app also configures structured JSON logging (with PII redaction) and turns
    off ADK's default capture of message content in spans, so PII stays out of both
    logs and traces. Both configurators are idempotent.
    """
    configure_logging()
    configure_telemetry()
    return App(
        name=APP_NAME,
        root_agent=root_agent,
        plugins=[PolicyPlugin(), ObservabilityPlugin()],
    )


# Module-level ``App`` so ADK's agent loader discovers it: ``adk web src`` /
# ``adk api_server src`` check for ``sous.app`` (an ``App``) *before* falling back
# to ``root_agent``. Without this, the primary run path would build its own Runner
# straight from ``root_agent`` — bypassing the plugins and the logging/telemetry
# config, which would then only apply when ``build_runner`` is called by hand.
# Building it at import also runs the observability configurators in every path.
app = build_app()
