"""Phase 4 — persistence tests: user-scoped state survives across sessions.

No LLM calls — this exercises the ADK SessionService directly to prove the memory
mechanism the concierge depends on.
"""

import pytest
from google.adk.events import Event, EventActions
from google.adk.sessions import DatabaseSessionService

from sous.runtime import APP_NAME, build_runner, get_session_service
from sous.tools import PANTRY_KEY

USER = "test-user"


@pytest.fixture
def db_service(tmp_path):
    db_file = tmp_path / "sessions.sqlite"
    return DatabaseSessionService(db_url=f"sqlite+aiosqlite:///{db_file}")


async def _write_state(service, session, delta: dict):
    """Persist a state delta the ADK-sanctioned way (via an event)."""
    await service.append_event(
        session,
        Event(author="user", actions=EventActions(state_delta=delta)),
    )


async def test_user_pantry_persists_across_sessions(db_service):
    # Session 1: user records their pantry.
    s1 = await db_service.create_session(app_name=APP_NAME, user_id=USER)
    await _write_state(db_service, s1, {PANTRY_KEY: ["rice", "eggs"]})

    # Session 2: a brand-new conversation for the same user.
    s2 = await db_service.create_session(app_name=APP_NAME, user_id=USER)
    fetched = await db_service.get_session(
        app_name=APP_NAME, user_id=USER, session_id=s2.id
    )
    assert fetched.state.get(PANTRY_KEY) == ["rice", "eggs"]


async def test_session_scoped_state_does_not_leak_across_sessions(db_service):
    s1 = await db_service.create_session(app_name=APP_NAME, user_id=USER)
    # No user: prefix -> session-scoped, must NOT appear in a different session.
    await _write_state(db_service, s1, {"draft_plan": "monday: tacos"})

    s2 = await db_service.create_session(app_name=APP_NAME, user_id=USER)
    fetched = await db_service.get_session(
        app_name=APP_NAME, user_id=USER, session_id=s2.id
    )
    assert "draft_plan" not in fetched.state


def test_get_session_service_defaults_to_in_memory(monkeypatch):
    monkeypatch.delenv("SOUS_SESSION_DB", raising=False)
    from google.adk.sessions import InMemorySessionService

    assert isinstance(get_session_service(), InMemorySessionService)


def test_get_session_service_uses_db_when_url_given(tmp_path):
    url = f"sqlite:///{tmp_path/'s.sqlite'}"
    assert isinstance(get_session_service(db_url=url), DatabaseSessionService)


def test_build_runner_wires_agent_and_app(db_service):
    runner = build_runner(session_service=db_service)
    assert runner.app_name == APP_NAME
    assert runner.agent.name == "sous_coordinator"
