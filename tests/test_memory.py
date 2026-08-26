"""Phase 6 — long-term memory service wiring + ingestion (Context & Memory, issue #5).

Covers the env-driven ``MemoryService`` factory, Runner wiring, the async
``after_agent`` ingestion callback, and cross-session fact recall. The recall test
exercises ADK's ``InMemoryMemoryService`` directly (keyword search) — no LLM calls.
"""

from google.adk.events import Event
from google.adk.memory import InMemoryMemoryService, VertexAiMemoryBankService
from google.genai import types
from tests.conftest import FakeCallbackContext

from sous.memory import remember_session
from sous.runtime import APP_NAME, build_runner, get_memory_service

USER = "test-user"


# --- factory (mirrors get_session_service tests in test_state.py) ---------------


def test_get_memory_service_defaults_to_in_memory(monkeypatch):
    monkeypatch.delenv("SOUS_MEMORY_BACKEND", raising=False)
    assert isinstance(get_memory_service(), InMemoryMemoryService)


def test_get_memory_service_uses_vertex_when_configured(monkeypatch):
    monkeypatch.setenv("SOUS_MEMORY_BACKEND", "vertex")
    monkeypatch.setenv("SOUS_VERTEX_PROJECT", "demo-project")
    monkeypatch.setenv("SOUS_VERTEX_LOCATION", "us-central1")
    monkeypatch.setenv("SOUS_VERTEX_AGENT_ENGINE_ID", "123")
    # Type check only — no live Vertex call is made at construction time.
    assert isinstance(get_memory_service(), VertexAiMemoryBankService)


def test_build_runner_wires_memory_service():
    memory = InMemoryMemoryService()
    runner = build_runner(memory_service=memory)
    assert runner.memory_service is memory


# --- async ingestion callback ---------------------------------------------------


async def test_remember_session_ingests_via_callback():
    ctx = FakeCallbackContext()

    result = await remember_session(ctx)

    assert result is None  # does not override the agent's reply
    assert ctx.add_session_calls == 1


async def test_remember_session_tolerates_missing_memory_service():
    # If no memory service is wired, ingestion must fail soft, not crash the turn.
    ctx = FakeCallbackContext(memory_available=False)

    result = await remember_session(ctx)

    assert result is None
    assert ctx.add_session_calls == 0


# --- cross-session fact recall (no LLM) -----------------------------------------


async def test_fact_stated_in_one_session_is_recalled_in_another():
    memory = InMemoryMemoryService()

    # Session 1: the user reveals a durable fact in conversation.
    from google.adk.sessions import InMemorySessionService

    sessions = InMemorySessionService()
    s1 = await sessions.create_session(app_name=APP_NAME, user_id=USER)
    await sessions.append_event(
        s1,
        Event(
            author="user",
            content=types.Content(
                role="user",
                parts=[types.Part(text="I am allergic to shellfish.")],
            ),
        ),
    )
    # Re-fetch so the session carries the appended event, then ingest it.
    s1 = await sessions.get_session(app_name=APP_NAME, user_id=USER, session_id=s1.id)
    await memory.add_session_to_memory(s1)

    # Session 2: a brand-new conversation searches memory for the fact.
    found = await memory.search_memory(app_name=APP_NAME, user_id=USER, query="shellfish")

    assert found.memories
    texts = " ".join(
        part.text for m in found.memories for part in m.content.parts if part.text
    )
    assert "shellfish" in texts.lower()


async def test_memory_is_scoped_per_user():
    memory = InMemoryMemoryService()
    from google.adk.sessions import InMemorySessionService

    sessions = InMemorySessionService()
    s1 = await sessions.create_session(app_name=APP_NAME, user_id=USER)
    await sessions.append_event(
        s1,
        Event(
            author="user",
            content=types.Content(role="user", parts=[types.Part(text="I love tofu.")]),
        ),
    )
    s1 = await sessions.get_session(app_name=APP_NAME, user_id=USER, session_id=s1.id)
    await memory.add_session_to_memory(s1)

    # A different user must not see the first user's facts.
    found = await memory.search_memory(app_name=APP_NAME, user_id="someone-else", query="tofu")
    assert not found.memories
