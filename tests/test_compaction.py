"""Phase 6 — conversation history compaction (Context & Memory, issue #5).

The coordinator's ``before_model_callback`` bounds the token footprint of long
conversations. These tests exercise the pure trimming/summarisation logic with a
synthetic ``LlmRequest`` — no LLM calls.
"""

from google.adk.models import LlmRequest
from google.genai import types

from conftest import FakeCallbackContext
from sous.memory import HISTORY_SUMMARY_KEY, compact_history


def _history(n: int) -> list[types.Content]:
    """A synthetic alternating user/model conversation of ``n`` turns."""
    out = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "model"
        out.append(types.Content(role=role, parts=[types.Part(text=f"turn {i}")]))
    return out


def _text(role: str, text: str) -> types.Content:
    return types.Content(role=role, parts=[types.Part(text=text)])


def _fn_response(name: str) -> types.Content:
    return types.Content(
        role="user",
        parts=[types.Part(function_response=types.FunctionResponse(name=name, response={}))],
    )


def test_short_history_is_left_untouched():
    req = LlmRequest(contents=_history(4))
    ctx = FakeCallbackContext()

    result = compact_history(ctx, req, window=12)

    assert result is None  # proceed with the model call
    assert len(req.contents) == 4  # nothing trimmed


def test_long_history_trimmed_to_window_sliding_window_only():
    req = LlmRequest(contents=_history(30))
    ctx = FakeCallbackContext(state={"user:pantry": ["rice", "eggs"]})

    # No summarizer -> pure sliding window: keep only the last `window` turns.
    result = compact_history(ctx, req, window=12, summarize_fn=None)

    assert result is None
    assert len(req.contents) == 12
    # The kept turns are the most recent ones.
    assert req.contents[-1].parts[0].text == "turn 29"
    assert req.contents[0].parts[0].text == "turn 18"


def test_compaction_preserves_user_scoped_state():
    req = LlmRequest(contents=_history(30))
    ctx = FakeCallbackContext(state={"user:pantry": ["rice", "eggs"]})

    compact_history(ctx, req, window=12, summarize_fn=None)

    # Durable user-scoped state is never touched by history trimming.
    assert ctx.state["user:pantry"] == ["rice", "eggs"]


def test_summarizer_prepends_summary_of_dropped_prefix():
    req = LlmRequest(contents=_history(30))
    ctx = FakeCallbackContext()
    seen = {}

    def fake_summarizer(dropped, prior):
        seen["dropped"] = dropped
        seen["prior"] = prior
        return "USER LIKES TACOS"

    result = compact_history(ctx, req, window=12, summarize_fn=fake_summarizer)

    assert result is None
    # window recent turns + exactly one prepended summary content.
    assert len(req.contents) == 13
    assert "USER LIKES TACOS" in req.contents[0].parts[0].text
    assert req.contents[-1].parts[0].text == "turn 29"
    # The dropped prefix (everything but the last window) was handed to the summarizer.
    assert len(seen["dropped"]) == 18
    assert seen["dropped"][0].parts[0].text == "turn 0"


def test_running_summary_stored_in_session_scoped_state():
    req = LlmRequest(contents=_history(30))
    ctx = FakeCallbackContext()

    compact_history(ctx, req, window=12, summarize_fn=lambda dropped, prior: "RUNNING")

    # Stored under a session-scoped key (no user: prefix) so it does not leak across
    # sessions the way durable profile data does.
    assert ctx.state[HISTORY_SUMMARY_KEY] == "RUNNING"
    assert not HISTORY_SUMMARY_KEY.startswith("user:")


def test_prior_running_summary_is_fed_back_to_summarizer():
    req = LlmRequest(contents=_history(30))
    ctx = FakeCallbackContext(state={HISTORY_SUMMARY_KEY: "OLD SUMMARY"})
    seen = {}

    def fake_summarizer(dropped, prior):
        seen["prior"] = prior
        return "NEW SUMMARY"

    compact_history(ctx, req, window=12, summarize_fn=fake_summarizer)

    assert seen["prior"] == "OLD SUMMARY"


def test_summarizer_not_called_below_threshold():
    req = LlmRequest(contents=_history(5))
    ctx = FakeCallbackContext()
    calls = []

    compact_history(ctx, req, window=12, summarize_fn=lambda d, p: calls.append(1))

    assert calls == []
    assert len(req.contents) == 5


def test_summarizer_failure_falls_back_to_sliding_window():
    req = LlmRequest(contents=_history(30))
    ctx = FakeCallbackContext()

    def boom(dropped, prior):
        raise RuntimeError("no API key")

    # A failing summarizer must not crash the turn — degrade to a plain trim.
    result = compact_history(ctx, req, window=12, summarize_fn=boom)

    assert result is None
    assert len(req.contents) == 12
    assert req.contents[-1].parts[0].text == "turn 29"


def test_orphan_function_response_is_dropped_from_window_start():
    # Window boundary lands right after a function_call: the kept slice would start
    # with an orphaned function_response, which Gemini rejects. It must be dropped.
    contents = _history(6) + [
        _fn_response("read_pantry"),  # orphan — its call is in the dropped prefix
        _text("user", "what can I cook?"),
        _text("model", "let's see"),
        _text("user", "thanks"),
    ]
    req = LlmRequest(contents=contents)
    ctx = FakeCallbackContext()

    compact_history(ctx, req, window=4, summarize_fn=None)

    # The leading orphan response is gone; history opens on a real user turn.
    assert req.contents[0].parts[0].text == "what can I cook?"
    assert all(
        part.function_response is None
        for part in req.contents[0].parts
    )


def test_pure_trim_opens_on_a_user_turn():
    # If the raw window would start on a model turn, drop it so the model gets a
    # user-first history.
    contents = _history(6) + [
        _text("model", "leading model turn"),
        _text("user", "hi"),
        _text("model", "hello"),
        _text("user", "plan please"),
    ]
    req = LlmRequest(contents=contents)
    ctx = FakeCallbackContext()

    compact_history(ctx, req, window=4, summarize_fn=None)

    assert req.contents[0].role == "user"
    assert req.contents[0].parts[0].text == "hi"


def test_summary_mode_also_strips_orphan_function_response():
    contents = _history(6) + [
        _fn_response("update_pantry"),  # orphan at the window start
        _text("user", "add eggs"),
        _text("model", "done"),
        _text("user", "great"),
    ]
    req = LlmRequest(contents=contents)
    ctx = FakeCallbackContext()

    compact_history(ctx, req, window=4, summarize_fn=lambda dropped, prior: "SUMMARY")

    # Summary is first (provides the user start); the orphan response is stripped.
    assert "SUMMARY" in req.contents[0].parts[0].text
    assert req.contents[1].parts[0].text == "add eggs"
    assert all(part.function_response is None for part in req.contents[1].parts)


def test_window_defaults_from_env(monkeypatch):
    monkeypatch.setenv("SOUS_HISTORY_WINDOW", "6")
    req = LlmRequest(contents=_history(30))
    ctx = FakeCallbackContext()

    compact_history(ctx, req, summarize_fn=None)

    assert len(req.contents) == 6
