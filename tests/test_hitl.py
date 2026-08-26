"""Human-in-the-loop confirmation tests (issue #7) — no live LLM.

ADK's ``FunctionTool`` owns the pause/resume gating; these tests drive it
directly with a fake tool context so we can assert the three paths without a
Runner or a model:

* no confirmation yet  → the call pauses and the underlying function never runs;
* confirmed=True       → the function runs and state is mutated;
* confirmed=False      → the call is rejected and state is untouched.
"""

from types import SimpleNamespace

from google.adk.tools import FunctionTool
from google.adk.tools.tool_confirmation import ToolConfirmation

from conftest import FakeCallbackContext, FakeToolContext
from sous.agent import (
    PLAN_APPROVED_KEY,
    finalize_plan_tool,
    guard_presentation,
    record_plan_approval,
    update_pantry_tool,
)
from sous.tools import PANTRY_KEY, update_pantry


class _Actions:
    def __init__(self) -> None:
        self.skip_summarization = False
        self.requested_tool_confirmations: dict = {}


class FakeConfirmContext:
    """Minimal ToolContext stand-in supporting the confirmation handshake."""

    def __init__(self, state: dict | None = None, tool_confirmation=None) -> None:
        self.state: dict = dict(state or {})
        self.tool_confirmation = tool_confirmation
        self.actions = _Actions()
        self.confirmation_requests: list[dict] = []

    def request_confirmation(self, *, hint=None, payload=None) -> None:
        self.confirmation_requests.append({"hint": hint, "payload": payload})


# --- pantry write confirmation -------------------------------------------------


async def test_pantry_write_pauses_for_confirmation():
    ctx = FakeConfirmContext(state={PANTRY_KEY: ["rice"]})
    result = await update_pantry_tool.run_async(
        args={"items": ["eggs"], "action": "add"}, tool_context=ctx
    )
    # Paused: a confirmation was requested and the mutation did NOT happen.
    assert ctx.confirmation_requests, "expected a confirmation to be requested"
    assert ctx.state[PANTRY_KEY] == ["rice"], "state must not change before approval"
    assert "confirmation" in result["error"].lower()


async def test_pantry_write_applies_after_approval():
    ctx = FakeConfirmContext(
        state={PANTRY_KEY: ["rice"]},
        tool_confirmation=ToolConfirmation(confirmed=True),
    )
    result = await update_pantry_tool.run_async(
        args={"items": ["eggs"], "action": "add"}, tool_context=ctx
    )
    assert result["status"] == "success"
    assert ctx.state[PANTRY_KEY] == ["rice", "eggs"]


async def test_pantry_write_skipped_on_rejection():
    ctx = FakeConfirmContext(
        state={PANTRY_KEY: ["rice"]},
        tool_confirmation=ToolConfirmation(confirmed=False),
    )
    result = await update_pantry_tool.run_async(
        args={"items": ["eggs"], "action": "add"}, tool_context=ctx
    )
    assert "rejected" in result["error"].lower()
    assert ctx.state[PANTRY_KEY] == ["rice"], "rejected write must leave state intact"


def test_update_pantry_tool_is_confirmation_gated():
    from sous.agent import pantry_agent, root_agent

    assert isinstance(update_pantry_tool, FunctionTool)
    assert update_pantry_tool.name == "update_pantry"
    # Both agents that can mutate the pantry expose the *gated* tool, not the raw fn.
    for agent in (root_agent, pantry_agent):
        assert update_pantry_tool in agent.tools
        assert update_pantry not in agent.tools


# --- final plan approval -------------------------------------------------------


async def test_finalize_plan_pauses_for_confirmation():
    ctx = FakeConfirmContext(state={})
    result = await finalize_plan_tool.run_async(
        args={"summary": "3 dinners, 8 grocery items"}, tool_context=ctx
    )
    assert ctx.confirmation_requests, "finalizing should ask the user first"
    assert "confirmation" in result["error"].lower()


async def test_finalize_plan_approved():
    ctx = FakeConfirmContext(
        state={}, tool_confirmation=ToolConfirmation(confirmed=True)
    )
    result = await finalize_plan_tool.run_async(
        args={"summary": "3 dinners, 8 grocery items"}, tool_context=ctx
    )
    assert result["status"] == "approved"


async def test_finalize_plan_rejected():
    ctx = FakeConfirmContext(
        state={}, tool_confirmation=ToolConfirmation(confirmed=False)
    )
    result = await finalize_plan_tool.run_async(
        args={"summary": "3 dinners, 8 grocery items"}, tool_context=ctx
    )
    assert "rejected" in result["error"].lower()


# --- rejection halts presentation (issue #7 review) ----------------------------


def _finalize_tool():
    return SimpleNamespace(name="finalize_plan")


def test_record_plan_approval_marks_approved():
    ctx = FakeToolContext()
    record_plan_approval(
        tool=_finalize_tool(),
        args={},
        tool_context=ctx,
        tool_response={"status": "approved", "summary": "x"},
    )
    assert ctx.state[PLAN_APPROVED_KEY] is True


def test_record_plan_approval_marks_rejected():
    ctx = FakeToolContext()
    record_plan_approval(
        tool=_finalize_tool(),
        args={},
        tool_context=ctx,
        tool_response={"error": "This tool call is rejected."},
    )
    assert ctx.state[PLAN_APPROVED_KEY] is False


def test_record_plan_approval_ignores_pending_confirmation():
    # The first pass (awaiting confirmation) must NOT record an outcome — the run
    # is merely paused, not decided.
    ctx = FakeToolContext()
    record_plan_approval(
        tool=_finalize_tool(),
        args={},
        tool_context=ctx,
        tool_response={"error": "This tool call requires confirmation."},
    )
    assert PLAN_APPROVED_KEY not in ctx.state


def test_guard_presentation_blocks_when_rejected():
    ctx = FakeCallbackContext(state={PLAN_APPROVED_KEY: False})
    skip = guard_presentation(ctx)
    # Returning Content short-circuits the presenter — the rejected plan is not rendered.
    assert skip is not None
    assert "won't finalise" in skip.parts[0].text.lower()


def test_guard_presentation_allows_when_approved_or_absent():
    assert guard_presentation(FakeCallbackContext(state={PLAN_APPROVED_KEY: True})) is None
    # Fail-open: if the gate was never reached, don't block the happy path.
    assert guard_presentation(FakeCallbackContext(state={})) is None


def test_presenter_and_finalize_wire_the_guard_callbacks():
    from sous.agent import finalize_agent, presenter_agent

    assert finalize_agent.after_tool_callback is record_plan_approval
    assert presenter_agent.before_agent_callback is guard_presentation
