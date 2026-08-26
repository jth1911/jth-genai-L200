"""Phase 5 — ADK evaluation (LLM-gated).

These call a live Gemini model, so they are marked ``llm`` and skipped unless
``GOOGLE_API_KEY`` is set. Run them explicitly with:

    uv run pytest -m llm

The eval set + thresholds live in ``src/sous/eval/`` and can also be run with the
ADK CLI:  ``adk eval sous src/sous/eval/pantry_smoke.evalset.json``
"""

import os
from pathlib import Path

import pytest
from google.adk.evaluation.agent_evaluator import AgentEvaluator
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from sous.agent import plan_workflow, recipe_agent
from sous.schemas import GroceryPlan, RecipePlan


def _as_model(model_cls, value):
    """Validate a state value that may be a JSON string or an already-parsed dict."""
    if isinstance(value, str):
        return model_cls.model_validate_json(value)
    return model_cls.model_validate(value)

EVAL_DIR = Path(__file__).resolve().parents[1] / "src" / "sous" / "eval"

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(
        not os.environ.get("GOOGLE_API_KEY"),
        reason="GOOGLE_API_KEY not set; skipping live-LLM eval.",
    ),
]


async def test_pantry_smoke_evalset():
    await AgentEvaluator.evaluate(
        agent_module="sous",
        eval_dataset_file_path_or_dir=str(EVAL_DIR / "pantry_smoke.evalset.json"),
        num_runs=2,
    )


async def test_recipe_agent_emits_validated_structured_output():
    """The recipe stage must call its tool AND return schema-valid RecipePlan JSON.

    This exercises the Gemini-3.x `output_schema` + tools capability end-to-end.
    """
    service = InMemorySessionService()
    await service.create_session(app_name="probe", user_id="u", session_id="s")
    runner = Runner(agent=recipe_agent, app_name="probe", session_service=service)

    final = None
    async for event in runner.run_async(
        user_id="u",
        session_id="s",
        new_message=types.Content(
            role="user",
            parts=[types.Part(text="Plan 2 high-protein dinners for the week.")],
        ),
    ):
        if event.is_final_response() and event.content:
            final = "".join(p.text or "" for p in event.content.parts)

    assert final, "no final response produced"
    plan = RecipePlan.model_validate_json(final)  # raises if not schema-valid JSON
    assert plan.meals, "expected at least one planned meal"


async def test_full_plan_workflow_end_to_end():
    """Drive the whole Sequential(Parallel(...)) pipeline end-to-end.

    Asserts the structured stages land validated data in state (RecipePlan +
    GroceryPlan) and that the terminal presenter renders a friendly prose reply
    rather than raw JSON.
    """
    service = InMemorySessionService()
    await service.create_session(app_name="wf", user_id="u", session_id="s")
    runner = Runner(agent=plan_workflow, app_name="wf", session_service=service)

    final = None
    async for event in runner.run_async(
        user_id="u",
        session_id="s",
        new_message=types.Content(
            role="user",
            parts=[
                types.Part(
                    text="Plan 3 high-protein dinners to maintain my weight at 80kg, "
                    "no shellfish."
                )
            ],
        ),
    ):
        if event.is_final_response() and event.content:
            final = "".join(p.text or "" for p in event.content.parts)

    session = await service.get_session(app_name="wf", user_id="u", session_id="s")
    # Structured stages produced schema-valid data in state.
    plan = _as_model(RecipePlan, session.state["recipe_plan"])
    _as_model(GroceryPlan, session.state["grocery_list"])
    assert plan.meals, "expected planned meals"
    # Terminal reply is conversational prose, not a raw JSON blob.
    assert final and not final.lstrip().startswith("{")
