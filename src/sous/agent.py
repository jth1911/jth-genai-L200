"""Sous multi-agent system.

Architecture (see docs/architecture.md):

    root_agent  (sous_coordinator, LlmAgent — conversational routing + memory)
      └─ delegates to → plan_workflow  (SequentialAgent)
            ├─ gather_step  (ParallelAgent)
            │     ├─ nutrition_agent  → state["nutrition_targets"]
            │     └─ pantry_agent     → state["pantry_summary"]
            ├─ recipe_agent    → state["recipe_plan"]   (validated RecipePlan JSON)
            ├─ grocery_agent   → state["grocery_list"]  (validated GroceryPlan JSON)
            ├─ finalize_agent  → HITL: pauses for the user to approve the plan
            └─ presenter_agent → renders the approved plan as the final reply

This mixes both orchestration styles the project targets:
  * LLM-driven delegation — the coordinator decides when to hand off to the workflow.
  * A deterministic workflow — Sequential(Parallel(...), ...) runs a fixed pipeline,
    with each stage passing results to the next through session state and
    {key} instruction templating.

Note on the ADK API: ADK 2.x flags ``SequentialAgent``/``ParallelAgent`` as
deprecated in favour of the new graph ``Workflow``. We deliberately keep the
classic workflow agents here because (a) the new ``Workflow`` cannot yet be used
as an ``LlmAgent`` sub-agent — which would break coordinator delegation — and
(b) the classic Sequential/Parallel/Loop agents are the canonical way to express
the deterministic pipeline this project is graded on. They remain fully functional.
"""

from __future__ import annotations

import os

from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.tools import FunctionTool, load_memory
from google.genai import types

from .memory import compact_history, remember_session
from .schemas import GroceryPlan, RecipePlan
from .tools import (
    build_grocery_list,
    compute_nutrition_targets,
    finalize_plan,
    read_pantry,
    search_recipes,
    update_pantry,
)

# Strategic model routing (issue #7). Rather than run every agent on one model,
# we route by task complexity across two tiers:
#   * FAST  — cheap/low-latency, for simple specialists and final rendering.
#   * SMART — stronger reasoning, for the steps that actually plan and aggregate.
# Both are Gemini via Google AI Studio by default (see .env.example) and fully
# env-overridable so the same code runs against Vertex or another model without
# edits. `SOUS_MODEL` is kept as a back-compat single knob that pins both tiers.
_DEFAULT_FAST_MODEL = "gemini-3.6-flash"
# `gemini-pro-latest` tracks the current production Pro model (there is no
# 3.6-pro; the 3.6/3.7 line is flash-only), so the smart tier stays valid without
# pinning to a preview. Override with SOUS_SMART_MODEL as needed.
_DEFAULT_SMART_MODEL = "gemini-pro-latest"


def _tier_models() -> tuple[str, str]:
    """Resolve the (fast, smart) model ids from the environment.

    Precedence per tier: ``SOUS_FAST_MODEL`` / ``SOUS_SMART_MODEL`` → the legacy
    ``SOUS_MODEL`` override → the built-in default. Kept as a pure function so the
    routing logic is unit-testable without importing a live model.
    """
    override = os.environ.get("SOUS_MODEL")
    fast = os.environ.get("SOUS_FAST_MODEL", override or _DEFAULT_FAST_MODEL)
    smart = os.environ.get("SOUS_SMART_MODEL", override or _DEFAULT_SMART_MODEL)
    return fast, smart


FAST_MODEL, SMART_MODEL = _tier_models()


# Human-in-the-loop (issue #7). Writing to the pantry mutates persisted `user:`
# state, so it is a high-stakes action: gate it behind an explicit user
# confirmation. `update_pantry` is only ever a write (reads go through
# `read_pantry`, which stays ungated), so requiring confirmation unconditionally
# is exactly "confirm before every pantry change". ADK pauses the run when the
# tool is called, surfaces the pending call for approval, and only invokes the
# underlying function once the user confirms.
update_pantry_tool = FunctionTool(update_pantry, require_confirmation=True)

# HITL: committing to a finished plan is the second high-stakes action. The
# finalize step must be approved by the user before the presenter renders it,
# so the plan is never "confirmed" behind the user's back.
finalize_plan_tool = FunctionTool(finalize_plan, require_confirmation=True)

# State key recording the outcome of the finalize gate (True approved / False
# rejected / absent if never reached).
PLAN_APPROVED_KEY = "plan_approved"


def record_plan_approval(*, tool, args, tool_context, tool_response):
    """Persist the finalize gate's outcome so a rejection actually stops the plan.

    Runs after the confirmation-gated ``finalize_plan`` tool. That tool yields one
    of three results: a *pending-confirmation* notice (the run is paused — ignore
    it), an explicit *rejection*, or the *approved* payload. A ``SequentialAgent``
    does not branch, so recording the definitive outcome in session state is what
    lets ``guard_presentation`` skip the presenter when the user says no. Returns
    ``None`` to leave the tool response untouched.
    """
    if getattr(tool, "name", None) != "finalize_plan" or not isinstance(tool_response, dict):
        return None
    if tool_response.get("status") == "approved":
        tool_context.state[PLAN_APPROVED_KEY] = True
    elif "rejected" in str(tool_response.get("error", "")).lower():
        tool_context.state[PLAN_APPROVED_KEY] = False
    return None


def guard_presentation(callback_context):
    """Skip presenting the plan when the user rejected it at the finalize gate.

    Returning ``Content`` from a ``before_agent_callback`` short-circuits the
    presenter's model call, so an unapproved plan is never rendered (issue #7
    review). An absent or approved outcome lets the presenter run normally, so the
    guard fails open on the happy path and closed only on an explicit rejection.
    """
    if callback_context.state.get(PLAN_APPROVED_KEY) is False:
        return types.Content(
            role="model",
            parts=[
                types.Part(
                    text=(
                        "No problem — I won't finalise that plan. Tell me what you'd "
                        "like to change (different meals, budget, cook time, …) and "
                        "I'll put together a new one."
                    )
                )
            ],
        )
    return None


# --- specialist agents ---------------------------------------------------------

nutrition_agent = LlmAgent(
    name="nutrition_agent",
    model=FAST_MODEL,
    description="Calculates the user's daily calorie and macronutrient targets.",
    instruction=(
        "You are a nutrition specialist. From the user's goal (lose/maintain/gain) "
        "and body weight, call `compute_nutrition_targets` to produce daily calorie "
        "and macro targets. If weight or goal is unknown, state a sensible default "
        "(maintain, 75kg) and say so. Return a short summary of the targets."
    ),
    tools=[compute_nutrition_targets],
    output_key="nutrition_targets",
)

pantry_agent = LlmAgent(
    name="pantry_agent",
    model=FAST_MODEL,
    description="Tracks and reports what ingredients the user already has at home.",
    instruction=(
        "You manage the user's pantry. Call `read_pantry` to see what they have and "
        "`update_pantry` to add or remove items when the user mentions changes. "
        "Return a concise list of the current pantry contents."
    ),
    tools=[read_pantry, update_pantry_tool],
    output_key="pantry_summary",
)

recipe_agent = LlmAgent(
    name="recipe_agent",
    model=SMART_MODEL,
    description="Selects recipes that fit the user's goals, constraints and pantry.",
    instruction=(
        "You are a meal planner. Using the nutrition targets below and the pantry "
        "contents, call `search_recipes` (filtering by tags, excluded allergens, cost "
        "and cook time) and choose a set of meals that fit the user's request. Prefer "
        "recipes that reuse pantry items. Output the chosen plan as a list of recipe "
        "names with their `id`s so the grocery step can use them.\n\n"
        "Nutrition targets: {nutrition_targets?}\n"
        "Pantry: {pantry_summary?}"
    ),
    tools=[search_recipes],
    # Structured output: the chosen plan is emitted as validated RecipePlan JSON
    # rather than free text, so the grocery stage gets a clean contract. Relies on
    # Gemini 3.x supporting output_schema alongside tools in one request.
    output_schema=RecipePlan,
    output_key="recipe_plan",
)

grocery_agent = LlmAgent(
    name="grocery_agent",
    model=SMART_MODEL,
    description="Turns the chosen meal plan into a consolidated grocery list.",
    instruction=(
        "You build shopping lists. From the chosen meal plan below, extract the recipe "
        "`id`s and call `build_grocery_list` with them and the user's pantry so already "
        "owned ingredients are excluded. Present the final de-duplicated grocery list.\n\n"
        "Chosen plan: {recipe_plan?}\n"
        "Pantry: {pantry_summary?}"
    ),
    tools=[build_grocery_list],
    # Structured output: the final list is validated GroceryPlan JSON.
    output_schema=GroceryPlan,
    output_key="grocery_list",
)

finalize_agent = LlmAgent(
    name="finalize_agent",
    model=FAST_MODEL,
    description="Asks the user to approve the finished plan before it is presented.",
    instruction=(
        "You are the approval gate. In one short line, summarise the chosen meals "
        "and how many grocery items the plan needs, then call `finalize_plan` with "
        "that summary. This pauses for the user's explicit approval before the plan "
        "is presented — do not describe the full plan yourself.\n\n"
        "Chosen plan (JSON): {recipe_plan?}\n"
        "Grocery list (JSON): {grocery_list?}"
    ),
    tools=[finalize_plan_tool],
    output_key="plan_approval",
    # Record approve/reject into state so a rejection can stop the pipeline.
    after_tool_callback=record_plan_approval,
)

presenter_agent = LlmAgent(
    name="presenter_agent",
    model=FAST_MODEL,
    description="Presents the finished plan and grocery list to the user in friendly prose.",
    instruction=(
        "You are the concierge's voice. Turn the structured plan and grocery list "
        "below into a warm, clear reply: first the chosen meals, then the shopping "
        "list grouped sensibly. Keep it concise and friendly — do not output JSON.\n\n"
        "Chosen plan (JSON): {recipe_plan?}\n"
        "Grocery list (JSON): {grocery_list?}"
    ),
    # No output_schema/tools: the earlier stages hold the validated structured data
    # in state; this stage just renders it conversationally as the final response.
    output_key="final_message",
    # HITL: don't render a plan the user rejected at the finalize gate.
    before_agent_callback=guard_presentation,
)


# --- deterministic workflow ----------------------------------------------------

gather_step = ParallelAgent(
    name="gather_step",
    description="Fetches nutrition targets and pantry contents concurrently.",
    sub_agents=[nutrition_agent, pantry_agent],
)

plan_workflow = SequentialAgent(
    name="plan_workflow",
    description=(
        "End-to-end weekly meal planning pipeline: gather targets + pantry, choose "
        "recipes, build the grocery list, then present it to the user."
    ),
    # recipe_agent and grocery_agent emit validated structured JSON into state;
    # finalize_agent gets explicit user approval (HITL); presenter_agent then
    # renders the approved state as the friendly final reply.
    sub_agents=[gather_step, recipe_agent, grocery_agent, finalize_agent, presenter_agent],
)


# --- coordinator ---------------------------------------------------------------

root_agent = LlmAgent(
    name="sous_coordinator",
    model=SMART_MODEL,
    description="Meal & nutrition concierge that plans meals and builds grocery lists.",
    instruction=(
        "You are Sous, a friendly meal & nutrition concierge. "
        "For a request to plan meals or build a grocery list, delegate to "
        "`plan_workflow`, which will compute targets, check the pantry, choose recipes "
        "and produce a shopping list. For simple pantry updates or questions about what "
        "the user already has, use `read_pantry`/`update_pantry` directly. Before asking "
        "the user to repeat preferences or constraints (allergies, dislikes, budget, cook "
        "time, past favourites), call `load_memory` to recall what they told you in earlier "
        "conversations. Always confirm the user's goal, constraints and how many meals they "
        "want before planning, using remembered preferences when available."
    ),
    tools=[read_pantry, update_pantry_tool, load_memory],
    sub_agents=[plan_workflow],
    # Context & memory (issue #5): trim history before each model call to bound the
    # token footprint, and ingest each finished turn into long-term memory so facts
    # the user revealed are recallable in future sessions.
    before_model_callback=compact_history,
    after_agent_callback=remember_session,
)
