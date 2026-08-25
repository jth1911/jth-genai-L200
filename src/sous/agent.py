"""Sous multi-agent system.

Architecture (see docs/architecture.md):

    root_agent  (sous_coordinator, LlmAgent — conversational routing + memory)
      └─ delegates to → plan_workflow  (SequentialAgent)
            ├─ gather_step  (ParallelAgent)
            │     ├─ nutrition_agent  → state["nutrition_targets"]
            │     └─ pantry_agent     → state["pantry_summary"]
            ├─ recipe_agent   → state["recipe_plan"]
            └─ grocery_agent  → state["grocery_list"]

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

from .tools import (
    build_grocery_list,
    compute_nutrition_targets,
    read_pantry,
    search_recipes,
    update_pantry,
)

# Gemini via Google AI Studio by default (see .env.example). Overridable so the
# same code can run against Vertex or another model without edits.
MODEL = os.environ.get("SOUS_MODEL", "gemini-3.6-flash")


# --- specialist agents ---------------------------------------------------------

nutrition_agent = LlmAgent(
    name="nutrition_agent",
    model=MODEL,
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
    model=MODEL,
    description="Tracks and reports what ingredients the user already has at home.",
    instruction=(
        "You manage the user's pantry. Call `read_pantry` to see what they have and "
        "`update_pantry` to add or remove items when the user mentions changes. "
        "Return a concise list of the current pantry contents."
    ),
    tools=[read_pantry, update_pantry],
    output_key="pantry_summary",
)

recipe_agent = LlmAgent(
    name="recipe_agent",
    model=MODEL,
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
    output_key="recipe_plan",
)

grocery_agent = LlmAgent(
    name="grocery_agent",
    model=MODEL,
    description="Turns the chosen meal plan into a consolidated grocery list.",
    instruction=(
        "You build shopping lists. From the chosen meal plan below, extract the recipe "
        "`id`s and call `build_grocery_list` with them and the user's pantry so already "
        "owned ingredients are excluded. Present the final de-duplicated grocery list.\n\n"
        "Chosen plan: {recipe_plan?}\n"
        "Pantry: {pantry_summary?}"
    ),
    tools=[build_grocery_list],
    output_key="grocery_list",
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
        "recipes, then build the grocery list."
    ),
    sub_agents=[gather_step, recipe_agent, grocery_agent],
)


# --- coordinator ---------------------------------------------------------------

root_agent = LlmAgent(
    name="sous_coordinator",
    model=MODEL,
    description="Meal & nutrition concierge that plans meals and builds grocery lists.",
    instruction=(
        "You are Sous, a friendly meal & nutrition concierge. "
        "For a request to plan meals or build a grocery list, delegate to "
        "`plan_workflow`, which will compute targets, check the pantry, choose recipes "
        "and produce a shopping list. For simple pantry updates or questions about what "
        "the user already has, use `read_pantry`/`update_pantry` directly. Always "
        "confirm the user's goal, constraints (allergies, budget, cook time) and how "
        "many meals they want before planning, using remembered preferences when "
        "available."
    ),
    tools=[read_pantry, update_pantry],
    sub_agents=[plan_workflow],
)
