"""Phase 3 — agent & orchestration structure tests (no API calls)."""

from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent

from sous.agent import (
    grocery_agent,
    nutrition_agent,
    pantry_agent,
    plan_workflow,
    presenter_agent,
    recipe_agent,
    root_agent,
)


def _tool_names(agent) -> list[str]:
    names = []
    for t in agent.tools:
        if hasattr(t, "func"):
            names.append(t.func.__name__)
        elif hasattr(t, "__name__"):
            names.append(t.__name__)
        else:
            names.append(getattr(t, "name", None))
    return names


def test_root_is_llm_coordinator():
    assert isinstance(root_agent, LlmAgent)
    assert root_agent.name == "sous_coordinator"


def test_coordinator_delegates_to_plan_workflow():
    # LLM-driven delegation: the workflow is a sub-agent the coordinator can transfer to.
    assert plan_workflow in root_agent.sub_agents


def test_coordinator_handles_pantry_directly():
    # Quick memory updates don't need the full planning workflow.
    assert set(_tool_names(root_agent)) >= {"read_pantry", "update_pantry"}


def test_coordinator_can_recall_long_term_memory():
    # The coordinator can search past conversations for remembered preferences.
    assert "load_memory" in _tool_names(root_agent)


def test_coordinator_has_context_and_memory_callbacks():
    # before_model compaction bounds the token footprint; after_agent ingestion
    # grows long-term memory (issue #5).
    from sous.memory import compact_history, remember_session

    assert root_agent.before_model_callback is compact_history
    assert root_agent.after_agent_callback is remember_session


def test_plan_workflow_is_sequential():
    assert isinstance(plan_workflow, SequentialAgent)
    # gather -> recipe -> grocery -> finalize (HITL) -> presenter
    assert len(plan_workflow.sub_agents) == 5


def test_plan_workflow_starts_with_parallel_gather():
    gather = plan_workflow.sub_agents[0]
    assert isinstance(gather, ParallelAgent)
    gather_names = {a.name for a in gather.sub_agents}
    assert gather_names == {"nutrition_agent", "pantry_agent"}


def test_recipe_then_grocery_then_finalize_then_presenter_order():
    assert plan_workflow.sub_agents[1] is recipe_agent
    assert plan_workflow.sub_agents[2] is grocery_agent
    # HITL approval sits between the plan being built and it being presented.
    assert plan_workflow.sub_agents[3].name == "finalize_agent"
    assert plan_workflow.sub_agents[4] is presenter_agent


def test_presenter_renders_prose_not_structured_json():
    # The terminal stage stays free-text so the user gets a friendly reply, while
    # the structured data lives in state (recipe_plan / grocery_list).
    assert presenter_agent.output_schema is None
    assert not presenter_agent.tools


def test_each_specialist_has_its_tools():
    assert "compute_nutrition_targets" in _tool_names(nutrition_agent)
    assert set(_tool_names(pantry_agent)) >= {"read_pantry", "update_pantry"}
    assert "search_recipes" in _tool_names(recipe_agent)
    assert "build_grocery_list" in _tool_names(grocery_agent)


def test_pipeline_agents_write_output_keys():
    # State keys let each stage pass results to the next (context sharing).
    assert nutrition_agent.output_key
    assert pantry_agent.output_key
    assert recipe_agent.output_key
    assert grocery_agent.output_key


def test_pipeline_agents_have_structured_output_schemas():
    from sous.schemas import GroceryPlan, RecipePlan

    # recipe/grocery stages emit validated JSON, not free text.
    assert recipe_agent.output_schema is RecipePlan
    assert grocery_agent.output_schema is GroceryPlan


def test_all_agents_have_descriptions_for_delegation():
    # Descriptions are what the coordinator LLM uses to decide where to route.
    for agent in (nutrition_agent, pantry_agent, recipe_agent, grocery_agent, plan_workflow):
        assert agent.description


# --- strategic model routing (issue #7) ---------------------------------------


def test_model_tiers_route_by_task_complexity():
    """Reasoning-heavy stages run on the smart tier; simple specialists on fast."""
    from sous.agent import FAST_MODEL, SMART_MODEL

    assert FAST_MODEL and SMART_MODEL and FAST_MODEL != SMART_MODEL
    # Reasoning-heavy: recipe selection, grocery aggregation, coordinator routing.
    assert recipe_agent.model == SMART_MODEL
    assert grocery_agent.model == SMART_MODEL
    assert root_agent.model == SMART_MODEL
    # Simple specialists / rendering: cheaper, faster tier.
    assert nutrition_agent.model == FAST_MODEL
    assert pantry_agent.model == FAST_MODEL
    assert presenter_agent.model == FAST_MODEL


def test_model_tier_resolution_honours_env(monkeypatch):
    from sous.agent import _tier_models

    monkeypatch.setenv("SOUS_FAST_MODEL", "fast-x")
    monkeypatch.setenv("SOUS_SMART_MODEL", "smart-y")
    assert _tier_models() == ("fast-x", "smart-y")


def test_sous_model_is_backcompat_override_for_both_tiers(monkeypatch):
    # The pre-existing single SOUS_MODEL knob still works, pinning both tiers.
    from sous.agent import _tier_models

    monkeypatch.delenv("SOUS_FAST_MODEL", raising=False)
    monkeypatch.delenv("SOUS_SMART_MODEL", raising=False)
    monkeypatch.setenv("SOUS_MODEL", "legacy-model")
    assert _tier_models() == ("legacy-model", "legacy-model")
