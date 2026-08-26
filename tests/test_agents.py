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


def test_plan_workflow_is_sequential():
    assert isinstance(plan_workflow, SequentialAgent)
    # gather -> recipe -> grocery -> presenter
    assert len(plan_workflow.sub_agents) == 4


def test_plan_workflow_starts_with_parallel_gather():
    gather = plan_workflow.sub_agents[0]
    assert isinstance(gather, ParallelAgent)
    gather_names = {a.name for a in gather.sub_agents}
    assert gather_names == {"nutrition_agent", "pantry_agent"}


def test_recipe_then_grocery_then_presenter_order():
    assert plan_workflow.sub_agents[1] is recipe_agent
    assert plan_workflow.sub_agents[2] is grocery_agent
    assert plan_workflow.sub_agents[3] is presenter_agent


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
