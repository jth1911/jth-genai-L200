"""Phase 2 — tool tests (pure logic, no LLM, no network)."""

from conftest import FakeToolContext
from sous.tools import (
    build_grocery_list,
    compute_nutrition_targets,
    read_pantry,
    search_recipes,
    update_pantry,
)

# --- search_recipes ------------------------------------------------------------


def test_search_by_tag_returns_only_matching():
    res = search_recipes(tags=["vegetarian"])
    assert res["status"] == "success"
    assert res["recipes"]
    assert all("vegetarian" in r["tags"] for r in res["recipes"])


def test_search_requires_all_requested_tags():
    res = search_recipes(tags=["high-protein", "low-carb"])
    assert all(
        "high-protein" in r["tags"] and "low-carb" in r["tags"] for r in res["recipes"]
    )


def test_search_excludes_allergens():
    res = search_recipes(exclude_allergens=["shellfish", "fish"])
    for r in res["recipes"]:
        assert "shellfish" not in r["allergens"]
        assert "fish" not in r["allergens"]


def test_search_respects_max_cost():
    res = search_recipes(max_cost_usd=3.0)
    assert res["recipes"]
    assert all(r["est_cost_usd"] <= 3.0 for r in res["recipes"])


def test_search_respects_max_cook_time():
    res = search_recipes(max_cook_time_min=15)
    assert all(r["cook_time_min"] <= 15 for r in res["recipes"])


def test_search_no_match_returns_empty_success():
    res = search_recipes(tags=["keto"], max_cost_usd=0.01)
    assert res["status"] == "success"
    assert res["recipes"] == []


# --- compute_nutrition_targets -------------------------------------------------


def test_nutrition_targets_lose_below_maintain():
    lose = compute_nutrition_targets(goal="lose", weight_kg=80)
    maintain = compute_nutrition_targets(goal="maintain", weight_kg=80)
    assert lose["daily_kcal"] < maintain["daily_kcal"]


def test_nutrition_targets_gain_above_maintain():
    gain = compute_nutrition_targets(goal="gain", weight_kg=80)
    maintain = compute_nutrition_targets(goal="maintain", weight_kg=80)
    assert gain["daily_kcal"] > maintain["daily_kcal"]


def test_nutrition_targets_macros_sum_to_calories():
    t = compute_nutrition_targets(goal="maintain", weight_kg=75)
    kcal_from_macros = (
        t["protein_g"] * 4 + t["carbs_g"] * 4 + t["fat_g"] * 9
    )
    assert abs(kcal_from_macros - t["daily_kcal"]) <= 30  # rounding tolerance


def test_nutrition_targets_invalid_goal_errors():
    res = compute_nutrition_targets(goal="bulk-forever", weight_kg=80)
    assert res["status"] == "error"


def test_nutrition_targets_per_meal_present():
    t = compute_nutrition_targets(goal="maintain", weight_kg=75, meals_per_day=3)
    assert t["per_meal_kcal"] == round(t["daily_kcal"] / 3)


# --- pantry state --------------------------------------------------------------


def test_read_pantry_empty_by_default():
    ctx = FakeToolContext()
    res = read_pantry(tool_context=ctx)
    assert res["status"] == "success"
    assert res["pantry"] == []


def test_update_pantry_add_persists_to_user_state():
    ctx = FakeToolContext()
    update_pantry(items=["rice", "eggs"], action="add", tool_context=ctx)
    # Persisted under a user-scoped key so it survives across sessions.
    assert "user:pantry" in ctx.state
    assert set(read_pantry(tool_context=ctx)["pantry"]) == {"rice", "eggs"}


def test_update_pantry_add_is_idempotent_and_normalized():
    ctx = FakeToolContext()
    update_pantry(items=["Rice"], action="add", tool_context=ctx)
    update_pantry(items=["rice"], action="add", tool_context=ctx)
    assert read_pantry(tool_context=ctx)["pantry"] == ["rice"]


def test_update_pantry_remove():
    ctx = FakeToolContext(state={"user:pantry": ["rice", "eggs", "milk"]})
    update_pantry(items=["milk"], action="remove", tool_context=ctx)
    assert set(read_pantry(tool_context=ctx)["pantry"]) == {"rice", "eggs"}


def test_update_pantry_invalid_action_errors():
    ctx = FakeToolContext()
    res = update_pantry(items=["rice"], action="sprinkle", tool_context=ctx)
    assert res["status"] == "error"


# --- build_grocery_list --------------------------------------------------------


def test_grocery_list_aggregates_across_recipes():
    # Both bowls use quinoa/brown rice etc.; pick two known recipes.
    res = build_grocery_list(
        recipe_ids=["grilled-chicken-quinoa-bowl", "salmon-sweet-potato"],
        pantry=[],
    )
    assert res["status"] == "success"
    items = {i["item"] for i in res["grocery_list"]}
    assert "chicken breast" in items
    assert "salmon fillet" in items


def test_grocery_list_combines_duplicate_ingredients():
    # olive oil appears in both recipes -> should be a single combined line.
    res = build_grocery_list(
        recipe_ids=["grilled-chicken-quinoa-bowl", "steak-green-salad"],
        pantry=[],
    )
    olive = [i for i in res["grocery_list"] if i["item"] == "olive oil"]
    assert len(olive) == 1
    assert olive[0]["qty"] == 2  # 1 tbsp + 1 tbsp


def test_grocery_list_subtracts_pantry_items():
    res = build_grocery_list(
        recipe_ids=["grilled-chicken-quinoa-bowl"],
        pantry=["olive oil", "quinoa"],
    )
    items = {i["item"] for i in res["grocery_list"]}
    assert "olive oil" not in items
    assert "quinoa" not in items
    assert "chicken breast" in items


def test_grocery_list_unknown_recipe_reports_error():
    res = build_grocery_list(recipe_ids=["does-not-exist"], pantry=[])
    assert res["status"] == "error"
