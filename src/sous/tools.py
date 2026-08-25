"""Custom function tools for the Sous agents.

Each tool is a plain Python function with type hints and a descriptive docstring
(the ADK convention — the docstring guides the LLM on when to call it). Tools
return a dict with a ``"status"`` key (``"success"`` or ``"error"``) so the model
can reason about failures.

The tools contain no LLM calls, so they are fully unit-testable on their own.
Stateful tools (pantry) take an ADK ``ToolContext`` and read/write
``tool_context.state`` using the ``user:`` prefix, which persists across sessions.
"""

from __future__ import annotations

from google.adk.tools.tool_context import ToolContext

from .data import Recipe, load_recipes

PANTRY_KEY = "user:pantry"

# --- recipe search -------------------------------------------------------------


def _recipe_summary(r: Recipe) -> dict:
    return {
        "id": r.id,
        "name": r.name,
        "tags": list(r.tags),
        "allergens": list(r.allergens),
        "macros": {
            "kcal": r.macros.kcal,
            "protein_g": r.macros.protein_g,
            "carbs_g": r.macros.carbs_g,
            "fat_g": r.macros.fat_g,
        },
        "cook_time_min": r.cook_time_min,
        "est_cost_usd": r.est_cost_usd,
    }


def search_recipes(
    tags: list[str] | None = None,
    exclude_allergens: list[str] | None = None,
    max_cost_usd: float | None = None,
    max_cook_time_min: int | None = None,
) -> dict:
    """Search the recipe catalogue by dietary tags, allergens, cost and cook time.

    Args:
        tags: Recipe must have ALL of these tags (e.g. ["high-protein", "vegetarian"]).
        exclude_allergens: Drop recipes containing ANY of these allergens
            (e.g. ["shellfish", "peanut"]).
        max_cost_usd: Maximum estimated cost per serving in USD.
        max_cook_time_min: Maximum cook time in minutes.

    Returns:
        dict: {"status": "success", "recipes": [<recipe summary>, ...]}.
    """
    tagset = {t.lower() for t in (tags or [])}
    allergen_block = {a.lower() for a in (exclude_allergens or [])}

    results = []
    for r in load_recipes():
        r_tags = {t.lower() for t in r.tags}
        r_allergens = {a.lower() for a in r.allergens}
        if tagset and not tagset.issubset(r_tags):
            continue
        if allergen_block and allergen_block & r_allergens:
            continue
        if max_cost_usd is not None and r.est_cost_usd > max_cost_usd:
            continue
        if max_cook_time_min is not None and r.cook_time_min > max_cook_time_min:
            continue
        results.append(_recipe_summary(r))

    return {"status": "success", "recipes": results, "count": len(results)}


# --- nutrition targets ---------------------------------------------------------

_GOAL_ADJUST = {"lose": -400, "maintain": 0, "gain": 400}


def compute_nutrition_targets(
    goal: str,
    weight_kg: float,
    meals_per_day: int = 3,
) -> dict:
    """Compute daily calorie and macronutrient targets for a health goal.

    Uses a simple, transparent heuristic (≈30 kcal/kg maintenance) adjusted for the
    goal, with a protein-forward macro split (30% protein / 40% carbs / 30% fat).

    Args:
        goal: One of "lose", "maintain", or "gain".
        weight_kg: Body weight in kilograms.
        meals_per_day: Number of meals to divide the daily target across.

    Returns:
        dict: targets including daily_kcal and per-macro grams, or an error.
    """
    if goal not in _GOAL_ADJUST:
        return {
            "status": "error",
            "error_message": f"Unknown goal {goal!r}; expected one of {list(_GOAL_ADJUST)}.",
        }
    if weight_kg <= 0 or meals_per_day <= 0:
        return {"status": "error", "error_message": "weight_kg and meals_per_day must be > 0."}

    daily_kcal = round(weight_kg * 30 + _GOAL_ADJUST[goal])
    protein_g = round(daily_kcal * 0.30 / 4)
    carbs_g = round(daily_kcal * 0.40 / 4)
    fat_g = round(daily_kcal * 0.30 / 9)

    return {
        "status": "success",
        "goal": goal,
        "daily_kcal": daily_kcal,
        "protein_g": protein_g,
        "carbs_g": carbs_g,
        "fat_g": fat_g,
        "per_meal_kcal": round(daily_kcal / meals_per_day),
    }


# --- pantry state --------------------------------------------------------------


def _normalize(items: list[str]) -> list[str]:
    seen: list[str] = []
    for it in items:
        norm = it.strip().lower()
        if norm and norm not in seen:
            seen.append(norm)
    return seen


def read_pantry(tool_context: ToolContext) -> dict:
    """Return the items currently recorded in the user's pantry.

    Returns:
        dict: {"status": "success", "pantry": ["rice", "eggs", ...]}.
    """
    pantry = list(tool_context.state.get(PANTRY_KEY, []))
    return {"status": "success", "pantry": pantry}


def update_pantry(items: list[str], action: str, tool_context: ToolContext) -> dict:
    """Add or remove items from the user's pantry (persists across sessions).

    Args:
        items: Ingredient names to add or remove (e.g. ["chicken breast", "rice"]).
        action: "add" or "remove".
        tool_context: ADK tool context (state is written under the user: scope).

    Returns:
        dict: {"status": "success", "pantry": [...]} or an error.
    """
    if action not in ("add", "remove"):
        return {
            "status": "error",
            "error_message": f"Unknown action {action!r}; expected 'add' or 'remove'.",
        }

    current = _normalize(list(tool_context.state.get(PANTRY_KEY, [])))
    incoming = _normalize(items)

    if action == "add":
        updated = _normalize(current + incoming)
    else:  # remove
        updated = [i for i in current if i not in incoming]

    tool_context.state[PANTRY_KEY] = updated
    return {"status": "success", "pantry": updated}


# --- grocery list --------------------------------------------------------------


def build_grocery_list(recipe_ids: list[str], pantry: list[str]) -> dict:
    """Build a consolidated grocery list for a set of recipes, minus pantry items.

    Ingredients shared across recipes are combined (quantities summed when units
    match), and anything already in the pantry is dropped.

    Args:
        recipe_ids: Recipe ids to include in the plan.
        pantry: Ingredient names the user already has.

    Returns:
        dict: {"status": "success", "grocery_list": [{"item","qty","unit"}, ...]}
            or an error listing unknown recipe ids.
    """
    by_id = {r.id: r for r in load_recipes()}
    unknown = [rid for rid in recipe_ids if rid not in by_id]
    if unknown:
        return {
            "status": "error",
            "error_message": f"Unknown recipe id(s): {unknown}.",
        }

    have = {p.strip().lower() for p in pantry}
    # Aggregate by (item, unit) so mismatched units aren't silently merged.
    aggregated: dict[tuple[str, str], float] = {}
    for rid in recipe_ids:
        for ing in by_id[rid].ingredients:
            item = ing.item.strip().lower()
            if item in have:
                continue
            key = (item, ing.unit)
            aggregated[key] = aggregated.get(key, 0) + ing.qty

    grocery_list = [
        {"item": item, "qty": round(qty, 2), "unit": unit}
        for (item, unit), qty in sorted(aggregated.items())
    ]
    return {"status": "success", "grocery_list": grocery_list, "count": len(grocery_list)}
