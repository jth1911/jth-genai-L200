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
from pydantic import BaseModel, ValidationError

from .data import Recipe, load_recipes
from .schemas import (
    ErrorResult,
    Goal,
    GroceryInput,
    GroceryItem,
    GroceryList,
    NutritionInput,
    NutritionTargets,
    PantryAction,
    PantryState,
    PantryUpdateInput,
    RecipeSummary,
    SearchRecipesInput,
    SearchRecipesResult,
)

PANTRY_KEY = "user:pantry"


def _validate(model_cls: type[BaseModel], **kwargs):
    """Validate tool arguments against a schema.

    Returns ``(model, None)`` on success or ``(None, error_dict)`` on failure —
    the error dict is the tools' guided ``{"status": "error", ...}`` shape, so a
    validation failure reaches the LLM as an actionable message, never a traceback.
    """
    try:
        return model_cls(**kwargs), None
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'input'}: {err['msg']}"
            for err in exc.errors()
        )
        return None, ErrorResult(error_message=f"Invalid arguments — {details}").model_dump()

# --- recipe search -------------------------------------------------------------


def _recipe_summary(r: Recipe) -> RecipeSummary:
    return RecipeSummary(
        id=r.id,
        name=r.name,
        tags=list(r.tags),
        allergens=list(r.allergens),
        macros=r.macros.model_dump(),
        cook_time_min=r.cook_time_min,
        est_cost_usd=r.est_cost_usd,
    )


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
    args, err = _validate(
        SearchRecipesInput,
        tags=tags,
        exclude_allergens=exclude_allergens,
        max_cost_usd=max_cost_usd,
        max_cook_time_min=max_cook_time_min,
    )
    if err:
        return err
    tags, exclude_allergens = args.tags, args.exclude_allergens
    max_cost_usd, max_cook_time_min = args.max_cost_usd, args.max_cook_time_min

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

    return SearchRecipesResult(recipes=results, count=len(results)).model_dump()


# --- nutrition targets ---------------------------------------------------------

_GOAL_ADJUST = {"lose": -400, "maintain": 0, "gain": 400}


def compute_nutrition_targets(
    goal: Goal,
    weight_kg: float,
    meals_per_day: int = 3,
) -> dict:
    """Compute daily calorie and macronutrient targets for a health goal.

    Uses a simple, transparent heuristic (≈30 kcal/kg maintenance) adjusted for the
    goal, with a protein-forward macro split (30% protein / 40% carbs / 30% fat).

    Args:
        goal: One of "lose", "maintain", or "gain".
        weight_kg: Body weight in kilograms (0 < weight_kg <= 500).
        meals_per_day: Number of meals to divide the daily target across (1-6).

    Returns:
        dict: targets including daily_kcal and per-macro grams, or an error.
    """
    args, err = _validate(
        NutritionInput, goal=goal, weight_kg=weight_kg, meals_per_day=meals_per_day
    )
    if err:
        return err
    goal, weight_kg, meals_per_day = args.goal, args.weight_kg, args.meals_per_day

    daily_kcal = round(weight_kg * 30 + _GOAL_ADJUST[goal])
    protein_g = round(daily_kcal * 0.30 / 4)
    carbs_g = round(daily_kcal * 0.40 / 4)
    fat_g = round(daily_kcal * 0.30 / 9)

    return NutritionTargets(
        goal=goal,
        daily_kcal=daily_kcal,
        protein_g=protein_g,
        carbs_g=carbs_g,
        fat_g=fat_g,
        per_meal_kcal=round(daily_kcal / meals_per_day),
    ).model_dump()


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
    return PantryState(pantry=pantry).model_dump()


def update_pantry(items: list[str], action: PantryAction, tool_context: ToolContext) -> dict:
    """Add or remove items from the user's pantry (persists across sessions).

    Args:
        items: Ingredient names to add or remove (e.g. ["chicken breast", "rice"]).
        action: "add" or "remove".
        tool_context: ADK tool context (state is written under the user: scope).

    Returns:
        dict: {"status": "success", "pantry": [...]} or an error.
    """
    args, err = _validate(PantryUpdateInput, items=items, action=action)
    if err:
        return err
    items, action = args.items, args.action

    current = _normalize(list(tool_context.state.get(PANTRY_KEY, [])))
    incoming = _normalize(items)

    if action == "add":
        updated = _normalize(current + incoming)
    else:  # remove
        updated = [i for i in current if i not in incoming]

    tool_context.state[PANTRY_KEY] = updated
    return PantryState(pantry=updated).model_dump()


# --- plan finalization (human-in-the-loop) -------------------------------------


def finalize_plan(summary: str) -> dict:
    """Finalize the meal plan and grocery list once the user has approved them.

    This is the last gate before the plan is presented. It carries no side effects
    of its own — its purpose is to force an explicit user approval (the tool is
    confirmation-gated), so the plan is only committed to when the user says yes.

    Args:
        summary: A short, human-readable summary of the plan and grocery list the
            user is being asked to approve.

    Returns:
        dict: {"status": "approved", "summary": <summary>}.
    """
    return {"status": "approved", "summary": summary}


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
    args, err = _validate(GroceryInput, recipe_ids=recipe_ids, pantry=pantry)
    if err:
        return err
    recipe_ids, pantry = args.recipe_ids, args.pantry

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
        GroceryItem(item=item, qty=round(qty, 2), unit=unit)
        for (item, unit), qty in sorted(aggregated.items())
    ]
    return GroceryList(grocery_list=grocery_list, count=len(grocery_list)).model_dump()
