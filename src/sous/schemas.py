"""Strict Pydantic schemas for tool input and output validation.

These give the tools machine-checked contracts on top of their type hints:
inputs are range/enum constrained and reject unknown fields; outputs are built
from typed models so their shape can't silently drift. Validation failures are
turned into the tools' guided ``{"status": "error", ...}`` responses rather than
raised, so the LLM gets an actionable message instead of a traceback.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Constrained enums — used directly in tool signatures so ADK surfaces them as
# JSON-schema ``enum`` constraints to the model.
Goal = Literal["lose", "maintain", "gain"]
PantryAction = Literal["add", "remove"]


class _StrictModel(BaseModel):
    """Reject unknown fields so typos/hallucinated args fail loudly."""

    model_config = ConfigDict(extra="forbid")


# --- tool inputs ---------------------------------------------------------------


class SearchRecipesInput(_StrictModel):
    tags: list[str] | None = None
    exclude_allergens: list[str] | None = None
    max_cost_usd: float | None = Field(default=None, gt=0)
    max_cook_time_min: int | None = Field(default=None, gt=0)


class NutritionInput(_StrictModel):
    goal: Goal
    weight_kg: float = Field(gt=0, le=500)
    meals_per_day: int = Field(default=3, ge=1, le=6)


class PantryUpdateInput(_StrictModel):
    items: list[str] = Field(min_length=1)
    action: PantryAction


class GroceryInput(_StrictModel):
    recipe_ids: list[str] = Field(min_length=1)
    pantry: list[str] = Field(default_factory=list)


# --- tool outputs --------------------------------------------------------------


class MacrosOut(_StrictModel):
    kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float


class RecipeSummary(_StrictModel):
    id: str
    name: str
    tags: list[str]
    allergens: list[str]
    macros: MacrosOut
    cook_time_min: int
    est_cost_usd: float


class SearchRecipesResult(_StrictModel):
    status: Literal["success"] = "success"
    recipes: list[RecipeSummary]
    count: int


class NutritionTargets(_StrictModel):
    status: Literal["success"] = "success"
    goal: Goal
    daily_kcal: int
    protein_g: int
    carbs_g: int
    fat_g: int
    per_meal_kcal: int


class PantryState(_StrictModel):
    status: Literal["success"] = "success"
    pantry: list[str]


class GroceryItem(_StrictModel):
    item: str
    qty: float
    unit: str


class GroceryList(_StrictModel):
    status: Literal["success"] = "success"
    grocery_list: list[GroceryItem]
    count: int


class ErrorResult(_StrictModel):
    status: Literal["error"] = "error"
    error_message: str


# --- agent (stage) output schemas ---------------------------------------------


class PlannedMeal(_StrictModel):
    recipe_id: str
    name: str


class RecipePlan(_StrictModel):
    """Structured output of the recipe-planning stage."""

    meals: list[PlannedMeal]
    notes: str | None = None


class GroceryLineItem(_StrictModel):
    item: str
    qty: float
    unit: str


class GroceryPlan(_StrictModel):
    """Structured output of the grocery stage."""

    grocery_list: list[GroceryLineItem]
