"""Recipe dataset loader.

The dataset is a small, curated JSON file shipped *inside* the package
(``sous/resources/recipes.json``). Keeping it local makes the whole data + tool
layer deterministic and unit-testable with no API key or network — the project's
value is in orchestration and memory, not the data source. Shipping it as package
data means it resolves identically whether installed as a wheel or run from source.
"""

from __future__ import annotations

import functools
import json
from importlib import resources

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Controlled vocabulary for recipe tags. Keeps the dataset consistent and lets the
# search tool validate constraints against a known set.
RECIPE_TAGS: frozenset[str] = frozenset(
    {
        "high-protein",
        "low-carb",
        "keto",
        "vegetarian",
        "vegan",
        "pescatarian",
        "gluten-free",
        "dairy-free",
        "quick",  # <= 20 min
        "budget",  # cheap per serving
        "meal-prep",
    }
)

# Controlled vocabulary for allergens present in the dataset.
ALLERGENS: frozenset[str] = frozenset(
    {"fish", "shellfish", "soy", "gluten", "sesame", "peanut", "egg", "dairy"}
)


class _Strict(BaseModel):
    """Base for dataset models: reject unknown fields and stay immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Ingredient(_Strict):
    item: str = Field(min_length=1)
    qty: float = Field(gt=0)
    unit: str = Field(min_length=1)


class Macros(_Strict):
    kcal: float = Field(gt=0)
    protein_g: float = Field(ge=0)
    carbs_g: float = Field(ge=0)
    fat_g: float = Field(ge=0)


class Recipe(_Strict):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    tags: list[str] = Field(min_length=1)
    ingredients: list[Ingredient] = Field(min_length=1)
    macros: Macros
    cook_time_min: int = Field(gt=0)
    est_cost_usd: float = Field(gt=0)
    allergens: list[str] = Field(default_factory=list)

    @field_validator("tags")
    @classmethod
    def _known_tags(cls, tags: list[str]) -> list[str]:
        unknown = [t for t in tags if t not in RECIPE_TAGS]
        if unknown:
            raise ValueError(f"unknown tag(s) {unknown}; allowed: {sorted(RECIPE_TAGS)}")
        return tags

    @field_validator("allergens")
    @classmethod
    def _known_allergens(cls, allergens: list[str]) -> list[str]:
        unknown = [a for a in allergens if a not in ALLERGENS]
        if unknown:
            raise ValueError(f"unknown allergen(s) {unknown}; allowed: {sorted(ALLERGENS)}")
        return allergens


@functools.lru_cache(maxsize=1)
def load_recipes() -> list[Recipe]:
    """Load, parse and validate the recipe dataset. Cached — the dataset is static.

    Resolves the JSON via ``importlib.resources`` so it works identically from a
    source checkout or an installed wheel. Each record is validated against the
    strict ``Recipe`` schema, so malformed data fails fast at load time.
    """
    source = resources.files("sous").joinpath("resources/recipes.json")
    raw_recipes = json.loads(source.read_text(encoding="utf-8"))
    return [Recipe.model_validate(r) for r in raw_recipes]
