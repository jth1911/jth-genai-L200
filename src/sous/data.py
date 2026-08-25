"""Recipe dataset loader.

The dataset is a small, curated, local JSON file (``data/recipes.json``). Keeping
it local makes the whole data + tool layer deterministic and unit-testable with no
API key or network — the project's value is in orchestration and memory, not the
data source.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

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


@dataclass(frozen=True)
class Ingredient:
    item: str
    qty: float
    unit: str


@dataclass(frozen=True)
class Macros:
    kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float


@dataclass(frozen=True)
class Recipe:
    id: str
    name: str
    tags: list[str]
    ingredients: list[Ingredient]
    macros: Macros
    cook_time_min: int
    est_cost_usd: float
    allergens: list[str] = field(default_factory=list)


def _dataset_path() -> Path:
    """Locate ``data/recipes.json`` whether running from a source checkout or an
    installed wheel."""
    # Source checkout: repo_root/data/recipes.json (src/sous/data.py -> repo root)
    checkout = Path(__file__).resolve().parents[2] / "data" / "recipes.json"
    if checkout.exists():
        return checkout
    # Installed package: bundled alongside the package.
    return Path(str(resources.files("sous").joinpath("data/recipes.json")))


def _parse_recipe(raw: dict) -> Recipe:
    return Recipe(
        id=raw["id"],
        name=raw["name"],
        tags=list(raw["tags"]),
        ingredients=[Ingredient(**i) for i in raw["ingredients"]],
        macros=Macros(**raw["macros"]),
        cook_time_min=int(raw["cook_time_min"]),
        est_cost_usd=float(raw["est_cost_usd"]),
        allergens=list(raw.get("allergens", [])),
    )


@functools.lru_cache(maxsize=1)
def load_recipes() -> list[Recipe]:
    """Load and parse the recipe dataset. Cached — the dataset is static."""
    path = _dataset_path()
    with open(path, encoding="utf-8") as fh:
        raw_recipes = json.load(fh)
    return [_parse_recipe(r) for r in raw_recipes]
