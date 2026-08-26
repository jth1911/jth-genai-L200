"""Phase 1 — data layer tests (pure, no LLM, no network)."""

import pytest
from pydantic import ValidationError

from sous.data import ALLERGENS, RECIPE_TAGS, Ingredient, Macros, Recipe, load_recipes


def _valid_recipe_dict() -> dict:
    return {
        "id": "test-dish",
        "name": "Test Dish",
        "tags": ["high-protein"],
        "ingredients": [{"item": "rice", "qty": 100, "unit": "g"}],
        "macros": {"kcal": 500, "protein_g": 30, "carbs_g": 50, "fat_g": 20},
        "cook_time_min": 20,
        "est_cost_usd": 3.0,
        "allergens": [],
    }


def test_load_recipes_returns_nonempty_list():
    recipes = load_recipes()
    assert isinstance(recipes, list)
    assert len(recipes) >= 15  # enough variety for meaningful weekly planning


def test_every_recipe_has_required_fields_and_types():
    for r in load_recipes():
        assert isinstance(r, Recipe)
        assert r.id and isinstance(r.id, str)
        assert r.name and isinstance(r.name, str)
        assert isinstance(r.tags, list) and r.tags
        assert isinstance(r.ingredients, list) and r.ingredients
        assert r.cook_time_min > 0
        assert r.est_cost_usd > 0


def test_recipe_ids_are_unique():
    ids = [r.id for r in load_recipes()]
    assert len(ids) == len(set(ids))


def test_macros_present_and_positive():
    for r in load_recipes():
        assert r.macros.kcal > 0
        assert r.macros.protein_g >= 0
        assert r.macros.carbs_g >= 0
        assert r.macros.fat_g >= 0


def test_tags_come_from_known_vocabulary():
    for r in load_recipes():
        for tag in r.tags:
            assert tag in RECIPE_TAGS, f"unknown tag {tag!r} in {r.id}"


def test_allergens_is_a_list_of_strings():
    for r in load_recipes():
        assert isinstance(r.allergens, list)
        assert all(isinstance(a, str) for a in r.allergens)


def test_dataset_covers_high_protein_and_vegetarian():
    recipes = load_recipes()
    assert any("high-protein" in r.tags for r in recipes)
    assert any("vegetarian" in r.tags for r in recipes)


def test_ingredients_have_item_qty_unit():
    for r in load_recipes():
        for ing in r.ingredients:
            assert ing.item and isinstance(ing.item, str)
            assert ing.qty > 0
            assert ing.unit and isinstance(ing.unit, str)


# --- strict validation of the data model --------------------------------------


def test_valid_recipe_dict_parses():
    r = Recipe.model_validate(_valid_recipe_dict())
    assert r.id == "test-dish"


def test_unknown_tag_is_rejected():
    bad = _valid_recipe_dict()
    bad["tags"] = ["high-protein", "not-a-real-tag"]
    with pytest.raises(ValidationError):
        Recipe.model_validate(bad)


def test_empty_tags_is_rejected():
    bad = _valid_recipe_dict()
    bad["tags"] = []
    with pytest.raises(ValidationError):
        Recipe.model_validate(bad)


def test_unknown_allergen_is_rejected():
    bad = _valid_recipe_dict()
    bad["allergens"] = ["moon-dust"]
    with pytest.raises(ValidationError):
        Recipe.model_validate(bad)


def test_extra_field_is_forbidden():
    bad = _valid_recipe_dict()
    bad["calories"] = 500  # typo'd/legacy key must not slip through
    with pytest.raises(ValidationError):
        Recipe.model_validate(bad)


@pytest.mark.parametrize("field, value", [("kcal", 0), ("protein_g", -1)])
def test_macros_bounds(field, value):
    m = {"kcal": 500, "protein_g": 30, "carbs_g": 50, "fat_g": 20}
    m[field] = value
    with pytest.raises(ValidationError):
        Macros.model_validate(m)


@pytest.mark.parametrize("field, value", [("qty", 0), ("qty", -5), ("item", ""), ("unit", "")])
def test_ingredient_bounds(field, value):
    ing = {"item": "rice", "qty": 100, "unit": "g"}
    ing[field] = value
    with pytest.raises(ValidationError):
        Ingredient.model_validate(ing)


@pytest.mark.parametrize("field, value", [("cook_time_min", 0), ("est_cost_usd", 0)])
def test_recipe_positive_bounds(field, value):
    bad = _valid_recipe_dict()
    bad[field] = value
    with pytest.raises(ValidationError):
        Recipe.model_validate(bad)


def test_vocabularies_are_frozensets_of_str():
    assert isinstance(RECIPE_TAGS, frozenset) and all(isinstance(t, str) for t in RECIPE_TAGS)
    assert isinstance(ALLERGENS, frozenset) and all(isinstance(a, str) for a in ALLERGENS)
