"""Phase 1 — data layer tests (pure, no LLM, no network)."""

from sous.data import RECIPE_TAGS, Recipe, load_recipes


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
