"""Runtime guardrail plugin tests (issue #7) — no live LLM.

The plugin's ``before_tool_callback`` is a plain async method, so we can drive it
directly with a fake tool and args — the same "test the callback, not the Runner"
approach used for the memory callbacks.
"""

from types import SimpleNamespace

from conftest import FakeToolContext
from sous.plugins import MAX_PANTRY_ITEMS_PER_CALL, PolicyPlugin


def _tool(name: str) -> SimpleNamespace:
    # before_tool_callback only reads ``tool.name``.
    return SimpleNamespace(name=name)


async def test_allows_normal_pantry_update():
    plugin = PolicyPlugin()
    result = await plugin.before_tool_callback(
        tool=_tool("update_pantry"),
        tool_args={"items": ["rice", "eggs"], "action": "add"},
        tool_context=FakeToolContext(),
    )
    # None => the guardrail abstains and the real tool runs.
    assert result is None


async def test_blocks_oversized_pantry_update():
    plugin = PolicyPlugin()
    oversized = [f"item{i}" for i in range(MAX_PANTRY_ITEMS_PER_CALL + 1)]
    result = await plugin.before_tool_callback(
        tool=_tool("update_pantry"),
        tool_args={"items": oversized, "action": "add"},
        tool_context=FakeToolContext(),
    )
    # A returned dict short-circuits execution: it becomes the tool's result and
    # the real update_pantry never runs (no state mutation).
    assert result is not None
    assert result["status"] == "error"
    assert str(MAX_PANTRY_ITEMS_PER_CALL) in result["error_message"]


async def test_ignores_unrelated_tools():
    plugin = PolicyPlugin()
    result = await plugin.before_tool_callback(
        tool=_tool("search_recipes"),
        tool_args={"tags": ["x"] * 999},
        tool_context=FakeToolContext(),
    )
    assert result is None


def test_plugin_registered_on_runner():
    # The guardrail must actually be wired onto the Runner to have any effect.
    from sous.runtime import build_runner

    runner = build_runner()
    assert any(isinstance(p, PolicyPlugin) for p in runner.plugin_manager.plugins)
