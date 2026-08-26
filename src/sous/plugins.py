"""Runtime policy guardrails for the Sous agents (issue #7).

Batch evaluation (``tests/test_eval.py``) checks behaviour offline; this plugin
enforces policy *at runtime*, while the agent is actually running. Unlike an
agent-local callback, a :class:`BasePlugin` is registered **once** on the
``Runner`` and its callbacks apply globally to every agent and tool — so
cross-cutting policy lives in one place instead of being threaded through each
agent. Plugin callbacks also run *before* agent-local callbacks, so this sits in
front of the coordinator's ``compact_history`` hook.

The enforcement primitive is the return value: ``before_tool_callback`` returning
``None`` lets the call proceed, while returning a ``dict`` short-circuits it —
the dict becomes the tool's result and the real tool never runs.
"""

from __future__ import annotations

from typing import Any

from google.adk.plugins import BasePlugin
from google.adk.tools import BaseTool
from google.adk.tools.tool_context import ToolContext

from .schemas import ErrorResult

# A single pantry write should only ever touch a handful of items. A far larger
# list is a sign of a hallucinated or runaway call, so we refuse it before it can
# mutate persisted state. This complements the HITL confirmation on the same tool:
# confirmation asks the user, this bounds the blast radius unconditionally.
MAX_PANTRY_ITEMS_PER_CALL = 25


class PolicyPlugin(BasePlugin):
    """Runtime guardrail: inspect and short-circuit out-of-policy tool calls."""

    def __init__(self) -> None:
        super().__init__(name="policy")

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any] | None:
        if tool.name == "update_pantry":
            items = tool_args.get("items") or []
            if len(items) > MAX_PANTRY_ITEMS_PER_CALL:
                return ErrorResult(
                    error_message=(
                        f"Refused by policy: a single pantry update may change at most "
                        f"{MAX_PANTRY_ITEMS_PER_CALL} items (got {len(items)}). "
                        "Split it into smaller updates."
                    )
                ).model_dump()
        return None  # abstain — let the tool run
