"""Shared test fixtures/doubles."""


class FakeToolContext:
    """Minimal stand-in for ADK's ToolContext.

    Tools only ever touch ``tool_context.state`` (a dict-like), so a plain dict
    wrapper is enough to unit-test stateful tools without spinning up a Runner.
    """

    def __init__(self, state: dict | None = None):
        self.state: dict = dict(state or {})


class FakeCallbackContext:
    """Minimal stand-in for ADK's ``CallbackContext``.

    The memory callbacks only touch ``callback_context.state`` and the async
    ``add_session_to_memory()`` coroutine, so this records the latter's calls and
    exposes a dict-like ``state`` — enough to unit-test the callbacks without a
    Runner or a live memory service.
    """

    def __init__(self, state: dict | None = None, *, memory_available: bool = True):
        self.state: dict = dict(state or {})
        self.memory_available = memory_available
        self.add_session_calls = 0

    async def add_session_to_memory(self) -> None:
        if not self.memory_available:
            raise ValueError("memory service is not available")
        self.add_session_calls += 1
