"""Shared test fixtures/doubles."""


class FakeToolContext:
    """Minimal stand-in for ADK's ToolContext.

    Tools only ever touch ``tool_context.state`` (a dict-like), so a plain dict
    wrapper is enough to unit-test stateful tools without spinning up a Runner.
    """

    def __init__(self, state: dict | None = None):
        self.state: dict = dict(state or {})
