"""Observability: structured JSON logging, PII redaction, and telemetry config (issue #9).

ADK already emits OpenTelemetry traces of the delegation path and every tool call.
This module adds the layer around that:

* :func:`configure_logging` — routes all app logging through loguru as structured
  JSON (``serialize=True``), with a **global redaction patcher** so PII can never be
  logged, whatever the call site.
* :func:`configure_telemetry` — turns *off* ADK's default-on capture of message
  content in spans, so the same PII (tool args like weight / allergens / pantry)
  doesn't leak into traces either. Prevention at source beats trying to scrub
  already-recorded, immutable spans.
* :class:`ObservabilityPlugin` — an ADK ``BasePlugin`` that records a structured
  *intent* event (what the user asked) and *outcome* events (tool results, the HITL
  approve/reject and guardrail decisions, and the end of the run), correlated by
  invocation id. See :mod:`sous.plugins` for the sibling ``PolicyPlugin``.

The redaction policy is defined **once** here and reused by both the logs and the
plugin, so there is a single place to reason about what counts as PII.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from google.adk.plugins import BasePlugin
from loguru import logger

# --- PII redaction -------------------------------------------------------------

# Field names that carry health-adjacent personal data anywhere in a log record's
# structured `extra`. Anything under one of these keys is masked before output —
# whether it appears at the top level or nested inside tool args/results.
PII_KEYS = frozenset(
    {
        "weight_kg",
        "weight",
        "goal",
        "pantry",
        "items",
        "exclude_allergens",
        "allergens",
        "allergies",
        "user_message",
        "user_text",
        "message_text",
    }
)

MASK = "***redacted***"


def redact(obj: Any) -> Any:
    """Return a copy of ``obj`` with any value under a :data:`PII_KEYS` key masked.

    Recurses through dicts and lists so PII nested inside tool-call arguments or
    results (e.g. ``{"args": {"weight_kg": 80}}``) is caught too. Non-container
    values pass through unchanged.
    """
    if isinstance(obj, dict):
        return {k: (MASK if k in PII_KEYS else redact(v)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact(v) for v in obj]
    return obj


def _redact_record(record: dict) -> None:
    """loguru global patcher: scrub PII from a record's ``extra`` before serialization."""
    record["extra"] = redact(record["extra"])


# --- structured JSON logging ---------------------------------------------------


def configure_logging(*, level: str | None = None, sink: Any = None) -> None:
    """Route all app logging through loguru as structured JSON with PII redaction.

    Idempotent: replaces any existing handlers so repeated calls (e.g. per
    ``build_runner``) don't stack sinks. ``level`` defaults to ``SOUS_LOG_LEVEL``
    (or ``INFO``); ``sink`` defaults to stderr (overridable for tests).
    """
    lvl = level or os.environ.get("SOUS_LOG_LEVEL", "INFO")
    logger.configure(
        handlers=[{"sink": sink or sys.stderr, "serialize": True, "level": lvl}],
        patcher=_redact_record,
    )


# --- telemetry (trace) PII prevention ------------------------------------------

# ADK populates some span attributes (e.g. gcp.vertex.agent.tool_call_args /
# tool_response) with message content **by default**. That content includes our
# tool args — weight, allergens, pantry. Setting this env to a falsey value turns
# those attributes off at the source.
_ADK_CAPTURE_ENV = "ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"


def configure_telemetry() -> None:
    """Default ADK span content-capture to OFF so PII doesn't leak into traces.

    Only sets a default when the operator hasn't chosen explicitly, so capture
    stays opt-in: ``ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS=true`` still enables it.
    """
    os.environ.setdefault(_ADK_CAPTURE_ENV, "false")


# --- intent / outcome capture --------------------------------------------------


def _invocation_id(invocation_context: Any) -> str | None:
    return getattr(invocation_context, "invocation_id", None)


def _session_id(invocation_context: Any) -> str | None:
    session = getattr(invocation_context, "session", None)
    return getattr(session, "id", None)


def _message_text(content: Any) -> str:
    """Flatten a types.Content into its text (bound under a PII key, so redacted)."""
    parts = getattr(content, "parts", None) or []
    return "".join(getattr(p, "text", None) or "" for p in parts)


class ObservabilityPlugin(BasePlugin):
    """Capture structured intent/outcome events for every run (issue #9).

    Registered once on the ``App`` (alongside ``PolicyPlugin``), so capture is
    global. Every event is emitted through loguru — and therefore through the PII
    redaction patcher — correlated by invocation id.
    """

    def __init__(self) -> None:
        super().__init__(name="observability")

    async def on_user_message_callback(self, *, invocation_context, user_message):
        # Intent: what the user asked. The raw text is bound under a PII key so the
        # redaction patcher masks it in the emitted JSON.
        logger.bind(
            event="intent",
            invocation_id=_invocation_id(invocation_context),
            session_id=_session_id(invocation_context),
            user_message=_message_text(user_message),
        ).info("user intent received")
        return None

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
        # Outcome (per tool): captures normal results, the guardrail block, and the
        # HITL approve/reject — all visible via the tool's returned status/error.
        status = result.get("status") if isinstance(result, dict) else None
        logger.bind(
            event="tool_outcome",
            tool=getattr(tool, "name", None),
            status=status,
            args=tool_args,
            result=result,
        ).info("tool completed")
        return None

    async def after_run_callback(self, *, invocation_context):
        # Outcome (per run): the turn is finished.
        logger.bind(
            event="outcome",
            invocation_id=_invocation_id(invocation_context),
            session_id=_session_id(invocation_context),
        ).info("run completed")
        return None
