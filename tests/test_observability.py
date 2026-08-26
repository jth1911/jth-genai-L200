"""Observability tests (issue #9) — no live LLM.

Structured JSON logging, PII redaction, telemetry content-capture defaults, and the
intent/outcome plugin are all exercised by capturing loguru output into a buffer and
driving the plugin callbacks directly.
"""

import io
import json
from types import SimpleNamespace

import pytest
from google.genai import types

from sous.observability import (
    MASK,
    ObservabilityPlugin,
    configure_logging,
    configure_telemetry,
    redact,
)


@pytest.fixture
def capture_logs():
    """Route loguru to a buffer for the test, then restore a safe default sink."""
    buf = io.StringIO()
    configure_logging(sink=buf, level="DEBUG")
    try:
        yield buf
    finally:
        # Reset to a valid sink so later tests logging via loguru don't hit a
        # closed buffer.
        configure_logging()


def _records(buf: io.StringIO) -> list[dict]:
    return [json.loads(line)["record"] for line in buf.getvalue().splitlines() if line.strip()]


# --- redaction -----------------------------------------------------------------


def test_redact_masks_pii_keys_and_recurses():
    out = redact(
        {
            "weight_kg": 80,
            "note": "ok",
            "args": {"exclude_allergens": ["peanut"], "tags": ["vegan"]},
            "list": [{"pantry": ["rice"]}],
        }
    )
    assert out["weight_kg"] == MASK
    assert out["note"] == "ok"  # non-PII untouched
    assert out["args"]["exclude_allergens"] == MASK  # nested PII masked
    assert out["args"]["tags"] == ["vegan"]  # nested non-PII kept
    assert out["list"][0]["pantry"] == MASK  # PII inside a list masked


# --- structured JSON logging + redaction ---------------------------------------


def test_logging_emits_json_with_context(capture_logs):
    from loguru import logger

    logger.bind(event="probe", note="hello").info("structured event")
    recs = _records(capture_logs)
    assert recs and recs[-1]["message"] == "structured event"
    assert recs[-1]["extra"]["event"] == "probe"
    assert recs[-1]["extra"]["note"] == "hello"


def test_logging_redacts_pii_globally(capture_logs):
    from loguru import logger

    logger.bind(user_message="I am allergic to peanuts", weight_kg=137.5).info("intent")
    raw = capture_logs.getvalue()
    # The raw PII values never reach the sink...
    assert "peanuts" not in raw
    assert "137.5" not in raw
    # ...and the structured fields are masked.
    extra = _records(capture_logs)[-1]["extra"]
    assert extra["user_message"] == MASK
    assert extra["weight_kg"] == MASK


# --- telemetry (trace) content-capture default ---------------------------------


def test_configure_telemetry_defaults_capture_off(monkeypatch):
    monkeypatch.delenv("ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS", raising=False)
    monkeypatch.delenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", raising=False)
    configure_telemetry()
    from google.adk.telemetry.context import TelemetryConfig

    assert TelemetryConfig().should_add_content_to_legacy_spans is False


def test_configure_telemetry_respects_explicit_optin(monkeypatch):
    monkeypatch.setenv("ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS", "true")
    configure_telemetry()  # must not override an explicit opt-in
    from google.adk.telemetry.context import TelemetryConfig

    assert TelemetryConfig().should_add_content_to_legacy_spans is True


# --- intent / outcome capture --------------------------------------------------


def _ctx(invocation_id="inv-1", session_id="sess-1"):
    return SimpleNamespace(
        invocation_id=invocation_id, session=SimpleNamespace(id=session_id)
    )


async def test_plugin_logs_intent_with_redacted_message(capture_logs):
    plugin = ObservabilityPlugin()
    msg = types.Content(role="user", parts=[types.Part(text="allergic to peanuts")])
    await plugin.on_user_message_callback(invocation_context=_ctx(), user_message=msg)

    rec = _records(capture_logs)[-1]
    assert rec["extra"]["event"] == "intent"
    assert rec["extra"]["invocation_id"] == "inv-1"
    assert rec["extra"]["user_message"] == MASK  # PII redacted
    assert "peanuts" not in capture_logs.getvalue()


async def test_plugin_logs_tool_outcome_with_redacted_args(capture_logs):
    plugin = ObservabilityPlugin()
    await plugin.after_tool_callback(
        tool=SimpleNamespace(name="update_pantry"),
        tool_args={"items": ["rice", "eggs"], "action": "add"},
        tool_context=SimpleNamespace(state={}),
        result={"status": "success", "pantry": ["rice", "eggs"]},
    )
    rec = _records(capture_logs)[-1]
    assert rec["extra"]["event"] == "tool_outcome"
    assert rec["extra"]["tool"] == "update_pantry"
    assert rec["extra"]["status"] == "success"
    # PII in args and result is masked (items / pantry are PII keys).
    assert rec["extra"]["args"]["items"] == MASK
    assert rec["extra"]["result"]["pantry"] == MASK


async def test_plugin_logs_run_outcome(capture_logs):
    plugin = ObservabilityPlugin()
    await plugin.after_run_callback(invocation_context=_ctx(invocation_id="inv-9"))
    rec = _records(capture_logs)[-1]
    assert rec["extra"]["event"] == "outcome"
    assert rec["extra"]["invocation_id"] == "inv-9"


def test_observability_plugin_registered_on_runner():
    from sous.observability import ObservabilityPlugin as OP
    from sous.runtime import build_runner

    runner = build_runner()
    assert any(isinstance(p, OP) for p in runner.plugin_manager.plugins)
