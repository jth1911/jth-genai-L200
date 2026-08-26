"""Context & memory management for the coordinator (issue #5).

Two ADK-native callbacks keep the concierge's context bounded and its long-term
memory growing:

* :func:`compact_history` — a ``before_model_callback`` that trims the coordinator's
  conversation history to a sliding window before each LLM call, optionally
  summarising the dropped prefix into a compact running summary. This bounds the
  token footprint of long chats. It only touches the transient conversation
  (``llm_request.contents``); durable ``user:``-scoped state lives in
  ``session.state`` and is never affected.
* :func:`remember_session` — an ``after_agent_callback`` that ingests the finished
  turn into the ``MemoryService`` so facts the user revealed are recallable in
  future sessions. It runs after the user-facing reply is produced.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence

from google.genai import types

logger = logging.getLogger(__name__)

# Session-scoped (no ``user:`` prefix) so the rolling summary of *this* conversation
# does not leak into other sessions the way the durable profile/pantry does.
HISTORY_SUMMARY_KEY = "history_summary"

DEFAULT_WINDOW = 12

# A summarizer maps (dropped_turns, prior_summary) -> new summary text.
Summarizer = Callable[[Sequence[types.Content], str | None], str | None]

# Sentinel so ``summarize_fn=None`` (an explicit "no summariser, just trim") is
# distinguishable from "argument not supplied, use the module default".
_UNSET = object()


def _window() -> int:
    """History window size, overridable via ``SOUS_HISTORY_WINDOW``."""
    raw = os.environ.get("SOUS_HISTORY_WINDOW")
    try:
        return int(raw) if raw else DEFAULT_WINDOW
    except ValueError:
        return DEFAULT_WINDOW


def _summary_content(text: str) -> types.Content:
    """Wrap a running summary as a single synthetic user turn."""
    return types.Content(
        role="user",
        parts=[types.Part(text=f"[Summary of earlier conversation]\n{text}")],
    )


def _has_function_response(content: types.Content) -> bool:
    return any(getattr(p, "function_response", None) is not None for p in (content.parts or []))


def _clean_window_start(
    kept: list[types.Content], *, require_user_start: bool
) -> list[types.Content]:
    """Advance the window start to a boundary the model will accept.

    Slicing the history at an arbitrary index can leave the window starting on a
    ``function_response`` whose originating ``function_call`` was trimmed away —
    Gemini rejects that ("function response without preceding function call"). Drop
    such orphaned responses; when ``require_user_start`` is set (pure trim, no summary
    prepended), also drop any leading ``model`` turns so the history opens on a user
    turn. Dropping a leading ``model`` function-call turn orphans its response, which
    the loop then drops in turn — so cascades resolve naturally.
    """
    start = 0
    while start < len(kept):
        content = kept[start]
        if _has_function_response(content):
            start += 1
            continue
        if require_user_start and content.role != "user":
            start += 1
            continue
        break
    return kept[start:]


def compact_history(
    callback_context,
    llm_request,
    *,
    window: int | None = None,
    summarize_fn: Summarizer | None | object = _UNSET,
):
    """Trim the coordinator's history to a sliding window before each model call.

    Keeps the most recent ``window`` turns verbatim. When a summariser is
    configured, the dropped prefix is condensed into a running summary (folded in
    with any prior summary) and prepended as one synthetic turn; otherwise the
    prefix is simply dropped. Returns ``None`` so the model call proceeds.

    ``window`` and ``summarize_fn`` are injectable for tests; ADK calls the
    callback with just ``(callback_context, llm_request)``, in which case the
    window comes from the environment and the module default summariser is used.
    """
    win = window if window is not None else _window()
    contents = list(llm_request.contents or [])
    if len(contents) <= win:
        return None  # nothing to compact

    if summarize_fn is _UNSET:
        summarize_fn = _default_summarizer
    keep = contents[-win:]
    dropped = contents[:-win]

    if summarize_fn is not None:
        prior = callback_context.state.get(HISTORY_SUMMARY_KEY)
        try:
            summary = summarize_fn(dropped, prior)
        except Exception:  # a summariser failure must never break the turn
            logger.warning("history summariser failed; falling back to plain trim", exc_info=True)
            summary = None
        if summary:
            callback_context.state[HISTORY_SUMMARY_KEY] = summary
            # The prepended summary provides the user-role start, so we only need to
            # strip orphaned function responses from the kept window.
            body = _clean_window_start(keep, require_user_start=False)
            llm_request.contents = [_summary_content(summary), *(body or keep)]
            return None

    # Pure sliding window: also ensure the history opens on a user turn. Fall back to
    # the raw window if sanitising somehow empties it (better than no contents at all).
    llm_request.contents = _clean_window_start(keep, require_user_start=True) or keep
    return None


def _default_summarizer(
    dropped: Sequence[types.Content], prior: str | None
) -> str | None:
    """Default running-summary generator.

    Kept intentionally dependency-light: summarisation via a live LLM call is opt-in
    (production wires a Gemini-backed summariser). When unset, compaction degrades to
    a pure sliding window, which is the safe, offline-friendly default.
    """
    return None


async def remember_session(callback_context) -> None:
    """Ingest the finished turn into long-term memory (``after_agent_callback``).

    Runs after the user-facing reply is produced, so memory generation is off the
    response critical path. Fails soft: if no ``MemoryService`` is wired, the turn
    still completes normally.
    """
    try:
        await callback_context.add_session_to_memory()
    except ValueError:
        # No memory service configured (e.g. local run without one) — skip quietly.
        logger.debug("no memory service available; skipping session ingestion")
    except Exception:
        logger.warning("failed to ingest session into memory", exc_info=True)
    return None
