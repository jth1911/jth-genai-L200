"""Phase 5 — ADK evaluation (LLM-gated).

These call a live Gemini model, so they are marked ``llm`` and skipped unless
``GOOGLE_API_KEY`` is set. Run them explicitly with:

    uv run pytest -m llm

The eval set + thresholds live in ``src/sous/eval/`` and can also be run with the
ADK CLI:  ``adk eval sous src/sous/eval/pantry_smoke.evalset.json``
"""

import os
from pathlib import Path

import pytest
from google.adk.evaluation.agent_evaluator import AgentEvaluator

EVAL_DIR = Path(__file__).resolve().parents[1] / "src" / "sous" / "eval"

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(
        not os.environ.get("GOOGLE_API_KEY"),
        reason="GOOGLE_API_KEY not set; skipping live-LLM eval.",
    ),
]


async def test_pantry_smoke_evalset():
    await AgentEvaluator.evaluate(
        agent_module="sous",
        eval_dataset_file_path_or_dir=str(EVAL_DIR / "pantry_smoke.evalset.json"),
        num_runs=2,
    )
