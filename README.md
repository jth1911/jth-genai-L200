# Sous — Meal & Nutrition Concierge

A multi-agent **Meal & Nutrition Concierge** built on [Google's Agent Development
Kit (ADK)](https://adk.dev/). Sous turns a person's health goals, dietary
constraints and current pantry into an adaptive weekly meal plan and a
ready-to-use grocery list.

> L200 Agent project — Concierge track, Lifestyle/Health domain. See [issue #1](https://github.com/jth1911/jth-genai-L200/issues/1).

## Why

Eating well is a recurring, multi-constraint planning problem: calorie/macro
goals, allergies and diet style, budget, cook time, and *what's already in the
fridge*. Doing it by hand is tedious, so people default to repetitive meals,
takeout and food waste. A coordinated multi-agent system with persistent memory
holds all of that together across a real week.

## Architecture

```
root_agent  (sous_coordinator, LlmAgent — routing + memory; manages pantry directly)
  └─ delegates to → plan_workflow  (SequentialAgent)
        ├─ gather_step  (ParallelAgent)
        │     ├─ nutrition_agent  → state["nutrition_targets"]
        │     └─ pantry_agent     → state["pantry_summary"]
        ├─ recipe_agent   → state["recipe_plan"]
        └─ grocery_agent  → state["grocery_list"]
```

Two orchestration styles in one system:

- **LLM-driven delegation** — the coordinator decides when to hand off to the workflow.
- **Deterministic workflow** — `Sequential(Parallel(nutrition, pantry), recipe, grocery)`
  runs a fixed pipeline, each stage passing results to the next via session state
  and `{key}` instruction templating.

**Memory:** the pantry and dietary profile are stored under `user:`-scoped session
state, so with the SQLite-backed `DatabaseSessionService` they persist across
separate conversations.

See [`docs/architecture.md`](docs/architecture.md) for the full design.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync --extra dev
cp .env.example .env   # then add your Google AI Studio key
```

Get a Gemini API key at <https://aistudio.google.com/apikey> and set it in `.env`:

```
GOOGLE_GENAI_USE_VERTEXAI=FALSE
GOOGLE_API_KEY=your-key
```

## Run

```bash
# Interactive web UI (chat + trace view) — `src` is the agents directory:
uv run adk web src

# Or an HTTP API server (this is exactly what the container serves):
uv run adk api_server src

# Or a terminal REPL against the single agent:
uv run adk run src/sous
```

Try: *"Plan me 5 high-protein dinners for the week, no shellfish, around $60,
using what I already have."* Then follow up: *"I don't have chicken anymore."*

## Test

```bash
uv run pytest -m "not llm"   # fast unit tests, no API key / network
uv run pytest -m llm         # live-LLM evals (needs GOOGLE_API_KEY)
uv run ruff check src tests  # lint
```

## Evaluate

The eval set and thresholds live in [`src/sous/eval/`](src/sous/eval/):

```bash
uv run adk eval src/sous src/sous/eval/pantry_smoke.evalset.json
```

## Observability

ADK emits OpenTelemetry traces of the full delegation path and every tool call.
- `adk web` includes a **Trace** tab showing each agent hop and tool invocation.
- Add `--trace_to_cloud` to `adk web`/`adk api_server` to export traces to Google
  Cloud Trace when deployed.

## Deploy

Containerised for Cloud Run:

```bash
docker build -t sous .
# Locally:
docker run -p 8080:8080 --env-file .env sous
# Cloud Run (ADK helper):
uv run adk deploy cloud_run src/sous
```

For persistent memory in production, set `SOUS_SESSION_DB` to a database URL
instead of the default in-memory store. SQLite and PostgreSQL (e.g. Cloud SQL)
are supported out of the box — plain `sqlite:///...` and `postgresql://...` URLs
are automatically upgraded to their async drivers (`aiosqlite` / `asyncpg`):

```
SOUS_SESSION_DB=postgresql://user:pass@host:5432/sous
```

## Project layout

```
src/sous/
  data.py                  # recipe dataset loader (dataclasses)
  tools.py                 # function tools: search, nutrition, pantry, grocery list
  agent.py                 # agents + orchestration (root_agent)
  runtime.py               # session service + Runner wiring
  resources/recipes.json   # local, curated recipe/nutrition dataset (shipped in-package)
  eval/                    # ADK eval set + criteria
tests/                     # TDD suite (data, tools, agents, state, eval)
```
