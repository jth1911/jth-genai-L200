# Sous — Meal & Nutrition Concierge

A multi-agent meal & nutrition concierge built on Google's ADK. It turns health
goals, dietary constraints and your current pantry into an adaptive weekly meal
plan and a grocery list.

- **[Architecture](architecture.md)** — agents, orchestration, memory, tools.
- **Repository README** — setup, run, test, deploy instructions.

## Quick start

```bash
uv sync --extra dev
cp .env.example .env    # add your Google AI Studio key
uv run adk web src      # chat UI with a trace view
```

Then ask: *"Plan me 5 high-protein dinners for the week, no shellfish, around
$60, using what I already have."*
