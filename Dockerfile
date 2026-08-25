# Sous — Meal & Nutrition Concierge (ADK) container image.
# Serves the agent via `adk api_server` on Cloud Run.
FROM python:3.12-slim

# Install uv (fast, reproducible installs from the lockfile).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PORT=8080

WORKDIR /app

# Install dependencies first (better layer caching). The recipe dataset ships
# inside the package (src/sous/resources), so no separate data copy is needed.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

# ADK serves an OpenAPI/UI over HTTP. Cloud Run provides $PORT. `src` is the
# agents directory (the `sous` package is discovered as the agent) — matching the
# `adk web src` command in the README.
EXPOSE 8080
CMD ["sh", "-c", "uv run adk api_server --host 0.0.0.0 --port ${PORT} src"]
