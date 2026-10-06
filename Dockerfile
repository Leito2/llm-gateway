FROM python:3.12-slim AS base
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv sync --no-dev
COPY config ./config
EXPOSE 8080
CMD ["uv", "run", "uvicorn", "llm_gateway.main:app", "--host", "0.0.0.0", "--port", "8080"]
