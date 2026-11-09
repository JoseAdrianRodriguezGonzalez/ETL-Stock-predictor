FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY deployment/pyproject.toml deployment/uv.lock deployment/README.md ./

RUN uv sync --locked --no-dev --no-install-project

COPY deployment/src ./src

RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"

CMD ["deployment"]
