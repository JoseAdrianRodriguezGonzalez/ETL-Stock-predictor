FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Primero copiamos los archivos de dependencias
COPY training/pyproject.toml training/uv.lock training/README.md ./

# Instalamos las dependencias de forma reproducible
RUN uv sync --locked --no-dev --no-install-project

# Copiamos el código fuente
COPY training/src ./src

# Instalamos el proyecto
RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"

CMD ["training"]
