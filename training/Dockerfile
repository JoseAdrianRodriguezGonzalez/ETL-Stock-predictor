FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Dependencias
COPY training/pyproject.toml training/uv.lock training/README.md ./

RUN uv sync --locked --no-dev --no-install-project

# Código fuente
COPY training/src ./src

# Configuraciones externas al paquete
COPY training/configs ./configs

# Instalación del proyecto
RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["training"]
CMD ["--config", "configs/lstm.yaml"]

