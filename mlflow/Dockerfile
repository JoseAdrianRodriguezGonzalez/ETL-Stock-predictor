ARG MLFLOW_VERSION=3.18.0

FROM python:3.12-slim

ARG MLFLOW_VERSION

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
    "mlflow==${MLFLOW_VERSION}" \
    "psycopg2-binary==2.9.10"

EXPOSE 5000

ENTRYPOINT ["mlflow"]
