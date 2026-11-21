FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/models/sentence-transformers

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

COPY navigator/ ./navigator/
COPY config/ ./config/

RUN useradd --system --uid 10001 --create-home app \
    && mkdir -p /models && chown -R app:app /app /models
USER app

EXPOSE 8082 9092

ENTRYPOINT ["python", "-m", "navigator.main"]
CMD ["--config", "/etc/navigator/config.yaml"]
