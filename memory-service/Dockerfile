FROM python:3.12-slim AS base

ARG EMBED_MODEL=snowflake/snowflake-arctic-embed-xs

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    EMBED_MODEL=${EMBED_MODEL} \
    EMBED_CACHE_DIR=/opt/models \
    DB_PATH=/data/memory.db \
    OMP_NUM_THREADS=2 \
    EMBED_THREADS=2

WORKDIR /app

COPY requirements.txt requirements.lock.txt ./
RUN pip install -r requirements.lock.txt

# Bake the embedding model into the image so runtime needs no network.
RUN python -c "import os; from fastembed import TextEmbedding; TextEmbedding(os.environ['EMBED_MODEL'], cache_dir='/opt/models')" \
    && chmod -R a+rX /opt/models \
    && du -sh /opt/models

COPY memory_service ./memory_service

# `docker build --target test` runs the unit tests (synthetic data only).
FROM base AS test
COPY requirements-dev.txt ./
RUN pip install -r requirements-dev.txt
COPY tests ./tests
RUN python -m pytest -q tests

FROM base AS runtime

RUN useradd --create-home --uid 10002 memsvc \
    && mkdir -p /data \
    && chown -R memsvc:memsvc /data /app
USER memsvc

VOLUME ["/data"]
EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8765/healthz', timeout=3).status==200 else 1)"

CMD ["python", "-m", "memory_service", "serve"]
