FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    OKFOLIO_PROJECT_ROOT=/app \
    DATA_DIR=/app/runtime/data \
    PROMPTS_DIR=/app/prompts \
    RELEASES_DIR=/app/runtime/releases \
    MCP_ENABLE_WRITES=false \
    MCP_ENABLE_DOCKER=false \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=3099 \
    DATA_ASSET_MODE=local \
    MINERU_COMMAND=mineru

WORKDIR /app

RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' \
      /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates \
      poppler-utils \
    && rm -rf /var/lib/apt/lists/*

ENV SOURCES_DIR=/app/runtime/data/normalized-sources

COPY requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir -r /app/requirements.lock

COPY okfolio/ /app/okfolio/
COPY scripts/ /app/scripts/
COPY prompts/ /app/prompts/
COPY mkdocs.yml /app/mkdocs.yml
COPY VERSION /app/VERSION

RUN mkdir -p \
      /app/runtime/data/inbox \
      /app/runtime/data/parser-jobs \
      /app/runtime/data/processed \
      /app/runtime/data/normalized-sources/images \
      /app/runtime/releases


FROM base AS compiler

ENTRYPOINT ["bash", "/app/scripts/process_inbox.sh"]


FROM base AS release

ARG RELEASE_PATH
ENV DATA_DIR=/app/data \
    OKFOLIO_PORT=8080

COPY ${RELEASE_PATH}/data/ /app/data/
COPY ${RELEASE_PATH}/release-manifest.json /app/release-manifest.json
COPY ${RELEASE_PATH}/MANIFEST.sha256 /app/MANIFEST.sha256
COPY ${RELEASE_PATH}/acceptance.json /app/acceptance.json

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/graph.html', timeout=3)"

ENTRYPOINT ["python3", "/app/scripts/serve_release.py"]


FROM base AS service

EXPOSE 3099

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python3 -c "import socket; socket.create_connection(('127.0.0.1', 3099), 3).close()"

ENTRYPOINT ["python3", "-m", "okfolio.mcp.server"]
CMD ["--transport", "streamable-http", "--host", "0.0.0.0", "--port", "3099"]
