ARG PYTHON_IMAGE=ghcr.io/astral-sh/uv:0.9.5-python3.12-bookworm-slim
FROM ${PYTHON_IMAGE} AS build
ENV UV_TOOL_DIR=/opt/hatch-tool UV_TOOL_BIN_DIR=/usr/local/bin
RUN uv tool install hatch==1.18.1
WORKDIR /opt/service
COPY pyproject.toml ./
RUN hatch env create runtime

FROM ${PYTHON_IMAGE}
ENV PATH=/opt/service/.venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 SQLITE_PATH=/data/memory.sqlite3
WORKDIR /opt/service
COPY --from=build /opt/service/.venv /opt/service/.venv
COPY memory_service ./memory_service
COPY deployment ./deployment
COPY LICENSE NOTICE ./
RUN useradd --uid 10001 --create-home memory && mkdir -p /data && chown memory:memory /data
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD ["python", "-c", "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health',timeout=3).close()"]
CMD ["python", "-m", "deployment.server"]
