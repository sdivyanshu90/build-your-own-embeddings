FROM python:3.11-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /build
COPY pyproject.toml ./
COPY constraints ./constraints
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade "pip==25.3" \
    && /opt/venv/bin/pip install "hatchling==1.27.0" \
    && /opt/venv/bin/pip install \
      --index-url https://download.pytorch.org/whl/cpu "torch==2.7.1" \
    && PIP_CONSTRAINT=/build/constraints/runtime.txt \
      /opt/venv/bin/pip install \
        faiss-cpu fastapi httpx numpy pandas prometheus-client pydantic \
        pydantic-settings PyYAML safetensors scikit-learn typer uvicorn
COPY README.md LICENSE ./
COPY src ./src
RUN PIP_CONSTRAINT=/build/constraints/runtime.txt \
    /opt/venv/bin/pip wheel \
      --no-build-isolation --no-deps --wheel-dir /wheels .

FROM python:3.11-slim-bookworm AS runtime

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/models/model \
    INDEX_PATH=/models/index
RUN groupadd --system --gid 10001 embedding \
    && useradd --system --uid 10001 --gid embedding --home /nonexistent embedding
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /wheels /tmp/wheels
RUN /opt/venv/bin/pip install --no-index --no-deps /tmp/wheels/*.whl \
    && rm -r /tmp/wheels
COPY docker/entrypoint.sh /usr/local/bin/embedding-entrypoint
RUN chmod 0555 /usr/local/bin/embedding-entrypoint
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2)"]
ENTRYPOINT ["/usr/local/bin/embedding-entrypoint"]
